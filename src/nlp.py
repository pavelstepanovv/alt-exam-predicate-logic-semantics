"""Разбор предложения и локальный кэш."""

from copy import deepcopy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re

PARAMETERS = {"schema_version": 1, "normalization": "none", "segment": "razdel_default",
              "pipeline": ["segment", "morph", "lemma", "syntax"], "attempts_max": 2,
              "graph_policy": "one_root_and_acyclic_or_missing",
              "missing_marker": "<SYNTAX_UNAVAILABLE>"}


TOKEN_FIELDS = {"id", "start", "stop", "text", "lemma", "pos", "feats", "head_id", "rel"}


POS_TAGS = set("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_id(text, identity, parameters=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Для разбора требуется непустой текст")
    return hashlib.sha256(_json({"text": text, "identity": identity,
                                "parameters": PARAMETERS if parameters is None else parameters}).encode("utf-8")).hexdigest()


def validate_parse(record):
    """Структурная проверка. Не исправляет леммы, падежи или зависимости."""
    def require(ok, message):
        if not ok:
            raise ValueError(message)

    text, tokens, sentences = record["text"], record["tokens"], record["sentences"]
    require(isinstance(text, str) and bool(text.strip()), "Пустой исходный текст")
    if record["status"] == "missing":
        require(tokens == [] and sentences == [] and record["missing_marker"] == PARAMETERS["missing_marker"], "Неверная запись отсутствующего разбора")
        require(record["attempts"] == 2 and bool(record["error"]["type"]), "Не зафиксированы попытки и сбой")
        return
    require(record["status"] == "ok" and bool(tokens) and bool(sentences), "Нет слов или предложений")
    require(len({t["id"] for t in tokens}) == len(tokens), "Повторяются номера слов")
    previous = 0
    covered = set()
    for token in tokens:
        require(set(token) == TOKEN_FIELDS, "Лишние или отсутствующие поля слова")
        start, stop = token["start"], token["stop"]
        require(type(start) is int and type(stop) is int and previous <= start < stop <= len(text), "Неверные границы слова")
        require(text[start:stop] == token["text"], "Слово не совпадает с исходным текстом")
        require(not text[previous:start].strip(), "Пропущен фрагмент исходного текста")
        previous = stop
        require(isinstance(token["lemma"], str) and bool(token["lemma"]), "Нет начальной формы слова")
        require(token["pos"] in POS_TAGS, "Неизвестная часть речи")
        require(isinstance(token["feats"], dict) and all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in token["feats"].items()), "Некорректные грамматические признаки")
        require(isinstance(token["rel"], str) and bool(re.fullmatch(r"[a-z]+(?::[a-z]+)?", token["rel"])), "Нет типа связи")
    require(not text[previous:].strip(), "Пропущен конец текста")
    by_id = {t["id"]: t for t in tokens}
    previous = 0
    for number, sentence in enumerate(sentences, 1):
        require(set(sentence) == {"id", "start", "stop", "token_ids"}, "Неверные поля предложения")
        require(sentence["id"] == number and previous <= sentence["start"] < sentence["stop"] <= len(text), "Неверные границы предложения")
        previous = sentence["stop"]
        ids = sentence["token_ids"]
        require(ids == [f"{number}_{i}" for i in range(1, len(ids)+1)] and bool(ids), "Неверная нумерация слов предложения")
        require(not covered.intersection(ids) and all(i in by_id for i in ids), "Неверный состав предложения")
        covered.update(ids)
        root = f"{number}_0"
        require(sum(by_id[i]["head_id"] == root for i in ids) == 1, "Должна быть одна главная вершина предложения")
        for identifier in ids:
            token = by_id[identifier]
            require(sentence["start"] <= token["start"] < token["stop"] <= sentence["stop"], "Слово вне предложения")
            require(token["head_id"] in ids or token["head_id"] == root, "Ссылка на отсутствующее слово или другое предложение")
            require((token["rel"] == "root") == (token["head_id"] == root), "Не согласована главная вершина")
            chain, current = set(), identifier
            while current != root:
                require(current not in chain, "Циклическая связь между словами")
                require(current in by_id and current in ids, "Неверная цепочка связей")
                chain.add(current)
                current = by_id[current]["head_id"]
    require(covered == set(by_id), "Не все слова включены в предложения")


class NatashaAnalyzer:
    def __init__(self):
        # Импорты остаются здесь: обычные CPU-тесты работают без Natasha.
        from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, MorphVocab
        from natasha.data import NEWS_EMBEDDING, NEWS_MORPH, NEWS_SYNTAX
        dependencies = ("natasha", "razdel", "slovnet", "navec", "pymorphy2", "pymorphy2-dicts-ru", "numpy")
        self.identity = {"implementation": "ru-natasha-v2", "packages": {name: metadata.version(name) for name in dependencies},
                         "models_sha256": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                           for p in (NEWS_EMBEDDING, NEWS_MORPH, NEWS_SYNTAX)}}
        self.segmenter = Segmenter()
        embedding = NewsEmbedding()
        self.morph = NewsMorphTagger(embedding)
        self.syntax = NewsSyntaxParser(embedding)
        self.vocab = MorphVocab()

    def analyze(self, text):
        from natasha import Doc
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph)
        for token in doc.tokens:
            token.lemmatize(self.vocab)
        doc.parse_syntax(self.syntax)
        return {"tokens": [{key: deepcopy(getattr(token, key)) for key in TOKEN_FIELDS} for token in doc.tokens],
                "sentences": [{"id": i, "start": sent.start, "stop": sent.stop, "token_ids": [t.id for t in sent.tokens]}
                              for i, sent in enumerate(doc.sents, 1)]}


class CachedParser:
    def __init__(self, analyzer, directory):
        self.analyzer, self.directory = analyzer, Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits, self.computed = 0, 0

    def parse_record(self, row):
        return self.parse_text(row["text"])

    def parse_text(self, text):
        identity = self.analyzer.identity
        identifier = cache_id(text, identity)
        path = self.directory / (identifier + ".json")
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("cache_id") != identifier or record.get("text") != text or record.get("identity") != identity or record.get("parameters") != PARAMETERS:
                raise ValueError("Сохранённый разбор не соответствует тексту или версии анализатора")
            validate_parse(record)
            self.hits += 1
            return record
        record = {"schema_version": 1, "cache_id": identifier, "text": text,
                  "identity": deepcopy(identity), "parameters": deepcopy(PARAMETERS)}
        for attempt in (1, 2):
            body = None
            try:
                body = self.analyzer.analyze(text)
                candidate = dict(record, status="ok", tokens=body["tokens"], sentences=body["sentences"], attempts=attempt)
                validate_parse(candidate)
                record = candidate
                break
            except Exception as exc:
                if attempt == 2:
                    record.update(status="missing", tokens=[], sentences=[], attempts=2,
                                  missing_marker=PARAMETERS["missing_marker"],
                                  error={"type": type(exc).__name__, "message": str(exc)[:300],
                                         "stage": "structure_validation" if body is not None else "analyzer"})
                    if body is not None:
                        record["raw_analysis"] = {key: deepcopy(body[key]) for key in ("tokens", "sentences") if key in body}
                    validate_parse(record)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(_json(record) + "\n", encoding="utf-8")
        temporary.replace(path)
        self.computed += 1
        return record
