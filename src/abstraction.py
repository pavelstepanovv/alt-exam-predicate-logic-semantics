"""Замена слов обозначениями и обратное восстановление названий."""

from copy import deepcopy
import hashlib
import random

from .open_vocabulary import describe_words
from .serialization import INSTRUCTION, compact_json
from .logic import validate_formula

VERSION = "abstract-words-v1"


SLOTS = 8


PREFIX = {"class": "C", "property": "P", "relation": "R", "entity": "a"}


def abstract_input(text, analysis, lexicon, *, seed=42):
    """Таблица создаётся до чтения цели. Номера не кодируют порядок ролей."""
    classified = deepcopy(analysis)
    classified_source = classified if classified["status"] == "ok" else classified.get("raw_analysis", {})
    # Незнакомое подлежащее/дополнение с определителем остаётся названием
    # класса даже при противоречивой отметке части речи (например ADV).
    classified_tokens = classified_source.get("tokens", [])
    for token in classified_tokens:
        if token["pos"] not in ("NOUN", "PROPN") and token["rel"] in ("nsubj", "obj") and any(
                t["head_id"] == token["id"] and t["rel"] in ("det", "nummod")
                and t["lemma"].casefold() in ("каждый", "весь", "один", "некоторый", "любой", "всякий")
                for t in classified_tokens):
            token["pos"] = "NOUN"
    description = describe_words(text, classified, lexicon)
    if description["unresolved"]:
        raise ValueError("Не определена роль слов: " + ", ".join(description["unresolved"]))
    source = analysis if analysis["status"] == "ok" else analysis.get("raw_analysis", {})
    tokens, terms = source.get("tokens", []), deepcopy(description["terms"])
    original_lexicon = deepcopy(description["lexicon"])
    entities = dict(description["row"]["entities"])
    # Новые имена связываются по начальной форме, а не по падежу в предложении.
    for token in tokens:
        term = terms.get(token["id"])
        if term and term["kind"] == "entity" and term["symbol"] not in lexicon["entities"]:
            old = term["symbol"]
            identifier = "name_" + token["lemma"].casefold().replace("ё", "е")
            label = token["lemma"][:1].upper() + token["lemma"][1:]
            entities.pop(old, None)
            entities[identifier] = label
            term["symbol"] = identifier
    groups = {kind: sorted({t["symbol"] for t in terms.values() if t["kind"] == kind}) for kind in PREFIX}
    signature = compact_json(groups)
    rng = random.Random(hashlib.sha256((str(seed) + signature).encode("utf-8")).hexdigest())
    mapping = {}
    for kind, values in groups.items():
        if len(values) > SLOTS:
            raise ValueError("В предложении больше восьми разных обозначений одного вида")
        for original, number in zip(values, rng.sample(range(1, SLOTS + 1), len(values))):
            mapping[original] = PREFIX[kind] + str(number)
    inverse = {value: key for key, value in mapping.items()}
    predicates = {}
    for kind, symbols in groups.items():
        if kind != "entity":
            for symbol in symbols:
                predicates[mapping[symbol]] = {"kind": kind, "arity": 2 if kind == "relation" else 1,
                                                "lemmas": [mapping[symbol]], "forms": {}}
    names = {mapping[i]: mapping[i] for i in groups["entity"]}
    abstract_lexicon = {"predicates": predicates, "entities": {i: {"name": i} for i in names}}
    pieces, previous, lines = [], 0, []
    for token in tokens:
        term = terms.get(token["id"])
        word = mapping[term["symbol"]] if term else token["text"]
        pieces.extend((text[previous:token["start"]], word))
        previous = token["stop"]
        if token["pos"] == "PUNCT":
            continue
        lemma = word if term else token["lemma"]
        feats = ",".join(k + "=" + v for k, v in sorted(token["feats"].items())) or "_"
        # При отсутствующем дереве используем только морфологию, без плохих связей.
        head, rel = (token["head_id"], token["rel"]) if analysis["status"] == "ok" else ("_", "_")
        lines.append("|".join((token["id"], word, lemma, token["pos"], feats, head, rel)))
    pieces.append(text[previous:])
    normalized = "".join(pieces)
    catalog = ";".join(f"{s}/{e['arity']}:{e['kind']}" for s, e in sorted(predicates.items()))
    prompt = (INSTRUCTION + "\nC — класс, P — свойство одного участника, R — отношение двух участников, a — имя. "
              "Номера произвольны и не обозначают роли. Используй только перечисленные обозначения.\n"
              + "Обозначения:\n" + catalog + "\nИмена:\n" + compact_json(names)
              + "\nПредложение:\n" + normalized + "\nРазбор (id|слово|лемма|POS|признаки|head|relation):\n"
              + ("SYNTAX_UNAVAILABLE; только морфология\n" if analysis["status"] != "ok" else "") + "\n".join(lines))
    return {"input": prompt, "normalized_text": normalized, "mapping": mapping, "inverse": inverse,
            "lexicon": abstract_lexicon, "entities": names, "original_lexicon": original_lexicon,
            "original_entities": entities, "analysis_status": analysis["status"], "seed": seed}


def rename_formula(formula, mapping):
    """Только переименование предикатов/констант, без изменения структуры и ролей."""
    result = deepcopy(formula)

    def visit(node):
        if node[0] == "pred":
            node[1] = mapping[node[1]]
            for term in node[2:]:
                if term[0] == "const":
                    term[1] = mapping[term[1]]
        elif node[0] in ("forall", "exists"):
            visit(node[2])
        else:
            for child in node[1:]:
                visit(child)
    visit(result)
    return result
