"""Определение классов, свойств, отношений и имён по разбору."""

from copy import deepcopy
import hashlib

from .text import FUNCTION_WORDS, strings, words

def describe_words(text, analysis, base_lexicon):
    if analysis["text"] != text:
        raise ValueError("Разбор относится к другому тексту")
    tokens = analysis["tokens"] if analysis["status"] == "ok" else analysis.get("raw_analysis", {}).get("tokens", [])
    if not tokens:
        raise ValueError("Нет слов для выделения обозначений")
    lexicon = deepcopy(base_lexicon)
    by_id = {t["id"]: t for t in tokens}
    children = {t["id"]: [c for c in tokens if c["head_id"] == t["id"] and c["id"] != t["id"]] for t in tokens}
    aliases = {}
    for symbol, entry in base_lexicon["predicates"].items():
        for form in strings([entry.get("forms", {}), entry["lemmas"], symbol]):
            aliases.setdefault(form.casefold(), set()).add(symbol)
    names = {form.casefold(): identifier for identifier, entry in base_lexicon["entities"].items()
             for form in strings(entry["forms"])}
    terms, entities, additions, assumptions, unresolved = {}, {}, [], [], []
    noun_aliases, predicate_aliases = {}, {}
    for token in tokens:
        raw, lemma = token["text"], token["lemma"]
        if token["pos"] == "PUNCT" or raw.casefold() in FUNCTION_WORDS or lemma.casefold() in FUNCTION_WORDS:
            continue
        named = names.get(raw.casefold())
        if named is not None:
            entities[named] = lexicon["entities"][named]["name"]
            terms[token["id"]] = {"symbol": named, "kind": "entity", "text": raw}
            continue
        candidates = aliases.get(raw.casefold(), set()) | aliases.get(lemma.casefold(), set())
        if len(candidates) == 1:
            symbol = next(iter(candidates))
            entry = lexicon["predicates"][symbol]
            terms[token["id"]] = {"symbol": symbol, "kind": entry["kind"], "text": raw}
            continue
        quantified = any(c["rel"] in ("det", "nummod") and c["lemma"].casefold() in
                         ("каждый", "весь", "один", "некоторый", "любой", "всякий") for c in children[token["id"]])
        nominal_predicate = token["pos"] == "NOUN" and any(c["rel"] == "nsubj" for c in children[token["id"]])
        if token["pos"] in ("NOUN", "PROPN") and not quantified and not nominal_predicate:
            # Без определителя существительное трактуется как имя, явно отмечая
            # это соглашение; начальную форму нового имени не выдумываем.
            identifier = "u" + hashlib.sha256(raw.casefold().encode()).hexdigest()[:8]
            name = raw[:1].upper() + raw[1:]
            lexicon["entities"][identifier] = {"name": name, "forms": {"nom": raw, "acc": raw}}
            entities[identifier] = name
            terms[token["id"]] = {"symbol": identifier, "kind": "entity", "text": raw}
            additions.append({"word": raw, "symbol": identifier, "kind": "entity", "label": name})
            if token["pos"] != "PROPN":
                assumptions.append(f"«{raw}» без названного квантора считается именем конкретного объекта.")
            continue
        if token["pos"] in ("NOUN", "PROPN"):
            kind, arity = "class", 1
            symbol = noun_aliases.setdefault(lemma.casefold(), raw[:1].upper() + raw[1:])
        elif token["pos"] in ("VERB", "ADJ"):
            inherited = by_id.get(token["head_id"]) if token["rel"] == "conj" else None
            has_object = any(c["rel"] == "obj" for c in children[token["id"]])
            if inherited:
                has_object |= any(c["rel"] == "obj" for c in children[inherited["id"]])
            arity = 2 if token["pos"] == "VERB" and has_object else 1
            kind = "relation" if arity == 2 else "property"
            symbol = predicate_aliases.setdefault((lemma.casefold(), arity), lemma[:1].upper() + lemma[1:])
        else:
            unresolved.append(raw)
            continue
        if symbol in lexicon["predicates"]:
            if lexicon["predicates"][symbol]["arity"] != arity:
                raise ValueError("Неоднозначное число участников нового предиката")
        else:
            lexicon["predicates"][symbol] = {"arity": arity, "kind": kind, "lemmas": [lemma, raw.casefold()],
                                              "forms": {"observed": [raw]}}
            additions.append({"word": raw, "symbol": symbol, "kind": kind, "arity": arity, "label": symbol})
        terms[token["id"]] = {"symbol": symbol, "kind": kind, "text": raw}
    if any(t["lemma"].casefold() == "весь" and t["pos"] == "PRON"
           and not (t["rel"] == "det" and terms.get(t["head_id"], {}).get("kind") == "class") for t in tokens):
        assumptions.append("«Все» без названия класса означает все объекты рассматриваемой области.")
    return {"row": {"id": "probe", "text": text, "entities": dict(sorted(entities.items()))},
            "lexicon": lexicon, "terms": terms, "additions": additions,
            "assumptions": assumptions, "unresolved": unresolved}
