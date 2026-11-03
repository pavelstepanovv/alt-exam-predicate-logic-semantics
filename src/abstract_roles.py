"""Более явные связи рядом со словами; логическая формула здесь не строится."""

from .abstraction import abstract_input as base_input


VERSION = "abstract-roles-v2"
TABLE_HEADER = "\nРазбор (id|слово|лемма|POS|признаки|head|relation):\n"
CASES = {"Nom": "им", "Acc": "вин", "Gen": "род", "Dat": "дат", "Ins": "твор", "Loc": "предл", "Voc": "зват"}
RELATIONS = {"root": "главное", "nsubj": "подлежащее", "obj": "дополнение", "iobj": "косвенное-дополнение",
             "conj": "сочинение", "cc": "союз", "acl:relcl": "уточнение", "acl": "уточнение",
             "det": "определитель", "nummod": "количество", "advmod": "обстоятельство", "cop": "связка"}


def abstract_input(text, analysis, lexicon, *, seed=42):
    context = base_input(text, analysis, lexicon, seed=seed)
    # Это собственная фиксированная таблица v1, уже без исходных названий.
    # Её слова и номера связей согласованы с таблицей обратного переименования.
    prefix, table = context["input"].split(TABLE_HEADER, 1)
    prefix = prefix.split("\nПредложение:\n", 1)[0]
    rows = {}
    for line in table.splitlines():
        if line.startswith("SYNTAX_UNAVAILABLE"):
            continue
        fields = line.split("|")
        if len(fields) != 7:
            raise ValueError("Изменился внутренний формат обобщённых слов")
        identifier, word, lemma, pos, feats, head, relation = fields
        rows[identifier] = {"word": word, "pos": pos, "feats": dict(pair.split("=", 1) for pair in feats.split(",") if "=" in pair),
                            "head": head, "relation": relation}
    source = analysis if analysis["status"] == "ok" else analysis.get("raw_analysis", {})
    tokens = source.get("tokens", [])
    pieces, previous, annotations = [], 0, {}
    for token in tokens:
        row = rows.get(token["id"])
        word = token["text"] if row is None else row["word"]
        if row is not None:
            # Связь стоит первой, прямо возле обозначения; подробные признаки
            # сохраняются следом, включая возможные неточности анализатора.
            if analysis["status"] == "ok":
                head = rows.get(row["head"])
                relation = RELATIONS.get(row["relation"], row["relation"])
                role = relation + ("→" + row["head"] + ":" + head["word"] if head else "→0")
            else:
                role = "связь=недоступна"
            features = [role, token["id"], row["pos"]]
            case = row["feats"].get("Case")
            if case:
                features.append("падеж=" + CASES.get(case, case))
            features.extend(key + "=" + value for key, value in sorted(row["feats"].items()) if key != "Case")
            annotation = ";".join(features)
            annotations[token["id"]] = annotation
            word += "{" + annotation + "}"
        pieces.extend((text[previous:token["start"]], word))
        previous = token["stop"]
    pieces.append(text[previous:])
    inline_text = "".join(pieces)
    missing = "\nSYNTAX_UNAVAILABLE: связи недоступны, указаны только признаки слов." if analysis["status"] != "ok" else ""
    prompt = (prefix + "\nВ фигурных скобках — признаки и связи из разбора. "
              "им — именительный падеж, вин — винительный. Подлежащее и дополнение определяются связями, а не порядком слов. "
              "Уточнение указывает на определяемое слово; сочинение соединяет однородные части. "
              "Эти отметки не являются частью обозначений предикатов и имён в ответе."
              + "\nПредложение со связями:\n" + inline_text + missing)
    return {**context, "input": prompt, "representation": VERSION, "inline_text": inline_text,
            "inline_annotations": annotations}
