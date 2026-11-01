"""Формат входа модели и проверка длины."""

import json

INSTRUCTION = (
    "Переведи предложение в одну формулу JSON из списков, без пояснений.\n"
    'Узлы: ["pred",символ,терм,...], ["not",F], ["and",F,G], ["or",F,G], '
    '["implies",F,G], ["forall",переменная,F], ["exists",переменная,F].\n'
    'Терм: ["var",переменная] или ["const",идентификатор имени]. '
    "Переменная действующего участника x, второго участника y. "
    "У отношения первый аргумент — действующий, второй — тот, на кого направлено действие.\n"
    "Область непуста. Классы могут быть пустыми и пересекаться; имена могут совпадать. "
    "«Каждый» не утверждает существование. «Ни один ... не» выражает одно отрицание. "
    "«Одну и ту же» означает существование общего объекта без требования единственности."
)


DEFAULT_LIMITS = {"max_input_tokens": 1152, "max_target_tokens": 128}


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tokenize_input(text, tokenizer, max_tokens):
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(ids) > max_tokens:
        raise ValueError(f"Вход не помещается целиком: {len(ids)} > {max_tokens}; обрезание запрещено")
    if tokenizer.unk_token_id in ids:
        raise ValueError("Токенизатор теряет символ входа")
    return ids
