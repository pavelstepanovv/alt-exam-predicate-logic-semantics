"""Обработка введённого текста."""

import re

FUNCTION_WORDS = set("каждый каждая каждое каждого каждую каждых все всех всякий всякая любой любая "
                     "не ни один одна одно одного одну одной одни нет существует существуют хотя бы "
                     "по крайней мере который которая которое которые которого которую и или "
                     "одновременно является являются одну ту же одного того тот та то".split())


WORDS = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


def words(text):
    return WORDS.findall(text.casefold())


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def normalize_sentence(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Введите предложение.")
    text = " ".join(text.split())
    if len(text) > 400:
        raise ValueError("Введите одно предложение длиной до 400 символов.")
    if re.search(r"[.!?]\s*\S", text):
        raise ValueError("Вводите по одному предложению.")
    if not text.endswith((".", "!", "?")):
        text += "."
    if not words(text):
        raise ValueError("Введите предложение со словами.")
    return text
