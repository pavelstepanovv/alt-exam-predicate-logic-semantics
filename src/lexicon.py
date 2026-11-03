"""Словарь словоформ и обозначений."""

import json
from pathlib import Path

def _require(condition):
    if not condition:
        raise ValueError("Неверное значение словарного поля")


def validate_lexicon(lexicon):
    if not isinstance(lexicon, dict) or not isinstance(lexicon.get("predicates"), dict):
        raise ValueError("Ожидается словарь с predicates")
    if not isinstance(lexicon.get("entities"), dict):
        raise ValueError("Отсутствует таблица имён entities")
    for symbol, entry in lexicon["predicates"].items():
        if not isinstance(symbol, str) or not symbol or not isinstance(entry, dict):
            raise ValueError("Некорректное описание предиката")
        kind = entry.get("kind")
        if kind not in ("class", "property", "relation"):
            raise ValueError(f"{symbol}: неизвестный вид записи")
        if type(entry.get("arity")) is not int or entry["arity"] != (2 if kind == "relation" else 1):
            raise ValueError(f"{symbol}: неверное число аргументов")
        if not entry.get("lemmas") or not entry.get("forms"):
            raise ValueError(f"{symbol}: не заданы леммы или словоформы")
        forms = entry["forms"]
        try:
            if kind == "class":
                _require(entry["gender"] in ("m", "f"))
                _require(entry["domain"] in ("human", "document"))
                required = [forms[number][case] for number in ("sg", "pl") for case in ("nom", "acc", "gen")]
            elif kind == "relation":
                _require(entry["argument_order"] == ["agent", "patient"])
                _require(entry["object_case"] == "acc")
                _require(entry["subject_domains"] and entry["object_domains"])
                required = [forms["sg"], forms["pl"]]
            elif entry["style"] == "verb":
                _require(entry["domains"])
                required = [forms["sg"], forms["pl"]]
            else:
                _require(entry["style"] == "adjective" and entry["domains"])
                required = [forms["short"][g] for g in ("m", "f", "pl")]
                required += [forms["instrumental"][g] for g in ("m", "f", "pl")]
                required += [forms["full"][g][case] for g in ("m", "f", "pl") for case in ("nom", "acc", "acc_animate", "gen")]
            _require(all(isinstance(value, str) and value.strip() for value in required))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{symbol}: неполное описание словоформ") from error
    for identifier, entry in lexicon["entities"].items():
        try:
            _require(isinstance(identifier, str) and identifier)
            _require(entry["gender"] in ("m", "f") and entry["domain"] in ("human", "document"))
            _require(entry["name"] == entry["forms"]["nom"])
            _require(all(isinstance(entry["forms"][c], str) and entry["forms"][c] for c in ("nom", "acc")))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{identifier}: неверное описание имени") from error


def load_lexicon():
    path = Path(__file__).resolve().parents[1] / "data/lexicon.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_lexicon(result)
    return result
