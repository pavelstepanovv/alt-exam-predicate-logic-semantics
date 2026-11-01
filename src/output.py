"""Один общий разбор ответа всех систем, без исправления смысла."""

import json
import re

from .logic import validate_formula


BRACKET_POLICY = "trailing_square_brackets_v1"
PREDICATE_POLICY = "lexicon_grammatical_forms_v1"
OUTPUT_POLICY = BRACKET_POLICY + "+" + PREDICATE_POLICY


def normalize_predicate_forms(formula, lexicon):
    """Только объявленные словоформы; имена участников и разные понятия не меняются."""
    candidates = {}

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    key = lambda text: text.casefold().replace("ё", "е")
    for symbol, entry in lexicon["predicates"].items():
        for form in (symbol, *strings(entry.get("forms", {})), *strings(entry.get("lemmas", []))):
            candidates.setdefault(key(form), set()).add(symbol)
    aliases = {form: next(iter(symbols)) for form, symbols in candidates.items() if len(symbols) == 1}
    changes = []
    pending = [formula]
    while pending:
        node = pending.pop()
        if not isinstance(node, list):
            continue
        if len(node) >= 2 and node[0] == "pred" and isinstance(node[1], str) and node[1] not in lexicon["predicates"]:
            canonical = aliases.get(key(node[1]))
            if canonical is not None:
                changes.append({"from": node[1], "to": canonical})
                node[1] = canonical
        pending.extend(child for child in node if isinstance(child, list))
    return changes


def _reject_constant(value):
    raise ValueError(f"Недопустимое значение JSON: {value}")


def _repair_trailing_brackets(text):
    """Один кандидат без словаря/gold: только ] в конце, с учётом строк JSON.

    Не переставляет скобки внутри дерева и не дописывает оборванные строки.
    Кандидат ещё должен пройти обычный JSON-разбор и проверку формулы.
    """
    if not text.startswith("["):
        return text, 0, 0
    depth, quoted, escaped = 0, False, False
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                tail = text[index + 1:]
                if tail and tail.strip() and all(c == "]" or c.isspace() for c in tail):
                    return text[:index + 1], 0, tail.count("]")
                return text, 0, 0
    if depth > 0 and not quoted:
        return text + "]" * depth, depth, 0
    return text, 0, 0


def parse_response(raw_response, lexicon, entities, *, max_chars=40_000, max_depth=64, max_nodes=10_000,
                   repair_brackets=True):
    """Ответ — строка с одним JSON AST; допустима одна внешняя ограда ```json.

    По умолчанию исправляет только конечные квадратные скобки. Не извлекает
    формулу из пояснений, не выбирает по эталону и не меняет содержание.
    repair_brackets=False отключает исправление скобок; словоформы обрабатываются одинаково.
    """
    for value in (max_chars, max_depth, max_nodes):
        if type(value) is not int or value < 1:
            raise ValueError("Лимиты разбора должны быть положительными целыми")
    result = {"raw_response": raw_response, "response_for_parsing": None,
              "raw_json_valid": None,
              "bracket_repair": {"policy": BRACKET_POLICY if repair_brackets else "disabled",
                                 "applied": False, "added_closing": 0, "removed_closing": 0},
              "predicate_normalization": {"policy": PREDICATE_POLICY, "changes": []},
              "parsed_ast": None,
              "validation_status": "invalid_or_missing", "reason_code": None, "reason": None}

    def invalid(code, reason):
        result.update(reason_code=code, reason=reason)
        return result

    if raw_response is None or (isinstance(raw_response, str) and not raw_response.strip()):
        return invalid("missing", "Ответ отсутствует")
    if not isinstance(raw_response, str):
        return invalid("response_type", "Ожидается строка с одной JSON-формулой")
    if len(raw_response) > max_chars:
        return invalid("output_length_limit", "Ответ превысил общий лимит длины")
    text = raw_response.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    result["response_for_parsing"] = text
    try:
        formula = json.loads(text, parse_constant=_reject_constant)
        result["raw_json_valid"] = True
    except (ValueError, RecursionError) as error:
        result["raw_json_valid"] = False
        if not repair_brackets:
            return invalid("invalid_json", str(error))
        candidate, added, removed = _repair_trailing_brackets(text)
        if candidate == text:
            return invalid("invalid_json", str(error))
        if len(candidate) > max_chars:
            return invalid("output_length_limit", "Исправленная запись превысила лимит длины")
        try:
            formula = json.loads(candidate, parse_constant=_reject_constant)
        except (ValueError, RecursionError):
            return invalid("invalid_json", str(error))
        result["response_for_parsing"] = candidate
        result["bracket_repair"].update(applied=True, added_closing=added, removed_closing=removed)
    pending, count = [(formula, 0)], 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > max_nodes or depth > max_depth:
            return invalid("output_structure_limit", "Ответ превысил общий лимит структуры")
        if isinstance(node, list):
            pending.extend((child, depth + 1) for child in node)
    changes = normalize_predicate_forms(formula, lexicon)
    result["predicate_normalization"]["changes"] = changes
    try:
        validate_formula(formula, lexicon, entities)
    except (ValueError, RecursionError) as error:
        reason = str(error)
        code = "invalid_ast"
        for marker, category in (("неизвестный предикат", "unknown_predicate"),
                                 ("ожидает аргументов", "wrong_arity"),
                                 ("не связана", "free_variable"),
                                 ("не объявлен", "unknown_entity")):
            if marker in reason:
                code = category
                break
        return invalid(code, reason)
    result.update(parsed_ast=formula, validation_status="valid")
    return result
