"""Поиск конечного мира, в котором две замкнутые формулы различаются.

Граница автоматически вычисляется для проверенного языка исследования.
Факты ищет SAT-решатель, назначения констант перебираются, включая совпадения.
Эквивалентность подтверждается только полной SAT-проверкой достаточной
границы. Меньший пользовательский предел даёт ограниченный ответ.
"""

from itertools import product

from .bounds import comparison_bound
from .evaluate import evaluate
from .sat import solve


def _encode_pair(formula_a, formula_b, domain, constants, tick):
    """Раскрыть кванторы и получить КНФ различия A XOR B.

    Вспомогательные переменные Цейтина обозначают значения подформул.
    Одна таблица atoms связывает одинаковые факты обеих формул.
    Возвращает число SAT-переменных, скобки и отображение фактов в номера.
    """
    atoms, gates, clauses = {}, {}, []
    count = 0
    truth = None

    def fresh():
        nonlocal count
        count += 1
        return count

    def gate(operation, literals):
        nonlocal truth
        # Удаление повторов и общий номер одинаковых подформул уменьшают КНФ.
        distinct = set(literals)
        arguments = tuple(sorted(distinct))
        if len(arguments) == 1:
            return arguments[0]
        if any(-x in distinct for x in arguments):
            # A ∨ ¬A истинно, A ∧ ¬A ложно. Одной единичной скобки
            # достаточно; не заставляем решатель заново доказывать это.
            if truth is None:
                truth = fresh()
                clauses.append((truth,))
            return truth if operation == "or" else -truth
        key = (operation, arguments)
        if key in gates:
            return gates[key]
        result = fresh()
        gates[key] = result
        if operation == "and":
            # result ↔ (a1 ∧ ... ∧ an)
            clauses.extend((-result, x) for x in arguments)
            clauses.append((result, *(-x for x in arguments)))
        else:
            # result ↔ (a1 ∨ ... ∨ an)
            clauses.extend((result, -x) for x in arguments)
            clauses.append((-result, *arguments))
        return result

    def visit(node, environment):
        tick()
        operation = node[0]
        if operation == "pred":
            arguments = tuple(environment[name] if kind == "var" else constants[name]
                              for kind, name in node[2:])
            key = (node[1], arguments)
            if key not in atoms:
                atoms[key] = fresh()
            return atoms[key]
        if operation == "not":
            return -visit(node[1], environment)
        if operation in ("forall", "exists"):
            literals = [visit(node[2], {**environment, node[1]: obj}) for obj in domain]
            return gate("and" if operation == "forall" else "or", literals)
        left = visit(node[1], environment)
        right = visit(node[2], environment)
        if operation == "implies":
            return gate("or", [-left, right])
        return gate(operation, [left, right])

    left, right = visit(formula_a, {}), visit(formula_b, {})
    # Различие: хотя бы одно истинно, хотя бы одно ложно.
    clauses.extend(((left, right), (-left, -right)))
    return count, clauses, atoms


class _EncodingLimit(Exception):
    """Внутренний сигнал исчерпания бюджета раскрытия кванторов."""


def find_counterexample(formula_a, formula_b, lexicon, entities, max_domain_size=None,
                        *, max_nodes=100_000, max_encoding_steps=100_000):
    """Сравнить две формулы с уже согласованными идентификаторами участников.

    Результат — обычный словарь:
      status: equivalent / counterexample_found /
              no_counterexample_up_to_bound / inconclusive;
      checked_sizes: только полностью проверенные размеры;
      current_size: размер найденного мира или незавершённой проверки, иначе None;
      world, value_a, value_b: контрпример и значения, иначе None;
      reason: пояснение при inconclusive;
      stats: число SAT-вызовов, узлов поиска и посещений дерева при кодировании.

    При max_domain_size=None граница выбирается по обеим формулам, выход за
    язык исследования вызывает ValueError. Явное целое задаёт предел поиска;
    вне языка остаётся доступен прежний поиск без гарантии полноты.
    bound_info содержит достаточную границу и её обоснование, proof_method —
    способ доказательства: только small_model_bound после полной SAT-проверки.
    В SAT передаются исходные деревья, найденный мир проверяется через evaluate.

    max_nodes и max_encoding_steps — общие бюджеты на весь вызов, не на один
    размер или назначение. None снимает соответствующий лимит. Это ограничения
    работы, а не таймер. Ошибки входа вызывают ValueError. Данные не изменяются.
    """
    bound_info = comparison_bound(formula_a, formula_b, lexicon, entities)
    if max_domain_size is not None and (type(max_domain_size) is not int or max_domain_size < 1):
        raise ValueError("max_domain_size должно быть целым числом не меньше 1 или None.")
    for name, value in (("max_nodes", max_nodes), ("max_encoding_steps", max_encoding_steps)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} должно быть целым неотрицательным числом или None.")

    required = bound_info["required_domain_size"]
    if max_domain_size is None and not bound_info["supported"]:
        raise ValueError(f"Нельзя выбрать гарантированную границу: {bound_info['reason']}")
    limit = required if max_domain_size is None else max_domain_size
    if required is not None:
        limit = min(limit, required)

    predicates = bound_info["predicates"]
    constant_names = bound_info["constants"]
    result = {
        "status": "no_counterexample_up_to_bound", "max_domain_size": limit,
        "requested_max_domain_size": max_domain_size, "bound_info": bound_info,
        "proof_method": None,
        "checked_sizes": [], "current_size": None,
        "world": None, "value_a": None, "value_b": None, "reason": None,
        "stats": {"sat_calls": 0, "sat_nodes": 0, "encoding_steps": 0},
    }
    stats = result["stats"]

    def tick():
        if max_encoding_steps is not None and stats["encoding_steps"] >= max_encoding_steps:
            raise _EncodingLimit
        stats["encoding_steps"] += 1

    def inconclusive(reason):
        result.update(status="inconclusive", reason=reason)
        return result

    for size in range(1, limit + 1):
        result["current_size"] = size
        domain = [f"d{i}" for i in range(size)]
        # repeat=0 даёт одно пустое назначение, если в формулах нет констант.
        for values in product(domain, repeat=len(constant_names)):
            assignment = dict(zip(constant_names, values))
            try:
                n, clauses, atoms = _encode_pair(formula_a, formula_b, domain, assignment, tick)
            except _EncodingLimit:
                return inconclusive("Исчерпан max_encoding_steps при раскрытии формул.")
            remaining = None if max_nodes is None else max_nodes - stats["sat_nodes"]
            answer = solve(n, clauses, max_nodes=remaining)
            stats["sat_calls"] += 1
            stats["sat_nodes"] += answer["nodes"]
            if answer["status"] == "UNKNOWN":
                return inconclusive("Исчерпан max_nodes в SAT-поиске.")
            if answer["status"] == "SAT":
                world = {"domain": domain, "constants": assignment,
                         "predicates": {name: [] for name in sorted(predicates)}}
                # Не встретившиеся при раскрытии факты считаем ложными.
                for (name, arguments), variable in atoms.items():
                    if answer["model"][variable]:
                        world["predicates"][name].append(list(arguments))
                value_a = evaluate(formula_a, world, lexicon, entities)
                value_b = evaluate(formula_b, world, lexicon, entities)
                if value_a == value_b:
                    raise RuntimeError("Ошибка SAT-кодирования: мир не различает исходные формулы.")
                result.update(status="counterexample_found", world=world,
                              value_a=value_a, value_b=value_b)
                return result
        result["checked_sizes"].append(size)
    result["current_size"] = None
    if required is not None and limit >= required:
        result.update(status="equivalent", proof_method="small_model_bound")
    return result
