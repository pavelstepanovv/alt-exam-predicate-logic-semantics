"""Проверка языка исследования и доказанная граница размера контрпримера.

Язык: замкнутые формулы без функций и встроенного равенства, предикаты
с одним/двумя аргументами, не более двух свободных переменных в подформуле.
Последнее допускает корректное переименование в x/y, в том числе при
повторном связывании. Число имён переменных во всём JSON не ограничиваем.

Обоснование численной оценки — в docs/equivalence_guarantee.md.
Общий validate_formula и вычислитель остаются пригодны для более широкого языка.
"""

from .logic import validate_formula


def _profile(formula):
    predicates, constants = set(), set()
    quantifiers = 0
    width = 0

    def visit(node):
        nonlocal quantifiers, width
        operation = node[0]
        if operation == "pred":
            predicates.add(node[1])
            constants.update(name for kind, name in node[2:] if kind == "const")
            free = {name for kind, name in node[2:] if kind == "var"}
        elif operation == "not":
            free = visit(node[1])
        elif operation in ("forall", "exists"):
            quantifiers += 1
            free = visit(node[2]) - {node[1]}
        else:
            free = visit(node[1]) | visit(node[2])
        width = max(width, len(free))
        return free

    visit(formula)
    return {"predicates": predicates, "constants": constants,
            "quantifiers": quantifiers, "variable_width": width}


def comparison_bound(formula_a, formula_b, lexicon, entities):
    """Проанализировать ОБЕ формулы; вернуть сведения о языке и границе.

    supported=False означает выход за язык исследования, а не плохое дерево.
    Неверные деревья/таблицы по-прежнему вызывают ValueError.
    Учитываем только встречающиеся символы; входные данные не изменяем.
    """
    validate_formula(formula_a, lexicon, entities)
    validate_formula(formula_b, lexicon, entities)
    a, b = _profile(formula_a), _profile(formula_b)
    names = sorted(a["predicates"] | b["predicates"])
    constants = sorted(a["constants"] | b["constants"])
    arities = {name: lexicon["predicates"][name]["arity"] for name in names}
    result = {
        "supported": False, "method": None, "required_domain_size": None,
        "predicates": arities, "constants": constants,
        "variable_widths": [a["variable_width"], b["variable_width"]],
        "parameters": {}, "reason": None,
    }
    if any(arity not in (1, 2) for arity in arities.values()):
        result["reason"] = "Язык исследования допускает только предикаты с 1 или 2 аргументами."
        return result
    if max(result["variable_widths"]) > 2:
        result["reason"] = "В подформуле требуется больше двух свободных переменных."
        return result

    k = sum(arity == 1 for arity in arities.values())
    r = sum(arity == 2 for arity in arities.values())
    c = len(constants)
    q = a["quantifiers"] + b["quantifiers"]
    result["parameters"] = {"unary_predicates": k, "binary_predicates": r,
                            "constants": c, "quantifiers": q}
    # Из подходящих теорем выбираем наименьшую безопасную оценку.
    if q == 0:
        method, bound = "ground_constants", max(1, c)
    elif r == 0:
        method, bound = "monadic_types", 2 ** k
    else:
        # До q меток подформул, r диагоналей и 2*c*r связей с константами.
        # Три слоя по m свидетелей каждого типа + сами константы.
        p = k + q + r + 2 * c * r
        m = max(1, q)
        method, bound = "fo2_types_and_witnesses", c + 3 * m * 2 ** p
        result["parameters"].update(type_bits=p, witness_slots=m)
    result.update(supported=True, method=method, required_domain_size=bound)
    return result


def validate_research_formula(formula, lexicon, entities):
    """Проверить дерево и ограничения языка исследования; вернуть None.

    Отдельная функция: базовое ядро намеренно поддерживает и другие формулы.
    """
    info = comparison_bound(formula, formula, lexicon, entities)
    if not info["supported"]:
        raise ValueError(f"Формула вне языка исследования: {info['reason']}")


