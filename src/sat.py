"""Точный SAT-поиск по КНФ на основе решателя из домашней работы.

Сохранены выбор самой частой переменной и перебор False, True. Добавлены
распространение единичных скобок, явный стек и ограничение числа узлов.
Литерал i означает переменную i, -i — её отрицание. Нумерация начинается с 1.
"""


def solve(n, clauses, *, max_nodes=100_000):
    """Вернуть dict(status, model, nodes); status — SAT, UNSAT или UNKNOWN.

    model — полный словарь {номер: bool} для SAT, иначе None.
    Пустая КНФ истинна, пустая скобка ложна. UNKNOWN означает исчерпание
    max_nodes, а не UNSAT. max_nodes=None снимает ограничение.
    Переданные списки не изменяются.
    """
    if type(n) is not int or n < 0:
        raise ValueError("n должно быть целым неотрицательным числом.")
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 0):
        raise ValueError("max_nodes должно быть целым неотрицательным числом или None.")
    if not isinstance(clauses, (list, tuple)):
        raise ValueError("clauses должно быть списком или кортежем скобок.")
    original, normalized = [], []
    for clause in clauses:
        if not isinstance(clause, (list, tuple)):
            raise ValueError("Каждая скобка должна быть списком или кортежем литералов.")
        if any(type(x) is not int or x == 0 or abs(x) > n for x in clause):
            raise ValueError("Литерал должен быть целым ненулевым числом в диапазоне -n..n.")
        original.append(tuple(clause))
        unique = tuple(dict.fromkeys(clause))
        if not any(-x in unique for x in unique):
            normalized.append(unique)

    def assign(formula, literal):
        # Убираем истинные скобки и ложные литералы из оставшихся скобок.
        return [tuple(x for x in clause if x != -literal)
                for clause in formula if literal not in clause]

    pending = [(normalized, {})]
    nodes = 0
    while pending:
        if max_nodes is not None and nodes >= max_nodes:
            return {"status": "UNKNOWN", "model": None, "nodes": nodes}
        formula, model = pending.pop()
        nodes += 1
        while formula and not any(not clause for clause in formula):
            unit = next((c[0] for c in formula if len(c) == 1), None)
            if unit is None:
                break
            model[abs(unit)] = unit > 0
            formula = assign(formula, unit)

        if any(not clause for clause in formula):
            continue
        if not formula:
            model = {i: model.get(i, False) for i in range(1, n + 1)}
            if not all(any(model[abs(x)] == (x > 0) for x in c) for c in original):
                raise RuntimeError("SAT-решатель вернул неверную модель.")
            return {"status": "SAT", "model": model, "nodes": nodes}

        counts = {}
        for clause in formula:
            for literal in clause:
                variable = abs(literal)
                counts[variable] = counts.get(variable, 0) + 1
        variable = max(counts, key=lambda v: (counts[v], -v))
        # Стек LIFO: False будет рассмотрено первым, как в исходном решателе.
        for value in (True, False):
            literal = variable if value else -variable
            pending.append((assign(formula, literal), {**model, variable: value}))

    return {"status": "UNSAT", "model": None, "nodes": nodes}
