"""Проверка деревьев логических формул и их читаемый вывод.

validate_formula проверяет структуру, словарь, участников и связывание
переменных. format_formula печатает дерево, но не знает словаря предикатов.
Обе функции принимают одну формулу, а не весь список formulas.
Корректность структуры не означает истинность или правильность перевода.
"""


_CONNECTIVES = {"and": "∧", "or": "∨", "implies": "→"}
_QUANTIFIERS = {"forall": "∀", "exists": "∃"}


def _check_name(value, path):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}: ожидается непустая строка.")


def _check_entities(entities):
    if not isinstance(entities, dict):
        raise ValueError("entities: ожидается словарь идентификаторов и имён участников.")
    for entity_id, name in entities.items():
        _check_name(entity_id, "Идентификатор участника в entities")
        _check_name(name, f"entities[{entity_id!r}]")


def _get_predicates(lexicon):
    if not isinstance(lexicon, dict) or not isinstance(lexicon.get("predicates"), dict):
        raise ValueError("lexicon: ожидается словарь с разделом predicates.")
    predicates = lexicon["predicates"]
    for name, definition in predicates.items():
        _check_name(name, "Имя предиката в словаре")
        if not isinstance(definition, dict):
            raise ValueError(f"Предикат {name}: описание должно быть словарём.")
        arity = definition.get("arity")
        # bool является подклассом int в Python, но не числом аргументов.
        if type(arity) is not int or arity < 0:
            raise ValueError(f"Предикат {name}: arity должно быть целым неотрицательным числом.")
    return predicates


def _validate_tree(formula, entities, predicates=None):
    """Без predicates проверяет дерево, но не объявление и арность предикатов."""
    active = set()

    def visit(node, bound, path):
        if not isinstance(node, list) or not node:
            raise ValueError(f"{path}: формула должна быть непустым списком.")
        if id(node) in active:
            raise ValueError(f"{path}: циклическая ссылка вместо дерева формулы.")
        active.add(id(node))
        try:
            operation = node[0]
            _check_name(operation, f"{path}[0]: операция")

            if operation == "pred":
                if len(node) < 2:
                    raise ValueError(f"{path}: у pred не указано имя предиката.")
                name = node[1]
                _check_name(name, f"{path}[1]: имя предиката")
                if predicates is not None:
                    if name not in predicates:
                        raise ValueError(f"{path}: неизвестный предикат {name}.")
                    expected = predicates[name]["arity"]
                    received = len(node) - 2
                    if received != expected:
                        raise ValueError(
                            f"{path}: предикат {name} ожидает аргументов: {expected}; "
                            f"получено: {received}."
                        )
                for index, term in enumerate(node[2:], start=2):
                    term_path = f"{path}[{index}]"
                    if (not isinstance(term, list) or len(term) != 2
                            or term[0] not in ("var", "const")):
                        raise ValueError(f"{term_path}: ожидается терм ['var', имя] или ['const', id].")
                    kind, value = term
                    _check_name(value, term_path)
                    if kind == "var" and value not in bound:
                        raise ValueError(f"{term_path}: переменная {value} не связана квантором.")
                    if kind == "const" and value not in entities:
                        raise ValueError(f"{term_path}: участник {value} не объявлен в entities.")

            elif operation == "not":
                if len(node) != 2:
                    raise ValueError(f"{path}: not ожидает одну вложенную формулу.")
                visit(node[1], bound, f"{path}[1]")

            elif operation in _CONNECTIVES:
                if len(node) != 3:
                    raise ValueError(f"{path}: {operation} ожидает две вложенные формулы.")
                visit(node[1], bound, f"{path}[1]")
                visit(node[2], bound, f"{path}[2]")

            elif operation in _QUANTIFIERS:
                if len(node) != 3:
                    raise ValueError(f"{path}: {operation} ожидает имя переменной и тело.")
                _check_name(node[1], f"{path}[1]: имя переменной")
                # Новый набор действует только в теле этого квантора.
                visit(node[2], bound | {node[1]}, f"{path}[2]")

            else:
                raise ValueError(f"{path}: неизвестная операция формулы {operation}.")
        finally:
            active.remove(id(node))

    visit(formula, frozenset(), "formula")


def validate_formula(formula, lexicon, entities):
    """Проверить одну замкнутую формулу; вернуть None или вызвать ValueError.

    lexicon содержит раздел predicates с числом аргументов arity.
    entities — таблица участников данного примера, например {'e1': 'Анна'}.
    Не ограничивает число кванторов, имена или число аргументов предикатов.
    Проверяет базовый формат этих таблиц, но не леммы и смысловые описания.
    Не изменяет входные данные и не проверяет истинность или эквивалентность.
    """
    predicates = _get_predicates(lexicon)
    _check_entities(entities)
    _validate_tree(formula, entities, predicates)


def format_formula(formula, entities):
    """Вывести одну формулу символами логики, заменив id участников их именами.

    Проверяет структуру, замкнутость и entities. Без lexicon не может проверить
    объявление предиката и число его аргументов: для этого сначала вызовите
    validate_formula. Скобки сохраняют области действия всех операций.
    Вывод предназначен для чтения, а не для обратного разбора в дерево.
    """
    _check_entities(entities)
    _validate_tree(formula, entities)

    def scope(node):
        text = render(node)
        # Бинарные связки уже окружены скобками.
        return text if node[0] in _CONNECTIVES else f"({text})"

    def render(node):
        operation = node[0]
        if operation == "pred":
            arguments = [
                name if kind == "var" else entities[name]
                for kind, name in node[2:]
            ]
            return f"{node[1]}({', '.join(arguments)})"
        if operation == "not":
            return f"¬{scope(node[1])}"
        if operation in _CONNECTIVES:
            return f"({render(node[1])} {_CONNECTIVES[operation]} {render(node[2])})"
        return f"{_QUANTIFIERS[operation]}{node[1]} {scope(node[2])}"

    return render(formula)
