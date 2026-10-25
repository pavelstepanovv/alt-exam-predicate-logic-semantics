"""Сборка JSON-деревьев из частей или параметров смыслового шаблона.

Общие функции не ограничивают лексику, число аргументов и вложенность.
build_formulas поддерживает текущие восемь шаблонов и всегда возвращает
список формул. Русский текст и файлы данных этот модуль не читает.

Полную проверку дерева по словарю и entities выполняет validate_formula
из src.logic. Здесь проверяются имена и параметры шаблонов.
Промежуточные деревья могут содержать ещё не связанные переменные.
Переданные поддеревья не копируются и не изменяются; после сборки следует
считать их неизменяемыми, чтобы правка общего списка не затронула две формулы.
"""


def _check_name(name, label):
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{label}: ожидается непустая строка.")


def var(name):
    """Переменная, например x. Связать её квантором можно позже."""
    _check_name(name, "Имя переменной")
    return ["var", name]


def const(entity_id):
    """Константа — идентификатор из entities, а не грамматическая роль."""
    _check_name(entity_id, "Идентификатор участника")
    return ["const", entity_id]


def pred(name, *terms):
    """Предикат с аргументами var/const; число аргументов проверяется по словарю позже."""
    _check_name(name, "Имя предиката")
    for term in terms:
        if (not isinstance(term, list) or len(term) != 2
                or term[0] not in ("var", "const")):
            raise ValueError("Аргумент предиката должен быть создан через var или const.")
        _check_name(term[1], "Имя аргумента")
    return ["pred", name, *terms]


def not_(formula):
    """Отрицание формулы."""
    return ["not", formula]


def and_(left, right):
    """Конъюнкция двух формул. Для трёх и более используем вложенность."""
    return ["and", left, right]


def or_(left, right):
    """Дизъюнкция двух формул."""
    return ["or", left, right]


def implies(left, right):
    """Импликация: если left, то right."""
    return ["implies", left, right]


def forall(variable, body):
    """Квантор всеобщности; variable — имя переменной строкой."""
    _check_name(variable, "Имя переменной")
    return ["forall", variable, body]


def exists(variable, body):
    """Квантор существования; variable — имя переменной строкой."""
    _check_name(variable, "Имя переменной")
    return ["exists", variable, body]


_PROPERTY_CODES = ("ALL", "EXISTS", "NONE", "NOT_ALL")
_RELATION_CODES = ("AE", "EA", "AMBIG_AE_EA")


def _check_slots(code, slots):
    _check_name(code, "Код шаблона")
    if code in _PROPERTY_CODES:
        required = {"class_predicate", "property_predicate"}
    elif code == "ATOM":
        required = {"relation", "subject", "object"}
    elif code in _RELATION_CODES:
        required = {"subject_class", "object_class", "relation"}
    else:
        raise ValueError(f"Неизвестный код шаблона: {code}")

    if not isinstance(slots, dict):
        raise ValueError("slots должен быть словарём параметров.")
    missing = required - slots.keys()
    if missing:
        raise ValueError(f"Для {code} не хватает параметров: {', '.join(sorted(missing))}")
    extra = slots.keys() - required
    if extra:
        names = ", ".join(sorted(str(key) for key in extra))
        raise ValueError(f"Для {code} переданы лишние параметры: {names}")
    for key in sorted(required):
        _check_name(slots[key], f"Параметр {key}")


def build_formulas(code, slots):
    """Построить список формул по вручную заданным code и slots.

    ALL/EXISTS/NONE/NOT_ALL: class_predicate, property_predicate.
    ATOM: relation, subject, object (два идентификатора из entities).
    AE/EA/AMBIG_AE_EA: subject_class, object_class, relation.

    AMBIG_AE_EA возвращает две отдельные формулы в порядке AE, EA;
    функция не устанавливает неоднозначность исходного предложения.
    При неизвестном коде или неправильных параметрах вызывает ValueError.
    Существование предикатов и участников проверяет отдельный валидатор.
    """
    _check_slots(code, slots)

    if code in _PROPERTY_CODES:
        member = pred(slots["class_predicate"], var("x"))
        property_ = pred(slots["property_predicate"], var("x"))
        if code == "ALL":
            formula = forall("x", implies(member, property_))
        elif code == "EXISTS":
            formula = exists("x", and_(member, property_))
        elif code == "NONE":
            formula = forall("x", implies(member, not_(property_)))
        else:  # NOT_ALL
            formula = not_(forall("x", implies(member, property_)))
        return [formula]

    if code == "ATOM":
        return [pred(slots["relation"], const(slots["subject"]), const(slots["object"]))]

    if code == "AMBIG_AE_EA":
        # Два независимых дерева: изменение одного не затрагивает другое.
        return build_formulas("AE", slots) + build_formulas("EA", slots)

    subject = pred(slots["subject_class"], var("x"))
    object_ = pred(slots["object_class"], var("y"))
    relation = pred(slots["relation"], var("x"), var("y"))
    if code == "AE":
        return [forall("x", implies(subject, exists("y", and_(object_, relation))))]
    # После проверки кода здесь остаётся только EA.
    return [exists("y", and_(object_, forall("x", implies(subject, relation))))]
