"""Вычисление истинности одной формулы в полностью заданном конечном мире.

Модуль не ищет миры и не доказывает эквивалентность формул.
Все необходимые определения проверяются до вычисления, включая ветки,
до которых сокращённое вычисление and/or/implies могло бы не дойти.
"""

from .logic import validate_formula


def validate_world(world, lexicon, formula):
    """Проверить мир для данной формулы; вернуть None или вызвать ValueError.

    domain — непустой список уникальных непустых строк (объектов).
    constants — словарь локальных идентификаторов участников и объектов.
    predicates — словарь расширений предикатов: списков наборов аргументов.
    Все переданные определения должны быть корректны, но наличие предикатов
    и констант требуется только для символов данной формулы.

    Проверяется и сама формула по словарю и назначениям constants.
    Имена участников из entities эта функция не проверяет — это делает evaluate.
    id и description мира для вычисления не нужны.
    """
    if not isinstance(world, dict):
        raise ValueError("world: ожидается словарь с описанием мира.")
    domain = world.get("domain")
    if not isinstance(domain, list) or not domain:
        raise ValueError("world.domain: ожидается непустой список объектов.")
    for obj in domain:
        if not isinstance(obj, str) or not obj.strip():
            raise ValueError("world.domain: идентификаторы объектов должны быть непустыми строками.")
    objects = set(domain)
    if len(objects) != len(domain):
        raise ValueError("world.domain: объекты не должны повторяться.")

    constants = world.get("constants")
    if not isinstance(constants, dict):
        raise ValueError("world.constants: ожидается словарь, в том числе пустой при отсутствии констант.")
    for name, obj in constants.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("world.constants: идентификатор участника должен быть непустой строкой.")
        if not isinstance(obj, str) or obj not in objects:
            raise ValueError(f"world.constants[{name!r}]: значение должно быть объектом из domain.")

    extensions = world.get("predicates")
    if not isinstance(extensions, dict):
        raise ValueError("world.predicates: ожидается словарь расширений предикатов.")

    # Для проверки интерпретации нужны идентификаторы, а не отображаемые имена.
    # Используем публичный валидатор, чтобы не дублировать проверку дерева.
    try:
        validate_formula(formula, lexicon, {name: name for name in constants})
    except ValueError as error:
        raise ValueError(f"Формула не согласована со словарём или world.constants: {error}") from error

    definitions = lexicon["predicates"]
    for name, facts in extensions.items():
        if not isinstance(name, str) or name not in definitions:
            raise ValueError(f"world.predicates: неизвестный предикат {name!r}.")
        if not isinstance(facts, list):
            raise ValueError(f"world.predicates[{name!r}]: ожидается список фактов.")
        arity = definitions[name]["arity"]
        for index, arguments in enumerate(facts):
            path = f"world.predicates[{name!r}][{index}]"
            if not isinstance(arguments, list) or len(arguments) != arity:
                raise ValueError(f"{path}: ожидается список из {arity} аргументов.")
            if any(not isinstance(obj, str) or obj not in objects for obj in arguments):
                raise ValueError(f"{path}: все аргументы должны входить в domain.")

    def check_required(node):
        operation = node[0]
        if operation == "pred":
            if node[1] not in extensions:
                raise ValueError(f"world.predicates: не задан предикат {node[1]}; пустое расширение задаётся как [].")
        elif operation == "not":
            check_required(node[1])
        elif operation in ("forall", "exists"):
            check_required(node[2])
        else:
            check_required(node[1])
            check_required(node[2])

    check_required(formula)


def evaluate(formula, world, lexicon, entities):
    """Вернуть True/False для одной замкнутой формулы в данном мире.

    Сначала проверяет формулу, локальные entities и весь переданный мир.
    Ошибки входа вызывают ValueError и не превращаются в False.
    Предикаты интерпретируются по данным, без специальных случаев для имён.
    Входные словари и деревья не изменяются.
    """
    validate_formula(formula, lexicon, entities)
    validate_world(world, lexicon, formula)

    domain = world["domain"]
    constants = world["constants"]
    # Множества ускоряют проверку фактов; повторные записи факта не меняют смысл.
    extensions = {
        name: {tuple(arguments) for arguments in facts}
        for name, facts in world["predicates"].items()
    }

    def visit(node, environment):
        operation = node[0]
        if operation == "pred":
            arguments = tuple(
                environment[name] if kind == "var" else constants[name]
                for kind, name in node[2:]
            )
            return arguments in extensions[node[1]]
        if operation == "not":
            return not visit(node[1], environment)
        if operation == "and":
            return visit(node[1], environment) and visit(node[2], environment)
        if operation == "or":
            return visit(node[1], environment) or visit(node[2], environment)
        if operation == "implies":
            return not visit(node[1], environment) or visit(node[2], environment)
        if operation == "forall":
            return all(
                visit(node[2], {**environment, node[1]: obj})
                for obj in domain
            )
        # После валидации единственная оставшаяся операция — exists.
        return any(
            visit(node[2], {**environment, node[1]: obj})
            for obj in domain
        )

    return visit(formula, {})
