"""Проверки сборки по ручным эталонам и ожидаемому смыслу формул."""

import copy
import json
from pathlib import Path
import unittest

from src.builders import (
    and_, build_formulas, const, exists, forall, implies, not_, or_, pred, var,
)


DATA = Path(__file__).resolve().parents[1] / "data"


def truth(formula, world, environment=None):
    """Небольшой независимый вычислитель только для тестов сборщика.

    Принимает заранее корректные деревья и миры; оставлен независимой
    проверкой сборщика. В рабочем коде используем src.evaluate.
    """
    environment = {} if environment is None else environment
    operation = formula[0]
    if operation == "pred":
        arguments = [
            environment[name] if kind == "var" else world["constants"][name]
            for kind, name in formula[2:]
        ]
        return arguments in world["predicates"][formula[1]]
    if operation == "not":
        return not truth(formula[1], world, environment)
    if operation in ("forall", "exists"):
        values = [
            truth(formula[2], world, {**environment, formula[1]: obj})
            for obj in world["domain"]
        ]
        return all(values) if operation == "forall" else any(values)
    left = truth(formula[1], world, environment)
    right = truth(formula[2], world, environment)
    if operation == "and":
        return left and right
    if operation == "or":
        return left or right
    if operation == "implies":
        return not left or right
    raise ValueError(f"Неизвестная операция: {operation}")


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pilot = json.loads((DATA / "pilot.json").read_text(encoding="utf-8"))
        cls.worlds = {
            world["id"]: world
            for world in json.loads((DATA / "worlds.json").read_text(encoding="utf-8"))
        }

    def test_templates_match_manual_trees(self):
        # 06 и 08 намеренно записаны иначе. Их смысл проверяется ниже.
        for example in self.pilot:
            if example["id"] in ("pilot_06", "pilot_08"):
                continue
            with self.subTest(example=example["id"]):
                self.assertEqual(
                    build_formulas(example["code"], example["slots"]),
                    example["formulas"],
                )

    def test_templates_have_expected_meaning_in_worlds(self):
        world_ids = [
            "all_students_work", "mixed_students", "no_students_work",
            "no_students", "separate_articles", "shared_article",
            "no_students_no_articles",
        ]
        # Ожидания заданы вручную, а не получены сборщиком.
        expected = {
            "ALL": [True, False, False, True, False, False, True],
            "EXISTS": [True, True, False, False, False, False, False],
            "NONE": [False, False, True, True, True, True, True],
            "NOT_ALL": [False, True, True, False, True, True, False],
            "ATOM": [False, True, False, False, False, False, False],
            "AE": [False, False, False, True, True, True, True],
            "EA": [False, False, False, True, False, True, False],
        }
        for example in self.pilot:
            built = build_formulas(example["code"], example["slots"])
            codes = ("AE", "EA") if example["code"] == "AMBIG_AE_EA" else (example["code"],)
            self.assertEqual(len(built), len(codes))
            self.assertEqual(len(example["formulas"]), len(codes))
            for index, code in enumerate(codes):
                for world_id, answer in zip(world_ids, expected[code]):
                    with self.subTest(example=example["id"], reading=code, world=world_id):
                        world = self.worlds[world_id]
                        self.assertEqual(truth(built[index], world), answer)
                        self.assertEqual(truth(example["formulas"][index], world), answer)

    def test_new_predicates_and_entities_need_no_builder_changes(self):
        actual = build_formulas("ALL", {
            "class_predicate": "Преподаватель", "property_predicate": "Спит",
        })
        self.assertEqual(actual, [["forall", "x", ["implies",
            ["pred", "Преподаватель", ["var", "x"]],
            ["pred", "Спит", ["var", "x"]],
        ]]])
        self.assertEqual(build_formulas("ATOM", {
            "relation": "Помогает", "subject": "visitor_8", "object": "visitor_3",
        }), [["pred", "Помогает", ["const", "visitor_8"], ["const", "visitor_3"]]])

    def test_argument_order_and_repeated_participant(self):
        world = self.worlds["mixed_students"]
        forward = build_formulas("ATOM", {
            "relation": "Консультирует", "subject": "entity_1", "object": "entity_2",
        })[0]
        reverse = build_formulas("ATOM", {
            "relation": "Консультирует", "subject": "entity_2", "object": "entity_1",
        })[0]
        self.assertTrue(truth(forward, world))
        self.assertFalse(truth(reverse, world))
        self.assertEqual(build_formulas("ATOM", {
            "relation": "Помогает", "subject": "someone", "object": "someone",
        }), [["pred", "Помогает", ["const", "someone"], ["const", "someone"]]])

    def test_composition_outside_templates(self):
        # Смешиваем константу, переменную и связки без нового кода шаблона.
        formula = exists("person", and_(
            pred("Преподаватель", var("person")),
            and_(
                pred("Консультирует", const("visitor"), var("person")),
                implies(pred("Работает", var("person")),
                        or_(pred("Спит", var("person")),
                            not_(pred("Устал", var("person"))))),
            ),
        ))
        world = {
            "domain": ["a", "b"], "constants": {"visitor": "a"},
            "predicates": {"Преподаватель": [["b"]], "Консультирует": [["a", "b"]],
                           "Работает": [["b"]], "Спит": [], "Устал": []},
        }
        self.assertTrue(truth(formula, world))
        world["predicates"]["Устал"] = [["b"]]
        self.assertFalse(truth(formula, world))

    def test_three_quantifiers_and_ternary_predicate(self):
        formula = forall("giver", exists("item", exists("recipient",
            pred("Передаёт", var("giver"), var("item"), var("recipient")),
        )))
        world = {"domain": ["a", "b"], "constants": {},
                 "predicates": {"Передаёт": [["a", "a", "b"], ["b", "a", "a"]]}}
        self.assertTrue(truth(formula, world))
        world["predicates"]["Передаёт"].pop()
        self.assertFalse(truth(formula, world))
        self.assertEqual(json.loads(json.dumps(formula, ensure_ascii=False)), formula)

    def test_builds_do_not_mutate_slots_or_share_readings(self):
        slots = {"subject_class": "Студент", "object_class": "Статья", "relation": "Рецензирует"}
        original = copy.deepcopy(slots)
        first = build_formulas("AMBIG_AE_EA", slots)
        second = build_formulas("AMBIG_AE_EA", slots)
        expected_ea = copy.deepcopy(first[1])
        first[0][2][1][1] = "ДругойКласс"
        self.assertEqual(first[1], expected_ea)
        self.assertEqual(second[0][2][1][1], "Студент")
        self.assertEqual(slots, original)

    def test_bad_template_parameters(self):
        cases = [
            ("UNKNOWN", {}), (None, {}), ([], {}),
            ("ALL", None), ("ALL", []), ("ALL", {}),
            ("ALL", {"class_predicate": "Студент", "property_predicate": " "}),
            ("ALL", {"class_predicate": "Студент", "property_predicate": 1}),
            ("ALL", {"class_predicate": "Студент", "property_predicate": "Работает", "extra": "x"}),
            ("ATOM", {"relation": "R", "subject": "a"}),
            ("EA", {"subject_class": "N", "object_class": "M", "relation": None}),
        ]
        for code, slots in cases:
            with self.subTest(code=code, slots=slots), self.assertRaises(ValueError):
                build_formulas(code, slots)

    def test_bad_names_and_predicate_arguments(self):
        for name in ("", "  ", None, 7, []):
            for function in (var, const, pred):
                with self.subTest(function=function.__name__, name=name), self.assertRaises(ValueError):
                    function(name)
            for function in (forall, exists):
                with self.subTest(function=function.__name__, name=name), self.assertRaises(ValueError):
                    function(name, ["pred", "P", ["var", "x"]])
        for term in ("x", [], ["var"], ["var", ""], ["pred", "P"], ["const", "a", "b"]):
            with self.subTest(term=term), self.assertRaises(ValueError):
                pred("P", term)


if __name__ == "__main__":
    unittest.main()
