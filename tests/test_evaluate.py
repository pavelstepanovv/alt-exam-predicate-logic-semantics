"""Вычисление на ручных мирах, связывание переменных и ошибки интерпретации."""

import copy
import json
from pathlib import Path
import unittest

from src.builders import build_formulas
from src.evaluate import evaluate, validate_world


DATA = Path(__file__).resolve().parents[1] / "data"


class EvaluateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = json.loads((DATA / "lexicon.json").read_text(encoding="utf-8"))
        cls.pilot = json.loads((DATA / "pilot.json").read_text(encoding="utf-8"))
        cls.worlds = {
            world["id"]: world
            for world in json.loads((DATA / "worlds.json").read_text(encoding="utf-8"))
        }

    def setUp(self):
        self.symbols = {"predicates": {"P": {"arity": 1}, "Q": {"arity": 1}, "R": {"arity": 2}}}
        self.world = {"domain": ["a", "b"], "constants": {"e": "a"},
                      "predicates": {"P": [["a"]], "Q": [["a"]], "R": [["a", "b"]]}}
        self.formula = ["pred", "P", ["const", "e"]]
        self.entities = {"e": "Произвольное имя"}

    def test_pilot_and_built_formulas_on_seven_worlds(self):
        world_ids = [
            "all_students_work", "mixed_students", "no_students_work", "no_students",
            "separate_articles", "shared_article", "no_students_no_articles",
        ]
        # Значения заданы вручную; не вычисляем expected проверяемым модулем.
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
            readings = ("AE", "EA") if example["code"] == "AMBIG_AE_EA" else (example["code"],)
            for source, formulas in (
                ("manual", example["formulas"]),
                ("built", build_formulas(example["code"], example["slots"])),
            ):
                self.assertEqual(len(formulas), len(readings))
                for formula, code in zip(formulas, readings):
                    for world_id, answer in zip(world_ids, expected[code]):
                        with self.subTest(example=example["id"], source=source, code=code, world=world_id):
                            self.assertIs(evaluate(formula, self.worlds[world_id], self.lexicon,
                                                   example["entities"]), answer)

    def test_truth_tables_and_zero_argument_predicates(self):
        lexicon = {"predicates": {"A": {"arity": 0}, "B": {"arity": 0}}}
        for a in (False, True):
            for b in (False, True):
                world = {"domain": ["object"], "constants": {},
                         "predicates": {"A": [[]] if a else [], "B": [[]] if b else []}}
                cases = [
                    (["pred", "A"], a), (["not", ["pred", "A"]], not a),
                    (["and", ["pred", "A"], ["pred", "B"]], a and b),
                    (["or", ["pred", "A"], ["pred", "B"]], a or b),
                    (["implies", ["pred", "A"], ["pred", "B"]], not a or b),
                ]
                for formula, answer in cases:
                    with self.subTest(a=a, b=b, formula=formula):
                        self.assertIs(evaluate(formula, world, lexicon, {}), answer)

    def test_quantifiers_visit_domain_not_just_class_members(self):
        self.assertFalse(evaluate(["forall", "x", ["pred", "P", ["var", "x"]]],
                                  self.world, self.symbols, {}))
        self.assertTrue(evaluate(["exists", "x", ["pred", "P", ["var", "x"]]],
                                 self.world, self.symbols, {}))
        restricted = ["forall", "x", ["implies",
            ["pred", "P", ["var", "x"]], ["pred", "Q", ["var", "x"]],
        ]]
        self.assertTrue(evaluate(restricted, self.world, self.symbols, {}))

    def test_inner_shadowing_preserves_outer_assignment(self):
        # ∃x P(x) истинна. Внешняя ∀x Q(x) должна заметить Q(b) = ложь.
        formula = ["forall", "x", ["implies",
            ["exists", "x", ["pred", "P", ["var", "x"]]],
            ["pred", "Q", ["var", "x"]],
        ]]
        self.assertFalse(evaluate(formula, self.world, self.symbols, {}))
        # Вторая ветка получает своё окружение, а не остаток первой.
        formula = ["and", ["exists", "x", ["pred", "P", ["var", "x"]]],
                   ["forall", "x", ["pred", "Q", ["var", "x"]]]]
        self.assertFalse(evaluate(formula, self.world, self.symbols, {}))

    def test_three_quantifiers_and_ternary_relation(self):
        lexicon = {"predicates": {"Передаёт": {"arity": 3}}}
        formula = ["forall", "giver", ["exists", "item", ["exists", "receiver",
            ["pred", "Передаёт", ["var", "giver"], ["var", "item"], ["var", "receiver"]],
        ]]]
        world = {"domain": ["one", "two"], "constants": {},
                 "predicates": {"Передаёт": [["one", "one", "two"], ["two", "one", "one"]]}}
        self.assertTrue(evaluate(formula, world, lexicon, {}))
        world["predicates"]["Передаёт"].pop()
        self.assertFalse(evaluate(formula, world, lexicon, {}))

    def test_direction_and_aliases_of_constants(self):
        self.world["constants"] = {"first": "a", "second": "b"}
        entities = {"first": "Мария", "second": "Иван"}
        forward = ["pred", "R", ["const", "first"], ["const", "second"]]
        reverse = ["pred", "R", ["const", "second"], ["const", "first"]]
        self.assertTrue(evaluate(forward, self.world, self.symbols, entities))
        self.assertFalse(evaluate(reverse, self.world, self.symbols, entities))
        self.world["constants"]["second"] = "a"
        self.world["predicates"]["R"] = [["a", "a"]]
        self.assertTrue(evaluate(forward, self.world, self.symbols, entities))

    def test_variable_and_constant_names_are_separate(self):
        formula = ["forall", "x", ["pred", "R", ["var", "x"], ["const", "x"]]]
        world = {"domain": ["a", "b"], "constants": {"x": "a"},
                 "predicates": {"R": [["a", "a"], ["b", "a"]]}}
        self.assertTrue(evaluate(formula, world, self.symbols, {"x": "Имя"}))

    def test_only_used_symbols_are_required(self):
        world = {"domain": ["a"], "constants": {"e": "a"}, "predicates": {"P": [["a"]]}}
        self.assertIsNone(validate_world(world, self.symbols, self.formula))
        self.assertTrue(evaluate(self.formula, world, self.symbols, {**self.entities, "unused": "Другой"}))
        # Повторные факты обозначают то же множество.
        world["predicates"]["P"].append(["a"])
        self.assertTrue(evaluate(self.formula, world, self.symbols, self.entities))

    def test_empty_extension_is_false_but_missing_is_error(self):
        self.world["predicates"]["P"] = []
        self.assertIs(evaluate(self.formula, self.world, self.symbols, self.entities), False)
        del self.world["predicates"]["P"]
        with self.assertRaisesRegex(ValueError, "не задан предикат P"):
            evaluate(self.formula, self.world, self.symbols, self.entities)

    def test_missing_predicate_cannot_hide_in_skipped_branch(self):
        lexicon = {"predicates": {"A": {"arity": 0}, "B": {"arity": 0}}}
        for operation, facts in (("and", []), ("or", [[]]), ("implies", [])):
            world = {"domain": ["a"], "constants": {}, "predicates": {"A": facts}}
            formula = [operation, ["pred", "A"], ["pred", "B"]]
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "не задан предикат B"):
                evaluate(formula, world, lexicon, {})
        self.world["predicates"] = {"P": []}
        formula = ["forall", "x", ["implies",
            ["pred", "P", ["var", "x"]], ["pred", "Q", ["var", "x"]],
        ]]
        with self.assertRaisesRegex(ValueError, "не задан предикат Q"):
            evaluate(formula, self.world, self.symbols, {})

    def test_missing_constant_cannot_hide_in_skipped_branch(self):
        formula = ["or", self.formula, ["pred", "Q", ["const", "missing"]]]
        with self.assertRaisesRegex(ValueError, "world.constants.*missing"):
            evaluate(formula, self.world, self.symbols, {**self.entities, "missing": "Имя"})

    def test_invalid_domain_and_world_sections(self):
        for world in (None, [], {}, {"domain": ["a"]}):
            with self.subTest(world=world), self.assertRaises(ValueError):
                validate_world(world, self.symbols, self.formula)
        for domain in (None, [], "ab", ["a", "a"], [""], [" "], [1], [True], [[]]):
            world = {**self.world, "domain": domain}
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                validate_world(world, self.symbols, self.formula)
        for section in ("constants", "predicates"):
            for value in (None, [], "invalid"):
                world = {**self.world, section: value}
                with self.subTest(section=section, value=value), self.assertRaises(ValueError):
                    validate_world(world, self.symbols, self.formula)

    def test_invalid_constant_assignments(self):
        for constants in ({}, {"e": "outside"}, {"e": []}, {"e": True}, {"": "a"}, {1: "a"}):
            world = {**self.world, "constants": constants}
            with self.subTest(constants=constants), self.assertRaises(ValueError):
                validate_world(world, self.symbols, self.formula)

    def test_invalid_extensions_including_unused_data(self):
        cases = [
            {"P": None}, {"P": {}}, {"P": ["a"]}, {"P": [[]]},
            {"P": [["a", "b"]]}, {"P": [["outside"]]}, {"P": [[[]]]}, {"P": [[True]]},
            {"P": [["a"]], "Q": [["outside"]]},
            {"P": [["a"]], "Typo": []}, {"P": [["a"]], 7: []},
        ]
        for predicates in cases:
            world = {**self.world, "predicates": predicates}
            with self.subTest(predicates=predicates), self.assertRaises(ValueError):
                validate_world(world, self.symbols, self.formula)

    def test_invalid_formula_and_entities_are_rejected(self):
        cases = [None, [], ["unknown"], ["pred", "P", ["var", "x"]],
                 ["pred", "P"], ["pred", "Missing", ["const", "e"]]]
        cycle = ["not"]
        cycle.append(cycle)
        cases.append(cycle)
        for formula in cases:
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                evaluate(formula, self.world, self.symbols, self.entities)
        with self.assertRaisesRegex(ValueError, "участник e не объявлен"):
            evaluate(self.formula, self.world, self.symbols, {})

    def test_inputs_are_unchanged(self):
        before = copy.deepcopy((self.formula, self.world, self.symbols, self.entities))
        validate_world(self.world, self.symbols, self.formula)
        evaluate(self.formula, self.world, self.symbols, self.entities)
        self.assertEqual((self.formula, self.world, self.symbols, self.entities), before)


if __name__ == "__main__":
    unittest.main()
