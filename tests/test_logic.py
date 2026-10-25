"""Проверки структуры, областей связывания и вывода формул."""

import copy
import json
from pathlib import Path
import unittest

from src.builders import build_formulas
from src.logic import format_formula, validate_formula


DATA = Path(__file__).resolve().parents[1] / "data"


class LogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = json.loads((DATA / "lexicon.json").read_text(encoding="utf-8"))
        cls.pilot = json.loads((DATA / "pilot.json").read_text(encoding="utf-8"))

    def setUp(self):
        self.symbols = {"predicates": {
            "P": {"arity": 1}, "Q": {"arity": 1}, "R": {"arity": 2},
        }}

    def test_manual_and_built_formulas_are_valid(self):
        for example in self.pilot:
            formulas = example["formulas"] + build_formulas(example["code"], example["slots"])
            for formula in formulas:
                with self.subTest(example=example["id"], formula=formula):
                    self.assertIsNone(validate_formula(formula, self.lexicon, example["entities"]))
                    self.assertIsInstance(format_formula(formula, example["entities"]), str)

    def test_unknown_predicate(self):
        with self.assertRaisesRegex(ValueError, "неизвестный предикат Спит"):
            validate_formula(["forall", "x", ["pred", "Спит", ["var", "x"]]], self.lexicon, {})

    def test_wrong_number_of_arguments(self):
        for arguments in ([], [["const", "e"]], [["const", "e"]] * 3):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "ожидает аргументов: 2"):
                validate_formula(["pred", "R", *arguments], self.symbols, {"e": "Имя"})

    def test_unknown_entity(self):
        with self.assertRaisesRegex(ValueError, "участник stranger не объявлен"):
            validate_formula(["pred", "P", ["const", "stranger"]], self.symbols, {})

    def test_free_variable_reports_position(self):
        formula = ["forall", "x", ["implies",
            ["pred", "P", ["var", "x"]], ["pred", "Q", ["var", "y"]],
        ]]
        with self.assertRaisesRegex(ValueError, r"formula\[2\]\[2\]\[2\].*переменная y не связана"):
            validate_formula(formula, self.symbols, {})

    def test_binding_does_not_leak_to_sibling(self):
        # В том числе общий объект списка в двух разных областях действия.
        shared = ["pred", "P", ["var", "x"]]
        for operation in ("and", "or", "implies"):
            for quantified in ("forall", "exists"):
                formula = [operation, [quantified, "x", shared], shared]
                with self.subTest(operation=operation, quantified=quantified):
                    with self.assertRaisesRegex(ValueError, "переменная x не связана"):
                        validate_formula(formula, self.symbols, {})

    def test_outer_scope_and_shadowing_are_valid(self):
        formula = ["forall", "x", ["and",
            ["exists", "x", ["pred", "P", ["var", "x"]]],
            ["exists", "y", ["pred", "R", ["var", "x"], ["var", "y"]]],
        ]]
        self.assertIsNone(validate_formula(formula, self.symbols, {}))

    def test_inner_variable_does_not_escape(self):
        formula = ["forall", "x", ["and",
            ["exists", "y", ["pred", "R", ["var", "x"], ["var", "y"]]],
            ["pred", "Q", ["var", "y"]],
        ]]
        with self.assertRaisesRegex(ValueError, "переменная y не связана"):
            validate_formula(formula, self.symbols, {})

    def test_variable_and_constant_with_same_id_are_distinct(self):
        formula = ["forall", "x", ["pred", "R", ["var", "x"], ["const", "x"]]]
        entities = {"x": "Мария"}
        self.assertIsNone(validate_formula(formula, self.symbols, entities))
        self.assertEqual(format_formula(formula, entities), "∀x (R(x, Мария))")
        with self.assertRaisesRegex(ValueError, "участник x не объявлен"):
            validate_formula(formula, self.symbols, {})

    def test_general_arity_and_quantifier_depth(self):
        lexicon = {"predicates": {"Передаёт": {"arity": 3}, "Дождь": {"arity": 0}}}
        formula = ["forall", "giver", ["exists", "item", ["exists", "receiver",
            ["pred", "Передаёт", ["var", "giver"], ["var", "item"], ["var", "receiver"]],
        ]]]
        self.assertIsNone(validate_formula(formula, lexicon, {}))
        self.assertEqual(format_formula(formula, {}),
                         "∀giver (∃item (∃receiver (Передаёт(giver, item, receiver))))")
        self.assertIsNone(validate_formula(["pred", "Дождь"], lexicon, {}))
        self.assertEqual(format_formula(["pred", "Дождь"], {}), "Дождь()")

    def test_malformed_nodes(self):
        cases = [
            None, {}, "P(x)", [], [7], [["pred", "P"]], ["unknown"],
            ["var", "x"], ["const", "e"], ["pred"], ["pred", []],
            ["not"], ["not", ["pred", "P"], ["pred", "Q"]],
            ["and", ["pred", "P"]], ["or", None, None],
            ["implies", None, None, None], ["forall", "x"],
            ["exists", [], ["pred", "P"]], ["forall", " ", ["pred", "P"]],
            ["forall", "x", "body"], ["forall", "x", ["pred", "P", "x"]],
            ["forall", "x", ["pred", "P", ["var"]]],
            ["forall", "x", ["pred", "P", ["var", "x", "extra"]]],
            ["forall", "x", ["pred", "P", ["var", []]]],
            ["forall", "x", ["pred", "P", ["var", " "]]],
            ["forall", "x", ["pred", "P", ["pred", "Q"]]],
            ["forall", "x", ["pred", "P", [[], "x"]]],
        ]
        for formula in cases:
            with self.subTest(formula=formula), self.assertRaises(ValueError):
                validate_formula(formula, self.symbols, {})

    def test_invalid_symbol_tables(self):
        formula = ["pred", "P", ["const", "e"]]
        bad_lexicons = [None, [], {}, {"predicates": []}, {"predicates": {"": {"arity": 1}}}]
        for definition in (None, [], {}, {"arity": -1}, {"arity": True}, {"arity": "1"}, {"arity": 1.0}):
            bad_lexicons.append({"predicates": {"P": definition}})
        for lexicon in bad_lexicons:
            with self.subTest(lexicon=lexicon), self.assertRaises(ValueError):
                validate_formula(formula, lexicon, {"e": "Имя"})
        for entities in (None, [], {"": "Имя"}, {1: "Имя"}, {"e": " "}, {"e": {"name": "Имя"}}):
            with self.subTest(entities=entities), self.assertRaises(ValueError):
                validate_formula(formula, self.symbols, entities)

    def test_shared_subtrees_are_allowed_but_cycles_are_not(self):
        shared = ["pred", "P", ["const", "e"]]
        self.assertIsNone(validate_formula(["and", shared, shared], self.symbols, {"e": "Имя"}))
        cycle = ["not"]
        cycle.append(cycle)
        with self.assertRaisesRegex(ValueError, "циклическая ссылка"):
            validate_formula(cycle, self.symbols, {})
        with self.assertRaisesRegex(ValueError, "циклическая ссылка"):
            format_formula(cycle, {})

    def test_format_preserves_quantifier_and_negation_scope(self):
        expected = {
            "pilot_01": "∀x (Студент(x) → Работает(x))",
            "pilot_03": "∃x (Студент(x) ∧ Работает(x))",
            "pilot_05": "∀x (Студент(x) → ¬(Работает(x)))",
            "pilot_07": "¬(∀x (Студент(x) → Работает(x)))",
            "pilot_08": "∃x (Студент(x) ∧ ¬(Работает(x)))",
            "pilot_09": "Консультирует(Анна, Борис)",
            "pilot_11": "∀x (Студент(x) → ∃y (Статья(y) ∧ Рецензирует(x, y)))",
            "pilot_13": "∃y (Статья(y) ∧ ∀x (Студент(x) → Рецензирует(x, y)))",
        }
        for example in self.pilot:
            if example["id"] in expected:
                with self.subTest(example=example["id"]):
                    self.assertEqual(format_formula(example["formulas"][0], example["entities"]),
                                     expected[example["id"]])

    def test_format_nested_connectives_and_new_names(self):
        formula = ["not", ["or",
            ["pred", "P", ["const", "visitor"]],
            ["implies", ["pred", "Q", ["const", "visitor"]],
             ["not", ["pred", "P", ["const", "visitor"]]]],
        ]]
        self.assertEqual(format_formula(formula, {"visitor": "Иван"}),
                         "¬(P(Иван) ∨ (Q(Иван) → ¬(P(Иван))))")

    def test_formatter_is_not_a_lexicon_validator(self):
        formula = ["pred", "Рецензирует", ["const", "e"]]
        self.assertEqual(format_formula(formula, {"e": "Анна"}), "Рецензирует(Анна)")
        with self.assertRaisesRegex(ValueError, "ожидает аргументов: 2"):
            validate_formula(formula, self.lexicon, {"e": "Анна"})
        with self.assertRaises(ValueError):
            format_formula(formula, {})
        with self.assertRaises(ValueError):
            format_formula(["pred", "P", ["var", "x"]], {})

    def test_functions_leave_inputs_unchanged(self):
        example = copy.deepcopy(self.pilot[10])
        lexicon = copy.deepcopy(self.lexicon)
        before = copy.deepcopy((example, lexicon))
        validate_formula(example["formulas"][0], lexicon, example["entities"])
        format_formula(example["formulas"][0], example["entities"])
        self.assertEqual((example, lexicon), before)


if __name__ == "__main__":
    unittest.main()
