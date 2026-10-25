"""Проверка ограничений языка и гарантий автоматического сравнения."""

import copy
from itertools import product
import json
from pathlib import Path
import unittest

from src.bounds import comparison_bound, validate_research_formula
from src.builders import and_, const, exists, forall, implies, not_, or_, pred, var
from src.compare import find_counterexample
from src.evaluate import evaluate


DATA = Path(__file__).resolve().parents[1] / "data"


class BoundTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = {"predicates": {"P": {"arity": 1}, "Q": {"arity": 1},
                                      "R": {"arity": 2}, "Unused": {"arity": 3}}}
        self.p, self.q = pred("P", var("x")), pred("Q", var("x"))
        self.all_p = forall("x", self.p)

    def test_current_pilot_is_in_language(self):
        lexicon = json.loads((DATA / "lexicon.json").read_text(encoding="utf-8"))
        pilot = json.loads((DATA / "pilot.json").read_text(encoding="utf-8"))
        for example in pilot:
            for formula in example["formulas"]:
                validate_research_formula(formula, lexicon, example["entities"])
                info = comparison_bound(formula, formula, lexicon, example["entities"])
                self.assertTrue(info["supported"])
                self.assertGreaterEqual(info["required_domain_size"], 1)

    def test_monadic_bound_uses_union_of_used_symbols(self):
        info = comparison_bound(self.all_p, exists("x", self.q), self.lexicon, {})
        self.assertEqual(info["method"], "monadic_types")
        self.assertEqual(info["required_domain_size"], 4)
        self.assertEqual(info["predicates"], {"P": 1, "Q": 1})
        same = comparison_bound(self.all_p, self.all_p, self.lexicon, {})
        self.assertEqual(same["required_domain_size"], 2)

    def test_ground_bound_and_constant_aliases(self):
        a = pred("R", const("a"), const("b"))
        b = pred("R", const("b"), const("a"))
        entities = {"a": "Первый", "b": "Второй", "unused": "Третий"}
        info = comparison_bound(a, b, self.lexicon, entities)
        self.assertEqual(info["method"], "ground_constants")
        self.assertEqual(info["required_domain_size"], 2)
        result = find_counterexample(a, b, self.lexicon, entities)
        self.assertEqual(result["status"], "counterexample_found")
        self.assertEqual(result["checked_sizes"], [1])

    def test_relational_bound_counts_diagonals_labels_and_constants(self):
        a = forall("x", pred("R", var("x"), const("a")))
        b = pred("R", const("b"), const("a"))
        info = comparison_bound(a, b, self.lexicon, {"a": "А", "b": "Б"})
        # k=0, q=1, r=1, c=2; p=0+1+1+2*2*1=6; B=2+3*1*64.
        self.assertEqual(info["method"], "fo2_types_and_witnesses")
        self.assertEqual(info["parameters"]["type_bits"], 6)
        self.assertEqual(info["required_domain_size"], 194)

    def test_three_variable_names_in_separate_scopes_are_allowed(self):
        formula = and_(forall("student", pred("P", var("student"))), and_(
            exists("teacher", pred("Q", var("teacher"))),
            forall("author", pred("P", var("author"))),
        ))
        validate_research_formula(formula, self.lexicon, {})
        info = comparison_bound(formula, formula, self.lexicon, {})
        self.assertEqual(info["variable_widths"], [1, 1])

    def test_three_simultaneous_variables_are_rejected(self):
        transitivity = forall("x", forall("y", forall("z", implies(
            and_(pred("R", var("x"), var("y")), pred("R", var("y"), var("z"))),
            pred("R", var("x"), var("z")),
        ))))
        info = comparison_bound(transitivity, transitivity, self.lexicon, {})
        self.assertFalse(info["supported"])
        self.assertIsNone(info["required_domain_size"])
        with self.assertRaisesRegex(ValueError, "больше двух"):
            validate_research_formula(transitivity, self.lexicon, {})
        with self.assertRaisesRegex(ValueError, "гарантированную границу"):
            find_counterexample(transitivity, transitivity, self.lexicon, {})
        limited = find_counterexample(transitivity, transitivity, self.lexicon, {}, 2)
        self.assertEqual(limited["status"], "no_counterexample_up_to_bound")
        self.assertIsNone(limited["proof_method"])

    def test_predicate_arity_restriction_and_invalid_input_are_distinct(self):
        lexicon = {"predicates": {"T": {"arity": 3}}}
        formula = forall("x", pred("T", var("x"), var("x"), var("x")))
        info = comparison_bound(formula, formula, lexicon, {})
        self.assertFalse(info["supported"])
        with self.assertRaisesRegex(ValueError, "1 или 2"):
            validate_research_formula(formula, lexicon, {})
        with self.assertRaisesRegex(ValueError, "не связана"):
            comparison_bound(self.p, self.all_p, self.lexicon, {})

    def test_full_monadic_bound_proves_equivalence(self):
        a = forall("x", implies(self.p, not_(self.q)))
        b = not_(exists("x", and_(self.p, self.q)))
        result = find_counterexample(a, b, self.lexicon, {})
        self.assertEqual(result["status"], "equivalent")
        self.assertEqual(result["proof_method"], "small_model_bound")
        self.assertEqual(result["checked_sizes"], [1, 2, 3, 4])
        self.assertEqual(result["max_domain_size"], 4)
        self.assertIsNone(result["world"])

    def test_user_limit_below_proven_bound_does_not_prove_equivalence(self):
        result = find_counterexample(self.all_p, self.all_p, self.lexicon, {}, 1)
        self.assertEqual(result["status"], "no_counterexample_up_to_bound")
        self.assertIsNone(result["proof_method"])
        capped = find_counterexample(self.all_p, self.all_p, self.lexicon, {}, 20)
        self.assertEqual(capped["status"], "equivalent")
        self.assertEqual(capped["max_domain_size"], 2)

    def test_budget_exhaustion_never_becomes_equivalence(self):
        for options in ({"max_nodes": 0}, {"max_encoding_steps": 0}):
            result = find_counterexample(self.all_p, self.all_p, self.lexicon, {}, **options)
            self.assertEqual(result["status"], "inconclusive")
            self.assertEqual(result["checked_sizes"], [])
            self.assertIsNone(result["proof_method"])

    def test_three_distinct_unary_types_need_three_objects(self):
        a = and_(exists("x", and_(self.p, self.q)), and_(
            exists("x", and_(self.p, not_(self.q))),
            exists("x", and_(not_(self.p), self.q)),
        ))
        b = exists("x", and_(self.p, not_(self.p)))
        limited = find_counterexample(a, b, self.lexicon, {}, 2)
        self.assertEqual(limited["status"], "no_counterexample_up_to_bound")
        result = find_counterexample(a, b, self.lexicon, {})
        self.assertEqual(result["bound_info"]["required_domain_size"], 4)
        self.assertEqual(result["status"], "counterexample_found")
        self.assertEqual(len(result["world"]["domain"]), 3)

    def test_relational_cycle_counterexample_needs_three_objects(self):
        rxy = pred("R", var("x"), var("y"))
        a = and_(forall("x", not_(pred("R", var("x"), var("x")))), and_(
            forall("x", exists("y", rxy)),
            forall("x", forall("y", implies(rxy, not_(pred("R", var("y"), var("x")))))),
        ))
        b = exists("x", and_(self.p, not_(self.p)))
        result = find_counterexample(a, b, self.lexicon, {})
        self.assertEqual(result["status"], "counterexample_found")
        self.assertEqual(result["checked_sizes"], [1, 2])
        self.assertEqual(len(result["world"]["domain"]), 3)

    def test_complete_relational_bound(self):
        diagonal = pred("R", var("x"), var("x"))
        a = forall("x", or_(diagonal, not_(diagonal)))
        ground = pred("R", const("a"), const("a"))
        b = or_(ground, not_(ground))
        # Обе формулы тождественно истинны; SAT полностью проверяет границу.
        result = find_counterexample(a, b, self.lexicon, {"a": "Объект"},
                                     max_encoding_steps=None)
        self.assertEqual(result["status"], "equivalent")
        self.assertEqual(result["proof_method"], "small_model_bound")
        self.assertEqual(result["checked_sizes"], list(range(1, 50)))
        self.assertEqual(result["bound_info"]["required_domain_size"], 49)

    def test_relational_equivalent_forms_use_sat(self):
        a = forall("author", exists("paper", pred("R", var("author"), var("paper"))))
        b = not_(exists("x", forall("y", not_(pred("R", var("x"), var("y"))))))
        result = find_counterexample(a, b, self.lexicon, {}, max_domain_size=2)
        self.assertEqual(result["status"], "no_counterexample_up_to_bound")
        self.assertIsNone(result["proof_method"])
        self.assertEqual(result["checked_sizes"], [1, 2])
        self.assertGreater(result["stats"]["sat_calls"], 0)
        # Вторая x связывает только себя. SAT должен обнаружить смену смысла.
        shadow = forall("x", exists("x", pred("R", var("x"), var("x"))))
        different = find_counterexample(a, shadow, self.lexicon, {}, max_domain_size=2)
        self.assertEqual(different["status"], "counterexample_found")
        self.assertNotEqual(evaluate(a, different["world"], self.lexicon, {}),
                            evaluate(shadow, different["world"], self.lexicon, {}))

    def test_equivalent_closed_formulas_use_complete_sat_check(self):
        a = forall("x", implies(self.p, self.q))
        b = not_(exists("y", and_(pred("P", var("y")), not_(pred("Q", var("y"))))))
        c = forall("unused", b)
        for formula in (b, c, and_(b, b)):
            result = find_counterexample(a, formula, self.lexicon, {})
            self.assertEqual(result["status"], "equivalent")
            self.assertEqual(result["proof_method"], "small_model_bound")
            self.assertEqual(result["checked_sizes"], [1, 2, 3, 4])
            self.assertGreater(result["stats"]["sat_calls"], 0)
        for values in product((False, True), repeat=4):
            world = {"domain": ["a", "b"], "constants": {}, "predicates": {"P": [], "Q": []}}
            for (name, obj), truth in zip(product(("P", "Q"), ("a", "b")), values):
                if truth:
                    world["predicates"][name].append([obj])
            expected = evaluate(a, world, self.lexicon, {})
            for formula in (b, c, and_(b, b)):
                self.assertIs(evaluate(formula, world, self.lexicon, {}), expected)

    def test_constant_names_and_inputs_are_preserved(self):
        a = exists("x", pred("R", var("x"), const("a")))
        b = exists("x", pred("R", var("x"), const("b")))
        entities = {"a": "Одно имя", "b": "Одно имя"}
        before = copy.deepcopy((a, b, self.lexicon, entities))
        result = find_counterexample(a, b, self.lexicon, entities)
        self.assertEqual(result["status"], "counterexample_found")
        self.assertEqual((a, b, self.lexicon, entities), before)


if __name__ == "__main__":
    unittest.main()
