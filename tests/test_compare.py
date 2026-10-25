"""Контрпримеры, границы проверки и сверка кодирования с перебором миров."""

import copy
from itertools import combinations_with_replacement, product
import json
from pathlib import Path
import unittest

from src.builders import and_, const, exists, forall, implies, not_, or_, pred, var
from src.compare import find_counterexample
from src.evaluate import evaluate, validate_world


DATA = Path(__file__).resolve().parents[1] / "data"


def brute_difference(a, b, lexicon, entities, size):
    """Независимый маленький перебор всех интерпретаций для тестов."""
    domain = [f"o{i}" for i in range(size)]
    facts = [(name, args) for name, entry in lexicon["predicates"].items()
             for args in product(domain, repeat=entry["arity"])]
    for values in product(domain, repeat=len(entities)):
        constants = dict(zip(entities, values))
        for truth_values in product((False, True), repeat=len(facts)):
            world = {"domain": domain, "constants": constants,
                     "predicates": {name: [] for name in lexicon["predicates"]}}
            for (name, args), truth in zip(facts, truth_values):
                if truth:
                    world["predicates"][name].append(list(args))
            if evaluate(a, world, lexicon, entities) != evaluate(b, world, lexicon, entities):
                return True
    return False


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lexicon = json.loads((DATA / "lexicon.json").read_text(encoding="utf-8"))
        cls.pilot = {p["id"]: p for p in json.loads((DATA / "pilot.json").read_text(encoding="utf-8"))}

    def setUp(self):
        self.small = {"predicates": {"P": {"arity": 1}, "Q": {"arity": 1}}}
        self.p = pred("P", var("x"))
        self.q = pred("Q", var("x"))

    def check_pair(self, a, b, lexicon, entities, bound, found):
        result = find_counterexample(a, b, lexicon, entities, bound,
                                     max_nodes=None, max_encoding_steps=None)
        full = result["bound_info"]["required_domain_size"]
        status = ("counterexample_found" if found else "equivalent"
                  if full is not None and bound >= full
                  else "no_counterexample_up_to_bound")
        self.assertEqual(result["status"], status)
        self.assertIsNone(result["reason"])
        if found:
            world = result["world"]
            for formula in (a, b):
                validate_world(world, lexicon, formula)
            actual_a = evaluate(a, world, lexicon, entities)
            actual_b = evaluate(b, world, lexicon, entities)
            self.assertIs(result["value_a"], actual_a)
            self.assertIs(result["value_b"], actual_b)
            self.assertNotEqual(actual_a, actual_b)
            self.assertEqual(result["current_size"], len(world["domain"]))
            self.assertEqual(result["checked_sizes"], list(range(1, len(world["domain"]))))
        else:
            self.assertEqual(result["checked_sizes"], list(range(1, result["max_domain_size"] + 1)))
            self.assertIsNone(result["world"])
            self.assertIsNone(result["value_a"])
            self.assertIsNone(result["value_b"])
            self.assertIsNone(result["current_size"])
        return result

    def test_identical_and_renamed_bound_variable(self):
        a = forall("x", implies(self.p, self.q))
        renamed = forall("other", implies(pred("P", var("other")), pred("Q", var("other"))))
        for b in (copy.deepcopy(a), renamed):
            self.check_pair(a, b, self.small, {}, 4, False)

    def test_manual_equivalent_forms_from_pilot(self):
        for left, right in (("pilot_05", "pilot_06"), ("pilot_07", "pilot_08")):
            self.check_pair(self.pilot[left]["formulas"][0], self.pilot[right]["formulas"][0],
                            self.lexicon, {}, 4, False)

    def test_different_meanings_from_pilot(self):
        for left, right in (("pilot_01", "pilot_03"), ("pilot_05", "pilot_07"),
                            ("pilot_11", "pilot_13")):
            self.check_pair(self.pilot[left]["formulas"][0], self.pilot[right]["formulas"][0],
                            self.lexicon, {}, 4, True)

    def test_direction_requires_two_objects_and_distinct_assignments(self):
        a = pred("Консультирует", const("e1"), const("e2"))
        b = pred("Консультирует", const("e2"), const("e1"))
        entities = {"e1": "Имя 1", "e2": "Имя 2", "unused": "Лишний участник"}
        self.check_pair(a, b, self.lexicon, entities, 1, False)
        result = self.check_pair(a, b, self.lexicon, entities, 2, True)
        self.assertEqual(set(result["world"]["constants"]), {"e1", "e2"})
        self.assertNotEqual(*result["world"]["constants"].values())
        self.assertEqual(set(result["world"]["predicates"]), {"Консультирует"})

    def test_constants_may_coincide(self):
        a = pred("P", const("e1"))
        b = not_(pred("P", const("e2")))
        result = self.check_pair(a, b, self.small, {"e1": "Анна", "e2": "Борис"}, 1, True)
        self.assertEqual(result["world"]["constants"], {"e1": "d0", "e2": "d0"})

    def test_bound_is_not_general_equivalence(self):
        # Три несовместимых набора свойств требуют хотя бы трёх объектов.
        needs_three = and_(exists("x", and_(self.p, self.q)), and_(
            exists("x", and_(self.p, not_(self.q))),
            exists("x", and_(not_(self.p), self.q)),
        ))
        contradiction = exists("x", and_(self.p, not_(self.p)))
        self.check_pair(needs_three, contradiction, self.small, {}, 2, False)
        result = self.check_pair(needs_three, contradiction, self.small, {}, 3, True)
        self.assertEqual(result["checked_sizes"], [1, 2])

    def test_shadowed_variable_and_variable_constant_namespaces(self):
        a = forall("x", implies(exists("x", self.p), self.q))
        b = forall("z", implies(exists("y", pred("P", var("y"))), pred("Q", var("z"))))
        self.check_pair(a, b, self.small, {}, 3, False)
        a = forall("e", pred("P", var("e")))
        b = pred("P", const("e"))
        self.check_pair(a, b, self.small, {"e": "Имя"}, 2, True)

    def test_zero_arity_and_three_arguments(self):
        lexicon = {"predicates": {"Signal": {"arity": 0}, "Передаёт": {"arity": 3}}}
        a = pred("Signal")
        self.check_pair(a, not_(a), lexicon, {}, 1, True)
        relation = pred("Передаёт", var("x"), var("y"), var("z"))
        a = forall("x", exists("y", forall("z", relation)))
        b = not_(exists("x", forall("y", exists("z", not_(relation)))))
        self.check_pair(a, b, lexicon, {}, 2, False)

    def test_small_unary_formulas_against_all_worlds(self):
        formulas = [
            forall("x", implies(self.p, self.q)), exists("x", and_(self.p, self.q)),
            forall("x", implies(self.p, not_(self.q))),
            not_(forall("x", implies(self.p, self.q))),
            exists("x", and_(self.p, not_(self.q))),
            forall("x", or_(self.p, self.q)),
            and_(exists("x", self.p), exists("x", self.q)),
        ]
        for a, b in combinations_with_replacement(formulas, 2):
            with self.subTest(a=a, b=b):
                expected = any(brute_difference(a, b, self.small, {}, n) for n in (1, 2))
                self.check_pair(a, b, self.small, {}, 2, expected)

    def test_binary_formulas_and_constants_against_all_worlds(self):
        lexicon = {"predicates": {"R": {"arity": 2}}}
        relation = pred("R", var("x"), var("y"))
        pairs = [
            (forall("x", exists("y", relation)), exists("y", forall("x", relation))),
            (pred("R", const("a"), const("b")), pred("R", const("b"), const("a"))),
        ]
        for a, b in pairs:
            entities = {} if a[0] == "forall" else {"a": "Первый", "b": "Второй"}
            expected = any(brute_difference(a, b, lexicon, entities, n) for n in (1, 2))
            self.check_pair(a, b, lexicon, entities, 2, expected)

    def test_limits_and_completed_sizes(self):
        a = forall("x", self.p)
        for option, limit, sizes in (("max_nodes", 0, []), ("max_encoding_steps", 0, []),
                                     ("max_nodes", 1, [1])):
            result = find_counterexample(a, a, self.small, {}, 3, **{option: limit})
            self.assertEqual(result["status"], "inconclusive")
            self.assertEqual(result["checked_sizes"], sizes)
            self.assertIn(option, result["reason"])
            self.assertEqual(result["current_size"], len(sizes) + 1)
            self.assertIsNone(result["world"])

    def test_bad_inputs_raise_and_inputs_are_preserved(self):
        a = forall("x", self.p)
        before = copy.deepcopy((a, self.small))
        find_counterexample(a, a, self.small, {})
        self.assertEqual((a, self.small), before)
        for name, value in (("max_domain_size", 0), ("max_domain_size", True),
                            ("max_nodes", -1), ("max_encoding_steps", True)):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    find_counterexample(a, a, self.small, {}, **{name: value})
        with self.assertRaisesRegex(ValueError, "не связана"):
            find_counterexample(self.p, a, self.small, {})
        with self.assertRaisesRegex(ValueError, "неизвестный предикат"):
            find_counterexample(a, pred("Unknown"), self.small, {})


if __name__ == "__main__":
    unittest.main()
