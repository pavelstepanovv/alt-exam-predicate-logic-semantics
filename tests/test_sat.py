"""Независимая проверка SAT-решателя полным перебором маленьких КНФ."""

import copy
from itertools import product
import random
import unittest

from src.sat import solve


def brute_model(n, clauses):
    for values in product((False, True), repeat=n):
        model = dict(enumerate(values, 1))
        if all(any(model[abs(x)] == (x > 0) for x in c) for c in clauses):
            return model
    return None


class SatTests(unittest.TestCase):
    def check_against_brute(self, n, clauses):
        expected = brute_model(n, clauses)
        result = solve(n, clauses, max_nodes=None)
        self.assertEqual(result["status"], "SAT" if expected is not None else "UNSAT")
        if expected is not None:
            model = result["model"]
            self.assertEqual(set(model), set(range(1, n + 1)))
            self.assertTrue(all(type(value) is bool for value in model.values()))
            self.assertTrue(all(any(model[abs(x)] == (x > 0) for x in c) for c in clauses))
        else:
            self.assertIsNone(result["model"])

    def test_all_cnf_over_two_variables(self):
        # Все 512 подмножеств девяти различных нетавтологических скобок.
        clauses = [tuple(sign * i for i, sign in enumerate(signs, 1) if sign)
                   for signs in product((-1, 0, 1), repeat=2)]
        for selected in product((False, True), repeat=len(clauses)):
            formula = [c for c, use in zip(clauses, selected) if use]
            with self.subTest(clauses=formula):
                self.check_against_brute(2, formula)

    def test_seeded_formulas_with_repetitions_and_tautologies(self):
        rng = random.Random(42)
        for _ in range(100):
            n = rng.randrange(1, 6)
            clauses = [tuple(rng.choice((-1, 1)) * rng.randrange(1, n + 1)
                             for _ in range(rng.randrange(1, 5)))
                       for _ in range(rng.randrange(1, 12))]
            self.check_against_brute(n, clauses)

    def test_empty_formula_and_empty_clause(self):
        for n, clauses in ((0, []), (0, [()]), (2, []), (2, [()])):
            self.check_against_brute(n, clauses)

    def test_unit_propagation_and_branch_conflicts(self):
        # Цепочка единичных скобок; затем противоречие без исходных единиц.
        self.check_against_brute(4, [(1,), (-1, 2), (-2, 3), (-3, 4), (-4,)])
        self.check_against_brute(2, [(1, 2), (-1, 2), (1, -2), (-1, -2)])

    def test_limit_is_unknown_and_inputs_are_preserved(self):
        clauses = [[1, 2], [-1, 2]]
        before = copy.deepcopy(clauses)
        result = solve(2, clauses, max_nodes=0)
        self.assertEqual(result, {"status": "UNKNOWN", "model": None, "nodes": 0})
        solve(2, clauses)
        self.assertEqual(clauses, before)

    def test_invalid_inputs(self):
        for n, clauses, budget in (
            (True, [], 10), (-1, [], 10), (1, [[0]], 10), (1, [[2]], 10),
            (1, [[True]], 10), (1, [[1.0]], 10), (1, [1], 10),
            (1, None, 10), (1, [], -1), (1, [], True),
        ):
            with self.subTest(n=n, clauses=clauses, budget=budget):
                with self.assertRaises(ValueError):
                    solve(n, clauses, max_nodes=budget)


if __name__ == "__main__":
    unittest.main()
