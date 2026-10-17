"""Small deterministic checks for the tour representation and GA operators."""

import unittest

import numpy as np
import pandas as pd

from genetic_algorithms import (
    TSPProblem,
    fitness_length,
    mutate_inversion,
    mutate_swap,
    ordered_crossover,
    run_ga,
    validate_solution,
)


class GeneticAlgorithmTests(unittest.TestCase):
    def setUp(self):
        coordinates = pd.DataFrame(
            {"x": [0, 3, 3, 0], "y": [0, 0, 4, 4]},
            index=pd.Index([1, 2, 3, 4], name="city_id"),
        )
        self.problem = TSPProblem(coordinates, "EUC_2D")

    def test_closed_tour_uses_tsplib_rounded_distances(self):
        tour = np.array([1, 2, 3, 4])
        self.assertEqual(fitness_length(tour, self.problem), 14.0)
        self.assertTrue(validate_solution(tour, self.problem))
        self.assertFalse(validate_solution(np.array([1, 2, 3, 3]), self.problem))

    def test_crossover_and_mutations_preserve_valid_permutations(self):
        rng = np.random.default_rng(42)
        first = np.array([1, 2, 3, 4])
        second = np.array([4, 3, 2, 1])
        for tour in (
            ordered_crossover(first, second, rng),
            mutate_swap(first, 1.0, rng),
            mutate_inversion(first, 1.0, rng),
        ):
            self.assertTrue(validate_solution(tour, self.problem))

    def test_small_ga_returns_a_valid_scored_tour(self):
        tour, score, history = run_ga(
            self.problem, pop_size=12, epochs=15, greedy_count=2, px=0.9,
            pm=0.05, elite_count=1, tournament_size=3, mutation="inversion",
            seed=42, print_every=0, patience=10,
        )
        self.assertTrue(validate_solution(tour, self.problem))
        self.assertEqual(score, fitness_length(tour, self.problem))
        self.assertGreaterEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
