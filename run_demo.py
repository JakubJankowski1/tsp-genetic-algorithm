"""Run one reproducible, moderately sized experiment on berlin52."""

from pathlib import Path

from genetic_algorithms import (
    OPTIMAL_SCORES,
    TSPProblem,
    fitness_length,
    parse_tsp,
    plot_history,
    plot_tour,
    run_ga,
    validate_solution,
)


ROOT = Path(__file__).resolve().parent


def main() -> None:
    instance = ROOT / "data" / "berlin52.tsp"
    if not instance.exists():
        raise SystemExit("Missing data/berlin52.tsp. Run: python prepare_data.py")

    coordinates, edge_weight_type = parse_tsp(str(instance))
    problem = TSPProblem(coordinates, edge_weight_type)
    best_tour, best_score, history = run_ga(
        problem=problem,
        pop_size=80,
        epochs=150,
        greedy_count=5,
        px=0.9,
        pm=0.05,
        elite_count=3,
        tournament_size=5,
        mutation="inversion",
        seed=999,
        print_every=0,
        patience=50,
    )
    if not validate_solution(best_tour, problem) or fitness_length(best_tour, problem) != best_score:
        raise RuntimeError("Invalid or incorrectly scored tour")

    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    plot_history(history, "berlin52 — GA progress", str(output / "berlin52_progress.png"), False)
    plot_tour(problem, best_tour, f"berlin52 — tour length {best_score:.0f}",
              str(output / "berlin52_tour.png"), False)
    print(f"Valid tour: {len(best_tour)} cities; length: {best_score:.0f}")
    print(f"TSPLIB optimum: {OPTIMAL_SCORES['berlin52']}; gap: {best_score - OPTIMAL_SCORES['berlin52']:.0f}")
    print(f"Charts saved in: {output}")


if __name__ == "__main__":
    main()
