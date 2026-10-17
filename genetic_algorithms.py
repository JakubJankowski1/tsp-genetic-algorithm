import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


FULL_LOG = False
SHOW_PLOTS = False


DATA_DIR = Path("data")

OPTIMAL_SCORES = {
    "berlin52": 7542,
    "kroA100": 21282,
    "kroA150": 26524,
}


# =========================================================
# Utilities — paths
# =========================================================
def _resolve_file(preferred: Path, alternatives: list[Path] | None = None) -> Path:

    alternatives = alternatives or []
    script_dir = Path(__file__).resolve().parent

    candidates = [preferred, *alternatives]

    # try as-is (relative to current working dir)
    for c in candidates:
        if c.exists():
            return c

    # try relative to script directory
    for c in candidates:
        c2 = script_dir / c
        if c2.exists():
            return c2

    # try inside DATA_DIR relative to script directory
    for c in candidates:
        if c.is_absolute():
            continue
        c3 = script_dir / DATA_DIR / c.name
        if c3.exists():
            return c3

    tried = "\n".join([f"- {str(x)}" for x in candidates])
    raise FileNotFoundError(
        f"Could not find file. Tried:\n{tried}\n\n"
        f"Fix: create folder 'data' next to your script and put .tsp files there, "
        f"or change DATA_DIR to Path('.') if files are next to the script."
    )


# =========================================================
# Part 1 — Parsing + problem representation
# =========================================================
def parse_tsp(filename: str) -> tuple[pd.DataFrame, str]:
    """
    Reads TSPLIB *.tsp file and returns:
      - DataFrame indexed by city_id with columns x,y
      - edge_weight_type (e.g. EUC_2D)
    """
    edge_weight_type = "EUC_2D"
    rows = []

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        in_coords = False
        for line in f:
            s = line.strip()
            if not s:
                continue

            if not in_coords:
                if s.upper().startswith("EDGE_WEIGHT_TYPE"):
                    parts = s.replace(":", " ").split()
                    if len(parts) >= 2:
                        edge_weight_type = parts[-1].strip()
                if s == "NODE_COORD_SECTION":
                    in_coords = True
                continue

            if s == "EOF":
                break
            p = s.split()
            if len(p) >= 3:
                rows.append((int(p[0]), float(p[1]), float(p[2])))

    df = pd.DataFrame(rows, columns=["city_id", "x", "y"]).set_index("city_id")
    return df, edge_weight_type


class TSPProblem:
    def __init__(self, df: pd.DataFrame, edge_weight_type: str = "EUC_2D"):
        self.df = df.copy()
        self.edge_weight_type = edge_weight_type

        self.ids = df.index.to_numpy(dtype=int)
        self.coords = df[["x", "y"]].to_numpy(dtype=float)
        self.id_to_idx = {cid: i for i, cid in enumerate(self.ids)}
        self.dist = self._build_dist_matrix(self.coords, self.edge_weight_type)

    @staticmethod
    def _build_dist_matrix(coords: np.ndarray, edge_weight_type: str) -> np.ndarray:
        """
        TSPLIB EUC_2D distance:
          dij = int( sqrt(dx^2 + dy^2) + 0.5 )
        which is equivalent to floor(d + 0.5).
        """
        dx = coords[:, 0][:, None] - coords[:, 0][None, :]
        dy = coords[:, 1][:, None] - coords[:, 1][None, :]

        ewt = edge_weight_type.upper()
        if ewt == "EUC_2D":
            d = np.sqrt(dx * dx + dy * dy)
            return np.floor(d + 0.5).astype(int)
        else:
            return np.hypot(dx, dy)


def validate_solution(sol: np.ndarray, problem: TSPProblem) -> bool:
    ids = problem.ids
    return (
        len(sol) == len(ids)
        and np.unique(sol).size == ids.size
        and set(sol.tolist()) == set(ids.tolist())
    )


def random_solution(problem: TSPProblem, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(problem.ids)


def city_distance(city_a: int, city_b: int, problem: TSPProblem) -> float | int:
    ia = problem.id_to_idx[int(city_a)]
    ib = problem.id_to_idx[int(city_b)]
    return problem.dist[ia, ib].item()


# =========================================================
# Part 2 — Fitness + greedy baseline + random baseline
# =========================================================
def fitness_length(solution: np.ndarray, problem: TSPProblem) -> float:
    idx = np.array([problem.id_to_idx[int(c)] for c in solution], dtype=int)
    nxt = np.roll(idx, -1)
    # dist may be int (EUC_2D) or float (fallback)
    return float(problem.dist[idx, nxt].sum())


def print_solution_info(solution: np.ndarray, problem: TSPProblem, label: str = "") -> None:
    seq = " ".join(map(str, solution.tolist()))
    score = fitness_length(solution, problem)
    print(f"{label}tour: {seq}")
    print(f"{label}score: {score:.6f}\n")


def greedy_nearest_neighbor(start_id: int, problem: TSPProblem) -> np.ndarray:
    n = len(problem.ids)
    start_idx = problem.id_to_idx[int(start_id)]

    remaining = set(range(n))
    remaining.remove(start_idx)

    tour_idx = [start_idx]
    cur = start_idx

    while remaining:
        rem_list = np.array(list(remaining), dtype=int)
        dists = problem.dist[cur, rem_list]
        best_pos = int(np.argmin(dists))
        nxt = int(rem_list[best_pos])
        tour_idx.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    return problem.ids[np.array(tour_idx, dtype=int)]


def greedy_all_starts_stats(problem: TSPProblem, full_log: bool = False) -> dict:
    scores = []
    tours = []

    for start in problem.ids:
        t = greedy_nearest_neighbor(int(start), problem)
        s = fitness_length(t, problem)
        tours.append(t)
        scores.append(s)

        # requirement: print results for each start city
        if full_log:
            print_solution_info(t, problem, label=f"[start {start}] ")
        else:
            print(f"[start {start}] score: {s:.6f}")

    scores = np.array(scores, dtype=float)
    order = np.argsort(scores)

    best_tour = tours[int(order[0])]
    best_score = float(scores[order[0]])

    print(
        f"Greedy all-starts summary -> min: {scores.min():.6f}, mean: {scores.mean():.6f}, "
        f"max: {scores.max():.6f}, std: {scores.std():.6f}, var: {scores.var():.6f}"
    )
    print("Greedy best 5:")
    for k in range(min(5, len(order))):
        i = int(order[k])
        print(f"  #{k+1}: start={problem.ids[i]} score={scores[i]:.6f}")
    print()

    return {"all_scores": scores, "best_tour": best_tour, "best_score": best_score}


def greedy_best_only(problem: TSPProblem) -> tuple[np.ndarray, float]:
    scores = []
    tours = []
    for start in problem.ids:
        t = greedy_nearest_neighbor(int(start), problem)
        s = fitness_length(t, problem)
        tours.append(t)
        scores.append(s)
    scores = np.array(scores, dtype=float)
    i = int(np.argmin(scores))
    return tours[i], float(scores[i])


def random_stats(problem: TSPProblem, n: int = 100, seed: int = 123, full_log: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    scores = np.empty(n, dtype=float)
    best_tour = None
    best_score = float("inf")

    for i in range(n):
        t = rng.permutation(problem.ids)
        s = fitness_length(t, problem)
        scores[i] = s

        # requirement: print info for random solutions
        if full_log:
            print_solution_info(t, problem, label=f"[rand {i+1}] ")
        else:
            print(f"[rand {i+1}] score: {s:.6f}")

        if s < best_score:
            best_score = float(s)
            best_tour = t.copy()

    print(
        f"Random {n} summary -> min: {scores.min():.6f}, mean: {scores.mean():.6f}, "
        f"max: {scores.max():.6f}, std: {scores.std():.6f}, var: {scores.var():.6f}"
    )
    if best_tour is not None:
        print_solution_info(best_tour, problem, label="[best random] ")

    return {"all_scores": scores, "best_tour": best_tour, "best_score": best_score}


def compare_to_opt(tag: str, score: float) -> None:
    if tag in OPTIMAL_SCORES:
        opt = float(OPTIMAL_SCORES[tag])
        gap = score - opt
        pct = (score / opt - 1.0) * 100.0
        print(f"[OPT] {tag}: optimal={opt:.0f} | got={score:.6f} | gap={gap:.6f} ({pct:.2f}%)\n")


def run_part2_checklist(problem: TSPProblem, tag: str, seed: int, random_n: int, full_log: bool) -> dict:
    print(f"=== Part 1: random solution on {tag} + validation ===")
    rs = random_solution(problem, seed=seed)
    print_solution_info(rs, problem, label=f"[random {tag}] ")
    print("valid=", validate_solution(rs, problem), "\n")

    print(f"=== Part 2 checklist: {tag} ===")
    g = greedy_all_starts_stats(problem, full_log=full_log)
    r = random_stats(problem, n=random_n, seed=seed, full_log=full_log)

    print(f"Compare ({tag}): GreedyBest={g['best_score']:.6f} | RandomBest={r['best_score']:.6f}\n")
    compare_to_opt(tag, g["best_score"])
    return {"greedy": g, "random": r}


# =========================================================
# Part 3 — GA operators
# =========================================================
Population = list[np.ndarray]


def initial_population(problem: TSPProblem, pop_size: int, greedy_count: int, seed: int) -> Population:
    rng = np.random.default_rng(seed)
    pop: Population = []
    seen = set()

    def _add(ind: np.ndarray):
        key = tuple(ind.tolist())
        if key not in seen:
            pop.append(ind)
            seen.add(key)

    greedy_count = max(0, min(greedy_count, pop_size))

    if greedy_count > 0:
        g_best, _ = greedy_best_only(problem)
        _add(g_best)

        starts = problem.ids.copy()
        rng.shuffle(starts)
        for s in starts:
            if len(pop) >= greedy_count:
                break
            _add(greedy_nearest_neighbor(int(s), problem))

    while len(pop) < pop_size:
        _add(rng.permutation(problem.ids))

    return pop


def population_scores(pop: Population, problem: TSPProblem) -> np.ndarray:
    return np.array([fitness_length(ind, problem) for ind in pop], dtype=float)


def tournament_select_idx(scores: np.ndarray, tournament_size: int, rng: np.random.Generator) -> int:
    idxs = rng.choice(len(scores), size=tournament_size, replace=False)
    return int(idxs[np.argmin(scores[idxs])])


def ordered_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(p1)
    i, j = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=int)

    child[i : j + 1] = p1[i : j + 1]
    used = set(child[i : j + 1].tolist())

    pos = (j + 1) % n
    p2_iter = list(p2[j + 1 :]) + list(p2[: j + 1])
    for gene in p2_iter:
        gene = int(gene)
        if gene in used:
            continue
        while child[pos] != -1:
            pos = (pos + 1) % n
        child[pos] = gene
        used.add(gene)

    return child


def mutate_swap(ind: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    out = ind.copy()
    n = len(out)
    for i in range(n):
        if rng.random() < pm:
            j = int(rng.integers(0, n))
            while j == i:
                j = int(rng.integers(0, n))
            out[i], out[j] = out[j], out[i]
    return out


def mutate_inversion(ind: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
    out = ind.copy()
    if rng.random() < pm:
        n = len(out)
        i, j = sorted(rng.choice(n, size=2, replace=False))
        out[i : j + 1] = out[i : j + 1][::-1]
    return out


def print_population_info(scores: np.ndarray, pop_size: int) -> None:
    print(
        f"  pop_size={pop_size} | best={scores.min():.6f} | median={np.median(scores):.6f} | "
        f"mean={scores.mean():.6f} | worst={scores.max():.6f} | std={scores.std():.6f}"
    )


# =========================================================
# Part 4 — GA run + history tracking
# =========================================================
def run_ga(
    problem: TSPProblem,
    pop_size: int,
    epochs: int,
    greedy_count: int,
    px: float,
    pm: float,
    elite_count: int,
    tournament_size: int,
    mutation: str,
    seed: int,
    print_every: int,
    patience: int,
    track_history: bool = True,
) -> tuple[np.ndarray, float, pd.DataFrame | None]:

    rng = np.random.default_rng(seed)
    pop = initial_population(problem, pop_size, greedy_count, seed=int(rng.integers(0, 1_000_000)))

    scores = population_scores(pop, problem)
    best_idx = int(np.argmin(scores))
    global_best = pop[best_idx].copy()
    global_best_score = float(scores[best_idx])

    history = [] if track_history else None
    no_improve = 0

    def _log(ep: int):
        if not track_history:
            return
        sc = population_scores(pop, problem)
        history.append(
            {
                "epoch": ep,
                "best_pop": float(np.min(sc)),
                "median": float(np.median(sc)),
                "mean": float(np.mean(sc)),
                "worst": float(np.max(sc)),
                "global_best": float(global_best_score),
            }
        )

    _log(0)

    for ep in range(1, epochs + 1):
        scores = population_scores(pop, problem)

        elite_count2 = max(0, min(elite_count, len(pop)))
        elite_idxs = np.argsort(scores)[:elite_count2]
        new_pop: Population = [pop[int(i)].copy() for i in elite_idxs]

        while len(new_pop) < len(pop):
            i1 = tournament_select_idx(scores, tournament_size, rng)
            i2 = tournament_select_idx(scores, tournament_size, rng)
            p1, p2 = pop[i1], pop[i2]

            if rng.random() < px:
                child = ordered_crossover(p1, p2, rng)
            else:
                child = p1.copy()

            if mutation == "swap":
                child = mutate_swap(child, pm, rng)
            elif mutation == "inversion":
                child = mutate_inversion(child, pm, rng)
            else:
                raise ValueError("mutation must be 'swap' or 'inversion'")

            new_pop.append(child)

        pop = new_pop

        scores = population_scores(pop, problem)
        best_idx = int(np.argmin(scores))
        ep_best_score = float(scores[best_idx])

        if ep_best_score < global_best_score:
            global_best_score = ep_best_score
            global_best = pop[best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1

        if print_every > 0 and (ep % print_every == 0 or ep == epochs):
            print(f"[epoch {ep}] best_pop={ep_best_score:.6f} global_best={global_best_score:.6f}")
            print_population_info(scores, pop_size)

        _log(ep)

        if no_improve >= patience:
            if print_every > 0:
                print(f"Stopping early (no improvement for {patience} epochs). Best: {global_best_score:.6f}\n")
            break

    hist_df = pd.DataFrame(history) if track_history else None
    return global_best, global_best_score, hist_df


# =========================================================
# Plotting
# =========================================================
def plot_history(history: pd.DataFrame, title: str, save_path: str | None, show: bool) -> None:
    plt.figure()
    plt.plot(history["epoch"], history["global_best"], label="global_best")
    plt.plot(history["epoch"], history["best_pop"], label="best_in_population")
    plt.plot(history["epoch"], history["mean"], label="mean")
    plt.plot(history["epoch"], history["median"], label="median")
    plt.plot(history["epoch"], history["worst"], label="worst")
    plt.xlabel("epoch")
    plt.ylabel("score (tour length)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_tour(problem: TSPProblem, tour: np.ndarray, title: str, save_path: str | None, show: bool) -> None:
    coords = problem.coords
    idx = np.array([problem.id_to_idx[int(c)] for c in tour], dtype=int)
    path = coords[idx]
    path2 = np.vstack([path, path[0]])

    plt.figure()
    plt.plot(path2[:, 0], path2[:, 1])
    plt.scatter(coords[:, 0], coords[:, 1])
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


# =========================================================
# Part 2 — Parameter comparison (same seed)
# =========================================================
def compare_configs_same_seed(
    problem: TSPProblem,
    configs: list[dict],
    seed: int,
    title: str,
    save_prefix: str | None,
    show: bool,
) -> pd.DataFrame:

    rows = []
    histories = {}

    for cfg in configs:
        name = cfg["name"]
        _, best_score, hist = run_ga(
            problem=problem,
            pop_size=cfg["pop_size"],
            epochs=cfg["epochs"],
            greedy_count=cfg["greedy_count"],
            px=cfg["px"],
            pm=cfg["pm"],
            elite_count=cfg["elite_count"],
            tournament_size=cfg["tournament_size"],
            mutation=cfg["mutation"],
            seed=seed,
            print_every=cfg["print_every"],
            patience=cfg["patience"],
            track_history=True,
        )

        rows.append({**cfg, "best_score": float(best_score)})
        histories[name] = hist

        if save_prefix:
            hist.to_csv(f"{save_prefix}_{name}_history.csv", index=False)

    res = pd.DataFrame(rows).sort_values("best_score")

    plt.figure()
    for name, hist in histories.items():
        plt.plot(hist["epoch"], hist["global_best"], label=name)
    plt.xlabel("epoch")
    plt.ylabel("score (tour length)")
    plt.title(title + " (global best)")
    plt.legend()
    plt.grid(True)
    if save_prefix:
        plt.savefig(f"{save_prefix}_curves.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure()
    plt.bar(res["name"], res["best_score"])
    plt.xlabel("config")
    plt.ylabel("best_score")
    plt.title(title + " (final best score)")
    plt.xticks(rotation=30, ha="right")
    plt.grid(True, axis="y")
    if save_prefix:
        plt.savefig(f"{save_prefix}_final_scores.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    if save_prefix:
        res.to_csv(f"{save_prefix}_summary.csv", index=False)

    return res


# =========================================================
# Part 3 — 10 GA runs + statistics
# =========================================================
def run_ga_10_times(problem: TSPProblem, best_cfg: dict, seed_base: int = 1000) -> pd.DataFrame:
    rows = []

    for i in range(10):
        seed = seed_base + i
        _, score, _ = run_ga(
            problem=problem,
            pop_size=int(best_cfg["pop_size"]),
            epochs=int(best_cfg["epochs"]),
            greedy_count=int(best_cfg["greedy_count"]),
            px=float(best_cfg["px"]),
            pm=float(best_cfg["pm"]),
            elite_count=int(best_cfg["elite_count"]),
            tournament_size=int(best_cfg["tournament_size"]),
            mutation=str(best_cfg["mutation"]),
            seed=seed,
            print_every=0,
            patience=int(best_cfg["patience"]),
            track_history=False,
        )
        rows.append({"run": i + 1, "seed": seed, "best_score": float(score)})

    df = pd.DataFrame(rows)
    scores = df["best_score"].to_numpy(dtype=float)

    print(
        "GA 10 runs summary -> "
        f"min: {scores.min():.6f}, mean: {scores.mean():.6f}, max: {scores.max():.6f}, "
        f"std: {scores.std():.6f}, var: {scores.var():.6f}\n"
    )
    return df


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    # Find tsp files (try both names for berlin11)
    berlin11_path = _resolve_file(DATA_DIR / "berlin11.tsp", [DATA_DIR / "berlin11_modified.tsp", Path("berlin11.tsp"), Path("berlin11_modified.tsp")])
    berlin52_path = _resolve_file(DATA_DIR / "berlin52.tsp", [Path("berlin52.tsp")])
    kroA100_path = _resolve_file(DATA_DIR / "kroA100.tsp", [Path("kroA100.tsp")])
    kroA150_path = _resolve_file(DATA_DIR / "kroA150.tsp", [Path("kroA150.tsp")])

    files = {
        "berlin11_modified": berlin11_path,
        "berlin52": berlin52_path,
        "kroA100": kroA100_path,
        "kroA150": kroA150_path,
    }

    problems = {}
    for name, path in files.items():
        df, ewt = parse_tsp(str(path))
        problems[name] = TSPProblem(df, edge_weight_type=ewt)

    print("=== Part 1: parser test (first 5 rows) ===")
    print("berlin11_modified head:\n", problems["berlin11_modified"].df.head(), "\n")
    print("berlin52 head:\n", problems["berlin52"].df.head(), "\n")

    # Part 2 checklist required
    base11 = run_part2_checklist(problems["berlin11_modified"], tag="berlin11_modified", seed=42, random_n=100, full_log=FULL_LOG)
    base52 = run_part2_checklist(problems["berlin52"], tag="berlin52", seed=42, random_n=100, full_log=FULL_LOG)

    print("=== Parameter comparison (kroA100, same seed) ===")
    seed_cmp = 111

    cfgs_kroA100 = [
        dict(
            name="inv_pm0.02",
            pop_size=160,
            epochs=350,
            greedy_count=5,
            px=0.9,
            pm=0.02,
            elite_count=3,
            tournament_size=5,
            mutation="inversion",
            print_every=0,
            patience=80,
        ),
        dict(
            name="inv_pm0.05",
            pop_size=160,
            epochs=350,
            greedy_count=5,
            px=0.9,
            pm=0.05,
            elite_count=3,
            tournament_size=5,
            mutation="inversion",
            print_every=0,
            patience=80,
        ),
        dict(
            name="swap_pm0.02",
            pop_size=160,
            epochs=350,
            greedy_count=5,
            px=0.9,
            pm=0.02,
            elite_count=3,
            tournament_size=5,
            mutation="swap",
            print_every=0,
            patience=80,
        ),
    ]

    summary_kroA100 = compare_configs_same_seed(
        problem=problems["kroA100"],
        configs=cfgs_kroA100,
        seed=seed_cmp,
        title="kroA100 parameter comparison",
        save_prefix="kroA100",
        show=SHOW_PLOTS,
    )

    print(
        summary_kroA100[["name", "best_score", "pop_size", "epochs", "pm", "tournament_size", "mutation"]]
        .to_string(index=False),
        "\n",
    )

    best_cfg = summary_kroA100.iloc[0].to_dict()

    print("=== GA vs greedy vs random (kroA100) ===")
    g = greedy_all_starts_stats(problems["kroA100"], full_log=False)
    r = random_stats(problems["kroA100"], n=1000, seed=42, full_log=False)
    print(f"Compare (kroA100): GreedyBest={g['best_score']:.6f} | RandomBest={r['best_score']:.6f}\n")
    compare_to_opt("kroA100", best_cfg["best_score"] if "best_score" in best_cfg else np.nan)

    ga10 = run_ga_10_times(problems["kroA100"], best_cfg=best_cfg, seed_base=2000)
    print("GA 10 runs table:\n", ga10.to_string(index=False), "\n")

    print("=== Best GA solution for 11, 52, 100, 150 ===")
    chosen_for_tour_plot = {"kroA100", "kroA150"}

    for name in ["berlin11_modified", "berlin52", "kroA100", "kroA150"]:
        prob = problems[name]

        if len(prob.ids) <= 60:
            cfg_run = dict(
                pop_size=120,
                epochs=500,
                greedy_count=5,
                px=0.9,
                pm=0.05,
                elite_count=3,
                tournament_size=5,
                mutation="inversion",
                patience=120,
            )
        else:
            cfg_run = dict(
                pop_size=int(best_cfg["pop_size"]),
                epochs=int(best_cfg["epochs"]),
                greedy_count=int(best_cfg["greedy_count"]),
                px=float(best_cfg["px"]),
                pm=float(best_cfg["pm"]),
                elite_count=int(best_cfg["elite_count"]),
                tournament_size=int(best_cfg["tournament_size"]),
                mutation=str(best_cfg["mutation"]),
                patience=int(best_cfg["patience"]),
            )

        best_tour, best_score, hist = run_ga(
            problem=prob,
            seed=999,
            print_every=50 if name in ("berlin11_modified", "berlin52") else 0,
            track_history=True,
            **cfg_run,
        )

        print(f"{name}: GA_best={best_score:.6f} valid={validate_solution(best_tour, prob)}")
        compare_to_opt(name, best_score)

        plot_history(hist, f"{name} - GA progress", save_path=f"{name}_progress.png", show=SHOW_PLOTS)

        if name in chosen_for_tour_plot:
            plot_tour(
                prob,
                best_tour,
                f"{name} - best tour ({best_score:.2f})",
                save_path=f"{name}_best_tour.png",
                show=SHOW_PLOTS,
            )

    print("\nDone.")
