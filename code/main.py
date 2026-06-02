import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from instance import read_json_instance
from construction import build_initial_solution
from tabu_search import advanced_tabu_search, TabuSearchConfig

# ─────────────────────────────────────────────────────────────────────────────
# Điền tên file JSON muốn test vào đây
# ─────────────────────────────────────────────────────────────────────────────
TEST_FILES = [
    "6.5.1.json",
    "10.5.1.json",
]

DATA_DIR = "WithTimeWindows"

CFG = TabuSearchConfig(
    max_iter       = 2000,
    max_no_improve = 300,
    tenure_base    = 7,
    time_limit     = 120.0,
    verbose        = True,
)
# ─────────────────────────────────────────────────────────────────────────────

def process(filepath: str):
    print(f"\n{'='*65}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*65}")

    inst = read_json_instance(filepath)
    print(f"  Trucks : {inst.num_trucks}  (cap={inst.truck_capacity}, speed={inst.truck_speed})")
    print(f"  Drones : {inst.num_drones}  (cap={inst.drone_capacity}, "
          f"range={inst.drone_range}, speed={inst.drone_speed})")
    print(f"  Khách  : {len(inst.customers)}  "
          f"(C1/chỉ-truck={sorted(inst.c1_ids)}, C2/drone-ok={sorted(inst.c2_ids)})")

    # Construction
    t0 = time.time()
    init_sol = build_initial_solution(inst)
    t_con = time.time() - t0
    print(f"\n  [Construction] {t_con:.3f}s"
          f"  Makespan={init_sol.makespan():.2f}"
          f"  Feasible={init_sol.is_feasible(inst)}"
          f"  AllServed={init_sol.all_served(inst)}")
    print(init_sol.summary(inst))

    # Tabu Search
    print(f"\n  [Tabu Search] đang chạy...")
    t0 = time.time()
    best_sol, history = advanced_tabu_search(init_sol, inst, CFG)
    t_ts = time.time() - t0

    print(f"\n  [Kết quả] {t_ts:.1f}s"
          f"  Makespan={best_sol.makespan():.2f}"
          f"  Feasible={best_sol.is_feasible(inst)}"
          f"  AllServed={best_sol.all_served(inst)}")
    if len(history) > 1:
        print(f"  Cải thiện: {history[0]:.2f} → {history[-1]:.2f} "
              f"(giảm {history[0]-history[-1]:.2f})")
    print(best_sol.summary(inst))


def main():
    for fname in TEST_FILES:
        candidates = [
            fname,
            os.path.join(DATA_DIR, fname),
            os.path.join(os.path.dirname(__file__), fname),
            os.path.join(os.path.dirname(__file__), DATA_DIR, fname),
        ]
        fp = next((p for p in candidates if os.path.isfile(p)), None)
        if fp is None:
            print(f"[KHÔNG TÌM THẤY] {fname}")
            continue
        try:
            process(fp)
        except Exception as e:
            import traceback
            print(f"[LỖI] {fname}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
