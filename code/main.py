import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from instance import read_json_instance, Instance
from solution import Solution
from solomon_i1 import solomon_i1_construction, multi_start_i1
from tabu_search import advanced_tabu_search, TabuConfig

# ─────────────────────────────────────────────────────────────────────────────
# CẤU HÌNH: Điền tên file JSON vào list bên dưới
# Ví dụ: ["6_5_1.json", "10_5_1.json", "15_5_1.json"]
# ─────────────────────────────────────────────────────────────────────────────

TEST_FILES = [
    "6.5.1.json",
    "10.5.1.json",
]

# Thư mục chứa các file JSON (thay đổi nếu cần)
DATA_DIR = "WithTimeWindows"

# ─────────────────────────────────────────────────────────────────────────────
# Cấu hình thuật toán
# ─────────────────────────────────────────────────────────────────────────────

TABU_CFG = TabuConfig(
    max_iterations   = 500,
    tabu_tenure      = 10,
    diversify_thresh = 40,
    verbose          = False,
)

# ─────────────────────────────────────────────────────────────────────────────

def process_instance(filepath: str):
    print(f"\n{'='*65}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*65}")

    inst = read_json_instance(filepath)

    print(f"  Trucks   : {inst.num_trucks}  (cap={inst.truck_capacity}, speed={inst.truck_speed})")
    print(f"  Drones   : {inst.num_drones}  (cap={inst.drone_capacity}, range={inst.drone_range}, speed={inst.drone_speed})")
    print(f"  Khách    : {len(inst.customers)}  (C1/chỉ-truck={len(inst.c1_ids)}, C2/drone-ok={len(inst.c2_ids)})")
    print(f"  C1 ids   : {sorted(inst.c1_ids)}")
    print(f"  Depot    : x={inst.depot.x:.2f}, y={inst.depot.y:.2f}, due={inst.depot.due:.2f}")

    # 1. Construction
    t0 = time.time()
    init_sol = solomon_i1_construction(inst)
    t_con = time.time() - t0

    print(f"\n  [Construction] {t_con:.3f}s  |  "
          f"Makespan: {init_sol.makespan():.2f}  |  "
          f"Feasible: {init_sol.is_feasible(inst)}  |  "
          f"All served: {init_sol.all_served(inst)}")

    # 2. Tabu Search
    t0 = time.time()
    best_sol = advanced_tabu_search(init_sol, inst, TABU_CFG)
    t_ts = time.time() - t0

    print(f"  [Tabu Search]  {t_ts:.3f}s  |  "
          f"Makespan: {best_sol.makespan():.2f}  |  "
          f"Feasible: {best_sol.is_feasible(inst)}  |  "
          f"All served: {best_sol.all_served(inst)}")

    print(f"\n{best_sol.summary(inst)}")


def main():
    print(f"Số file cần chạy: {len(TEST_FILES)}")

    for fname in TEST_FILES:
        # Thử tìm file theo thứ tự: cùng thư mục → DATA_DIR → đường dẫn tuyệt đối
        candidates = [
            fname,
            os.path.join(DATA_DIR, fname),
            os.path.join(os.path.dirname(__file__), fname),
            os.path.join(os.path.dirname(__file__), DATA_DIR, fname),
        ]
        filepath = next((p for p in candidates if os.path.isfile(p)), None)

        if filepath is None:
            print(f"\n[KHÔNG TÌM THẤY] {fname}  "
                  f"(đã tìm: {', '.join(candidates)})")
            continue

        try:
            process_instance(filepath)
        except Exception as e:
            import traceback
            print(f"\n[LỖI] {fname}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()