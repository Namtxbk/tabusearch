"""
main.py — Chạy thuật toán MVRPD-TW Tabu Search

Cách dùng:
    python main.py                          # chạy với file mẫu nội bộ
    python main.py --file mydata.txt        # Solomon format
    python main.py --file mydata.txt --format custom
    python main.py --file c101.txt --trucks 3 --drones 2 --range 80
"""

import argparse
import sys
import os

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(__file__))

from instance import read_solomon, read_custom, Instance, Customer
from solution import Solution
from tabu_search import tabu_search, TabuSearchConfig


# ─────────────────────────────────────────────────────────────────────────────
# Tạo instance mẫu nhỏ (không cần file)
# ─────────────────────────────────────────────────────────────────────────────

def make_sample_instance() -> Instance:
    """
    Instance nhỏ 10 khách hàng để test nhanh.
    C1 = {1,2,3} (demand lớn, chỉ truck)
    C2 = {4..10} (drone hoặc truck)
    """
    from instance import Instance, Customer
    import math

    depot = Customer(id=0, x=40, y=50, demand=0,
                     ready=0, due=1236, service=0)

    raw = [
        # id   x     y   demand  ready   due  service  c1?
        (1,  45,  68,   10,    912,   967,   90,  False),
        (2,  45,  70,   30,    825,   870,   90,  True),   # C1: demand 30
        (3,  42,  66,   10,    652,   721,   90,  False),
        (4,  42,  68,   10,    148,   194,   90,  False),
        (5,  42,  65,   10,    177,   218,   90,  False),
        (6,  40,  69,   20,    255,   324,   90,  True),   # C1: demand 20
        (7,  40,  66,   20,    587,   629,   90,  True),   # C1: demand 20
        (8,  38,  68,   20,    897,   941,   90,  True),   # C1: demand 20
        (9,  38,  70,   10,    743,   820,   90,  False),
        (10, 35,  66,   10,    557,   609,   90,  False),
    ]

    customers = []
    c1_ids, c2_ids = set(), set()
    for r in raw:
        c = Customer(id=r[0], x=r[1], y=r[2], demand=r[3],
                     ready=r[4], due=r[5], service=r[6], is_c1=r[7])
        customers.append(c)
        if c.is_c1:
            c1_ids.add(c.id)
        else:
            c2_ids.add(c.id)

    inst = Instance(
        name="Sample-10",
        num_trucks=2,
        num_drones=2,
        truck_capacity=200,
        drone_capacity=15,
        drone_range=60.0,
        depot=depot,
        customers=customers,
        c1_ids=c1_ids,
        c2_ids=c2_ids,
    )
    inst.build_dist()
    return inst


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MVRPD-TW Tabu Search Solver"
    )
    parser.add_argument('--file',    type=str, default=None,
                        help='Đường dẫn file instance (.txt)')
    parser.add_argument('--format',  type=str, default='solomon',
                        choices=['solomon', 'custom'],
                        help='Định dạng file: solomon hoặc custom')
    parser.add_argument('--trucks',  type=int, default=2,
                        help='Số xe tải K (default: 2)')
    parser.add_argument('--drones',  type=int, default=2,
                        help='Số drone D (default: 2)')
    parser.add_argument('--truck-cap', type=float, default=None,
                        help='Tải trọng truck (None → đọc từ file)')
    parser.add_argument('--drone-cap', type=float, default=30.0,
                        help='Tải trọng drone (default: 30)')
    parser.add_argument('--range',   type=float, default=100.0,
                        help='Tầm bay tối đa drone L_D (default: 100)')
    parser.add_argument('--iter',    type=int, default=500,
                        help='Số iteration tối đa (default: 500)')
    parser.add_argument('--tenure',  type=int, default=10,
                        help='Tabu tenure cơ sở (default: 10)')
    parser.add_argument('--time',    type=float, default=120.0,
                        help='Giới hạn thời gian (giây, default: 120)')
    parser.add_argument('--quiet',   action='store_true',
                        help='Tắt verbose output')

    args = parser.parse_args()

    # ── Đọc instance ─────────────────────────────────────────────────────
    if args.file:
        print(f"Đọc file: {args.file}  (format={args.format})")
        if args.format == 'solomon':
            inst = read_solomon(
                args.file,
                num_trucks=args.trucks,
                num_drones=args.drones,
                truck_capacity=args.truck_cap,
                drone_capacity=args.drone_cap,
                drone_range=args.range,
            )
        else:
            inst = read_custom(args.file)
    else:
        print("Không có file đầu vào → dùng instance mẫu 10 khách hàng.\n")
        inst = make_sample_instance()

    print(f"Instance: {inst}")
    print(f"  |C1| = {len(inst.c1_ids)} (chỉ truck): {sorted(inst.c1_ids)}")
    print(f"  |C2| = {len(inst.c2_ids)} (truck/drone): {sorted(inst.c2_ids)}")
    print()

    # ── Cấu hình và chạy ─────────────────────────────────────────────────
    cfg = TabuSearchConfig(
        max_iter        = args.iter,
        max_no_improve  = max(100, args.iter // 4),
        diversify_thresh= max(40,  args.iter // 8),
        tenure_base     = args.tenure,
        time_limit      = args.time,
        verbose         = not args.quiet,
    )

    best_sol, history = tabu_search(inst, cfg)

    # ── Kết quả cuối ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*60)
    print(best_sol.summary(inst))

    print(f"\nLịch sử makespan (5 điểm):")
    step = max(1, len(history) // 5)
    for i in range(0, len(history), step):
        print(f"  iter {i:4d}: {history[i]:.2f}")
    print(f"  iter {len(history)-1:4d}: {history[-1]:.2f}  (final)")


if __name__ == '__main__':
    main()
