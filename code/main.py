"""
main.py — Chạy MVRPD-TW Tabu Search với Solomon I1 Construction

Cách dùng:
    python main.py                              # instance mẫu 10 khách
    python main.py --file mydata.txt            # Solomon format
    python main.py --file mydata.txt --format custom
    python main.py --file c101.txt --trucks 3 --drones 2 --range 80
    python main.py --compare                    # so sánh NN vs Solomon I1
    python main.py --multi-start 5              # multi-start I1
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from instance import read_solomon, read_custom, Instance, Customer
from solution import Solution
from solomon_i1 import solomon_i1_construction, multi_start_i1
from tabu_search import tabu_search, TabuSearchConfig


def make_sample_instance() -> Instance:
    depot = Customer(id=0, x=40, y=50, demand=0,
                     ready=0, due=1236, service=0)
    raw = [
        (1,  45, 68,  10, 912, 967,  90, False),
        (2,  45, 70,  30, 825, 870,  90, True),
        (3,  42, 66,  10, 652, 721,  90, False),
        (4,  42, 68,  10, 148, 194,  90, False),
        (5,  42, 65,  10, 177, 218,  90, False),
        (6,  40, 69,  20, 255, 324,  90, True),
        (7,  40, 66,  20, 587, 629,  90, True),
        (8,  38, 68,  20, 897, 941,  90, True),
        (9,  38, 70,  10, 743, 820,  90, False),
        (10, 35, 66,  10, 557, 609,  90, False),
    ]
    customers, c1_ids, c2_ids = [], set(), set()
    for r in raw:
        c = Customer(id=r[0], x=r[1], y=r[2], demand=r[3],
                     ready=r[4], due=r[5], service=r[6], is_c1=r[7])
        customers.append(c)
        (c1_ids if c.is_c1 else c2_ids).add(c.id)
    inst = Instance(
        name="Sample-10", num_trucks=2, num_drones=2,
        truck_capacity=200, drone_capacity=15, drone_range=60.0,
        depot=depot, customers=customers,
        c1_ids=c1_ids, c2_ids=c2_ids,
    )
    inst.build_dist()
    return inst


def compare_constructions(inst: Instance):
    from construction import greedy_construction
    print("\n" + "="*65)
    print("SO SÁNH CÁC PHƯƠNG PHÁP CONSTRUCTION")
    print("="*65)
    print(f"{'Phương pháp':<28} {'Makespan':>10} {'TW Pen':>8} "
          f"{'Feasible':>9} {'Phủ đủ':>7}")
    print("-"*65)

    configs = [
        ("Nearest Neighbor (cũ)",
         lambda: greedy_construction(inst)),
        ("Solomon I1 (mu=1, dist)",
         lambda: solomon_i1_construction(inst, mu=1.0, lam=1.0)),
        ("Solomon I1 (mu=0, time)",
         lambda: solomon_i1_construction(inst, mu=0.0, lam=1.0)),
        ("Solomon I1 (mu=0.5, mix)",
         lambda: solomon_i1_construction(inst, mu=0.5, lam=1.0)),
        ("Solomon I1 (urgent seed)",
         lambda: solomon_i1_construction(
             inst, mu=0.5, lam=1.0,
             seed_criterion_truck='urgent')),
        ("Multi-start I1 (best of 5)",
         lambda: multi_start_i1(inst, n_starts=5)),
    ]

    best_sol, best_score = None, float('inf')
    best_name = None

    for name, fn in configs:
        try:
            sol = fn()
            mk  = sol.makespan()
            tw  = sol.penalty_tw(inst)
            feas = sol.is_feasible(inst)
            full = sol.all_served(inst)
            score = mk + tw * 1000
            flag = ""
            if score < best_score:
                best_score = score
                best_sol   = sol
                best_name  = name
                flag = " ← tốt nhất"
            print(f"  {name:<26} {mk:>10.1f} {tw:>8.1f} "
                  f"{'Có' if feas else 'Không':>9} "
                  f"{'Có' if full else 'Không':>7}{flag}")
        except Exception as e:
            print(f"  {name:<26} {'LỖI':>10}  ({e})")

    print("-"*65)
    print(f"\nPhương pháp tốt nhất: {best_name}")
    return best_sol


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',        type=str,   default=None)
    parser.add_argument('--format',      type=str,   default='solomon',
                        choices=['solomon', 'custom'])
    parser.add_argument('--trucks',      type=int,   default=2)
    parser.add_argument('--drones',      type=int,   default=2)
    parser.add_argument('--truck-cap',   type=float, default=None)
    parser.add_argument('--drone-cap',   type=float, default=30.0)
    parser.add_argument('--range',       type=float, default=100.0)
    parser.add_argument('--iter',        type=int,   default=500)
    parser.add_argument('--tenure',      type=int,   default=10)
    parser.add_argument('--time',        type=float, default=120.0)
    parser.add_argument('--mu',          type=float, default=0.5)
    parser.add_argument('--lam',         type=float, default=1.0)
    parser.add_argument('--multi-start', type=int,   default=1)
    parser.add_argument('--compare',     action='store_true')
    parser.add_argument('--quiet',       action='store_true')
    args = parser.parse_args()

    # Đọc instance
    if args.file:
        print(f"Đọc file: {args.file}  (format={args.format})")
        inst = read_solomon(
            args.file, num_trucks=args.trucks, num_drones=args.drones,
            truck_capacity=args.truck_cap,
            drone_capacity=args.drone_cap,
            drone_range=args.range,
        ) if args.format == 'solomon' else read_custom(args.file)
    else:
        print("Dùng instance mẫu 10 khách hàng.\n")
        inst = make_sample_instance()

    print(f"Instance : {inst}")
    print(f"  C1 = {sorted(inst.c1_ids)}")
    print(f"  C2 = {sorted(inst.c2_ids)}\n")

    # Construction
    if args.compare:
        init_sol = compare_constructions(inst)
    else:
        t0 = time.time()
        if args.multi_start > 1:
            print(f"Multi-start Solomon I1 ({args.multi_start} lần)...")
            init_sol = multi_start_i1(inst, n_starts=args.multi_start)
        else:
            print(f"Solomon I1 (mu={args.mu}, lam={args.lam})...")
            init_sol = solomon_i1_construction(
                inst, mu=args.mu, lam=args.lam)

        print(f"Construction: {time.time()-t0:.3f}s  |  "
              f"makespan={init_sol.makespan():.1f}  |  "
              f"feasible={init_sol.is_feasible(inst)}  |  "
              f"phủ đủ={init_sol.all_served(inst)}\n")

    # Tabu Search
    print("Bắt đầu Tabu Search...")
    cfg = TabuSearchConfig(
        max_iter        = args.iter,
        max_no_improve  = max(100, args.iter // 4),
        diversify_thresh= max(40,  args.iter // 8),
        tenure_base     = args.tenure,
        time_limit      = args.time,
        verbose         = not args.quiet,
    )
    best_sol, history = tabu_search(inst, cfg, init_solution=init_sol)

    print("\n" + "="*60)
    print("KẾT QUẢ CUỐI CÙNG")
    print("="*60)
    print(best_sol.summary(inst))
    if len(history) > 1:
        print(f"\nCải thiện: {history[0]:.1f} → {history[-1]:.1f}  "
              f"(giảm {history[0]-history[-1]:.1f})")


if __name__ == '__main__':
    main()
