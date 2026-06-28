"""
batch_compare.py — Chạy hàng loạt instance và so sánh với baseline CSV.

Cách dùng:
    python batch_compare.py \
        --data_dir WithTimeWindows \
        --baseline capacity-1400_baseline.csv \
        --output ket_qua_so_sanh.csv

    python batch_compare.py --data_dir WithTimeWindows --baseline result.csv --output ket_qua_so_sanh.csv

Yêu cầu:
    - Các file instance .json đặt trong --data_dir, tên dạng "6.5.1.json", "10.10.1.json", v.v.
    - File baseline CSV có cột "Problem" và "Truck working time" / "Drone working time"
      (mỗi instance có thể có nhiều dòng — script tự lấy makespan TỐT NHẤT trong các dòng đó).

Kết quả:
    Một file CSV duy nhất, mỗi dòng là 1 instance, gồm:
      Problem, Makespan_baseline, Makespan_thuat_toan, Gap (%),
      Feasible, AllServed, Construction_time(s), TS_time(s)
"""
import os
import sys
import re
import csv
import time
import json
import argparse
import ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instance import read_json_instance
from construction import build_initial_solution
from tabu_search import advanced_tabu_search, TabuSearchConfig


# ─────────────────────────────────────────────────────────────────────────────
# Đọc baseline CSV, lấy makespan TỐT NHẤT (nhỏ nhất) cho mỗi Problem
# ─────────────────────────────────────────────────────────────────────────────

def _parse_time_list(s: str) -> float:
    """Parse chuỗi dạng '[1234.56]' thành float. Trả về 0 nếu rỗng/lỗi."""
    try:
        val = ast.literal_eval(s.strip())
        if isinstance(val, list) and len(val) > 0:
            return max(float(x) for x in val)
        return float(val)
    except Exception:
        return 0.0


def load_baseline(csv_path: str) -> dict:
    """
    Trả về dict: { problem_name: best_makespan }
    Makespan của 1 dòng = max(truck_working_time, drone_working_time)
    Với mỗi Problem, lấy makespan NHỎ NHẤT trong số các dòng (baseline tốt nhất).
    """
    best = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem = row['Problem'].strip()
            truck_t = _parse_time_list(row.get('Truck working time', '[0]'))
            drone_t = _parse_time_list(row.get('Drone working time', '[0]'))
            makespan = max(truck_t, drone_t)

            if problem not in best or makespan < best[problem]:
                best[problem] = makespan

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Chạy thuật toán trên 1 file instance
# ─────────────────────────────────────────────────────────────────────────────

def run_one(filepath: str, cfg: TabuSearchConfig) -> dict:
    name = os.path.splitext(os.path.basename(filepath))[0]

    inst = read_json_instance(filepath)

    t0 = time.time()
    init_sol = build_initial_solution(inst)
    t_construction = time.time() - t0

    t0 = time.time()
    best_sol, history = advanced_tabu_search(init_sol, inst, cfg)
    t_ts = time.time() - t0

    return {
        'Problem':          name,
        'Makespan_algo':    round(best_sol.makespan(), 4),
        'Feasible':         best_sol.is_feasible(inst),
        'AllServed':        best_sol.all_served(inst),
        'Construction_s':   round(t_construction, 3),
        'TS_s':             round(t_ts, 3),
        'Penalty_TW':       round(best_sol.penalty_tw(inst), 4),
        'Penalty_Cap':      round(best_sol.penalty_cap(inst), 4),
        'Penalty_Range':    round(best_sol.penalty_range(inst), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def natural_key(name: str):
    """Sắp xếp tên instance kiểu '6.5.1', '10.5.1' theo thứ tự số tự nhiên."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', name)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',  required=True, help='Thư mục chứa các file .json instance')
    ap.add_argument('--baseline',  required=True, help='File CSV baseline để so sánh')
    ap.add_argument('--output',    default='ket_qua_so_sanh.csv', help='File CSV kết quả')
    ap.add_argument('--max_iter',        type=int,   default=2000)
    ap.add_argument('--max_no_improve',  type=int,   default=300)
    ap.add_argument('--tenure_base',     type=int,   default=7)
    ap.add_argument('--time_limit',      type=float, default=300.0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    cfg = TabuSearchConfig(
        max_iter=args.max_iter,
        max_no_improve=args.max_no_improve,
        tenure_base=args.tenure_base,
        time_limit=args.time_limit,
        verbose=args.verbose,
    )

    print(f"Đang đọc baseline từ: {args.baseline}")
    baseline = load_baseline(args.baseline)
    print(f"  -> Tìm thấy {len(baseline)} instance trong baseline.\n")

    json_files = sorted(
        [f for f in os.listdir(args.data_dir) if f.endswith('.json')],
        key=lambda f: natural_key(os.path.splitext(f)[0])
    )

    if not json_files:
        print(f"[LỖI] Không tìm thấy file .json nào trong {args.data_dir}")
        return

    rows = []
    n_total = len(json_files)

    for idx, fname in enumerate(json_files, start=1):
        fpath = os.path.join(args.data_dir, fname)
        problem_name = os.path.splitext(fname)[0]

        print(f"[{idx}/{n_total}] Đang chạy {problem_name} ...", end=' ', flush=True)
        try:
            result = run_one(fpath, cfg)
        except Exception as e:
            import traceback
            print(f"LỖI: {e}")
            traceback.print_exc()
            rows.append({
                'Problem': problem_name,
                'Makespan_baseline': baseline.get(problem_name, ''),
                'Makespan_algo': '',
                'Gap_%': '',
                'Feasible': 'ERROR',
                'AllServed': '',
                'Construction_s': '',
                'TS_s': '',
                'Penalty_TW': '',
                'Penalty_Cap': '',
                'Penalty_Range': '',
            })
            continue

        ms_base = baseline.get(problem_name)
        ms_algo = result['Makespan_algo']

        if ms_base is not None and ms_base > 0:
            gap = (ms_algo - ms_base) / ms_base * 100.0
        else:
            gap = ''

        print(f"Makespan={ms_algo:.2f}  Baseline={ms_base if ms_base else 'N/A'}  "
              f"Gap={gap:.2f}%" if gap != '' else f"Makespan={ms_algo:.2f}  Baseline=N/A")

        rows.append({
            'Problem':          problem_name,
            'Makespan_baseline': round(ms_base, 4) if ms_base is not None else '',
            'Makespan_algo':    ms_algo,
            'Gap_%':            round(gap, 4) if gap != '' else '',
            'Feasible':         result['Feasible'],
            'AllServed':        result['AllServed'],
            'Construction_s':   result['Construction_s'],
            'TS_s':             result['TS_s'],
            'Penalty_TW':       result['Penalty_TW'],
            'Penalty_Cap':      result['Penalty_Cap'],
            'Penalty_Range':    result['Penalty_Range'],
        })

    # Ghi file kết quả
    fieldnames = ['Problem', 'Makespan_baseline', 'Makespan_algo', 'Gap_%',
                  'Feasible', 'AllServed', 'Construction_s', 'TS_s',
                  'Penalty_TW', 'Penalty_Cap', 'Penalty_Range']

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Thống kê tổng hợp
    valid_gaps = [r['Gap_%'] for r in rows if isinstance(r['Gap_%'], (int, float))]
    n_feasible = sum(1 for r in rows if r['Feasible'] is True)
    n_served   = sum(1 for r in rows if r['AllServed'] is True)

    print(f"\n{'='*60}")
    print(f"TỔNG KẾT")
    print(f"{'='*60}")
    print(f"  Tổng số instance       : {n_total}")
    print(f"  Số nghiệm khả thi      : {n_feasible}/{n_total}")
    print(f"  Số nghiệm phục vụ đủ   : {n_served}/{n_total}")
    if valid_gaps:
        avg_gap = sum(valid_gaps) / len(valid_gaps)
        print(f"  Gap trung bình         : {avg_gap:.3f}%")
        print(f"  Gap nhỏ nhất           : {min(valid_gaps):.3f}%")
        print(f"  Gap lớn nhất           : {max(valid_gaps):.3f}%")
    print(f"\n  Kết quả chi tiết đã lưu tại: {args.output}")


if __name__ == '__main__':
    main()