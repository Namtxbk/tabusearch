import argparse
import sys
import os
import time
import glob

sys.path.insert(0, os.path.dirname(__file__))

from instance import read_solomon, read_json_instance, Instance, Customer
from solution import Solution
from solomon_i1 import solomon_i1_construction, multi_start_i1
from tabu_search import tabu_search, TabuSearchConfig

def process_single_instance(inst: Instance, args):
    print(f"\n" + "="*65)
    print(f"XỬ LÝ INSTANCE: {inst.name}")
    print(f"="*65)
    print(f"Cấu hình đọc từ JSON:")
    print(f"  - Số Truck: {inst.num_trucks} (Tải trọng: {inst.truck_capacity}) | Vận tốc: {inst.truck_speed}")
    print(f"  - Số Drone: {inst.num_drones} (Tải trọng: {inst.drone_capacity} | Tầm bay: {inst.drone_range}) | Vận tốc: {inst.drone_speed}")
    print(f"  - Số Khách hàng: {len(inst.customers)} (C1: {len(inst.c1_ids)}, C2: {len(inst.c2_ids)})")
    print(f"  - Danh sách C1 (Chỉ Truck): {sorted(inst.c1_ids)}\n")

    # 1. Construction
    t0 = time.time()
    if args.multi_start > 1:
        init_sol = multi_start_i1(inst, n_starts=args.multi_start)
    else:
        init_sol = solomon_i1_construction(inst, mu=args.mu, lam=args.lam)

    print(f" [Construction] Done trong {time.time()-t0:.3f}s | "
          f"Makespan ban đầu: {init_sol.makespan():.1f} | "
          f"Feasible: {init_sol.is_feasible(inst)}")

    # 2. Tabu Search
    print(" [Tabu Search] Đang chạy tối ưu...")
    cfg = TabuSearchConfig(
        max_iter        = args.iter,
        max_no_improve  = max(100, args.iter // 4),
        diversify_thresh= max(40,  args.iter // 8),
        tenure_base     = args.tenure,
        time_limit      = args.time,
        verbose         = not args.quiet,
    )
    
    try:
        best_sol, history = tabu_search(inst, cfg, init_solution=init_sol)
        print("\n [KẾT QUẢ CUỐI CÙNG]")
        print(best_sol.summary(inst))
        if len(history) > 1:
            print(f" Cải thiện: {history[0]:.1f} → {history[-1]:.1f} (Giảm {history[0]-history[-1]:.1f})")
    except Exception as e:
        print(f" [LỖI] Thất bại tại Tabu Search: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',        type=str,   default='WithTimeWindowns3')
    parser.add_argument('--format',      type=str,   default='json', choices=['solomon', 'custom', 'json'])
    parser.add_argument('--iter',        type=int,   default=500)
    parser.add_argument('--tenure',      type=int,   default=10)
    parser.add_argument('--time',        type=float, default=120.0)
    parser.add_argument('--mu',          type=float, default=0.5)
    parser.add_argument('--lam',         type=float, default=1.0)
    parser.add_argument('--multi-start', type=int,   default=1)
    parser.add_argument('--quiet',       action='store_true', default=True)
    
    # Giữ lại các tham số cũ làm phương án dự phòng cho file TXT/Solomon nếu cần
    parser.add_argument('--trucks',      type=int,   default=2)
    parser.add_argument('--drones',      type=int,   default=2)
    parser.add_argument('--truck-cap',   type=float, default=None)
    parser.add_argument('--drone-cap',   type=float, default=30.0)
    parser.add_argument('--range',       type=float, default=100.0)
    args = parser.parse_args()

    target_path = args.path

    if not os.path.exists(target_path):
        print(f"Đường dẫn không tồn tại: {target_path}")
        return

    if os.path.isdir(target_path):
        if args.format == 'json':
            files = glob.glob(os.path.join(target_path, "*.json"))
        else:
            files = glob.glob(os.path.join(target_path, "*.txt"))
            
        if not files:
            print(f"Không tìm thấy file định dạng '{args.format}' trong thư mục '{target_path}'")
            return
            
        print(f"Tìm thấy {len(files)} file cấu hình. Bắt đầu quét hàng loạt...")
        for filepath in sorted(files):
            try:
                if args.format == 'json':
                    inst = read_json_instance(filepath)
                elif args.format == 'solomon':
                    inst = read_solomon(filepath, num_trucks=args.trucks, num_drones=args.drones)
                else:
                    inst = read_custom(filepath)
                
                process_single_instance(inst, args)
            except Exception as e:
                print(f"Lỗi khi xử lý file {filepath}: {e}")
    else:
        if args.format == 'json':
            inst = read_json_instance(target_path)
        elif args.format == 'solomon':
            inst = read_solomon(target_path, num_trucks=args.trucks, num_drones=args.drones)
        else:
            inst = read_custom(target_path)
            
        process_single_instance(inst, args)

if __name__ == '__main__':
    main()