"""
tabu_search.py — Tabu Search nâng cao với Phạt Động và Toán tử Tái cấu trúc tuyến (ALNS)
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional

from instance import Instance, Customer
from solution import Trip, Vehicle, Solution, precompute_vehicle

@dataclass
class TabuSearchConfig:
    max_iter: int = 1000
    max_no_improve: int = 200
    tenure_base: int = 7
    time_limit: float = 60.0
    verbose: bool = True

class TabuSet:
    def __init__(self):
        self.matrix = {}
    def add(self, key: Tuple[int, int, int], current_iter: int):
        self.matrix[key] = current_iter
    def is_tabu(self, key: Tuple[int, int, int], current_iter: int) -> bool:
        if key in self.matrix:
            return current_iter <= self.matrix[key] + 10
        return False

def ruin_and_recreate_neighborhood(
    sol: Solution, inst: Instance, tabu: TabuSet, it: int, best_obj: float,
    w_cap: float, w_range: float, w_tw: float
) -> Tuple[Optional[Solution], float, Optional[Tuple[int, int, int, int]]]:
    """
    Toán tử Ruin & Recreate CẢI TIẾN: Hỗ trợ linh hoạt chèn vào chuyến cũ HOẶC TÁCH CHUYẾN MỚI.
    """
    new_sol = sol.copy()
    all_cust_ids = [c.id for c in inst.customers]
    
    # Tăng tỷ lệ nhấc khách lên 20-30% để khuấy động không gian tìm kiếm mạnh hơn
    num_to_remove = max(2, int(len(all_cust_ids) * 0.25))
    removed_custs = random.sample(all_cust_ids, num_to_remove)
    
    # 1. RUIN: Nhấc khách ra khỏi hệ thống
    for v in new_sol.trucks + new_sol.drones:
        for t in v.trips:
            t.sequence = [n for n in t.sequence if n not in removed_custs]
            
    # XÓA SẠCH các chuyến rỗng (chỉ còn [0, 0]) để tái cấu trúc
    for v in new_sol.trucks + new_sol.drones:
        v.trips = [t for t in v.trips if len(t.sequence) > 2]
        # Nếu xe không còn chuyến nào, cấp tạm 1 chuyến rỗng làm gốc
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
            
    # 2. RECREATE: Chèn lại thông minh (Hỗ trợ sinh Multi-Trip)
    for cust_id in removed_custs:
        cust_obj = inst.customers[cust_id - 1]
        best_insert_sol = None
        best_insert_score = float('inf')
        best_key = None
        
        vehicles_to_try = new_sol.drones if _drone_eligible(cust_obj, inst) else new_sol.trucks
        
        for v_idx, v in enumerate(vehicles_to_try):
            
            # CHIẾN LƯỢC A: Thử chèn vào bên trong các Trip ĐANG CÓ
            for t_idx, t in enumerate(v.trips):
                for pos in range(1, len(t.sequence)):
                    temp_sol = new_sol.copy()
                    target_v = temp_sol.drones[v_idx] if v.is_drone else temp_sol.trucks[v_idx]
                    target_v.trips[t_idx].sequence.insert(pos, cust_id)
                    
                    temp_sol.recompute_all(inst)
                    score = (temp_sol.makespan() + 
                             w_cap * temp_sol.penalty_cap(inst) + 
                             w_range * temp_sol.penalty_range(inst) + 
                             w_tw * temp_sol.penalty_tw(inst))
                    
                    key = (cust_id, v_idx, t_idx, pos)
                    if score < best_insert_score:
                        if not tabu.is_tabu(key, it) or score < best_obj:
                            best_insert_score = score
                            best_insert_sol = temp_sol
                            best_key = key
                            
            # CHIẾN LƯỢC B: Thử tạo hẳn một Trip MỚI (Tách chuyến)
            # Trip mới có thể được chèn vào trước, giữa, hoặc sau các chuyến hiện tại của xe
            for insert_t_idx in range(len(v.trips) + 1):
                temp_sol = new_sol.copy()
                target_v = temp_sol.drones[v_idx] if v.is_drone else temp_sol.trucks[v_idx]
                
                # Khởi tạo một chuyến đi chỉ có 1 khách này
                new_trip = Trip(sequence=[0, cust_id, 0], is_drone=target_v.is_drone)
                target_v.trips.insert(insert_t_idx, new_trip)
                
                temp_sol.recompute_all(inst)
                score = (temp_sol.makespan() + 
                         w_cap * temp_sol.penalty_cap(inst) + 
                         w_range * temp_sol.penalty_range(inst) + 
                         w_tw * temp_sol.penalty_tw(inst))
                
                # Dùng pos = -1 để đánh dấu đây là key của việc sinh chuyến mới
                key = (cust_id, v_idx, insert_t_idx, -1)
                if score < best_insert_score:
                    if not tabu.is_tabu(key, it) or score < best_obj:
                        best_insert_score = score
                        best_insert_sol = temp_sol
                        best_key = key
                        
        if best_insert_sol is not None:
            new_sol = best_insert_sol
            if best_key:
                tabu.add(best_key, it)
                
    new_sol.recompute_all(inst)
    total_score = (new_sol.makespan() + 
                   w_cap * new_sol.penalty_cap(inst) + 
                   w_range * new_sol.penalty_range(inst) + 
                   w_tw * new_sol.penalty_tw(inst))
    return new_sol, total_score, None

def advanced_tabu_search(init_sol: Solution, inst: Instance, cfg: TabuSearchConfig) -> Tuple[Solution, List[float]]:
    """
    Thuật toán Tabu Search tích hợp cơ chế Phạt Động (Strategic Oscillation).
    """
    current = init_sol.copy()
    best = init_sol.copy()
    
    # Khởi tạo trọng số phạt ban đầu mềm dẻo
    w_cap, w_range, w_tw = 50.0, 50.0, 50.0
    
    current.recompute_all(inst)
    best_obj_val = (current.makespan() + 
                    w_cap * current.penalty_cap(inst) + 
                    w_range * current.penalty_range(inst) + 
                    w_tw * current.penalty_tw(inst))
    
    tabu = TabuSet()
    history = [best.makespan()]
    
    no_improve = 0
    feasible_counter = 0
    infeasible_counter = 0
    
    for it in range(1, cfg.max_iter + 1):
        if no_improve >= cfg.max_no_improve:
            if cfg.verbose: print(f"  -> Dừng sớm tại vòng lặp {it} do không cải thiện.")
            break
            
        # Gọi cấu trúc lân cận Ruin & Recreate để tìm kiếm bước nhảy lớn
        nb_sol, nb_obj, _ = ruin_and_recreate_neighborhood(
            current, inst, tabu, it, best_obj_val, w_cap, w_range, w_tw
        )
        
        if nb_sol is None:
            break
            
        # Loại bỏ các trip rỗng thừa thãi phát sinh trong quá trình xáo trộn
        for v in nb_sol.trucks + nb_sol.drones:
            v.trips = [t for t in v.trips if len(t.customers()) > 0]
            if not v.trips:
                v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
            precompute_vehicle(v, inst)
            
        current = nb_sol
        
        # --- Cơ chế Phạt Động (Strategic Oscillation) ---
        if current.is_feasible(inst):
            feasible_counter += 1
            infeasible_counter = 0
            if feasible_counter >= 8:
                # Nghiệm liên tục khả thi tốt -> giảm phạt để tối ưu hóa mạnh hơn Makespan
                w_cap = max(10.0, w_cap * 0.8)
                w_range = max(10.0, w_range * 0.8)
                w_tw = max(10.0, w_tw * 0.8)
                feasible_counter = 0
        else:
            infeasible_counter += 1
            feasible_counter = 0
            if infeasible_counter >= 5:
                # Nghiệm liên tục bị vi phạm -> siết chặt phạt để ép về vùng khả thi
                w_cap = min(1000.0, w_cap * 1.5)
                w_range = min(1000.0, w_range * 1.5)
                w_tw = min(1000.0, w_tw * 1.5)
                infeasible_counter = 0
                
        # Tính toán lại hàm mục tiêu thực tế không phạt để so sánh lưu trữ nghiệm tốt nhất
        if current.is_feasible(inst) and current.all_served(inst):
            if current.makespan() < best.makespan() or not best.is_feasible(inst):
                best = current.copy()
                no_improve = 0
                history.append(best.makespan())
                if cfg.verbose:
                    print(f"  [{it:4d}] ⭐ CẬP NHẬT TỐT NHẤT: Makespan={best.makespan():.2f} | Mọi khách đã được phục vụ.")
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        if cfg.verbose and it % 50 == 0:
            print(f"  [{it:4d}] Giám sát hiện tại -> Makespan={current.makespan():.2f} | Feasible={current.is_feasible(inst)} | Trọng số phạt TW={w_tw:.1f}")
            
    return best, history

def _drone_eligible(c: Customer, inst: Instance) -> bool:
    if c.is_c1: return False
    if c.demand > inst.drone_capacity: return False
    rt = inst.travel_time(0, c.id, True) + inst.travel_time(c.id, 0, True)
    return rt <= inst.drone_range