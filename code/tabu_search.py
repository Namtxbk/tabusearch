

from __future__ import annotations
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

from instance import Instance, Customer
from solution import Route, Solution, precompute


@dataclass
class TabuConfig:
    max_iterations: int = 200
    tabu_tenure: int = 15
    diversify_thresh: int = 30
    verbose: bool = True


class TabuList:
    def __init__(self):
        self.matrix: Dict[Tuple[str, int, int], int] = {}

    def add(self, move_type: str, node_id: int, target_route: int, current_iter: int, tenure: int):
        self.matrix[(move_type, node_id, target_route)] = current_iter + tenure

    def is_tabu(self, move_type: str, node_id: int, target_route: int, current_iter: int) -> bool:
        return self.matrix.get((move_type, node_id, target_route), 0) > current_iter


def _evaluate_penalty_obj(sol: Solution, inst: Instance, penalty_factor: float) -> float:
    """
    Hàm mục tiêu tổng hợp tính toán Makespan cộng với 
    điểm phạt nặng cho các Tuyến ảo vượt quá giới hạn cấu hình (K và D).
    """
    makespan = sol.makespan()
    
    # Phạt nếu số tuyến xe tải vượt quá giới hạn cấu hình K
    excess_truck_routes = max(0, len(sol.truck_routes) - inst.num_trucks)
    # Phạt nếu số tuyến drone vượt quá giới hạn cấu hình D
    excess_drone_routes = max(0, len(sol.drone_routes) - inst.num_drones)
    
    # Thêm các hàm phạt cứng từ file solution của bạn (TW và Capacity)
    violation_penalty = sol.penalty_tw(inst) * 1000 + sol.penalty_cap(inst) * 1000
    
    return makespan + violation_penalty + (excess_truck_routes + excess_drone_routes) * penalty_factor


def _try_eliminate_virtual_routes(sol: Solution, inst: Instance, penalty_factor: float) -> Optional[Solution]:
    """
    TOÁN TỬ ĐỘT PHÁ: Chủ động bóc tách các tuyến vượt ngưỡng (Tuyến ảo) 
    và tìm cách hòa tan hành khách vào các xe chính thức bằng FTS.
    """
    # Nếu số lượng xe nằm trong phạm vi cho phép -> Không cần xử lý
    if len(sol.truck_routes) <= inst.num_trucks:
        return None
        
    test_sol = sol.copy()
    cdata = {c.id: c for c in inst.all_nodes}
    
    # Lấy tuyến ảo cuối cùng ra để phá hủy
    virtual_route = test_sol.truck_routes[-1]
    customers_to_relocate = virtual_route.sequence[1:-1]
    
    success_count = 0
    for cust_id in customers_to_relocate:
        cust = cdata[cust_id]
        inserted = False
        
        # Quét qua các xe tải chính thức (nằm trong giới hạn K) để lách vào
        for r_idx in range(inst.num_trucks):
            r = test_sol.truck_routes[r_idx]
            if r.total_load + cust.demand <= inst.truck_capacity:
                for i in range(len(r.sequence) - 1):
                    arrive_u = r.a[i] + cdata[r.sequence[i]].service + inst.travel_time(r.sequence[i], cust.id, is_drone=False)
                    a_u = max(arrive_u, cust.ready)
                    if a_u <= cust.due:
                        delay = max(a_u + cust.service + inst.travel_time(cust.id, r.sequence[i+1], is_drone=False), cdata[r.sequence[i+1]].ready) - r.a[i+1]
                        if delay <= r.F[i+1]: # Bảo đảm an toàn tuyệt đối Time Window bằng FTS
                            r.sequence.insert(i + 1, cust.id)
                            precompute(r, inst)
                            inserted = True
                            success_count += 1
                            break
            if inserted: break
            
    # Nếu toàn bộ hành khách của tuyến ảo đã được hấp thụ an toàn vào các xe chính thức
    if success_count == len(customers_to_relocate):
        test_sol.truck_routes.pop() # Xóa sổ tuyến ảo hoàn toàn
        return test_sol
        
    return None


def advanced_tabu_search(initial_sol: Solution, inst: Instance, cfg: TabuConfig = TabuConfig()) -> Solution:
    """
    Thuật toán Tabu Search cải tiến phối hợp chiến lược Hủy tuyến ảo và Phạt động thích nghi
    """
    current_sol = initial_sol.copy()
    best_sol = initial_sol.copy()
    
    tabu_list = TabuList()
    penalty_factor = 5000.0  # Lực phạt ban đầu cho tuyến ảo
    
    best_obj = _evaluate_penalty_obj(best_sol, inst, penalty_factor)
    no_improve = 0
    
    cdata = {c.id: c for c in inst.all_nodes}

    for iteration in range(cfg.max_iterations):
        # ── HÀNH ĐỘNG ƯU TIÊN 1: Thử hủy tuyến ảo chủ động ──
        eliminated_sol = _try_eliminate_virtual_routes(current_sol, inst, penalty_factor)
        if eliminated_sol is not None:
            current_sol = eliminated_sol
            cur_obj = _evaluate_penalty_obj(current_sol, inst, penalty_factor)
            if cur_obj < best_obj:
                best_sol = current_sol.copy()
                best_obj = cur_obj
            if cfg.verbose:
                print(f"[{iteration:3d}] 🔥 Đã tiêu diệt và hòa tan thành công 1 Tuyến ảo!")
            continue

        # ── HÀNH ĐỘNG 2: Tìm kiếm lân cận chuẩn (Neighborhood Search) ──
        best_neighbor_sol = None
        best_neighbor_obj = float('inf')
        chosen_move_attr = (0, 0) # (node_id, target_route)

        # Quét Toán tử Relocate giữa các tuyến xe tải
        for r_from_idx, r_from in enumerate(current_sol.truck_routes):
            if len(r_from.sequence) <= 3: continue # Tuyến quá ngắn không bốc khách được
            
            for pos in range(1, len(r_from.sequence) - 1):
                cust_id = r_from.sequence[pos]
                cust = cdata[cust_id]
                
                for r_to_idx, r_to in enumerate(current_sol.truck_routes):
                    if r_from_idx == r_to_idx: continue
                    # Kiểm tra nhanh sức chứa xe đích
                    if r_to.total_load + cust.demand > inst.truck_capacity: continue
                    
                    # Thử chèn vào các vị trí có sẵn trên xe đích
                    for insert_pos in range(len(r_to.sequence) - 1):
                        neighbor_sol = current_sol.copy()
                        
                        # Thực hiện di chuyển nút
                        neighbor_sol.truck_routes[r_from_idx].sequence.pop(pos)
                        neighbor_sol.truck_routes[r_to_idx].sequence.insert(insert_pos + 1, cust_id)
                        
                        # Cập nhật lại mảng FTS và thời gian
                        precompute(neighbor_sol.truck_routes[r_from_idx], inst)
                        precompute(neighbor_sol.truck_routes[r_to_idx], inst)
                        
                        neighbor_obj = _evaluate_penalty_obj(neighbor_sol, inst, penalty_factor)
                        
                        # Kiểm tra luật cấm Tabu kết hợp điều kiện phá luật (Aspiration Criterion)
                        is_tabu = tabu_list.is_tabu('relocate', cust_id, r_to_idx, iteration)
                        if not is_tabu or neighbor_obj < best_obj:
                            if neighbor_obj < best_neighbor_obj:
                                best_neighbor_obj = neighbor_obj
                                best_neighbor_sol = neighbor_sol
                                chosen_move_attr = (cust_id, r_to_idx)

        # Nếu không tìm được bước đi lân cận nào hợp lệ, thoát vòng lặp
        if best_neighbor_sol is None:
            if cfg.verbose: print("Không tìm thấy lân cận hợp lệ. Dừng giải thuật.")
            break

        # Chấp nhận bước đi tốt nhất vùng lặp
        current_sol = best_neighbor_sol
        tabu_list.add('relocate', chosen_move_attr[0], chosen_move_attr[1], iteration, cfg.tabu_tenure)

        # ── CƠ CHẾ PHẠT ĐỘNG THÍCH NGHI (Strategic Oscillation) ──
        is_current_feasible = len(current_sol.truck_routes) <= inst.num_trucks and current_sol.is_feasible(inst)
        if not is_current_feasible:
            penalty_factor *= 1.1  # Nghiệm vi phạm xe tải phụ -> Tăng phạt để ép thu gọn tuyến
            no_improve += 1
        else:
            penalty_factor *= 0.95 # Nghiệm hoàn toàn sạch -> Giảm phạt để mở rộng tìm kiếm không gian biên
            
            current_actual_obj = current_sol.makespan()
            if current_actual_obj < best_obj:
                best_sol = current_sol.copy()
                best_obj = current_actual_obj
                no_improve = 0

        if cfg.verbose and iteration % 20 == 0:
            print(f"Iteration {iteration:3d} | Best Makespan: {best_sol.makespan():.2f} | Tuyến hiện tại: {len(current_sol.truck_routes)} (Mục tiêu: {inst.num_trucks})")

    # Cắt tỉa các tuyến rỗng dư thừa trước khi trả về nghiệm cuối cùng
    best_sol.truck_routes = [r for r in best_sol.truck_routes if len(r.sequence) > 2]
    while len(best_sol.truck_routes) < inst.num_trucks:
        r = Route(sequence=[0, 0], is_drone=False)
        precompute(r, inst)
        best_sol.truck_routes.append(r)

    return best_sol