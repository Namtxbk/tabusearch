"""
construction.py — Greedy Construction heuristic cho MVRPD-TW

Thuật toán:
  1. Phân loại C2 → drone-candidate hoặc truck-pool
  2. Xây route bằng Nearest Neighbor có Time Window
     (hàm điểm: alpha*dist + beta*wait + gamma*urgency)
  3. Mở route mới nếu không thể chèn thêm khách nào
"""

from __future__ import annotations
import math
import random
from typing import List, Set, Dict

from instance import Instance, Customer
from solution import Route, Solution, precompute


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra drone-eligible
# ─────────────────────────────────────────────────────────────────────────────

def is_drone_eligible(cust: Customer, inst: Instance) -> bool:
    """
    Khách hàng c có thể phục vụ bằng drone hay không.
    Điều kiện:
      - c ∈ C2 (không phải C1)
      - demand ≤ drone_capacity
      - dist(depot, c) + dist(c, depot) ≤ drone_range
    """
    if cust.is_c1:
        return False
    if cust.demand > inst.drone_capacity:
        return False
    round_trip = inst.dist(0, cust.id) + inst.dist(cust.id, 0)
    return round_trip <= inst.drone_range


# ─────────────────────────────────────────────────────────────────────────────
# Nearest Neighbor với Time Window
# ─────────────────────────────────────────────────────────────────────────────

def _score(current_node: int, candidate: Customer,
           current_time: float, inst: Instance,
           is_drone: bool,
           alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2) -> float:
    """
    Hàm điểm tham lam (càng nhỏ càng tốt):
      alpha * dist + beta * wait_time + gamma * urgency
    """
    t_travel  = inst.travel_time(current_node, candidate.id, is_drone=is_drone)
    arrive    = current_time + t_travel
    wait      = max(0.0, candidate.ready - arrive)
    slack     = candidate.due - max(arrive, candidate.ready)
    urgency   = 1.0 / (slack + 1e-6)
    dist_     = inst.dist(current_node, candidate.id)
    return alpha * dist_ + beta * wait + gamma * urgency


def _build_route(pool: List[Customer], is_drone: bool,
                 inst: Instance,
                 alpha: float, beta: float, gamma: float) -> List[Route]:
    """
    Xây tập route cho một loại phương tiện (truck hoặc drone)
    từ danh sách khách hàng pool.
    Trả về list Route (mỗi route có thể rỗng nếu pool rỗng).
    """
    remaining = list(pool)
    routes: List[Route] = []

    capacity = inst.drone_capacity if is_drone else inst.truck_capacity

    while remaining:
        seq   = [0]
        load  = 0.0
        t     = inst.depot.ready

        while True:
            current = seq[-1]
            # Lọc khách khả thi
            feasible = []
            for c in remaining:
                t_travel = inst.travel_time(current, c.id, is_drone=is_drone)
                arrive   = t + t_travel
                if arrive > c.due:
                    continue   # đến muộn → vi phạm TW
                if load + c.demand > capacity:
                    continue   # vượt tải trọng
                # Drone: kiểm tra range nếu quay về depot sau node này
                if is_drone:
                    # Quãng đường từ đây đến c rồi về depot
                    dist_so_far = sum(
                        inst.dist(seq[k], seq[k+1]) for k in range(len(seq)-1)
                    )
                    dist_add = inst.dist(current, c.id) + inst.dist(c.id, 0)
                    if dist_so_far + dist_add > inst.drone_range:
                        continue
                feasible.append(c)

            if not feasible:
                break

            # Chọn khách tốt nhất theo hàm điểm
            best = min(feasible,
                       key=lambda c: _score(current, c, t, inst, is_drone,
                                            alpha, beta, gamma))
            t_travel = inst.travel_time(current, best.id, is_drone=is_drone)
            t = max(t + t_travel, best.ready) + best.service
            load += best.demand
            seq.append(best.id)
            remaining.remove(best)

        seq.append(0)
        r = Route(sequence=seq, is_drone=is_drone)
        precompute(r, inst)
        routes.append(r)

        if not remaining:
            break

    return routes


# ─────────────────────────────────────────────────────────────────────────────
# Construction chính
# ─────────────────────────────────────────────────────────────────────────────

def greedy_construction(inst: Instance,
                        alpha: float = 0.5,
                        beta: float  = 0.3,
                        gamma: float = 0.2) -> Solution:
    """
    Xây nghiệm khởi tạo bằng Nearest Neighbor có TW.

    Bước 1: Phân công phương tiện cho C2.
    Bước 2: Xây routes cho drone (từ drone_pool).
    Bước 3: Xây routes cho truck (C1 + C2 còn lại).
    Bước 4: Đảm bảo đủ K truck routes và D drone routes.
    """
    # ── Bước 1: Phân loại ────────────────────────────────────────────────
    truck_pool: List[Customer] = []
    drone_pool: List[Customer] = []

    for c in inst.customers:
        if is_drone_eligible(c, inst):
            drone_pool.append(c)
        else:
            # C1 hoặc C2 không đủ điều kiện drone → truck
            truck_pool.append(c)

    # ── Bước 2: Xây drone routes ─────────────────────────────────────────
    drone_routes_raw = _build_route(drone_pool, is_drone=True,
                                    inst=inst,
                                    alpha=alpha, beta=beta, gamma=gamma)

    # Giới hạn số drone routes = D
    # Nếu quá nhiều route → gộp vào truck
    while len(drone_routes_raw) > inst.num_drones:
        overflow = drone_routes_raw.pop()
        # Khách từ route thừa → chuyển về truck_pool
        truck_pool.extend(
            inst.customers[nid - 1] for nid in overflow.customers()
            if nid - 1 < len(inst.customers)
        )
        # Tìm đúng Customer object
        overflow_custs = [c for c in inst.customers
                          if c.id in overflow.customers()]
        truck_pool.extend(overflow_custs)
        # Tránh duplicate
        seen = set()
        truck_pool_dedup = []
        for c in truck_pool:
            if c.id not in seen:
                seen.add(c.id)
                truck_pool_dedup.append(c)
        truck_pool = truck_pool_dedup

    # Padding drone routes rỗng nếu chưa đủ D
    while len(drone_routes_raw) < inst.num_drones:
        r = Route(sequence=[0, 0], is_drone=True)
        precompute(r, inst)
        drone_routes_raw.append(r)

    # ── Bước 3: Xây truck routes ─────────────────────────────────────────
    truck_routes_raw = _build_route(truck_pool, is_drone=False,
                                    inst=inst,
                                    alpha=alpha, beta=beta, gamma=gamma)

    # Giới hạn số truck routes = K
    while len(truck_routes_raw) > inst.num_trucks:
        # Nếu vượt → thêm route cuối vào route đầu tiên còn capacity
        overflow = truck_routes_raw.pop()
        merged   = False
        for r in truck_routes_raw:
            if r.total_load + overflow.total_load <= inst.truck_capacity:
                # Thêm khách của overflow vào cuối r (trước depot cuối)
                r.sequence = r.sequence[:-1] + overflow.sequence[1:]
                precompute(r, inst)
                merged = True
                break
        if not merged:
            # Không gộp được → giữ lại (vi phạm sẽ bị xử lý bởi penalty)
            truck_routes_raw.append(overflow)
            break

    # Padding truck routes rỗng nếu chưa đủ K
    while len(truck_routes_raw) < inst.num_trucks:
        r = Route(sequence=[0, 0], is_drone=False)
        precompute(r, inst)
        truck_routes_raw.append(r)

    sol = Solution(
        truck_routes=truck_routes_raw[:inst.num_trucks],
        drone_routes=drone_routes_raw[:inst.num_drones],
    )
    return sol
