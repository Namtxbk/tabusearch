"""
solomon_i1.py — Khởi tạo lời giải ban đầu bằng chiến lược Tuyến ảo & Hòa tan (Virtual Routes & Absorption)
Đảm bảo: Phục vụ 100% đơn hàng, các tuyến ban đầu sinh ra HOÀN TOÀN HỢP LỆ (Feasible).
"""

from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple

from instance import Instance, Customer
from solution import Route, Solution, precompute


def _drone_eligible(c: Customer, inst: Instance) -> bool:
    if c.is_c1:
        return False
    if c.demand > inst.drone_capacity:
        return False
    return inst.dist(0, c.id) + inst.dist(c.id, 0) <= inst.drone_range


def _compute_polar_angle(c: Customer, depot: Customer) -> float:
    dx = c.x - depot.x
    dy = c.y - depot.y
    angle = math.atan2(dy, dx)
    return angle if angle >= 0 else angle + 2 * math.pi


def _build_strictly_feasible_routes(customers_pool: List[Customer], is_drone: bool, 
                                    inst: Instance, alpha: float, mu: float) -> List[Route]:
    """
    Xây dựng các tuyến đường HOÀN TOÀN HỢP LỆ. 
    Nếu không chèn được vào tuyến hiện tại do vi phạm ràng buộc, thuật toán lập tức mở tuyến mới.
    """
    if not customers_pool:
        return []

    remaining = list(customers_pool)
    cdata = {c.id: c for c in inst.all_nodes}
    capacity = inst.drone_capacity if is_drone else inst.truck_capacity
    routes: List[Route] = []

    while remaining:
        # Chọn Seed cho tuyến mới: Khách xa nhất hoặc ngặt nghèo nhất về thời gian
        seed = max(remaining, key=lambda c: (inst.dist(0, c.id), -c.due))
        route = Route(sequence=[0, seed.id, 0], is_drone=is_drone)
        precompute(route, inst)
        remaining.remove(seed)

        inserted_in_this_run = True
        while inserted_in_this_run and remaining:
            inserted_in_this_run = False
            best_u, best_pos, best_cost = None, -1, float('inf')

            for u in remaining:
                # 1. Kiểm tra ràng buộc cứng về Tải trọng
                if route.total_load + u.demand > capacity:
                    continue

                for i in range(len(route.sequence) - 1):
                    i_id, j_id = route.sequence[i], route.sequence[i + 1]

                    # Tiêu chí khoảng cách Solomon
                    c11 = inst.dist(i_id, u.id) + inst.dist(u.id, j_id) - alpha * inst.dist(i_id, j_id)
                    arrive_u = route.a[i] + cdata[i_id].service + inst.travel_time(i_id, u.id, is_drone)
                    a_u = max(arrive_u, u.ready)

                    # 2. Kiểm tra ràng buộc cứng về Time Window tại nút u
                    if a_u > u.due:
                        continue

                    # 3. Kiểm tra độ trễ lan truyền bằng Forward Time Slack (FTS) tại nút j
                    delay = max(a_u + u.service + inst.travel_time(u.id, j_id, is_drone), cdata[j_id].ready) - route.a[i + 1]
                    if delay > route.F[i + 1]:
                        continue

                    cost = mu * c11 + (1.0 - mu) * delay
                    if cost < best_cost:
                        best_cost, best_u, best_pos = cost, u, i

            # Chỉ chèn nếu tìm thấy vị trí HOÀN TOÀN HỢP LỆ
            if best_u is not None:
                route.sequence.insert(best_pos + 1, best_u.id)
                precompute(route, inst)
                remaining.remove(best_u)
                inserted_in_this_run = True

        routes.append(route)
    return routes


def solomon_i1_construction(inst: Instance,
                             alpha: float = 1.0,
                             mu: float = 1.0,
                             lam: float = 1.0,
                             seed_criterion_truck: str = 'farthest',
                             seed_criterion_drone: str = 'farthest_drone',
                             ) -> Solution:
    depot = inst.all_nodes[0] if hasattr(inst, 'all_nodes') else Customer(0, 0.0, 0.0, 0, 0, 9999, 0)
    cdata = {c.id: c for c in inst.all_nodes}

    # Phân tách tập khách hàng dựa trên góc cực và khả năng bay của Drone
    drone_custs = [c for c in inst.customers if _drone_eligible(c, inst)]
    truck_custs = [c for c in inst.customers if not _drone_eligible(c, inst)]

    drone_custs.sort(key=lambda c: _compute_polar_angle(c, depot))
    truck_custs.sort(key=lambda c: _compute_polar_angle(c, depot))

    # Tạo ra các tập tuyến đường sạch, hoàn toàn feasible
    all_drone_routes = _build_strictly_feasible_routes(drone_custs, is_drone=True, inst=inst, alpha=alpha, mu=mu)
    all_truck_routes = _build_strictly_feasible_routes(truck_custs, is_drone=False, inst=inst, alpha=alpha, mu=mu)

    # Cấp phát vào các xe chính thức
    truck_routes = all_truck_routes[:inst.num_trucks]
    drone_routes = all_drone_routes[:inst.num_drones]

    # Các tuyến vượt ngưỡng chính là các "Tuyến ảo" chứa đơn hàng mồ côi
    virtual_truck_routes = all_truck_routes[inst.num_trucks:]
    virtual_drone_routes = all_drone_routes[inst.num_drones:]
    
    # Gom tất cả các khách hàng nằm trong các tuyến ảo này ra để tiến hành "Hòa tan"
    orphan_customers: List[Customer] = []
    for r in virtual_truck_routes + virtual_drone_routes:
        for cust_id in r.sequence[1:-1]:
            orphan_customers.append(cdata[cust_id])

    # ── PHA HÒA TAN (ABSORPTION PHASE) ───────────────────────────────────
    # Thử lách các đơn hàng mồ côi vào các khoảng trống (Slack) của các xe chính thức
    for cust in list(orphan_customers):
        inserted = False
        
        # Thử nhét vào Drone chính thức trước (nếu thỏa mãn điều kiện drone)
        if _drone_eligible(cust, inst):
            for r in drone_routes:
                if r.total_load + cust.demand <= inst.drone_capacity:
                    for i in range(len(r.sequence) - 1):
                        arrive_u = r.a[i] + cdata[r.sequence[i]].service + inst.travel_time(r.sequence[i], cust.id, is_drone=True)
                        a_u = max(arrive_u, cust.ready)
                        if a_u <= cust.due:
                            delay = max(a_u + cust.service + inst.travel_time(cust.id, r.sequence[i+1], is_drone=True), cdata[r.sequence[i+1]].ready) - r.a[i+1]
                            if delay <= r.F[i+1]: # Hợp lệ Time Window tuyệt đối bằng FTS
                                r.sequence.insert(i + 1, cust.id)
                                precompute(r, inst)
                                orphan_customers.remove(cust)
                                inserted = True
                                break
                if inserted: break

        # Nếu drone không nhận được, thử lách vào các xe tải chính thức
        if not inserted:
            for r in truck_routes:
                if r.total_load + cust.demand <= inst.truck_capacity:
                    for i in range(len(r.sequence) - 1):
                        arrive_u = r.a[i] + cdata[r.sequence[i]].service + inst.travel_time(r.sequence[i], cust.id, is_drone=False)
                        a_u = max(arrive_u, cust.ready)
                        if a_u <= cust.due:
                            delay = max(a_u + cust.service + inst.travel_time(cust.id, r.sequence[i+1], is_drone=False), cdata[r.sequence[i+1]].ready) - r.a[i+1]
                            if delay <= r.F[i+1]:
                                r.sequence.insert(i + 1, cust.id)
                                precompute(r, inst)
                                orphan_customers.remove(cust)
                                inserted = True
                                break
                if inserted: break

    # ── XỬ LÝ CUỐI CÙNG ──────────────────────────────────────────────────
    # Nếu sau pha hòa tan mà vẫn còn khách mồ côi bất khả kháng, 
    # dựng họ thành các tuyến xe tải chính thức vượt ngưỡng quy định.
    # Tuyến này hoàn toàn FEASIBLE về mặt kỹ thuật, chỉ vi phạm số lượng xe (sẽ bị Tabu xử lý).
    if orphan_customers:
        extra_routes = _build_strictly_feasible_routes(orphan_customers, is_drone=False, inst=inst, alpha=alpha, mu=mu)
        truck_routes.extend(extra_routes)

    # Padding các tuyến rỗng nếu số tuyến sinh ra ít hơn năng lực hệ thống
    while len(truck_routes) < inst.num_trucks:
        r = Route(sequence=[0, 0], is_drone=False)
        precompute(r, inst)
        truck_routes.append(r)

    while len(drone_routes) < inst.num_drones:
        r = Route(sequence=[0, 0], is_drone=True)
        precompute(r, inst)
        drone_routes.append(r)

    return Solution(truck_routes=truck_routes, drone_routes=drone_routes)


def multi_start_i1(inst: Instance, n_starts: int = 1) -> Solution:
    return solomon_i1_construction(inst)