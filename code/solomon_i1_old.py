"""
solomon_i1.py — Solomon I1 Insertion Heuristic mở rộng cho MVRPD-TW
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


def _c1(i_pos: int, u: Customer, j_pos: int,
        route: Route, inst: Instance,
        alpha: float = 1.0, mu: float = 1.0) -> float:
    seq      = route.sequence
    is_drone = route.is_drone
    cdata    = {c.id: c for c in inst.all_nodes}

    i_id = seq[i_pos]
    j_id = seq[j_pos]

    t_iu    = inst.travel_time(i_id, u.id, is_drone=is_drone)
    s_i     = cdata[i_id].service
    arrive_u = route.a[i_pos] + s_i + t_iu
    a_u      = max(arrive_u, u.ready)

    if a_u > u.due:
        return float('inf')

    c11 = (inst.dist(i_id, u.id)
           + inst.dist(u.id, j_id)
           - alpha * inst.dist(i_id, j_id))

    t_uj     = inst.travel_time(u.id, j_id, is_drone=is_drone)
    a_j_new  = max(a_u + u.service + t_uj, cdata[j_id].ready)
    delay    = a_j_new - route.a[j_pos]

    if delay > route.F[j_pos]:
        return float('inf')

    c12 = delay

    if is_drone:
        extra_dist = inst.dist(i_id, u.id) + inst.dist(u.id, j_id) \
                   - inst.dist(i_id, j_id)
        if route.total_dist + extra_dist > inst.drone_range:
            return float('inf')

    return mu * c11 + (1.0 - mu) * c12


def _best_insertion(u: Customer, route: Route,
                    inst: Instance,
                    alpha: float = 1.0,
                    mu: float = 1.0) -> Tuple[float, int]:
    seq = route.sequence
    capacity = inst.drone_capacity if route.is_drone else inst.truck_capacity

    if route.total_load + u.demand > capacity:
        return float('inf'), -1

    best_cost = float('inf')
    best_pos  = -1

    for i in range(len(seq) - 1):
        cost = _c1(i, u, i + 1, route, inst, alpha, mu)
        if cost < best_cost:
            best_cost = cost
            best_pos  = i

    return best_cost, best_pos


def _c2(u: Customer, c1_star: float,
        inst: Instance, lam: float = 1.0) -> float:
    return lam * inst.dist(0, u.id) - c1_star


def _pick_seed(unserved: List[Customer],
               inst: Instance,
               criterion: str = 'farthest') -> Optional[Customer]:
    if not unserved:
        return None

    if criterion == 'farthest':
        return max(unserved, key=lambda c: inst.dist(0, c.id))
    elif criterion == 'urgent':
        return min(unserved, key=lambda c: c.due)
    elif criterion == 'farthest_drone':
        eligible = [c for c in unserved if _drone_eligible(c, inst)]
        if not eligible:
            return None
        return max(eligible,
                   key=lambda c: inst.dist(0, c.id) + inst.dist(c.id, 0))

    return unserved[0]


def _i1_build_single_route(seed: Customer,
                            unserved: List[Customer],
                            is_drone: bool,
                            inst: Instance,
                            alpha: float = 1.0,
                            mu:    float = 1.0,
                            lam:   float = 1.0) -> Route:
    route = Route(sequence=[0, seed.id, 0], is_drone=is_drone)
    precompute(route, inst)
    unserved.remove(seed)

    while True:
        best_u    = None
        best_pos  = -1
        best_c2   = -float('inf')

        for u in unserved:
            if is_drone and not _drone_eligible(u, inst):
                continue

            c1_star, pos = _best_insertion(u, route, inst, alpha, mu)
            if c1_star == float('inf'):
                continue

            c2_val = _c2(u, c1_star, inst, lam)
            if c2_val > best_c2:
                best_c2  = c2_val
                best_u   = u
                best_pos = pos

        if best_u is None:
            break

        route.sequence.insert(best_pos + 1, best_u.id)
        precompute(route, inst)
        unserved.remove(best_u)

    return route


def _forced_insert(unserved: List[Customer],
                   truck_routes: List[Route],
                   drone_routes: List[Route],
                   inst: Instance,
                   num_trucks: int,
                   num_drones: int) -> None:
    """
    Chèn cưỡng bức các khách còn lại vào routes hiện có.
    Nếu không chèn được → mở route mới (gộp vào đúng loại truck/drone).
    KHÔNG cắt theo num_trucks/num_drones ở đây — Solution sẽ gộp sau.
    """
    still_unserved = list(unserved)
    unserved.clear()

    for u in still_unserved:
        best_route = None
        best_pos   = -1
        best_delta = float('inf')

        # Ưu tiên chèn vào các route hiện có (truck trước, drone sau)
        candidates = [(r, False) for r in truck_routes] + \
                     [(r, True)  for r in drone_routes]

        for r, is_drone_route in candidates:
            if r.is_drone and not _drone_eligible(u, inst):
                continue
            cap = inst.drone_capacity if r.is_drone else inst.truck_capacity
            if r.total_load + u.demand > cap:
                continue

            for p in range(len(r.sequence) - 1):
                i_id = r.sequence[p]
                j_id = r.sequence[p + 1]
                delta = (inst.dist(i_id, u.id) + inst.dist(u.id, j_id)
                         - inst.dist(i_id, j_id))
                if delta < best_delta:
                    best_delta = delta
                    best_route = r
                    best_pos   = p

        if best_route is not None:
            best_route.sequence.insert(best_pos + 1, u.id)
            precompute(best_route, inst)
        else:
            # Mở route mới — thêm vào truck (hoặc drone nếu eligible)
            # Route thêm không bị cắt bỏ: sẽ được gộp vào Solution
            is_drone = _drone_eligible(u, inst)
            r_new = Route(sequence=[0, u.id, 0], is_drone=is_drone)
            precompute(r_new, inst)
            if is_drone:
                drone_routes.append(r_new)
            else:
                truck_routes.append(r_new)


def _merge_overflow_routes(main_routes: List[Route],
                            overflow: List[Route],
                            capacity: float,
                            inst: Instance) -> None:
    """
    Gộp các route thừa (ngoài giới hạn K/D) vào các route còn chỗ.
    Nếu không gộp được → chèn cưỡng bức vào route có delta nhỏ nhất.
    """
    for r_over in overflow:
        for nid in list(r_over.customers()):
            # Tìm customer object
            cdata = {c.id: c for c in inst.customers}
            u = cdata[nid]

            best_route = None
            best_pos   = -1
            best_delta = float('inf')

            for r in main_routes:
                if r.total_load + u.demand > capacity:
                    continue
                for p in range(len(r.sequence) - 1):
                    i_id = r.sequence[p]
                    j_id = r.sequence[p + 1]
                    delta = (inst.dist(i_id, u.id) + inst.dist(u.id, j_id)
                             - inst.dist(i_id, j_id))
                    if delta < best_delta:
                        best_delta = delta
                        best_route = r
                        best_pos   = p

            if best_route is None:
                # Capacity đầy → chèn vào route gần nhất dù vi phạm
                for r in main_routes:
                    for p in range(len(r.sequence) - 1):
                        i_id = r.sequence[p]
                        j_id = r.sequence[p + 1]
                        delta = (inst.dist(i_id, u.id) + inst.dist(u.id, j_id)
                                 - inst.dist(i_id, j_id))
                        if delta < best_delta:
                            best_delta = delta
                            best_route = r
                            best_pos   = p

            if best_route is not None:
                best_route.sequence.insert(best_pos + 1, nid)
                precompute(best_route, inst)


def solomon_i1_construction(inst: Instance,
                             alpha: float = 1.0,
                             mu:    float = 1.0,
                             lam:   float = 1.0,
                             seed_criterion_truck: str = 'farthest',
                             seed_criterion_drone: str = 'farthest_drone',
                             ) -> Solution:
    """
    Solomon I1 Construction Heuristic cho MVRPD-TW.
    Đảm bảo: Solution.all_served() = True sau khi trả về.
    """
    # ── Bước 1: Tách pool ────────────────────────────────────────────────
    drone_pool = [c for c in inst.customers if _drone_eligible(c, inst)]
    truck_pool = [c for c in inst.customers if not _drone_eligible(c, inst)]

    unserved_drone = list(drone_pool)
    unserved_truck = list(truck_pool)

    # ── Bước 2: Xây D drone routes ───────────────────────────────────────
    drone_routes: List[Route] = []

    for _ in range(inst.num_drones):
        seed = _pick_seed(unserved_drone, inst, seed_criterion_drone)
        if seed is None:
            break
        r = _i1_build_single_route(
            seed, unserved_drone, is_drone=True,
            inst=inst, alpha=alpha, mu=mu, lam=lam
        )
        drone_routes.append(r)

    # Khách drone không chèn được → chuyển về truck pool
    for c in unserved_drone:
        if c not in unserved_truck:
            unserved_truck.append(c)
    unserved_drone.clear()

    # ── Bước 3: Xây K truck routes ───────────────────────────────────────
    truck_routes: List[Route] = []

    for _ in range(inst.num_trucks):
        seed = _pick_seed(unserved_truck, inst, seed_criterion_truck)
        if seed is None:
            break
        r = _i1_build_single_route(
            seed, unserved_truck, is_drone=False,
            inst=inst, alpha=alpha, mu=mu, lam=lam
        )
        truck_routes.append(r)

    # ── Bước 4: Forced insertion cho khách còn sót ───────────────────────
    if unserved_truck:
        _forced_insert(unserved_truck, truck_routes, drone_routes,
                       inst, inst.num_trucks, inst.num_drones)

    # ── Bước 5: Gộp route thừa về đúng K truck + D drone ─────────────────
    # Truck overflow
    if len(truck_routes) > inst.num_trucks:
        overflow_t = truck_routes[inst.num_trucks:]
        truck_routes = truck_routes[:inst.num_trucks]
        _merge_overflow_routes(truck_routes, overflow_t,
                               inst.truck_capacity, inst)

    # Drone overflow
    if len(drone_routes) > inst.num_drones:
        overflow_d = drone_routes[inst.num_drones:]
        drone_routes = drone_routes[:inst.num_drones]
        # Gộp khách drone thừa vào truck routes
        _merge_overflow_routes(truck_routes, overflow_d,
                               inst.truck_capacity, inst)

    # ── Bước 6: Padding routes rỗng để đủ K và D ─────────────────────────
    while len(truck_routes) < inst.num_trucks:
        r = Route(sequence=[0, 0], is_drone=False)
        precompute(r, inst)
        truck_routes.append(r)

    while len(drone_routes) < inst.num_drones:
        r = Route(sequence=[0, 0], is_drone=True)
        precompute(r, inst)
        drone_routes.append(r)

    # ── Kiểm tra an toàn: nếu vẫn còn khách bị bỏ sót ───────────────────
    sol_check = Solution(
        truck_routes=truck_routes[:inst.num_trucks],
        drone_routes=drone_routes[:inst.num_drones],
    )
    if not sol_check.all_served(inst):
        # Fallback: chèn thủ công từng khách còn thiếu vào truck route nào đó
        served = set()
        for r in sol_check.truck_routes + sol_check.drone_routes:
            for nid in r.sequence:
                if nid != 0:
                    served.add(nid)
        missing_ids = {c.id for c in inst.customers} - served
        cdata = {c.id: c for c in inst.customers}
        for mid in missing_ids:
            u = cdata[mid]
            # Chèn vào cuối truck route 0 (trước depot)
            target = truck_routes[0]
            insert_at = len(target.sequence) - 1
            target.sequence.insert(insert_at, u.id)
            precompute(target, inst)

    return Solution(
        truck_routes=truck_routes[:inst.num_trucks],
        drone_routes=drone_routes[:inst.num_drones],
    )


def multi_start_i1(inst: Instance, n_starts: int = 5) -> Solution:
    """
    Chạy Solomon I1 nhiều lần với các tổ hợp tham số khác nhau.
    Trả về nghiệm có makespan thấp nhất và all_served=True.
    """
    configs = [
        (1.0,  1.0,  1.0,  'farthest',  'farthest_drone',  'I1-dist'),
        (1.0,  0.0,  1.0,  'farthest',  'farthest_drone',  'I1-time'),
        (1.0,  0.5,  1.0,  'urgent',    'farthest_drone',  'I1-mix-urgent'),
        (1.0,  1.0,  2.0,  'farthest',  'farthest_drone',  'I1-lam2'),
        (1.0,  0.5,  2.0,  'urgent',    'farthest_drone',  'I1-mix-lam2'),
    ][:n_starts]

    best_sol = None
    best_obj = float('inf')

    for alpha, mu, lam, s_truck, s_drone, name in configs:
        try:
            sol = solomon_i1_construction(
                inst, alpha=alpha, mu=mu, lam=lam,
                seed_criterion_truck=s_truck,
                seed_criterion_drone=s_drone,
            )

            if not sol.all_served(inst):
                continue   # bỏ qua (safety net, không nên xảy ra)

            obj    = sol.makespan()
            tw_pen = sol.penalty_tw(inst)
            score  = obj + tw_pen * 1000

            if best_sol is None or score < best_obj:
                best_sol = sol
                best_obj = score

        except Exception:
            continue

    # Fallback nếu tất cả config đều thất bại
    if best_sol is None:
        best_sol = solomon_i1_construction(inst)

    return best_sol
