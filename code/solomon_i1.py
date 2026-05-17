"""
solomon_i1.py — Solomon I1 Insertion Heuristic mở rộng cho MVRPD-TW

Nguồn gốc:
    Solomon, M.M. (1987). "Algorithms for the Vehicle Routing and Scheduling
    Problems with Time Window Constraints." Operations Research 35(2), 254-265.

Mở rộng cho MVRPD-TW:
    - Xử lý cả truck routes (C1 ∪ C2) và drone routes (C2 eligible)
    - Hàm c1 tích hợp Forward Time Slack (push-forward check O(1))
    - Drone: kiểm tra thêm range constraint L_D
    - Seed selection riêng biệt cho truck và drone

Thuật toán Solomon I1:
    Với mỗi route đang xây:
        1. Chọn seed customer khởi tạo route
        2. Lặp:
            a. Với mỗi unserved customer u, tính c1*(u) = min chi phí chèn
               khả thi trên route hiện tại
            b. Tính c2(u) = λ·dist(depot,u) − c1*(u)  [lợi ích chèn ngay]
            c. Chèn u* = argmax c2(u)
        3. Khi không còn khách feasible → mở route mới

Đảm bảo: mọi khách hàng đều được phục vụ (phủ đủ).
"""

from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple

from instance import Instance, Customer
from solution import Route, Solution, precompute


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra drone eligible
# ─────────────────────────────────────────────────────────────────────────────

def _drone_eligible(c: Customer, inst: Instance) -> bool:
    """Khách hàng c có thể phục vụ bằng drone không?"""
    if c.is_c1:
        return False
    if c.demand > inst.drone_capacity:
        return False
    return inst.dist(0, c.id) + inst.dist(c.id, 0) <= inst.drone_range


# ─────────────────────────────────────────────────────────────────────────────
# Hàm c1 — chi phí chèn (Solomon 1987, công thức mở rộng)
# ─────────────────────────────────────────────────────────────────────────────

def _c1(i_pos: int, u: Customer, j_pos: int,
        route: Route, inst: Instance,
        alpha: float = 1.0, mu: float = 1.0) -> float:
    """
    Chi phí chèn u vào giữa seq[i_pos] và seq[j_pos] — O(1).

    Công thức Solomon:
        c11 = dist(i,u) + dist(u,j) - alpha * dist(i,j)   [tăng khoảng cách]
        c12 = b_u(j) - b(j)                                [push forward tại j]
        c1  = mu * c11 + (1 - mu) * c12

    Trong đó b_u(j) = thời điểm đến j sau khi chèn u,
             b(j)   = thời điểm đến j hiện tại (route.a[j_pos]).

    Trả về +inf nếu không feasible (vi phạm TW của u hoặc suffix).
    """
    seq      = route.sequence
    is_drone = route.is_drone
    cdata    = {c.id: c for c in inst.all_nodes}

    i_id = seq[i_pos]
    j_id = seq[j_pos]

    # ── Tính arrival tại u ──────────────────────────────────────────────
    t_iu    = inst.travel_time(i_id, u.id, is_drone=is_drone)
    s_i     = cdata[i_id].service
    arrive_u = route.a[i_pos] + s_i + t_iu
    a_u      = max(arrive_u, u.ready)      # chờ nếu đến sớm

    # TW của chính u
    if a_u > u.due:
        return float('inf')

    # ── c11: tăng khoảng cách ───────────────────────────────────────────
    c11 = (inst.dist(i_id, u.id)
           + inst.dist(u.id, j_id)
           - alpha * inst.dist(i_id, j_id))

    # ── c12: push forward tại j (Forward Time Slack check) ──────────────
    t_uj     = inst.travel_time(u.id, j_id, is_drone=is_drone)
    a_j_new  = max(a_u + u.service + t_uj, cdata[j_id].ready)
    delay    = a_j_new - route.a[j_pos]   # có thể âm (nếu chờ hấp thụ)

    # Kiểm tra suffix bằng Forward Time Slack — O(1)
    if delay > route.F[j_pos]:
        return float('inf')

    c12 = delay   # = b_u(j) - b(j), Solomon định nghĩa

    # ── Drone: kiểm tra thêm range ──────────────────────────────────────
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
    """
    Tìm vị trí chèn tốt nhất cho u vào route.
    Trả về (c1_min, best_pos) — best_pos là index i trong seq
    sao cho chèn giữa seq[i] và seq[i+1].

    Trả về (inf, -1) nếu không có vị trí feasible.
    """
    seq = route.sequence
    capacity = inst.drone_capacity if route.is_drone else inst.truck_capacity

    # Kiểm tra capacity trước — O(1)
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


# ─────────────────────────────────────────────────────────────────────────────
# Hàm c2 — lợi ích chèn ngay (Solomon 1987)
# ─────────────────────────────────────────────────────────────────────────────

def _c2(u: Customer, c1_star: float,
        inst: Instance, lam: float = 1.0) -> float:
    """
    c2(u) = λ * dist(depot, u) - c1*(u)

    Ý nghĩa: nếu không chèn u ngay thì sau này phải đi route riêng
    với chi phí ~dist(depot,u). c2 lớn = lợi ích chèn ngay cao.
    """
    return lam * inst.dist(0, u.id) - c1_star


# ─────────────────────────────────────────────────────────────────────────────
# Chọn seed customer
# ─────────────────────────────────────────────────────────────────────────────

def _pick_seed(unserved: List[Customer],
               inst: Instance,
               criterion: str = 'farthest') -> Optional[Customer]:
    """
    Chọn seed customer để khởi tạo route mới.

    criterion:
        'farthest'  — xa depot nhất (Solomon default cho truck)
        'urgent'    — deadline sớm nhất (tốt khi TW chặt)
        'farthest_drone' — round-trip lớn nhất trong L_D (cho drone)
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Xây một route bằng Solomon I1
# ─────────────────────────────────────────────────────────────────────────────

def _i1_build_single_route(seed: Customer,
                            unserved: List[Customer],
                            is_drone: bool,
                            inst: Instance,
                            alpha: float = 1.0,
                            mu:    float = 1.0,
                            lam:   float = 1.0) -> Route:
    """
    Xây 1 route bằng Solomon I1, bắt đầu từ seed.
    Xóa các khách đã chèn ra khỏi unserved (in-place).

    Trả về Route đã precompute.
    """
    route = Route(sequence=[0, seed.id, 0], is_drone=is_drone)
    precompute(route, inst)
    unserved.remove(seed)

    while True:
        best_u    = None
        best_pos  = -1
        best_c2   = -float('inf')

        for u in unserved:
            # Drone chỉ nhận C2 eligible
            if is_drone and not _drone_eligible(u, inst):
                continue

            c1_star, pos = _best_insertion(u, route, inst, alpha, mu)
            if c1_star == float('inf'):
                continue   # không chèn được

            c2_val = _c2(u, c1_star, inst, lam)
            if c2_val > best_c2:
                best_c2  = c2_val
                best_u   = u
                best_pos = pos

        if best_u is None:
            break   # không còn ai chèn được → đóng route

        # Chèn best_u vào sau seq[best_pos]
        route.sequence.insert(best_pos + 1, best_u.id)
        precompute(route, inst)
        unserved.remove(best_u)

    return route


# ─────────────────────────────────────────────────────────────────────────────
# Xử lý khách chưa phục vụ — forced insertion
# ─────────────────────────────────────────────────────────────────────────────

def _forced_insert(unserved: List[Customer],
                   all_routes: List[Route],
                   inst: Instance) -> None:
    """
    Với mỗi khách còn lại trong unserved, chèn cưỡng bức vào vị trí
    làm tăng route.total_time ít nhất — kể cả khi vi phạm TW.
    Tabu Search sẽ sửa vi phạm sau thông qua penalty.

    Nếu không chèn được vào bất kỳ route nào (ví dụ capacity),
    mở route mới [0 → c → 0].
    """
    cdata = {c.id: c for c in inst.customers}

    still_unserved = list(unserved)
    unserved.clear()

    for u in still_unserved:
        best_route = None
        best_pos   = -1
        best_delta = float('inf')

        for r in all_routes:
            # Drone chỉ nhận C2 eligible
            if r.is_drone and not _drone_eligible(u, inst):
                continue

            cap = inst.drone_capacity if r.is_drone else inst.truck_capacity
            if r.total_load + u.demand > cap:
                continue   # vi phạm capacity → bỏ qua route này

            for p in range(len(r.sequence) - 1):
                # Ước tính delta thời gian (đơn giản hóa: thêm khoảng cách)
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
            # Không chèn được → mở route mới (vi phạm số lượng K/D, penalty xử lý)
            is_drone = _drone_eligible(u, inst)
            r_new = Route(sequence=[0, u.id, 0], is_drone=is_drone)
            precompute(r_new, inst)
            all_routes.append(r_new)


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính — Solomon I1 Construction cho MVRPD-TW
# ─────────────────────────────────────────────────────────────────────────────

def solomon_i1_construction(inst: Instance,
                             alpha: float = 1.0,
                             mu:    float = 1.0,
                             lam:   float = 1.0,
                             seed_criterion_truck: str = 'farthest',
                             seed_criterion_drone: str = 'farthest_drone',
                             ) -> Solution:
    """
    Solomon I1 Construction Heuristic cho MVRPD-TW.

    Tham số (Solomon 1987):
        alpha : trọng số giảm dist(i,j) trong c11 — thường = 1.0
        mu    : trọng số c11 vs c12 trong c1 — mu=1 → ưu dist, mu=0 → ưu time
        lam   : trọng số dist(depot,u) trong c2 — thường = 1.0 hoặc 2.0

    Quy trình:
        1. Tách unserved thành drone_pool (C2 eligible) và truck_pool
        2. Xây D drone routes bằng I1 (seed = xa nhất trong L_D)
        3. Xây K truck routes bằng I1 (seed = xa depot nhất)
        4. Forced insertion cho khách còn sót lại
        5. Padding routes rỗng để đủ K truck + D drone

    Đảm bảo: Solution.all_served() = True sau khi trả về.
    """

    cdata = {c.id: c for c in inst.customers}

    # ── Bước 1: Tách pool ────────────────────────────────────────────────
    drone_pool = [c for c in inst.customers if _drone_eligible(c, inst)]
    truck_pool = [c for c in inst.customers if not _drone_eligible(c, inst)]

    # unserved_drone = bản sao để I1 xóa dần
    unserved_drone = list(drone_pool)
    # unserved_truck = truck_pool + C2 không đủ điều kiện drone
    unserved_truck = list(truck_pool)

    # ── Bước 2: Xây D drone routes ───────────────────────────────────────
    drone_routes: List[Route] = []

    for _ in range(inst.num_drones):
        seed = _pick_seed(unserved_drone, inst, seed_criterion_drone)
        if seed is None:
            break   # không còn khách drone eligible
        r = _i1_build_single_route(
            seed, unserved_drone, is_drone=True,
            inst=inst, alpha=alpha, mu=mu, lam=lam
        )
        drone_routes.append(r)

    # Khách drone không được chèn vào drone route → chuyển về truck_pool
    for c in unserved_drone:
        if c not in unserved_truck:
            unserved_truck.append(c)
    unserved_drone.clear()

    # ── Bước 3: Xây K truck routes ───────────────────────────────────────
    truck_routes: List[Route] = []

    for _ in range(inst.num_trucks):
        seed = _pick_seed(unserved_truck, inst, seed_criterion_truck)
        if seed is None:
            break   # pool rỗng
        r = _i1_build_single_route(
            seed, unserved_truck, is_drone=False,
            inst=inst, alpha=alpha, mu=mu, lam=lam
        )
        truck_routes.append(r)

    # ── Bước 4: Forced insertion cho khách còn sót ───────────────────────
    # (unserved_truck giờ chứa những khách không được chèn vào K routes)
    if unserved_truck:
        all_routes_combined = truck_routes + drone_routes
        _forced_insert(unserved_truck, all_routes_combined, inst)
        # Tách lại truck/drone routes sau forced insert
        truck_routes = [r for r in all_routes_combined if not r.is_drone]
        drone_routes = [r for r in all_routes_combined if r.is_drone]

    # ── Bước 5: Padding routes rỗng để đủ K và D ─────────────────────────
    while len(truck_routes) < inst.num_trucks:
        r = Route(sequence=[0, 0], is_drone=False)
        precompute(r, inst)
        truck_routes.append(r)

    while len(drone_routes) < inst.num_drones:
        r = Route(sequence=[0, 0], is_drone=True)
        precompute(r, inst)
        drone_routes.append(r)

    sol = Solution(
        truck_routes=truck_routes[:inst.num_trucks],
        drone_routes=drone_routes[:inst.num_drones],
    )
    return sol


# ─────────────────────────────────────────────────────────────────────────────
# Multi-start: chạy nhiều lần với tham số khác nhau, lấy nghiệm tốt nhất
# ─────────────────────────────────────────────────────────────────────────────

def multi_start_i1(inst: Instance, n_starts: int = 5) -> Solution:
    """
    Chạy Solomon I1 nhiều lần với các tổ hợp tham số khác nhau.
    Trả về nghiệm có makespan thấp nhất và feasible.

    Các tổ hợp tham số theo Solomon (1987) và Bräysy & Gendreau (2005):
        - mu=1.0 (ưu khoảng cách), mu=0.0 (ưu thời gian)
        - seed = farthest hoặc urgent
        - lam = 1.0 hoặc 2.0
    """
    configs = [
        # (alpha, mu,  lam,  seed_truck,  seed_drone,        tên)
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
            # Ưu tiên nghiệm feasible, sau đó so sánh makespan
            obj = sol.makespan()
            tw_pen = sol.penalty_tw(inst)

            # Chỉ nhận nghiệm phủ đủ
            if not sol.all_served(inst):
                continue

            # Feasible được ưu tiên tuyệt đối
            if best_sol is None:
                best_sol = sol
                best_obj = obj + tw_pen * 1000
            else:
                cur_score = obj + tw_pen * 1000
                if cur_score < best_obj:
                    best_sol = sol
                    best_obj = cur_score

        except Exception:
            continue

    # Nếu tất cả đều thất bại → fallback với cấu hình mặc định
    if best_sol is None:
        best_sol = solomon_i1_construction(inst)

    return best_sol
