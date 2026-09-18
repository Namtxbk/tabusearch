"""
<<<<<<< HEAD
construction.py — Khởi tạo tham lam (Greedy Insertion) cho MVRPD-TW Multi-Trip

FIX:
  - Không dùng biến `v` từ vòng drone khi fall-through sang truck
  - Kiểm tra feasibility đầy đủ (bao gồm drone_assign)
  - Sau khi chèn thất bại phải revert đúng cách
=======
construction.py — Greedy Insertion theo pseudocode của giảng viên

Thuật toán:
  1. Sắp xếp khách theo deadline tăng dần.
  2. Với mỗi khách i:
     - Sinh tất cả move từ mọi xe tương thích:
         + Chèn vào mọi vị trí trong mọi trip hiện có
         + Mở trip mới ở mọi vị trí trong chuỗi trip của xe
     - Phân loại: feasibleMoves (không vi phạm ràng buộc cứng, không tạo thêm
       vi phạm TW) và penalizedMoves (không vi phạm ràng buộc cứng nhưng có
       thêm vi phạm TW).
     - Chọn: ưu tiên feasibleMoves (min makespan); nếu không có thì chọn
       penalizedMoves theo lexicographic (min ΔTW → min makespan → min dist).
     - Áp dụng move đã chọn.
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from instance import Instance
from solution import Trip, Vehicle, Solution, precompute_trip, precompute_vehicle


<<<<<<< HEAD
def _trip_feasible(trip: Trip, inst: Instance, is_drone: bool) -> bool:
    """Kiểm tra một trip có hợp lệ không (tải, TW, pin, phân công drone)."""
    # 1. Tải trọng
=======
# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra ràng buộc CỨNG của 1 trip (tải, pin drone, phân công C1)
# KHÔNG bao gồm TW — TW được xử lý riêng qua compute_delta_tw
# ─────────────────────────────────────────────────────────────────────────────
def _hard_feasible(trip: Trip, inst: Instance, is_drone: bool) -> bool:
    """Kiểm tra ràng buộc cứng (KHÔNG bao gồm TW):
    - Tải trọng ≤ capacity
    - Tổng thời gian bay drone ≤ drone_range
    - Drone không phục vụ khách C1
    """
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
    cap = inst.drone_capacity if is_drone else inst.truck_capacity
    if trip.total_load > cap + 1e-9:
        return False

<<<<<<< HEAD
    # 2. Time windows — dùng trip.a[] đã được precompute
    cdata = {c.id: c for c in inst.all_nodes}
    for i, nid in enumerate(trip.sequence):
        if nid != 0:
            if i >= len(trip.a):
                return False
            if trip.a[i] > cdata[nid].due + 1e-9:
                return False

    # 3. Giới hạn pin drone (tổng thời gian bay)
    if is_drone:
        seq = trip.sequence
        flight_time = sum(
            inst.travel_time(seq[k], seq[k + 1], True)
=======
    if is_drone:
        seq = trip.sequence
        flight_time = sum(
            inst.travel_time(seq[k], seq[k+1], True)
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
            for k in range(len(seq) - 1)
        )
        if flight_time > inst.drone_range + 1e-9:
            return False
<<<<<<< HEAD

    # 4. Drone không được phục vụ khách C1
    if is_drone:
        for nid in trip.sequence:
=======
        for nid in seq:
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
            if nid != 0 and nid in inst.c1_ids:
                return False

    return True


<<<<<<< HEAD
def _try_insert_into_trip(trip: Trip, cust_id: int, inst: Instance,
                           is_drone: bool) -> bool:
    """
    Thử chèn cust_id vào vị trí tốt nhất trong trip (best-insertion).
    Trả về True nếu chèn thành công, trip được cập nhật in-place.
    """
    best_pos = None
    best_cost = float('inf')

    for pos in range(1, len(trip.sequence)):
        trip.sequence.insert(pos, cust_id)
        precompute_trip(trip, inst)
        if _trip_feasible(trip, inst, is_drone):
            # Chi phí chèn = tăng thêm về return_time
            cost = trip.return_time
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        # Hoàn tác để thử vị trí tiếp theo
        trip.sequence.pop(pos)
        precompute_trip(trip, inst)

    if best_pos is not None:
        trip.sequence.insert(best_pos, cust_id)
        precompute_trip(trip, inst)
        return True
    return False


def build_initial_solution(inst: Instance) -> Solution:
    """
    Greedy Insertion cho MVRPD-TW Multi-Trip:
      1. Sắp xếp khách theo due (time window) tăng dần.
      2. Với mỗi khách:
         a. Thử chèn vào trip đang có của drone (nếu đủ điều kiện).
         b. Nếu không được, mở trip mới cho drone.
         c. Nếu vẫn không được, thử truck (chèn vào trip có / mở trip mới).
         d. Fallback cuối: ép vào truck[0] trip mới (để tabu xử lý phạt).
    """
    sorted_customers = sorted(inst.customers, key=lambda c: c.due)

    trucks = [Vehicle(is_drone=False) for _ in range(inst.num_trucks)]
    drones = [Vehicle(is_drone=True)  for _ in range(inst.num_drones)]

    # Khởi tạo một trip rỗng cho mỗi phương tiện
=======
# ─────────────────────────────────────────────────────────────────────────────
# Tính ΔTW của toàn bộ solution sau khi áp dụng 1 move (theo pseudocode)
# ΔTW = Σ max(0, a_j - l_j) với mọi khách j đã có trong solution
# ─────────────────────────────────────────────────────────────────────────────
def _compute_delta_tw(sol: Solution, inst: Instance) -> float:
    """Tổng vi phạm TW của toàn bộ solution hiện tại.
    Khi ignore_tw=True: luôn trả về 0 (mọi move đều là feasible move).
    """
    if inst.ignore_tw:
        return 0.0
    cdata = {c.id: c for c in inst.all_nodes}
    total = 0.0
    for v in sol.trucks + sol.drones:
        for trip in v.trips:
            for pos, nid in enumerate(trip.sequence):
                if nid == 0:
                    continue
                if pos < len(trip.a):
                    total += max(0.0, trip.a[pos] - cdata[nid].due)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Kiểm tra xe có tương thích với khách không
# ─────────────────────────────────────────────────────────────────────────────
def _compatible(cust_id: int, is_drone: bool, inst: Instance) -> bool:
    """Kiểm tra xe loại is_drone có tương thích với khách cust_id không."""
    c = inst.customers[cust_id - 1]
    if is_drone:
        if c.is_c1:
            return False
        if c.demand > inst.drone_capacity:
            return False
        rt = inst.travel_time(0, cust_id, True) + inst.travel_time(cust_id, 0, True)
        if rt > inst.drone_range:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Cấu trúc lưu 1 move ứng viên
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _Move:
    sol: Solution          # solution sau khi áp move (deep copy)
    delta_tw: float        # tổng vi phạm TW toàn solution
    makespan: float        # makespan sau khi áp move
    total_dist: float      # tổng khoảng cách sau khi áp move
    has_tw_violation: bool # move này có tạo thêm vi phạm TW không


def _total_dist(sol: Solution) -> float:
    """Tính tổng khoảng cách di chuyển của toàn bộ solution."""
    d = 0.0
    for v in sol.trucks + sol.drones:
        for trip in v.trips:
            d += trip.total_dist
    return d


# ─────────────────────────────────────────────────────────────────────────────
# Sinh tất cả move cho 1 khách i trên 1 xe v
# ─────────────────────────────────────────────────────────────────────────────
def _generate_moves(
    cust_id: int,
    vi: int,
    is_drone: bool,
    sol: Solution,
    inst: Instance,
    tw_before: float,
) -> List[_Move]:
    """
    Sinh tất cả move hợp lệ (không vi phạm ràng buộc cứng) cho khách cust_id
    trên xe thứ vi (loại is_drone).

    Theo pseudocode: generate
      - every position in every existing trip of v
      - a new trip at every position in v's trip sequence
    """
    moves = []
    v_orig = sol.drones[vi] if is_drone else sol.trucks[vi]

    # ── Loại 1: chèn vào vị trí trong trip đang có ────────────────────────
    for ti, trip in enumerate(v_orig.trips):
        for pos in range(1, len(trip.sequence)):
            # Tạo solution copy, thực hiện chèn
            cand = sol.copy()
            cv = cand.drones[vi] if is_drone else cand.trucks[vi]
            cv.trips[ti].sequence.insert(pos, cust_id)
            precompute_vehicle(cv, inst)

            # Kiểm tra ràng buộc cứng trên trip bị ảnh hưởng
            if not _hard_feasible(cv.trips[ti], inst, is_drone):
                continue

            # Tính ΔTW toàn solution
            dtw = _compute_delta_tw(cand, inst)
            has_viol = dtw > tw_before + 1e-9

            moves.append(_Move(
                sol=cand,
                delta_tw=dtw,
                makespan=cand.makespan(),
                total_dist=_total_dist(cand),
                has_tw_violation=has_viol,
            ))

    # ── Loại 2: mở trip mới ở mọi vị trí trong chuỗi trip của xe ─────────
    # Ví dụ xe có trips [A, B]: trip mới có thể chen ở vị trí 0, 1, 2
    # → [new, A, B], [A, new, B], [A, B, new]
    # Thứ tự trip trong xe quyết định start_time (tuần tự), nên vị trí
    # chèn trong chuỗi trip ảnh hưởng đến makespan
    n_trips = len(v_orig.trips)
    for trip_pos in range(n_trips + 1):  # 0 .. n_trips
        new_trip = Trip(
            sequence=[0, cust_id, 0],
            is_drone=is_drone,
        )

        cand = sol.copy()
        cv = cand.drones[vi] if is_drone else cand.trucks[vi]
        cv.trips.insert(trip_pos, new_trip)
        precompute_vehicle(cv, inst)  # tính lại start_time cho mọi trip

        # Kiểm tra ràng buộc cứng trip mới (ở đúng vị trí đã được tính)
        if not _hard_feasible(cv.trips[trip_pos], inst, is_drone):
            continue

        dtw = _compute_delta_tw(cand, inst)
        has_viol = dtw > tw_before + 1e-9

        moves.append(_Move(
            sol=cand,
            delta_tw=dtw,
            makespan=cand.makespan(),
            total_dist=_total_dist(cand),
            has_tw_violation=has_viol,
        ))

    return moves


# ─────────────────────────────────────────────────────────────────────────────
# Hàm chính: build_initial_solution
# ─────────────────────────────────────────────────────────────────────────────
def build_initial_solution(inst: Instance) -> Solution:
    """
    Greedy Insertion theo pseudocode của giảng viên.

    Với mỗi khách (theo thứ tự due tăng dần):
      1. Sinh tất cả move từ mọi xe tương thích.
      2. Phân loại feasibleMoves / penalizedMoves.
      3. Chọn move tốt nhất theo tiêu chí lexicographic.
      4. Áp dụng move.
    """
    sorted_customers = sorted(inst.customers, key=lambda c: c.due)

    # Khởi tạo solution rỗng với số xe theo instance
    trucks = [Vehicle(is_drone=False) for _ in range(inst.num_trucks)]
    drones = [Vehicle(is_drone=True)  for _ in range(inst.num_drones)]
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
    for v in trucks + drones:
        t = Trip(sequence=[0, 0], is_drone=v.is_drone)
        precompute_trip(t, inst)
        v.trips.append(t)
<<<<<<< HEAD

    for c in sorted_customers:
        inserted = False

        # ── Bước 1: Thử drone ───────────────────────────────────────────
        if not c.is_c1 and c.demand <= inst.drone_capacity:
            # Kiểm tra khách có thể đi drone không (vòng tròn 0→c→0 ≤ range)
            rt = (inst.travel_time(0, c.id, True)
                  + inst.travel_time(c.id, 0, True))
            if rt <= inst.drone_range:
                for v in drones:
                    # Thử chèn vào trip CUỐI CÙNG đang có
                    if _try_insert_into_trip(v.trips[-1], c.id, inst, True):
                        inserted = True
                        break

                    # Thử mở trip MỚI cho drone này
                    new_trip = Trip(
                        sequence=[0, c.id, 0],
                        is_drone=True,
                        start_time=v.trips[-1].return_time
                    )
                    precompute_trip(new_trip, inst)
                    if _trip_feasible(new_trip, inst, True):
                        v.trips.append(new_trip)
                        inserted = True
                        break

        if inserted:
            continue

        # ── Bước 2: Thử truck ───────────────────────────────────────────
        for v in trucks:
            # Thử chèn vào trip CUỐI CÙNG đang có
            if _try_insert_into_trip(v.trips[-1], c.id, inst, False):
                inserted = True
                break

            # Thử mở trip MỚI cho truck này
            new_trip = Trip(
                sequence=[0, c.id, 0],
                is_drone=False,
                start_time=v.trips[-1].return_time
            )
            precompute_trip(new_trip, inst)
            if _trip_feasible(new_trip, inst, False):
                v.trips.append(new_trip)
                inserted = True
                break

        # ── Fallback: ép vào truck[0] để tabu xử lý ────────────────────
        if not inserted:
            v0 = trucks[0]
            fb_trip = Trip(
                sequence=[0, c.id, 0],
                is_drone=False,
                start_time=v0.trips[-1].return_time
            )
            precompute_trip(fb_trip, inst)
            v0.trips.append(fb_trip)

    # Dọn trip rỗng và precompute lại toàn bộ
    for v in trucks + drones:
=======

    sol = Solution(trucks=trucks, drones=drones)

    for c in sorted_customers:
        # ΔTW của solution TRƯỚC khi chèn khách c
        tw_before = _compute_delta_tw(sol, inst)

        feasible_moves: List[_Move] = []
        penalized_moves: List[_Move] = []

        # Sinh moves từ mọi xe tương thích
        for vi in range(len(sol.trucks)):
            if not _compatible(c.id, False, inst):
                continue
            moves = _generate_moves(c.id, vi, False, sol, inst, tw_before)
            for m in moves:
                if m.has_tw_violation:
                    penalized_moves.append(m)
                else:
                    feasible_moves.append(m)

        for vi in range(len(sol.drones)):
            if not _compatible(c.id, True, inst):
                continue
            moves = _generate_moves(c.id, vi, True, sol, inst, tw_before)
            for m in moves:
                if m.has_tw_violation:
                    penalized_moves.append(m)
                else:
                    feasible_moves.append(m)

        # Chọn move tốt nhất
        best_move: Optional[_Move] = None

        if feasible_moves:
            # Ưu tiên feasible: chọn min makespan
            best_move = min(feasible_moves, key=lambda m: m.makespan)
        elif penalized_moves:
            # Lexicographic: (min ΔTW, min makespan, min total_dist)
            best_move = min(
                penalized_moves,
                key=lambda m: (m.delta_tw, m.makespan, m.total_dist)
            )

        if best_move is not None:
            sol = best_move.sol
        else:
            # Không có move nào (không có xe tương thích) — không xảy ra
            # trong bài toán này vì truck luôn tương thích với mọi khách,
            # nhưng giữ fallback an toàn
            raise ValueError(
                f"Không tìm được move nào cho khách id={c.id}. "
                f"Kiểm tra lại dữ liệu instance."
            )

    # Dọn trip rỗng và precompute lại
    for v in sol.trucks + sol.drones:
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
        v.trips = [t for t in v.trips if len(t.sequence) > 2]
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
        precompute_vehicle(v, inst)

<<<<<<< HEAD
    return Solution(trucks=trucks, drones=drones)
=======
    sol.extra_trucks_used = 0  # không mở phương tiện ảo trong thiết kế mới
    sol.extra_drones_used = 0
    return sol
>>>>>>> 36a6bab7a9c8f9c11355472fa708ef5d3369aa1e
