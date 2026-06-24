"""
construction.py — Khởi tạo tham lam (Greedy Insertion) cho MVRPD-TW Multi-Trip

FIX:
  - Không dùng biến `v` từ vòng drone khi fall-through sang truck
  - Kiểm tra feasibility đầy đủ (bao gồm drone_assign)
  - Sau khi chèn thất bại phải revert đúng cách
"""
from __future__ import annotations
from instance import Instance
from solution import Trip, Vehicle, Solution, precompute_trip, precompute_vehicle


def _trip_feasible(trip: Trip, inst: Instance, is_drone: bool) -> bool:
    """Kiểm tra một trip có hợp lệ không (tải, TW, pin, phân công drone)."""
    # 1. Tải trọng
    cap = inst.drone_capacity if is_drone else inst.truck_capacity
    if trip.total_load > cap + 1e-9:
        return False

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
            for k in range(len(seq) - 1)
        )
        if flight_time > inst.drone_range + 1e-9:
            return False

    # 4. Drone không được phục vụ khách C1
    if is_drone:
        for nid in trip.sequence:
            if nid != 0 and nid in inst.c1_ids:
                return False

    return True


def _trip_tw_violation(trip: Trip, inst: Instance) -> float:
    """Tổng vi phạm time window (a[i] - due[i])+ của 1 trip. Dùng để so sánh
    'mức độ tệ' giữa các phương án fallback khi không có lựa chọn feasible."""
    cdata = {c.id: c for c in inst.all_nodes}
    total = 0.0
    for i, nid in enumerate(trip.sequence):
        if nid == 0:
            continue
        if i >= len(trip.a):
            total += 1e9
            continue
        total += max(0.0, trip.a[i] - cdata[nid].due)
    return total


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
    for v in trucks + drones:
        t = Trip(sequence=[0, 0], is_drone=v.is_drone)
        precompute_trip(t, inst)
        v.trips.append(t)

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

        # ── Fallback: chọn vị trí gây vi phạm TW ÍT NHẤT trong số mọi
        # truck/mọi trip/mọi vị trí chèn có thể (không chỉ ép cứng vào
        # cuối truck[0] như trước). Vẫn để tabu search xử lý phạt còn lại,
        # nhưng điểm khởi đầu sẽ tốt hơn nhiều, đỡ gánh nặng cho tabu.
        if not inserted:
            fb_best_viol = float('inf')
            fb_best_v = None
            fb_best_ti = None
            fb_best_pos = None
            fb_best_open_new = False

            for v in trucks:
                for ti, trip in enumerate(v.trips):
                    for pos in range(1, len(trip.sequence)):
                        trip.sequence.insert(pos, c.id)
                        precompute_trip(trip, inst)
                        # Bỏ qua nếu vi phạm tải (vẫn phải hợp lệ về capacity)
                        if trip.total_load <= inst.truck_capacity + 1e-9:
                            viol = _trip_tw_violation(trip, inst)
                            if viol < fb_best_viol:
                                fb_best_viol = viol
                                fb_best_v, fb_best_ti, fb_best_pos = v, ti, pos
                                fb_best_open_new = False
                        trip.sequence.pop(pos)
                        precompute_trip(trip, inst)

                # Phương án mở trip mới cho xe này
                new_trip = Trip(
                    sequence=[0, c.id, 0],
                    is_drone=False,
                    start_time=v.trips[-1].return_time
                )
                precompute_trip(new_trip, inst)
                if new_trip.total_load <= inst.truck_capacity + 1e-9:
                    viol = _trip_tw_violation(new_trip, inst)
                    if viol < fb_best_viol:
                        fb_best_viol = viol
                        fb_best_v, fb_best_ti, fb_best_pos = v, None, None
                        fb_best_open_new = True

            if fb_best_open_new:
                new_trip = Trip(
                    sequence=[0, c.id, 0],
                    is_drone=False,
                    start_time=fb_best_v.trips[-1].return_time
                )
                precompute_trip(new_trip, inst)
                fb_best_v.trips.append(new_trip)
            elif fb_best_v is not None:
                fb_best_v.trips[fb_best_ti].sequence.insert(fb_best_pos, c.id)
                precompute_trip(fb_best_v.trips[fb_best_ti], inst)
            else:
                # Trường hợp cực hiếm (vd. capacity quá nhỏ): giữ hành vi cũ
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
        v.trips = [t for t in v.trips if len(t.sequence) > 2]
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
        precompute_vehicle(v, inst)

    return Solution(trucks=trucks, drones=drones)
