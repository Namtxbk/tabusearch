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

        # ── Fallback: ĐẢM BẢO TW-FEASIBLE TUYỆT ĐỐI ────────────────────
        # Theo yêu cầu mới: construction phải LUÔN trả về nghiệm không vi
        # phạm time window. Vì multi-trip là tuần tự (1 xe không thể đi 2
        # trip cùng lúc), cách duy nhất khả thi khi mọi phương tiện hiện có
        # đều không kịp giờ là MỞ THÊM 1 PHƯƠNG TIỆN ẢO mới (rảnh từ thời
        # điểm 0). Ưu tiên thử drone ảo trước (nhất quán với thứ tự ưu tiên
        # drone > truck ở Bước 1/2, và vì drone thường nhanh hơn — có thể là
        # lựa chọn khả thi DUY NHẤT khi truck không kịp dù xuất phát ngay từ
        # đầu, ví dụ do khoảng cách quá xa so với due). Số phương tiện ảo
        # thêm được đếm và báo cáo riêng ở cuối hàm, không giấu đi.
        if not inserted:
            drone_can_try = (not c.is_c1 and c.demand <= inst.drone_capacity and
                              (inst.travel_time(0, c.id, True) + inst.travel_time(c.id, 0, True))
                              <= inst.drone_range)
            if drone_can_try:
                extra_d = Vehicle(is_drone=True)
                extra_trip = Trip(sequence=[0, c.id, 0], is_drone=True, start_time=0.0)
                precompute_trip(extra_trip, inst)
                if _trip_feasible(extra_trip, inst, True):
                    extra_d.trips.append(extra_trip)
                    drones.append(extra_d)
                    inserted = True

        if not inserted:
            extra_v = Vehicle(is_drone=False)
            extra_trip = Trip(sequence=[0, c.id, 0], is_drone=False, start_time=0.0)
            precompute_trip(extra_trip, inst)
            if not _trip_feasible(extra_trip, inst, False):
                # Cực hiếm: ngay cả 1 phương tiện rảnh từ đầu, đi thẳng đến
                # khách này cũng không kịp do due quá sớm so với khoảng cách
                # từ depot. Đây là giới hạn vật lý của instance, không phải
                # lỗi thuật toán.
                raise ValueError(
                    f"Khách hàng id={c.id} (due={c.due:.2f}) không thể phục vụ "
                    f"đúng hạn bởi BẤT KỲ phương tiện nào, kể cả phương tiện "
                    f"rảnh từ thời điểm 0 đi thẳng từ depot (instance vật lý "
                    f"không thể feasible với khách này)."
                )
            extra_v.trips.append(extra_trip)
            trucks.append(extra_v)
            inserted = True

    # Dọn trip rỗng và precompute lại toàn bộ
    for v in trucks + drones:
        v.trips = [t for t in v.trips if len(t.sequence) > 2]
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
        precompute_vehicle(v, inst)

    sol = Solution(trucks=trucks, drones=drones)
    # Báo cáo minh bạch số truck VƯỢT ĐỊNH (inst.num_trucks) đã phải mở
    # thêm để đảm bảo TW-feasible tuyệt đối. Đây không phải số truck thật
    # sẽ dùng trong vận hành — là chỉ số cho biết instance "khó" tới mức
    # nào với số xe ban đầu. Gắn làm attribute thay vì đổi signature hàm,
    # để không phá vỡ các nơi khác (tabu_search.py, main.py, ...) đang gọi
    # build_initial_solution(inst) và chỉ mong nhận về 1 Solution.
    sol.extra_trucks_used = max(0, len(trucks) - inst.num_trucks)
    sol.extra_drones_used = max(0, len(drones) - inst.num_drones)
    return sol
