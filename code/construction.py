"""
construction.py — Khởi tạo tham lam (Greedy Insertion) cho MVRPD-TW Multi-Trip
"""
from __future__ import annotations
from instance import Instance
from solution import Trip, Vehicle, Solution, precompute_trip, precompute_vehicle

def _trip_feasible(trip: Trip, inst: Instance, is_drone: bool) -> bool:
    """Kiểm tra chuyến đi có hợp lệ toàn diện hay không."""
    # 1. Kiểm tra tải trọng
    cap = inst.drone_capacity if is_drone else inst.truck_capacity
    if trip.total_load > cap:
        return False
        
    # 2. Kiểm tra Time Windows (Không đến trễ)
    cdata = {c.id: c for c in inst.all_nodes}
    for i, nid in enumerate(trip.sequence):
        if nid != 0 and trip.a[i] > cdata[nid].due:
            return False
            
    # 3. Kiểm tra Giới hạn Pin của Drone (Tính bằng thời gian bay)
    if is_drone:
        seq = trip.sequence
        flight_time = sum(inst.travel_time(seq[i], seq[i+1], True) for i in range(len(seq)-1))
        if flight_time > inst.drone_range:
            return False
            
    return True

def build_initial_solution(inst: Instance) -> Solution:
    """
    Chiến lược Gom tuyến:
    - Sắp xếp khách hàng theo giờ đóng cửa (Due) để ưu tiên giao trước.
    - Cố gắng chèn khách vào chuyến hiện tại của xe.
    - Nếu vi phạm (tải trọng/giờ/pin), tạo chuyến đi mới nối tiếp chuyến cũ.
    """
    sorted_customers = sorted(inst.customers, key=lambda c: c.due)
    
    trucks = [Vehicle(is_drone=False) for _ in range(inst.num_trucks)]
    drones = [Vehicle(is_drone=True) for _ in range(inst.num_drones)]
    
    # Khởi tạo chuyến rỗng đầu tiên cho mọi phương tiện
    for v in trucks + drones:
        t = Trip(sequence=[0, 0], is_drone=v.is_drone)
        precompute_trip(t, inst)
        v.trips.append(t)
        
    for c in sorted_customers:
        inserted = False
        
        # 1. Thử ưu tiên chèn vào nhóm Drone
        if not c.is_c1 and c.demand <= inst.drone_capacity:
            for v in drones:
                last_trip = v.trips[-1]
                # Thử nhét khách vào ngay trước khi quay về Depot
                last_trip.sequence.insert(-1, c.id)
                precompute_trip(last_trip, inst)
                
                if _trip_feasible(last_trip, inst, True):
                    inserted = True  # Nhét thành công, xe đi tiếp
                    break
                else:
                    # Rút khách ra (Hoàn tác)
                    last_trip.sequence.pop(-2)
                    precompute_trip(last_trip, inst)
                    
                    # Thử tạo Trip mới xuất phát ngay sau khi Trip cũ quay về Depot
                    new_trip = Trip(sequence=[0, c.id, 0], is_drone=True, start_time=last_trip.return_time)
                    precompute_trip(new_trip, inst)
                    if _trip_feasible(new_trip, inst, True):
                        v.trips.append(new_trip)
                        inserted = True
                        break
                        
        if inserted: continue
        
        # 2. Thử chèn vào nhóm Truck (Tương tự logic Drone)
        for v in trucks:
            last_trip = v.trips[-1]
            last_trip.sequence.insert(-1, c.id)
            precompute_trip(last_trip, inst)
            
            if _trip_feasible(last_trip, inst, False):
                inserted = True
                break
            else:
                last_trip.sequence.pop(-2)
                precompute_trip(last_trip, inst)
                
                new_trip = Trip(sequence=[0, c.id, 0], is_drone=False, start_time=last_trip.return_time)
                precompute_trip(new_trip, inst)
                if _trip_feasible(new_trip, inst, False):
                    v.trips.append(new_trip)
                    inserted = True
                    break
                    
        # 3. An toàn: Nếu mọi cách đều vi phạm (do dữ liệu siêu gắt), 
        # ép nó vào 1 chuyến mới của Truck để Tabu Search xử lý phạt sau.
        if not inserted:
            v = trucks[0]
            new_trip = Trip(sequence=[0, c.id, 0], is_drone=False, start_time=v.trips[-1].return_time)
            precompute_trip(new_trip, inst)
            v.trips.append(new_trip)

    # Dọn dẹp các chuyến rỗng (nếu có)
    for v in trucks + drones:
        v.trips = [t for t in v.trips if len(t.sequence) > 2]
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
        precompute_vehicle(v, inst)

    sol = Solution(trucks=trucks, drones=drones)
    return sol