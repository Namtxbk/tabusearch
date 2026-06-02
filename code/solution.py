"""
solution.py — Biểu diễn lời giải Multi-Trip cho MVRPD-TW
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

from instance import Instance


# ─────────────────────────────────────────────────────────────────────────────
# Trip — một chuyến đi xuất phát từ depot và trở về depot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trip:
    sequence: List[int] = field(default_factory=lambda: [0, 0])
    is_drone: bool = False
    start_time: float = 0.0      # thời điểm rời depot

    # Precomputed
    a: List[float] = field(default_factory=list, repr=False)
    F: List[float] = field(default_factory=list, repr=False)
    prefix_load: List[float] = field(default_factory=list, repr=False)
    suffix_load: List[float] = field(default_factory=list, repr=False)
    total_load: float = 0.0
    total_dist: float = 0.0
    return_time: float = 0.0     # = a[-1], thời điểm về depot

    def customers(self) -> List[int]:
        return self.sequence[1:-1]

    def __len__(self):
        return len(self.sequence) - 2

    def copy(self) -> 'Trip':
        return Trip(
            sequence=self.sequence[:],
            is_drone=self.is_drone,
            start_time=self.start_time,
            a=self.a[:],
            F=self.F[:],
            prefix_load=self.prefix_load[:],
            suffix_load=self.suffix_load[:],
            total_load=self.total_load,
            total_dist=self.total_dist,
            return_time=self.return_time,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle — một phương tiện gồm nhiều trips tuần tự
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    is_drone: bool = False
    trips: List[Trip] = field(default_factory=list)

    def finish_time(self) -> float:
        """Thời điểm phương tiện hoàn thành tất cả trips."""
        return self.trips[-1].return_time if self.trips else 0.0

    def all_customers(self) -> List[int]:
        result = []
        for t in self.trips:
            result.extend(t.customers())
        return result

    def copy(self) -> 'Vehicle':
        return Vehicle(
            is_drone=self.is_drone,
            trips=[t.copy() for t in self.trips],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Precompute một Trip (có tính start_time)
# ─────────────────────────────────────────────────────────────────────────────

def precompute_trip(trip: Trip, inst: Instance) -> None:
    """
    Tính a[], F[], prefix_load[], suffix_load[], total_load, total_dist, return_time.
    Thời điểm xuất phát từ depot = trip.start_time.
    """
    seq = trip.sequence
    n = len(seq)
    is_drone = trip.is_drone
    cdata = {c.id: c for c in inst.all_nodes}

    # ── Arrival time ─────────────────────────────────────────────────────
    a = [0.0] * n
    a[0] = trip.start_time  # xuất phát từ depot lúc start_time

    for i in range(1, n):
        prev = seq[i - 1]
        curr = seq[i]
        t_travel = inst.travel_time(prev, curr, is_drone=is_drone)
        s_prev = cdata[prev].service
        arrive = a[i - 1] + s_prev + t_travel
        a[i] = max(arrive, cdata[curr].ready)

    # ── Forward Time Slack ────────────────────────────────────────────────
    F = [0.0] * n
    F[n - 1] = cdata[seq[-1]].due - a[n - 1]

    for i in range(n - 2, -1, -1):
        curr = seq[i]
        nxt = seq[i + 1]
        s_i = cdata[curr].service
        t_nxt = inst.travel_time(curr, nxt, is_drone=is_drone)
        wait_nxt = max(0.0, cdata[nxt].ready - (a[i] + s_i + t_nxt))
        F[i] = min(cdata[curr].due - a[i], F[i + 1] - wait_nxt)

    # ── Prefix / Suffix load ──────────────────────────────────────────────
    prefix_load = [0.0] * n
    suffix_load = [0.0] * n
    for i in range(1, n - 1):
        prefix_load[i] = prefix_load[i - 1] + cdata[seq[i]].demand
    for i in range(n - 2, 0, -1):
        suffix_load[i] = suffix_load[i + 1] + cdata[seq[i]].demand

    # ── Gán vào trip ──────────────────────────────────────────────────────
    trip.a = a
    trip.F = F
    trip.prefix_load = prefix_load
    trip.suffix_load = suffix_load
    trip.total_load = prefix_load[n - 2] if n > 2 else 0.0
    trip.return_time = a[-1]

    dist = sum(inst.dist(seq[i], seq[i + 1]) for i in range(n - 1))
    trip.total_dist = dist


def precompute_vehicle(vehicle: Vehicle, inst: Instance) -> None:
    """
    Precompute toàn bộ trips của 1 phương tiện theo thứ tự tuần tự.
    Trip sau bắt đầu khi trip trước đã về depot.
    """
    current_time = 0.0
    for trip in vehicle.trips:
        trip.start_time = current_time
        precompute_trip(trip, inst)
        current_time = trip.return_time  # trip sau chờ trip này về


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat: Route = Trip (để các module cũ vẫn dùng được)
# ─────────────────────────────────────────────────────────────────────────────

Route = Trip

def precompute(trip: Trip, inst: Instance, start_time: float = 0.0) -> None:
    trip.start_time = start_time
    precompute_trip(trip, inst)


# ─────────────────────────────────────────────────────────────────────────────
# Solution
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Solution:
    """
    trucks : List[Vehicle] — K xe tải, mỗi xe có nhiều trips
    drones : List[Vehicle] — D drone, mỗi drone có nhiều trips
    """
    trucks: List[Vehicle] = field(default_factory=list)
    drones: List[Vehicle] = field(default_factory=list)

    # Backward-compat properties
    @property
    def truck_routes(self) -> List[Trip]:
        """Tất cả trips của tất cả trucks (flatten)."""
        return [t for v in self.trucks for t in v.trips]

    @property
    def drone_routes(self) -> List[Trip]:
        """Tất cả trips của tất cả drones (flatten)."""
        return [t for v in self.drones for t in v.trips]

    def copy(self) -> 'Solution':
        return Solution(
            trucks=[v.copy() for v in self.trucks],
            drones=[v.copy() for v in self.drones],
        )

    def makespan(self) -> float:
        """Thời điểm phương tiện cuối cùng về depot."""
        times = ([v.finish_time() for v in self.trucks] +
                 [v.finish_time() for v in self.drones])
        return max(times) if times else 0.0

    def penalty_tw(self, inst: Instance) -> float:
        total = 0.0
        cdata = {c.id: c for c in inst.all_nodes}
        for trip in self.truck_routes + self.drone_routes:
            for pos, nid in enumerate(trip.sequence):
                if nid == 0:
                    continue
                viol = max(0.0, trip.a[pos] - cdata[nid].due)
                total += viol
        return total

    def penalty_cap(self, inst: Instance) -> float:
        total = 0.0
        for trip in self.truck_routes:
            total += max(0.0, trip.total_load - inst.truck_capacity)
        for trip in self.drone_routes:
            total += max(0.0, trip.total_load - inst.drone_capacity)
        return total

    def penalty_range(self, inst: Instance) -> float:
        """Tính toán phạt nếu tổng thời gian bay của chuyến vượt quá giới hạn Pin."""
        total = 0.0
        for trip in self.drone_routes:
            seq = trip.sequence
            # Đo tổng thời gian drone di chuyển trên không
            flight_time = sum(inst.travel_time(seq[i], seq[i+1], is_drone=True) for i in range(len(seq)-1))
            total += max(0.0, flight_time - inst.drone_range)
        return total
    
    def objective(self, inst: Instance) -> float:
        return (self.makespan()
                + 50.0  * self.penalty_tw(inst)
                + 200.0 * self.penalty_cap(inst)
                + 200.0 * self.penalty_range(inst))

    def is_feasible(self, inst: Instance) -> bool:
        return (self.penalty_tw(inst) == 0.0
                and self.penalty_cap(inst) == 0.0
                and self.penalty_range(inst) == 0.0)

    def all_served(self, inst: Instance) -> bool:
        served = set()
        for trip in self.truck_routes + self.drone_routes:
            for nid in trip.sequence:
                if nid != 0:
                    if nid in served:
                        return False
                    served.add(nid)
        return served == {c.id for c in inst.customers}

    def recompute_all(self, inst: Instance):
        """
        Tính toán lại toàn bộ thông số thời gian, tải trọng của hệ thống xe
        Đảm bảo tính tuần tự tuyến tính: Trip sau bắt đầu khi trip trước quay về Depot.
        """
        for v in self.trucks + self.drones:
            current_time = 0.0
            for trip in v.trips:
                # Trip sau phải đợi trip trước quay về Depot hoàn tất
                trip.start_time = max(0.0, current_time)
                
                # Tính toán chi tiết tải trọng và thời gian đến từng node của trip này
                load = 0.0
                dist = 0.0
                seq = trip.sequence
                
                arrival_times = [0.0] * len(seq)
                arrival_times[0] = trip.start_time
                
                t_curr = trip.start_time
                for i in range(len(seq) - 1):
                    u, u_next = seq[i], seq[i+1]
                    
                    # Cộng dồn tải trọng hàng hóa
                    if u_next != 0:
                        cust = inst.customers[u_next - 1]
                        load += cust.demand
                        
                    # Tính toán thời gian di chuyển giữa 2 điểm liên tiếp
                    dt = inst.travel_time(u, u_next, trip.is_drone)
                    dist += inst.dist(u, u_next)
                    
                    t_arrival = t_curr + dt
                    arrival_times[i+1] = t_arrival
                    
                    # Cộng thêm thời gian phục vụ tại điểm nếu điểm đó là khách hàng
                    if u_next != 0:
                        cust = inst.customers[u_next - 1]
                        t_curr = max(t_arrival, cust.ready) + cust.service
                    else:
                        t_curr = t_arrival
                        
                trip.total_load = load
                trip.total_dist = dist
                trip.return_time = t_curr  # Điểm kết thúc của trip hiện tại
                
                # Lưu mốc thời gian quay về làm mốc bắt đầu chuyến kế tiếp (Multi-Trip)
                current_time = trip.return_time
                
            # Đồng bộ hóa dữ liệu tổng quan cho phương tiện
            precompute_vehicle(v, inst)

    def summary(self, inst: Instance) -> str:
        lines = [
            f"Instance     : {inst.name}",
            f"Makespan     : {self.makespan():.2f}",
            f"Feasible     : {self.is_feasible(inst)}",
            f"All served   : {self.all_served(inst)}",
            f"Penalty TW   : {self.penalty_tw(inst):.2f}",
            f"Penalty cap  : {self.penalty_cap(inst):.2f}",
            f"Penalty range: {self.penalty_range(inst):.2f}",
            "",
        ]
        for k, v in enumerate(self.trucks):
            for t_idx, trip in enumerate(v.trips):
                custs = trip.customers()
                if not custs:
                    continue
                route_str = ' -> '.join(str(n) for n in [0] + custs + [0])
                lines.append(
                    f"  Truck {k+1} Trip {t_idx+1}: {route_str}"
                    f"  (load={trip.total_load:.3f}/{inst.truck_capacity:.0f},"
                    f" start={trip.start_time:.1f}, return={trip.return_time:.1f})"
                )
        for d, v in enumerate(self.drones):
            for t_idx, trip in enumerate(v.trips):
                custs = trip.customers()
                if not custs:
                    continue
                route_str = ' -> '.join(str(n) for n in [0] + custs + [0])
                lines.append(
                    f"  Drone {d+1} Trip {t_idx+1}: {route_str}"
                    f"  (load={trip.total_load:.3f}/{inst.drone_capacity:.2f},"
                    f" dist={trip.total_dist:.1f}/{inst.drone_range:.1f},"
                    f" start={trip.start_time:.1f}, return={trip.return_time:.1f})"
                )
        return "\n".join(lines)