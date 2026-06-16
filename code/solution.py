"""
solution.py — Biểu diễn lời giải Multi-Trip cho MVRPD-TW
FIX: 
  - precompute_trip là NGUỒN SỰ THẬT DUY NHẤT cho a[], F[], load, dist, return_time
  - recompute_all chỉ gọi precompute_vehicle (không tự tính song song)
  - suffix_load tính đúng (bao gồm node i chính nó)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from instance import Instance


# ─────────────────────────────────────────────────────────────────────────────
# Trip
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trip:
    sequence: List[int] = field(default_factory=lambda: [0, 0])
    is_drone: bool = False
    start_time: float = 0.0

    # Precomputed
    a: List[float] = field(default_factory=list, repr=False)
    F: List[float] = field(default_factory=list, repr=False)
    prefix_load: List[float] = field(default_factory=list, repr=False)
    suffix_load: List[float] = field(default_factory=list, repr=False)
    total_load: float = 0.0
    total_dist: float = 0.0
    return_time: float = 0.0

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
# Vehicle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Vehicle:
    is_drone: bool = False
    trips: List[Trip] = field(default_factory=list)

    def finish_time(self) -> float:
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
# precompute_trip — NGUỒN SỰ THẬT DUY NHẤT
# ─────────────────────────────────────────────────────────────────────────────

def precompute_trip(trip: Trip, inst: Instance) -> None:
    """
    Tính TOÀN BỘ thông số của một trip từ trip.start_time.
    Đây là hàm DUY NHẤT được phép ghi vào trip.a, trip.F, trip.return_time, v.v.
    Không có hàm nào khác được tự tính lại các trường này.
    """
    seq = trip.sequence
    n = len(seq)
    is_drone = trip.is_drone
    cdata = {c.id: c for c in inst.all_nodes}

    # ── Arrival time ────────────────────────────────────────────────────────
    # a[0] = thời điểm rời depot (= start_time, depot không có service)
    # a[i] = max(a[i-1] + service[i-1] + travel(i-1, i), ready[i])
    a = [0.0] * n
    a[0] = trip.start_time

    for i in range(1, n):
        prev_id = seq[i - 1]
        curr_id = seq[i]
        t_travel = inst.travel_time(prev_id, curr_id, is_drone=is_drone)
        s_prev = cdata[prev_id].service   # depot.service = 0
        depart_prev = a[i - 1] + s_prev
        arrive = depart_prev + t_travel
        a[i] = max(arrive, cdata[curr_id].ready)

    # ── Forward Time Slack F[i] ─────────────────────────────────────────────
    # F[i] = min(due[i] - a[i],  F[i+1] - wait[i+1])
    # wait[i+1] = max(0, ready[i+1] - (a[i] + service[i] + travel(i, i+1)))
    F = [0.0] * n
    F[n - 1] = cdata[seq[-1]].due - a[n - 1]   # depot cuối: due rất lớn

    for i in range(n - 2, -1, -1):
        curr_id = seq[i]
        nxt_id  = seq[i + 1]
        s_i     = cdata[curr_id].service
        t_nxt   = inst.travel_time(curr_id, nxt_id, is_drone=is_drone)
        # thời điểm thực sự đến nxt (trước khi đợi)
        raw_arrive_nxt = a[i] + s_i + t_nxt
        wait_nxt = max(0.0, a[i + 1] - raw_arrive_nxt)
        F[i] = min(cdata[curr_id].due - a[i], F[i + 1] - wait_nxt)

    # ── Prefix / Suffix load ────────────────────────────────────────────────
    # prefix_load[i] = tổng demand từ seq[1]..seq[i]  (seq[0]=depot: 0)
    # suffix_load[i] = tổng demand từ seq[i]..seq[n-2] (seq[n-1]=depot: 0)
    prefix_load = [0.0] * n
    suffix_load = [0.0] * n

    for i in range(1, n - 1):
        prefix_load[i] = prefix_load[i - 1] + cdata[seq[i]].demand

    # FIX: suffix_load[i] phải bao gồm cả demand[seq[i]] chính nó
    for i in range(n - 2, 0, -1):
        suffix_load[i] = cdata[seq[i]].demand + suffix_load[i + 1]

    # ── Gán vào trip ────────────────────────────────────────────────────────
    trip.a           = a
    trip.F           = F
    trip.prefix_load = prefix_load
    trip.suffix_load = suffix_load
    trip.total_load  = prefix_load[n - 2] if n > 2 else 0.0
    trip.return_time = a[-1]
    trip.total_dist  = sum(inst.dist(seq[i], seq[i + 1]) for i in range(n - 1))


def precompute_vehicle(vehicle: Vehicle, inst: Instance) -> None:
    """
    Precompute toàn bộ trips của 1 phương tiện theo thứ tự tuần tự.
    Trip sau bắt đầu ĐÚNG lúc trip trước về depot.
    """
    current_time = 0.0
    for trip in vehicle.trips:
        trip.start_time = current_time
        precompute_trip(trip, inst)
        current_time = trip.return_time


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat
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
    trucks: List[Vehicle] = field(default_factory=list)
    drones: List[Vehicle] = field(default_factory=list)

    @property
    def truck_routes(self) -> List[Trip]:
        return [t for v in self.trucks for t in v.trips]

    @property
    def drone_routes(self) -> List[Trip]:
        return [t for v in self.drones for t in v.trips]

    def copy(self) -> 'Solution':
        return Solution(
            trucks=[v.copy() for v in self.trucks],
            drones=[v.copy() for v in self.drones],
        )

    def makespan(self) -> float:
        times = ([v.finish_time() for v in self.trucks] +
                 [v.finish_time() for v in self.drones])
        return max(times) if times else 0.0

    def penalty_tw(self, inst: Instance) -> float:
        """Phạt vi phạm time window: tổng (a[i] - due[i])+ cho mọi khách hàng."""
        total = 0.0
        cdata = {c.id: c for c in inst.all_nodes}
        for trip in self.truck_routes + self.drone_routes:
            for pos, nid in enumerate(trip.sequence):
                if nid == 0:
                    continue
                if pos >= len(trip.a):
                    # trip chưa precompute — coi là vi phạm nặng
                    total += 1e6
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
        """Phạt nếu tổng thời gian bay của drone vượt quá drone_range."""
        total = 0.0
        for trip in self.drone_routes:
            seq = trip.sequence
            flight_time = sum(
                inst.travel_time(seq[i], seq[i + 1], is_drone=True)
                for i in range(len(seq) - 1)
            )
            total += max(0.0, flight_time - inst.drone_range)
        return total

    def penalty_drone_assign(self, inst: Instance) -> float:
        """Phạt nếu drone phục vụ khách C1 (chỉ-truck)."""
        total = 0.0
        c1 = inst.c1_ids
        for trip in self.drone_routes:
            for nid in trip.customers():
                if nid in c1:
                    total += 1.0
        return total

    def objective(self, inst: Instance,
                  w_tw: float = 50.0, w_cap: float = 200.0,
                  w_range: float = 200.0, w_assign: float = 500.0) -> float:
        return (self.makespan()
                + w_tw     * self.penalty_tw(inst)
                + w_cap    * self.penalty_cap(inst)
                + w_range  * self.penalty_range(inst)
                + w_assign * self.penalty_drone_assign(inst))

    def is_feasible(self, inst: Instance) -> bool:
        return (self.penalty_tw(inst)           == 0.0
                and self.penalty_cap(inst)       == 0.0
                and self.penalty_range(inst)     == 0.0
                and self.penalty_drone_assign(inst) == 0.0)

    def all_served(self, inst: Instance) -> bool:
        served = set()
        for trip in self.truck_routes + self.drone_routes:
            for nid in trip.sequence:
                if nid != 0:
                    if nid in served:
                        return False   # phục vụ 2 lần
                    served.add(nid)
        return served == {c.id for c in inst.customers}

    def recompute_all(self, inst: Instance) -> None:
        """
        Tính lại toàn bộ hệ thống bằng cách gọi precompute_vehicle cho từng xe.
        ĐÂY LÀ HÀM DUY NHẤT được gọi sau khi thay đổi sequence bất kỳ.
        """
        for v in self.trucks + self.drones:
            # Dọn trip rỗng trước khi compute
            v.trips = [t for t in v.trips if len(t.sequence) > 2 or
                       t.sequence == [0, 0]]
            if not v.trips:
                v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
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
            f"Penalty assign: {self.penalty_drone_assign(inst):.2f}",
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
