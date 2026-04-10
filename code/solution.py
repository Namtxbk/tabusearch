"""
solution.py — Biểu diễn lời giải và Forward Time Slack cho MVRPD-TW

Cấu trúc nghiệm:
    Solution
    ├── truck_routes : List[Route]   (K routes, mỗi route là list node id)
    └── drone_routes : List[Route]   (D routes, mỗi route là list node id)

Mỗi Route lưu:
    - sequence  : [0, c1, c2, ..., 0]
    - a[]       : arrival time tại mỗi vị trí
    - F[]       : forward time slack tại mỗi vị trí
    - prefix_load[], suffix_load[]
"""

from __future__ import annotations
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from instance import Instance


# ─────────────────────────────────────────────────────────────────────────────
# Route — một hành trình của một phương tiện
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Route:
    """
    sequence: list các node id, bao gồm depot (0) ở đầu và cuối.
    Ví dụ: [0, 3, 7, 2, 5, 0]
    """
    sequence: List[int] = field(default_factory=lambda: [0, 0])
    is_drone: bool = False

    # Precomputed (tính sau mỗi lần thay đổi)
    a: List[float] = field(default_factory=list, repr=False)
    F: List[float] = field(default_factory=list, repr=False)
    prefix_load: List[float] = field(default_factory=list, repr=False)
    suffix_load: List[float] = field(default_factory=list, repr=False)
    total_time: float = 0.0
    total_load: float = 0.0
    total_dist: float = 0.0   # dùng cho drone range check

    def customers(self) -> List[int]:
        """Trả về danh sách khách hàng (không kể depot)."""
        return self.sequence[1:-1]

    def __len__(self):
        return len(self.sequence) - 2   # số khách hàng

    def copy(self) -> 'Route':
        return Route(
            sequence=self.sequence[:],
            is_drone=self.is_drone,
            a=self.a[:],
            F=self.F[:],
            prefix_load=self.prefix_load[:],
            suffix_load=self.suffix_load[:],
            total_time=self.total_time,
            total_load=self.total_load,
            total_dist=self.total_dist,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Forward Time Slack — Precompute
# ─────────────────────────────────────────────────────────────────────────────

def precompute(route: Route, inst: Instance) -> None:
    """
    Tính toàn bộ thông tin precompute cho một route — O(n).

    Sau khi gọi hàm này, route.a[], route.F[],
    route.prefix_load[], route.suffix_load[] đều được cập nhật.
    """
    seq = route.sequence
    n = len(seq)
    is_drone = route.is_drone

    customers_data = {c.id: c for c in inst.all_nodes}

    # ── Bước 1: Arrival time (duyệt forward) ─────────────────────────────
    a = [0.0] * n
    a[0] = inst.depot.ready   # thường = 0

    for i in range(1, n):
        prev_node = seq[i - 1]
        curr_node = seq[i]
        t_travel = inst.travel_time(prev_node, curr_node, is_drone=is_drone)
        s_prev   = customers_data[prev_node].service
        arrive   = a[i - 1] + s_prev + t_travel
        e_curr   = customers_data[curr_node].ready
        a[i]     = max(arrive, e_curr)   # chờ nếu đến sớm

    # ── Bước 2: Forward Time Slack (duyệt backward) ───────────────────────
    F = [0.0] * n
    # Tại depot cuối (vị trí n-1): slack = thời gian còn lại
    l_last = customers_data[seq[-1]].due
    F[n - 1] = l_last - a[n - 1]

    for i in range(n - 2, -1, -1):
        curr_node = seq[i]
        next_node = seq[i + 1]
        l_i      = customers_data[curr_node].due
        s_i      = customers_data[curr_node].service
        t_next   = inst.travel_time(curr_node, next_node, is_drone=is_drone)
        # waiting time tại node i+1 hấp thụ bớt delay
        wait_next = max(0.0, customers_data[next_node].ready - (a[i] + s_i + t_next))
        slack_i   = l_i - a[i]
        F[i]      = min(slack_i, F[i + 1] - wait_next)

    # ── Bước 3: Prefix / Suffix load ─────────────────────────────────────
    m = n - 2  # số khách hàng (bỏ 2 depot)
    prefix_load = [0.0] * n
    suffix_load = [0.0] * n

    for i in range(1, n - 1):
        prefix_load[i] = prefix_load[i - 1] + customers_data[seq[i]].demand

    for i in range(n - 2, 0, -1):
        suffix_load[i] = suffix_load[i + 1] + customers_data[seq[i]].demand

    # ── Cập nhật vào route ────────────────────────────────────────────────
    route.a            = a
    route.F            = F
    route.prefix_load  = prefix_load
    route.suffix_load  = suffix_load
    route.total_time   = a[-1]
    route.total_load   = prefix_load[n - 2] if n > 2 else 0.0

    # Tổng quãng đường (cho drone range check)
    dist = 0.0
    for i in range(n - 1):
        dist += inst.dist(seq[i], seq[i + 1])
    route.total_dist = dist


# ─────────────────────────────────────────────────────────────────────────────
# Feasibility check O(1) nhờ Forward Time Slack
# ─────────────────────────────────────────────────────────────────────────────

def check_insert(route: Route, node_id: int, pos: int,
                 inst: Instance) -> Tuple[bool, float]:
    """
    Kiểm tra chèn node_id vào giữa seq[pos] và seq[pos+1] — O(1).

    Trả về (feasible, delay) trong đó delay là độ trễ truyền xuống suffix.
    """
    seq      = route.sequence
    is_drone = route.is_drone
    cdata    = {c.id: c for c in inst.all_nodes}

    i_node   = seq[pos]
    j_node   = seq[pos + 1]
    node     = cdata[node_id]

    # Arrival tại node mới
    t_to_node  = inst.travel_time(i_node, node_id, is_drone=is_drone)
    s_i        = cdata[i_node].service
    a_new_node = max(route.a[pos] + s_i + t_to_node, node.ready)

    # Vi phạm time window của chính node được chèn?
    if a_new_node > node.due:
        return False, 0.0

    # Arrival tại j_node sau khi qua node mới
    t_to_j      = inst.travel_time(node_id, j_node, is_drone=is_drone)
    s_node      = node.service
    a_new_j     = max(a_new_node + s_node + t_to_j, cdata[j_node].ready)
    delay       = a_new_j - route.a[pos + 1]

    # Kiểm tra suffix bằng Forward Time Slack
    if delay <= route.F[pos + 1]:
        return True, delay
    return False, delay


def check_remove(route: Route, pos: int, inst: Instance) -> Tuple[bool, float]:
    """
    Kiểm tra xóa node tại vị trí pos khỏi route — O(1).
    Thường luôn giảm delay → feasible, nhưng vẫn kiểm tra đúng.
    """
    seq      = route.sequence
    is_drone = route.is_drone
    cdata    = {c.id: c for c in inst.all_nodes}

    prev_node = seq[pos - 1]
    next_node = seq[pos + 1]
    s_prev    = cdata[prev_node].service

    t_skip    = inst.travel_time(prev_node, next_node, is_drone=is_drone)
    a_new_next = max(route.a[pos - 1] + s_prev + t_skip,
                     cdata[next_node].ready)
    delay     = a_new_next - route.a[pos + 1]

    if delay <= route.F[pos + 1]:
        return True, delay
    return False, delay


def check_2opt_star(r1: Route, i: int, r2: Route, j: int,
                    inst: Instance, capacity: float) -> Tuple[bool, float]:
    """
    Kiểm tra 2-opt* swap tail: r1[0..i] + r2[j+1..] và r2[0..j] + r1[i+1..] — O(1).

    Trả về (feasible, delta_cost).
    """
    seq1, seq2 = r1.sequence, r2.sequence
    is_drone   = r1.is_drone
    cdata      = {c.id: c for c in inst.all_nodes}

    # Delta cost
    d_remove = inst.dist(seq1[i], seq1[i + 1]) + inst.dist(seq2[j], seq2[j + 1])
    d_add    = inst.dist(seq1[i], seq2[j + 1]) + inst.dist(seq2[j], seq1[i + 1])
    delta_c  = d_add - d_remove

    # Capacity check
    new_load_r1 = r1.prefix_load[i] + r2.suffix_load[j + 1]
    new_load_r2 = r2.prefix_load[j] + r1.suffix_load[i + 1]
    if new_load_r1 > capacity or new_load_r2 > capacity:
        return False, delta_c

    # Time window check — Route 1 nhận suffix từ r2[j+1]
    s_i1     = cdata[seq1[i]].service
    t1       = inst.travel_time(seq1[i], seq2[j + 1], is_drone=is_drone)
    a_new_1  = max(r1.a[i] + s_i1 + t1, cdata[seq2[j + 1]].ready)
    delay_1  = a_new_1 - r2.a[j + 1]
    if delay_1 > r2.F[j + 1]:
        return False, delta_c

    # Time window check — Route 2 nhận suffix từ r1[i+1]
    s_j2     = cdata[seq2[j]].service
    t2       = inst.travel_time(seq2[j], seq1[i + 1], is_drone=is_drone)
    a_new_2  = max(r2.a[j] + s_j2 + t2, cdata[seq1[i + 1]].ready)
    delay_2  = a_new_2 - r1.a[i + 1]
    if delay_2 > r1.F[i + 1]:
        return False, delta_c

    return True, delta_c


# ─────────────────────────────────────────────────────────────────────────────
# Solution — toàn bộ nghiệm
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Solution:
    truck_routes: List[Route] = field(default_factory=list)
    drone_routes: List[Route] = field(default_factory=list)

    # Penalty coefficients
    lambda_tw:  float = 50.0
    lambda_cap: float = 200.0

    def copy(self) -> 'Solution':
        s = Solution(
            truck_routes=[r.copy() for r in self.truck_routes],
            drone_routes=[r.copy() for r in self.drone_routes],
            lambda_tw=self.lambda_tw,
            lambda_cap=self.lambda_cap,
        )
        return s

    def makespan(self) -> float:
        """Thời điểm phương tiện cuối cùng về depot."""
        times = ([r.total_time for r in self.truck_routes] +
                 [r.total_time for r in self.drone_routes])
        return max(times) if times else 0.0

    def penalty_tw(self, inst: Instance) -> float:
        """Tổng vi phạm time window."""
        total = 0.0
        cdata = {c.id: c for c in inst.all_nodes}
        for route in self.truck_routes + self.drone_routes:
            for pos, nid in enumerate(route.sequence):
                if nid == 0:
                    continue
                viol = max(0.0, route.a[pos] - cdata[nid].due)
                total += viol
        return total

    def penalty_cap(self, inst: Instance) -> float:
        """Tổng vi phạm tải trọng."""
        total = 0.0
        for r in self.truck_routes:
            over = max(0.0, r.total_load - inst.truck_capacity)
            total += over
        for r in self.drone_routes:
            over = max(0.0, r.total_load - inst.drone_capacity)
            total += over
        return total

    def penalty_range(self, inst: Instance) -> float:
        """Tổng vi phạm tầm bay drone."""
        total = 0.0
        for r in self.drone_routes:
            over = max(0.0, r.total_dist - inst.drone_range)
            total += over
        return total

    def objective(self, inst: Instance) -> float:
        """Hàm mục tiêu có penalty."""
        return (self.makespan()
                + self.lambda_tw  * self.penalty_tw(inst)
                + self.lambda_cap * self.penalty_cap(inst)
                + self.lambda_cap * self.penalty_range(inst))

    def is_feasible(self, inst: Instance) -> bool:
        return (self.penalty_tw(inst) == 0.0
                and self.penalty_cap(inst) == 0.0
                and self.penalty_range(inst) == 0.0)

    def all_served(self, inst: Instance) -> bool:
        """Kiểm tra tất cả khách hàng đã được phục vụ đúng 1 lần."""
        served = set()
        for r in self.truck_routes + self.drone_routes:
            for nid in r.sequence:
                if nid != 0:
                    if nid in served:
                        return False   # phục vụ 2 lần
                    served.add(nid)
        all_ids = {c.id for c in inst.customers}
        return served == all_ids

    def recompute_all(self, inst: Instance) -> None:
        """Precompute lại toàn bộ nghiệm."""
        for r in self.truck_routes + self.drone_routes:
            precompute(r, inst)

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
        for k, r in enumerate(self.truck_routes):
            cust = r.customers()
            lines.append(f"  Truck {k+1}: {' -> '.join(str(n) for n in [0]+cust+[0])}"
                         f"  (load={r.total_load:.0f}/{inst.truck_capacity:.0f},"
                         f" time={r.total_time:.1f})")
        for d, r in enumerate(self.drone_routes):
            cust = r.customers()
            lines.append(f"  Drone {d+1}: {' -> '.join(str(n) for n in [0]+cust+[0])}"
                         f"  (load={r.total_load:.0f}/{inst.drone_capacity:.0f},"
                         f" dist={r.total_dist:.1f}/{inst.drone_range:.1f},"
                         f" time={r.total_time:.1f})")
        return "\n".join(lines)
