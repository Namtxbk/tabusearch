import json
import os
import math
from dataclasses import dataclass, field
from typing import List, Set

L_W = 0.0    # Service time tại điểm khách = 0 (bốc/dỡ tức thì)
L_W_MAX = 60.0  # Thời gian chờ tối đa của xe tại điểm khách (phút)
               # Ràng buộc: wait_i = max(0, e_i - arrive_i) <= L_W_MAX

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready: float      # opentime
    due: float        # closetime
    service: float    # L_w = 60 phút
    is_c1: bool = False  # True => chỉ truck, False => drone được phép

@dataclass
class Instance:
    name: str
    num_trucks: int
    num_drones: int
    truck_capacity: float
    drone_capacity: float
    drone_range: float
    max_wait: float = L_W_MAX  # Thời gian chờ tối đa tại điểm khách (L_w)
    truck_speed: float = 1.0
    drone_speed: float = 1.5
    depot: Customer = None
    customers: List[Customer] = field(default_factory=list)
    c1_ids: Set[int] = field(default_factory=set)
    c2_ids: Set[int] = field(default_factory=set)
    _dist: List[List[float]] = field(default_factory=list, repr=False)

    def build_dist(self):
        nodes = [self.depot] + self.customers
        n = len(nodes)
        self._dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                dx = nodes[i].x - nodes[j].x
                dy = nodes[i].y - nodes[j].y
                self._dist[i][j] = math.hypot(dx, dy)

    def dist(self, i: int, j: int) -> float:
        return self._dist[i][j]

    def travel_time(self, i: int, j: int, is_drone: bool = False) -> float:
        speed = self.drone_speed if is_drone else self.truck_speed
        return self._dist[i][j] / speed

    @property
    def all_nodes(self) -> List[Customer]:
        return [self.depot] + self.customers

    def __repr__(self):
        return (f"Instance({self.name!r}, trucks={self.num_trucks}, drones={self.num_drones}, "
                f"truck_speed={self.truck_speed}, drone_speed={self.drone_speed}, "
                f"|C|={len(self.customers)}, |C1|={len(self.c1_ids)}, |C2|={len(self.c2_ids)})")


def read_json_instance(filepath: str) -> Instance:
    """
    Đọc file JSON bài toán MVRPD-TW.

    Cấu trúc mỗi request (khách hàng):
        [x, y, demand, ableServiceByDrone, r_i (bỏ qua), opentime, closetime]

    Depot: tọa độ (0.0, 0.0) cố định, KHÔNG có trong requests.
    Khách hàng: tất cả requests[], đánh id từ 1 đến n.

    ableServiceByDrone:
        1 → khách hàng CHO PHÉP drone (is_c1=False)
        0 → khách hàng KHÔNG cho phép drone, chỉ truck (is_c1=True)

    service_time = L_w = 60 phút (hằng số).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instance_name = os.path.splitext(os.path.basename(filepath))[0]
    requests = data["requests"]
    if not requests:
        raise ValueError(f"File {filepath} không chứa dữ liệu requests.")

    # ── Thông số hệ thống ─────────────────────────────────────────────────
    num_trucks     = int(data.get("truck_num", 1))
    num_drones     = int(data.get("drone_num", 1))
    truck_capacity = float(data.get("truck_cap", 400.0))
    drone_capacity = float(data.get("drone_cap", 2.27))
    drone_range    = float(data.get("drone_lim", 700.0))
    truck_speed    = float(data.get("truck_vel", 1.0))
    drone_speed    = float(data.get("drone_vel", 1.5))
    depot_close    = float(data.get("close", 9999.0))

    # ── Depot: tọa độ (0, 0) cố định, không lấy từ requests ─────────────
    depot = Customer(
        id      = 0,
        x       = 0.0,
        y       = 0.0,
        demand  = 0.0,
        ready   = 0.0,
        due     = depot_close,
        service = 0.0,
        is_c1   = False
    )

    # ── Khách hàng: tất cả requests[], id từ 1 đến n ─────────────────────
    customers: List[Customer] = []
    c1_ids: Set[int] = set()
    c2_ids: Set[int] = set()

    for idx, r_raw in enumerate(requests, start=1):
        # index 3: ableServiceByDrone
        #   1 → cho phép drone  → is_c1 = False
        #   0 → chỉ truck       → is_c1 = True
        able_drone = int(r_raw[3])
        is_c1_flag = (able_drone == 0)

        # index 4: r_i (thời điểm phát sinh request) → BỎ QUA
        # index 5: opentime  → ready
        # index 6: closetime → due
        cust = Customer(
            id      = idx,
            x       = float(r_raw[0]),
            y       = float(r_raw[1]),
            demand  = float(r_raw[2]),
            ready   = float(r_raw[5]),
            due     = float(r_raw[6]),
            service = L_W,
            is_c1   = is_c1_flag
        )

        customers.append(cust)
        if is_c1_flag:
            c1_ids.add(idx)
        else:
            c2_ids.add(idx)

    inst = Instance(
        name           = instance_name,
        max_wait       = L_W_MAX,
        num_trucks     = num_trucks,
        num_drones     = num_drones,
        truck_capacity = truck_capacity,
        drone_capacity = drone_capacity,
        drone_range    = drone_range,
        truck_speed    = truck_speed,
        drone_speed    = drone_speed,
        depot          = depot,
        customers      = customers,
        c1_ids         = c1_ids,
        c2_ids         = c2_ids,
    )

    inst.build_dist()
    return inst
