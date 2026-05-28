import json
import os
import math
from dataclasses import dataclass, field
from typing import List, Set

L_W = 60.0  # Service time cố định cho tất cả các instance (phút)

@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready: float      # opentime  (earliest service time)
    due: float        # closetime (latest arrival time)
    service: float    # L_w = 60 phút (hằng số)
    is_c1: bool = False   # True => ableServiceByDrone=1 => chỉ truck phục vụ được

@dataclass
class Instance:
    name: str
    num_trucks: int
    num_drones: int
    truck_capacity: float
    drone_capacity: float
    drone_range: float        # L_D: tổng tầm bay tối đa
    truck_speed: float = 1.0
    drone_speed: float = 1.5
    depot: Customer = None
    customers: List[Customer] = field(default_factory=list)
    c1_ids: Set[int] = field(default_factory=set)
    c2_ids: Set[int] = field(default_factory=set)

    # Ma trận khoảng cách
    _dist: List[List[float]] = field(default_factory=list, repr=False)

    def build_dist(self):
        """Tính ma trận khoảng cách Euclidean giữa tất cả các node."""
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

    Cấu trúc mỗi request:
        [x, y, demand, ableServiceByDrone, r_i (bỏ qua), opentime, closetime]

    Quy tắc phân loại:
        ableServiceByDrone = 1  →  is_c1 = True  (chỉ truck phục vụ được)
        ableServiceByDrone = 0  →  is_c1 = False (truck hoặc drone đều được)

    service_time = L_w = 60 phút (hằng số cho tất cả khách hàng).

    Node đầu tiên trong requests là DEPOT:
        - ready = 0.0
        - due   = data["close"]
        - service = 0.0
        - Không tính vào danh sách customers
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    instance_name = os.path.splitext(os.path.basename(filepath))[0]
    requests = data["requests"]
    if not requests:
        raise ValueError(f"File {filepath} không chứa dữ liệu requests.")

    # ── Đọc thông số hệ thống từ JSON ────────────────────────────────────
    num_trucks     = int(data.get("truck_num", 1))
    num_drones     = int(data.get("drone_num", 1))
    truck_capacity = float(data.get("truck_cap", 400.0))
    drone_capacity = float(data.get("drone_cap", 2.27))
    drone_range    = float(data.get("drone_lim", 700.0))
    truck_speed    = float(data.get("truck_vel", 1.0))
    drone_speed    = float(data.get("drone_vel", 1.5))
    depot_close    = float(data.get("close", 9999.0))

    # ── Node 0: Depot ─────────────────────────────────────────────────────
    # requests[0] = [x, y, demand, ableServiceByDrone, r_i, opentime, closetime]
    # Depot dùng "close" toàn cục làm due time, service = 0
    d_raw = requests[0]
    depot = Customer(
        id      = 0,
        x       = float(d_raw[0]),
        y       = float(d_raw[1]),
        demand  = 0.0,          # Depot không có demand
        ready   = 0.0,          # Kho mở cửa từ t=0
        due     = depot_close,  # Thời điểm đóng cửa kho lấy từ trường "close"
        service = 0.0,          # Không tốn thời gian phục vụ tại kho
        is_c1   = False
    )

    # ── Các khách hàng (requests[1:]) ────────────────────────────────────
    customers: List[Customer] = []
    c1_ids: Set[int] = set()
    c2_ids: Set[int] = set()

    for idx, r_raw in enumerate(requests[1:], start=1):
        # Cột 3: ableServiceByDrone
        #   1 → khách hàng CHO PHÉP drone giao (drone nhận hay không tùy capacity/range)
        #   0 → khách hàng KHÔNG cho phép drone → chỉ truck (is_c1 = True)
        able_drone = int(r_raw[3])
        is_c1_flag = (able_drone == 0)

        # Cột 4: r_i — thời điểm phát sinh request → BỎ QUA
        # Cột 5: opentime  → ready
        # Cột 6: closetime → due
        open_time  = float(r_raw[5])
        close_time = float(r_raw[6])

        cust = Customer(
            id      = idx,
            x       = float(r_raw[0]),
            y       = float(r_raw[1]),
            demand  = float(r_raw[2]),
            ready   = open_time,
            due     = close_time,
            service = L_W,          # L_w = 60 phút, hằng số
            is_c1   = is_c1_flag
        )

        customers.append(cust)
        if is_c1_flag:
            c1_ids.add(idx)
        else:
            c2_ids.add(idx)

    inst = Instance(
        name           = instance_name,
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
