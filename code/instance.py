"""
instance.py — Đọc và lưu trữ dữ liệu bài toán MVRPD-TW

Hỗ trợ định dạng file Solomon-style (VRPTW benchmark):
  - Cột: ID  x  y  demand  ready_time  due_time  service_time
  - Dòng đầu: tên instance
  - Dòng thông số: num_vehicles  capacity
  - Các dòng còn lại: dữ liệu khách hàng (dòng 0 = depot)

Ví dụ file:
  C101
  VEHICLE   NUMBER  CAPACITY
  TRUCK      25      200
  DRONE       5       10
  ...
  CUSTOMER   CUST   XCOORD  YCOORD  DEMAND  READY   DUE   SERVICE
  ...
        0       0    40.0    50.0      0      0      1236      0
        1       1    45.0    68.0     10     912     967      90
  ...
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Customer:
    id: int
    x: float
    y: float
    demand: float
    ready: float      # a_i  (earliest service time)
    due: float        # b_i  (latest arrival time)
    service: float    # s_i  (service time)
    is_c1: bool = False   # True => chỉ truck phục vụ được


@dataclass
class Instance:
    name: str
    num_trucks: int
    num_drones: int
    truck_capacity: float
    drone_capacity: float
    drone_range: float        # L_D: tổng tầm bay tối đa (km)
    depot: Customer
    customers: List[Customer]   # không kể depot
    c1_ids: Set[int] = field(default_factory=set)   # ID khách hàng C1
    c2_ids: Set[int] = field(default_factory=set)   # ID khách hàng C2

    # Ma trận khoảng cách (tính sẵn)
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
        """Khoảng cách giữa node i và node j (0 = depot)."""
        return self._dist[i][j]

    def travel_time(self, i: int, j: int,
                    truck_speed: float = 1.0,
                    drone_speed: float = 1.5,
                    is_drone: bool = False) -> float:
        """Thời gian di chuyển (mặc định speed=1 → thời gian = khoảng cách)."""
        speed = drone_speed if is_drone else truck_speed
        return self._dist[i][j] / speed

    @property
    def all_nodes(self) -> List[Customer]:
        return [self.depot] + self.customers

    def __repr__(self):
        return (f"Instance({self.name!r}, "
                f"trucks={self.num_trucks}, drones={self.num_drones}, "
                f"|C|={len(self.customers)}, |C1|={len(self.c1_ids)}, |C2|={len(self.c2_ids)})")


# ─────────────────────────────────────────────────────────────────────────────
# Hàm đọc file
# ─────────────────────────────────────────────────────────────────────────────

def _nums(line: str) -> List[float]:
    """Trích xuất tất cả số trên một dòng."""
    return [float(x) for x in re.findall(r'[-+]?\d*\.?\d+', line)]


def read_solomon(
    filepath: str,
    num_trucks: int = 2,
    num_drones: int = 2,
    truck_capacity: float | None = None,
    drone_capacity: float = 30.0,
    drone_range: float = 100.0,
    drone_weight_threshold: float | None = None,
    truck_speed: float = 1.0,
    drone_speed: float = 1.5,
) -> Instance:
    """
    Đọc file định dạng Solomon VRPTW chuẩn.

    Tham số:
        filepath            : đường dẫn file .txt
        num_trucks          : số xe tải K
        num_drones          : số drone D
        truck_capacity      : tải trọng tối đa của truck (None → đọc từ file)
        drone_capacity      : tải trọng tối đa của drone M_D
        drone_range         : tổng tầm bay tối đa của drone L_D (km)
        drone_weight_threshold: ngưỡng demand để tự động phân C1/C2
                               (None → dùng drone_capacity)
        truck_speed         : tốc độ xe tải (đơn vị/phút)
        drone_speed         : tốc độ drone
    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    # Lọc dòng trống
    lines = [l for l in lines if l]

    name = lines[0]

    # Tìm dòng chứa thông số xe (có số, không phải header)
    cap_line = None
    data_start = 0
    for i, line in enumerate(lines[1:], 1):
        nums = _nums(line)
        if len(nums) >= 2 and not any(c.isalpha() for c in line.split()[0]):
            # Dòng số đầu tiên sau tên → thông số vehicle
            cap_line = nums
            data_start = i + 1
            break
        # Có thể là header dạng "VEHICLE NUMBER CAPACITY"
        if re.search(r'\d', line) and len(nums) >= 2:
            cap_line = nums
            data_start = i + 1
            break

    if cap_line is None:
        raise ValueError("Không tìm thấy dòng thông số vehicle trong file.")

    file_num_vehicles = int(cap_line[0])
    file_capacity = cap_line[1]

    if truck_capacity is None:
        truck_capacity = file_capacity

    # Đọc dữ liệu khách hàng
    customers_raw = []
    for line in lines[data_start:]:
        nums = _nums(line)
        if len(nums) >= 7:
            customers_raw.append(nums)

    if not customers_raw:
        raise ValueError("Không đọc được dữ liệu khách hàng.")

    # Dòng đầu = depot (id=0)
    d = customers_raw[0]
    depot = Customer(
        id=0, x=d[1], y=d[2], demand=d[3],
        ready=d[4], due=d[5], service=d[6]
    )

    threshold = drone_weight_threshold if drone_weight_threshold is not None \
                else drone_capacity

    customers = []
    c1_ids, c2_ids = set(), set()

    for row in customers_raw[1:]:
        cid = int(row[0])
        cust = Customer(
            id=cid, x=row[1], y=row[2], demand=row[3],
            ready=row[4], due=row[5], service=row[6]
        )
        # Phân loại C1 / C2: nếu demand > ngưỡng → C1 (chỉ truck)
        if cust.demand > threshold:
            cust.is_c1 = True
            c1_ids.add(cid)
        else:
            c2_ids.add(cid)
        customers.append(cust)

    inst = Instance(
        name=name,
        num_trucks=num_trucks,
        num_drones=num_drones,
        truck_capacity=truck_capacity,
        drone_capacity=drone_capacity,
        drone_range=drone_range,
        depot=depot,
        customers=customers,
        c1_ids=c1_ids,
        c2_ids=c2_ids,
    )
    inst.build_dist()
    return inst


def read_custom(filepath: str) -> Instance:
    """
    Đọc file định dạng tùy chỉnh:

    Dòng 1 : tên instance
    Dòng 2 : num_trucks  num_drones  truck_cap  drone_cap  drone_range
    Dòng 3+: id  x  y  demand  ready  due  service  [type]
              type = 0 → C2 (linh hoạt), 1 → C1 (chỉ truck)
              (dòng id=0 là depot, type bỏ qua)

    Ví dụ:
        MY_INSTANCE
        2 2 200 30 100
        0  40  50   0    0  1236  0
        1  45  68  10  912   967 90   0
        2  55  60  25  825   870 90   1
    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    name = lines[0]
    params = _nums(lines[1])
    num_trucks  = int(params[0])
    num_drones  = int(params[1])
    truck_cap   = params[2]
    drone_cap   = params[3]
    drone_range = params[4]

    depot = None
    customers = []
    c1_ids, c2_ids = set(), set()

    for line in lines[2:]:
        nums = _nums(line)
        if len(nums) < 7:
            continue
        cid = int(nums[0])
        ctype = int(nums[7]) if len(nums) >= 8 else 0
        c = Customer(
            id=cid, x=nums[1], y=nums[2], demand=nums[3],
            ready=nums[4], due=nums[5], service=nums[6],
            is_c1=(ctype == 1)
        )
        if cid == 0:
            depot = c
        else:
            customers.append(c)
            if c.is_c1:
                c1_ids.add(cid)
            else:
                c2_ids.add(cid)

    if depot is None:
        raise ValueError("Không tìm thấy depot (id=0) trong file.")

    inst = Instance(
        name=name,
        num_trucks=num_trucks,
        num_drones=num_drones,
        truck_capacity=truck_cap,
        drone_capacity=drone_cap,
        drone_range=drone_range,
        depot=depot,
        customers=customers,
        c1_ids=c1_ids,
        c2_ids=c2_ids,
    )
    inst.build_dist()
    return inst
