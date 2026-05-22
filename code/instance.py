import json
import os
import math
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
    drone_range: float        # L_D: tổng tầm bay tối đa
    truck_speed: float = 1.0  # <--- THÊM MỚI
    drone_speed: float = 1.5  # <--- THÊM MỚI
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
        """Sử dụng trực tiếp vận tốc được cấu hình trong Instance"""
        speed = self.drone_speed if is_drone else self.truck_speed
        return self._dist[i][j] / speed

    @property
    def all_nodes(self) -> List[Customer]:
        return [self.depot] + self.customers

    def __repr__(self):
        return (f"Instance({self.name!r}, trucks={self.num_trucks}, drones={self.num_drones}, "
                f"truck_speed={self.truck_speed}, drone_speed={self.drone_speed}, "
                f"|C|={len(self.customers)}, |C1|={len(self.c1_ids)}, |C2|={len(self.c2_ids)})")

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

def read_json_instance(filepath: str) -> Instance:
    """
    Đọc chính xác thông số cấu hình từ file JSON bài toán MVRPD-TW 
    Bỏ qua thời điểm phát sinh request (cột index 4).
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    instance_name = os.path.splitext(os.path.basename(filepath))[0]
    requests = data["requests"]
    if not requests:
        raise ValueError(f"File {filepath} không chứa dữ liệu requests.")
        
    # Đọc metadata cấu hình hệ thống xe
    num_trucks = int(data.get("truck_num", 1))
    num_drones = int(data.get("drone_num", 1))
    truck_capacity = float(data.get("truck_cap", 400.0))
    drone_capacity = float(data.get("drone_cap", 2.27))
    drone_range = float(data.get("drone_lim", 700.0))
    
    # Đọc vận tốc xe
    truck_speed = float(data.get("truck_vel", 1.0))
    drone_speed = float(data.get("drone_vel", 1.5))
    
    # Đọc thời gian đóng cửa kho trung tâm (Depot due time) từ trường "close" ngoài cùng
    depot_close_time = float(data.get("close", 998.708))
    
    # Node đầu tiên (index 0) luôn là Depot
    d_raw = requests[0]
    depot = Customer(
        id=0,
        x=float(d_raw[0]),
        y=float(d_raw[1]),
        demand=float(d_raw[2]),
        ready=0.0,                # Kho mở cửa từ thời điểm 0
        due=depot_close_time,     # Lấy chính xác từ trường "close" ngoài cùng
        service=0.0,              # Tại kho không tốn thời gian phục vụ
        is_c1=False
    )
    
    customers = []
    c1_ids, c2_ids = set(), set()
    
    # Duyệt qua các khách hàng (từ index 1 trở đi)
    for idx, r_raw in enumerate(requests[1:], start=1):
        is_c1_type = int(r_raw[3]) == 1
        
        # Trích xuất chính xác theo thứ tự cột mới
        ready_time = float(r_raw[5])  # Cột index 5 là Ready Time
        due_time = float(r_raw[6])    # Cột index 6 là Due Time
        
        # Tính toán Service Time = Due - Ready
        service_time = max(0.0, due_time - ready_time)
        
        cust = Customer(
            id=idx,
            x=float(r_raw[0]),
            y=float(r_raw[1]),
            demand=float(r_raw[2]),
            ready=ready_time,
            due=due_time,
            service=service_time,     # Gán service time vừa tính được
            is_c1=is_c1_type
        )
        
        customers.append(cust)
        if is_c1_type:
            c1_ids.add(idx)
        else:
            c2_ids.add(idx)
            
    inst = Instance(
        name=instance_name,
        num_trucks=num_trucks,
        num_drones=num_drones,
        truck_capacity=truck_capacity,
        drone_capacity=drone_capacity,
        drone_range=drone_range,
        truck_speed=truck_speed,
        drone_speed=drone_speed,
        depot=depot,
        customers=customers,
        c1_ids=c1_ids,
        c2_ids=c2_ids
    )
    
    inst.build_dist()
    return inst