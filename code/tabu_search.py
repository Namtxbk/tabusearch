"""
tabu_search.py — Thuật toán Tabu Search cho MVRPD-TW

Move operators:
  1. Relocate    — di chuyển 1 khách giữa 2 route cùng loại
  2. TwoOpt      — đảo đoạn nội tuyến trong 1 route
  3. OrOpt       — dịch chuỗi 1-3 khách sang vị trí khác
  4. TwoOptStar  — swap tail 2 truck routes
  5. Transfer    — chuyển C2 giữa truck route và drone route

Tabu List lưu (move_type, attribute) với tenure thích nghi.
Aspiration: chấp nhận tabu move nếu F(S') < F(S_best).
Diversification: sau MAX_NO_IMPROVE iter, xáo trộn và tiếp tục.
"""

from __future__ import annotations
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

from instance import Instance
from solution import (Route, Solution, precompute,
                      check_insert, check_remove, check_2opt_star)
from construction import greedy_construction, is_drone_eligible as _elig


# ─────────────────────────────────────────────────────────────────────────────
# Tabu List
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TabuMove:
    move_type: str          # 'relocate', '2opt', 'oropt', '2optstar', 'transfer'
    attribute: tuple        # (customer_id, from_route_idx, to_route_idx) hoặc tương tự
    expire_iter: int


class TabuList:
    def __init__(self, tenure_base: int = 10, tenure_delta: int = 3):
        self.moves: List[TabuMove] = []
        self.tenure_base  = tenure_base
        self.tenure_delta = tenure_delta

    def add(self, move_type: str, attribute: tuple, current_iter: int):
        tenure = (self.tenure_base
                  + random.randint(-self.tenure_delta, self.tenure_delta))
        self.moves.append(TabuMove(move_type, attribute,
                                   current_iter + tenure))

    def is_tabu(self, move_type: str, attribute: tuple,
                current_iter: int) -> bool:
        for m in self.moves:
            if m.expire_iter > current_iter:
                if m.move_type == move_type and m.attribute == attribute:
                    return True
        return False

    def cleanup(self, current_iter: int):
        self.moves = [m for m in self.moves if m.expire_iter > current_iter]

    def increase_tenure(self, factor: float = 1.2, max_tenure: int = 30):
        self.tenure_base = min(int(self.tenure_base * factor), max_tenure)


# ─────────────────────────────────────────────────────────────────────────────
# Move generators
# ─────────────────────────────────────────────────────────────────────────────

MoveResult = Tuple[float, str, tuple, 'Solution']
# (delta_obj, move_type, attribute, new_solution)


def _apply_relocate(sol: Solution, inst: Instance,
                    src_type: str, src_idx: int, src_pos: int,
                    dst_type: str, dst_idx: int, dst_pos: int) -> Optional[MoveResult]:
    """Di chuyển khách tại src_pos sang dst_pos."""
    src_routes = sol.truck_routes if src_type == 'T' else sol.drone_routes
    dst_routes = sol.truck_routes if dst_type == 'T' else sol.drone_routes

    src_r = src_routes[src_idx]
    dst_r = dst_routes[dst_idx]

    if len(src_r.sequence) < 3:   # route rỗng
        return None
    if src_pos < 1 or src_pos >= len(src_r.sequence) - 1:
        return None

    node_id = src_r.sequence[src_pos]

    # Không chuyển C1 sang drone
    cdata = {c.id: c for c in inst.customers}
    if dst_type == 'D' and node_id in inst.c1_ids:
        return None
    if dst_type == 'D' and not _elig(cdata[node_id], inst):
        return None

    # Kiểm tra xóa khỏi src
    feasible_remove, _ = check_remove(src_r, src_pos, inst)
    if not feasible_remove:
        return None

    # Kiểm tra chèn vào dst tại dst_pos
    if dst_pos < 0 or dst_pos >= len(dst_r.sequence) - 1:
        return None
    feasible_insert, _ = check_insert(dst_r, node_id, dst_pos, inst)
    if not feasible_insert:
        return None

    # Tạo nghiệm mới
    new_sol = sol.copy()
    new_src = (new_sol.truck_routes if src_type == 'T'
               else new_sol.drone_routes)[src_idx]
    new_dst = (new_sol.truck_routes if dst_type == 'T'
               else new_sol.drone_routes)[dst_idx]

    new_src.sequence.pop(src_pos)
    new_dst.sequence.insert(dst_pos + 1, node_id)

    precompute(new_src, inst)
    precompute(new_dst, inst)

    delta = new_sol.objective(inst) - sol.objective(inst)
    attr  = (node_id, f'{src_type}{src_idx}', f'{dst_type}{dst_idx}')
    return delta, 'relocate', attr, new_sol


def _apply_2opt(sol: Solution, inst: Instance,
                vtype: str, ridx: int, i: int, j: int) -> Optional[MoveResult]:
    """Đảo đoạn [i+1..j] trong route."""
    routes = sol.truck_routes if vtype == 'T' else sol.drone_routes
    r = routes[ridx]
    seq = r.sequence

    if i < 0 or j >= len(seq) - 1 or i >= j - 1:
        return None

    new_sol = sol.copy()
    new_r   = (new_sol.truck_routes if vtype == 'T'
               else new_sol.drone_routes)[ridx]

    # Đảo đoạn [i+1..j]
    new_r.sequence[i+1:j+1] = new_r.sequence[i+1:j+1][::-1]
    precompute(new_r, inst)

    # Kiểm tra TW toàn route sau đảo
    cdata = {c.id: c for c in inst.all_nodes}
    for pos, nid in enumerate(new_r.sequence):
        if new_r.a[pos] > cdata[nid].due + 1e-6:
            return None

    delta = new_sol.objective(inst) - sol.objective(inst)
    attr  = (f'{vtype}{ridx}', seq[i+1], seq[j])
    return delta, '2opt', attr, new_sol


def _apply_oropt(sol: Solution, inst: Instance,
                 vtype: str, ridx: int,
                 seg_start: int, seg_len: int,
                 insert_pos: int) -> Optional[MoveResult]:
    """Dịch chuỗi seg_len khách từ seg_start sang insert_pos."""
    routes = sol.truck_routes if vtype == 'T' else sol.drone_routes
    r = routes[ridx]
    seq = r.sequence

    seg_end = seg_start + seg_len - 1
    if seg_start < 1 or seg_end >= len(seq) - 1:
        return None
    if insert_pos < 0 or insert_pos >= len(seq) - 1:
        return None
    if seg_start <= insert_pos <= seg_end:
        return None   # chèn vào chính đoạn mình

    new_sol = sol.copy()
    new_r   = (new_sol.truck_routes if vtype == 'T'
               else new_sol.drone_routes)[ridx]

    segment = new_r.sequence[seg_start:seg_end + 1]
    del new_r.sequence[seg_start:seg_end + 1]

    # Tính lại insert_pos sau khi xóa
    actual_pos = insert_pos if insert_pos < seg_start else insert_pos - seg_len
    actual_pos = max(0, min(actual_pos, len(new_r.sequence) - 1))
    for k, nid in enumerate(segment):
        new_r.sequence.insert(actual_pos + 1 + k, nid)

    precompute(new_r, inst)

    # Kiểm tra TW
    cdata = {c.id: c for c in inst.all_nodes}
    for pos, nid in enumerate(new_r.sequence):
        if new_r.a[pos] > cdata[nid].due + 1e-6:
            return None

    delta = new_sol.objective(inst) - sol.objective(inst)
    attr  = (f'{vtype}{ridx}', seq[seg_start], seq[seg_end])
    return delta, 'oropt', attr, new_sol


def _apply_2opt_star(sol: Solution, inst: Instance,
                     r1_idx: int, i: int,
                     r2_idx: int, j: int) -> Optional[MoveResult]:
    """Swap tail của truck_routes[r1_idx] và truck_routes[r2_idx]."""
    if r1_idx == r2_idx:
        return None
    r1 = sol.truck_routes[r1_idx]
    r2 = sol.truck_routes[r2_idx]

    if i < 0 or i >= len(r1.sequence) - 1:
        return None
    if j < 0 or j >= len(r2.sequence) - 1:
        return None

    feasible, delta_c = check_2opt_star(r1, i, r2, j, inst,
                                         inst.truck_capacity)
    if not feasible:
        return None

    new_sol = sol.copy()
    nr1 = new_sol.truck_routes[r1_idx]
    nr2 = new_sol.truck_routes[r2_idx]

    tail1 = nr1.sequence[i + 1:]
    tail2 = nr2.sequence[j + 1:]
    nr1.sequence = nr1.sequence[:i + 1] + tail2
    nr2.sequence = nr2.sequence[:j + 1] + tail1

    precompute(nr1, inst)
    precompute(nr2, inst)

    delta = new_sol.objective(inst) - sol.objective(inst)
    attr  = (r1.sequence[i], r1_idx, r2.sequence[j], r2_idx)
    return delta, '2optstar', attr, new_sol


def _apply_transfer(sol: Solution, inst: Instance,
                    src_type: str, src_ridx: int, src_pos: int,
                    dst_type: str, dst_ridx: int) -> Optional[MoveResult]:
    """Chuyển C2 giữa truck route và drone route."""
    src_routes = sol.truck_routes if src_type == 'T' else sol.drone_routes
    dst_routes = sol.truck_routes if dst_type == 'T' else sol.drone_routes

    src_r = src_routes[src_ridx]
    if src_pos < 1 or src_pos >= len(src_r.sequence) - 1:
        return None

    node_id = src_r.sequence[src_pos]

    # Chỉ C2 mới chuyển được
    if node_id in inst.c1_ids:
        return None

    cdata = {c.id: c for c in inst.customers}
    node_cust = cdata.get(node_id)
    if node_cust is None:
        return None

    # Nếu chuyển sang drone: kiểm tra điều kiện drone
    if dst_type == 'D' and not _elig(node_cust, inst):
        return None

    dst_r = dst_routes[dst_ridx]

    # Tìm vị trí chèn tốt nhất trong dst route
    best_pos   = None
    best_delta = float('inf')

    for p in range(len(dst_r.sequence) - 1):
        ok, _ = check_insert(dst_r, node_id, p, inst)
        if ok:
            # Kiểm tra drone range nếu cần
            if dst_type == 'D':
                # Ước tính quãng đường thêm
                extra = (inst.dist(dst_r.sequence[p], node_id)
                         + inst.dist(node_id, dst_r.sequence[p + 1])
                         - inst.dist(dst_r.sequence[p], dst_r.sequence[p + 1]))
                if dst_r.total_dist + extra > inst.drone_range:
                    continue
            best_pos = p
            break   # lấy vị trí đầu tiên khả thi (greedy)

    if best_pos is None:
        return None

    # Kiểm tra xóa khỏi src
    feasible_remove, _ = check_remove(src_r, src_pos, inst)
    if not feasible_remove:
        return None

    new_sol = sol.copy()
    new_src = (new_sol.truck_routes if src_type == 'T'
               else new_sol.drone_routes)[src_ridx]
    new_dst = (new_sol.truck_routes if dst_type == 'T'
               else new_sol.drone_routes)[dst_ridx]

    new_src.sequence.pop(src_pos)
    new_dst.sequence.insert(best_pos + 1, node_id)

    precompute(new_src, inst)
    precompute(new_dst, inst)

    delta = new_sol.objective(inst) - sol.objective(inst)
    attr  = (node_id, f'{src_type}{src_ridx}', f'{dst_type}{dst_ridx}')
    return delta, 'transfer', attr, new_sol


# ─────────────────────────────────────────────────────────────────────────────
# Sinh toàn bộ lân cận
# ─────────────────────────────────────────────────────────────────────────────

def generate_neighborhood(sol: Solution, inst: Instance,
                           max_neighbors: int = 300) -> List[MoveResult]:
    """
    Sinh tập lân cận N(S) bằng cách áp dụng 5 loại move.
    Giới hạn max_neighbors để kiểm soát thời gian mỗi iteration.
    """
    neighbors: List[MoveResult] = []

    K = len(sol.truck_routes)
    D = len(sol.drone_routes)

    all_truck = list(range(K))
    all_drone = list(range(D))

    # ── Move 1: Relocate (truck↔truck, drone↔drone) ──────────────────────
    for ridx in all_truck:
        r = sol.truck_routes[ridx]
        for src_pos in range(1, len(r.sequence) - 1):
            for dst_ridx in all_truck:
                dst_r = sol.truck_routes[dst_ridx]
                for dst_pos in range(len(dst_r.sequence) - 1):
                    if ridx == dst_ridx and src_pos == dst_pos:
                        continue
                    res = _apply_relocate(sol, inst,
                                          'T', ridx, src_pos,
                                          'T', dst_ridx, dst_pos)
                    if res:
                        neighbors.append(res)
                        if len(neighbors) >= max_neighbors:
                            return neighbors

    for ridx in all_drone:
        r = sol.drone_routes[ridx]
        for src_pos in range(1, len(r.sequence) - 1):
            for dst_ridx in all_drone:
                dst_r = sol.drone_routes[dst_ridx]
                for dst_pos in range(len(dst_r.sequence) - 1):
                    if ridx == dst_ridx and src_pos == dst_pos:
                        continue
                    res = _apply_relocate(sol, inst,
                                          'D', ridx, src_pos,
                                          'D', dst_ridx, dst_pos)
                    if res:
                        neighbors.append(res)
                        if len(neighbors) >= max_neighbors:
                            return neighbors

    # ── Move 2: 2-opt nội tuyến ──────────────────────────────────────────
    for vtype, routes in [('T', sol.truck_routes), ('D', sol.drone_routes)]:
        for ridx, r in enumerate(routes):
            n = len(r.sequence)
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    res = _apply_2opt(sol, inst, vtype, ridx, i, j)
                    if res:
                        neighbors.append(res)
                        if len(neighbors) >= max_neighbors:
                            return neighbors

    # ── Move 3: Or-opt (seg_len 1,2,3) ───────────────────────────────────
    for vtype, routes in [('T', sol.truck_routes), ('D', sol.drone_routes)]:
        for ridx, r in enumerate(routes):
            n = len(r.sequence)
            for seg_len in [1, 2, 3]:
                for seg_start in range(1, n - seg_len):
                    for ins_pos in range(0, n - 1):
                        res = _apply_oropt(sol, inst, vtype, ridx,
                                           seg_start, seg_len, ins_pos)
                        if res:
                            neighbors.append(res)
                            if len(neighbors) >= max_neighbors:
                                return neighbors

    # ── Move 4: 2-opt* (cross truck routes) ──────────────────────────────
    for r1_idx in all_truck:
        r1 = sol.truck_routes[r1_idx]
        for r2_idx in range(r1_idx + 1, K):
            r2 = sol.truck_routes[r2_idx]
            for i in range(0, len(r1.sequence) - 1):
                for j in range(0, len(r2.sequence) - 1):
                    res = _apply_2opt_star(sol, inst, r1_idx, i, r2_idx, j)
                    if res:
                        neighbors.append(res)
                        if len(neighbors) >= max_neighbors:
                            return neighbors

    # ── Move 5: Transfer (C2 giữa truck và drone) ─────────────────────────
    # Truck → Drone
    for t_idx in all_truck:
        tr = sol.truck_routes[t_idx]
        for src_pos in range(1, len(tr.sequence) - 1):
            nid = tr.sequence[src_pos]
            if nid in inst.c1_ids:
                continue
            for d_idx in all_drone:
                res = _apply_transfer(sol, inst,
                                      'T', t_idx, src_pos,
                                      'D', d_idx)
                if res:
                    neighbors.append(res)
                    if len(neighbors) >= max_neighbors:
                        return neighbors

    # Drone → Truck
    for d_idx in all_drone:
        dr = sol.drone_routes[d_idx]
        for src_pos in range(1, len(dr.sequence) - 1):
            for t_idx in all_truck:
                res = _apply_transfer(sol, inst,
                                      'D', d_idx, src_pos,
                                      'T', t_idx)
                if res:
                    neighbors.append(res)
                    if len(neighbors) >= max_neighbors:
                        return neighbors

    return neighbors


# ─────────────────────────────────────────────────────────────────────────────
# Diversification
# ─────────────────────────────────────────────────────────────────────────────

def diversify(sol: Solution, inst: Instance) -> Solution:
    """
    Xáo trộn có kiểm soát từ S_best:
      - Chuyển ngẫu nhiên ~30% C2 giữa truck và drone
      - Áp dụng 5 Or-opt ngẫu nhiên
    """
    new_sol = sol.copy()
    cdata   = {c.id: c for c in inst.customers}

    # Lấy danh sách C2 đang trong truck routes
    c2_in_truck = []
    for tidx, r in enumerate(new_sol.truck_routes):
        for pos in range(1, len(r.sequence) - 1):
            nid = r.sequence[pos]
            if nid in inst.c2_ids:
                c2_in_truck.append(('T', tidx, pos, nid))

    # Chuyển ~30%
    n_transfer = max(1, len(c2_in_truck) // 3)
    selected   = random.sample(c2_in_truck,
                               min(n_transfer, len(c2_in_truck)))

    offset = 0   # vị trí trượt sau mỗi lần xóa
    for src_type, t_idx, pos, nid in selected:
        actual_pos = pos - offset
        tr = new_sol.truck_routes[t_idx]
        if actual_pos < 1 or actual_pos >= len(tr.sequence) - 1:
            continue
        if tr.sequence[actual_pos] != nid:
            continue
        c = cdata.get(nid)
        if c and _elig(c, inst):
            # Tìm drone route phù hợp
            for dr in new_sol.drone_routes:
                if dr.total_load + c.demand <= inst.drone_capacity:
                    round_trip = (dr.total_dist
                                  + inst.dist(0, nid) + inst.dist(nid, 0))
                    if round_trip <= inst.drone_range:
                        tr.sequence.pop(actual_pos)
                        dr.sequence.insert(-1, nid)
                        precompute(tr, inst)
                        precompute(dr, inst)
                        offset += 1
                        break

    # 5 Or-opt ngẫu nhiên
    for _ in range(5):
        all_routes = [('T', i) for i in range(len(new_sol.truck_routes))] + \
                     [('D', i) for i in range(len(new_sol.drone_routes))]
        vtype, ridx = random.choice(all_routes)
        routes = new_sol.truck_routes if vtype == 'T' else new_sol.drone_routes
        r = routes[ridx]
        if len(r.sequence) < 4:
            continue
        seg_len   = random.choice([1, 2])
        seg_start = random.randint(1, len(r.sequence) - 2 - seg_len)
        ins_pos   = random.randint(0, len(r.sequence) - 2)
        res = _apply_oropt(new_sol, inst, vtype, ridx,
                           seg_start, seg_len, ins_pos)
        if res:
            _, _, _, new_sol = res

    return new_sol


# ─────────────────────────────────────────────────────────────────────────────
# Tabu Search chính
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TabuSearchConfig:
    max_iter:          int   = 500
    max_no_improve:    int   = 150
    diversify_thresh:  int   = 60
    tenure_base:       int   = 10
    tenure_delta:      int   = 3
    max_neighbors:     int   = 300
    lambda_tw:         float = 50.0
    lambda_cap:        float = 200.0
    alpha:             float = 0.5    # greedy: trọng số dist
    beta:              float = 0.3    # greedy: trọng số wait
    gamma:             float = 0.2    # greedy: trọng số urgency
    verbose:           bool  = True
    time_limit:        float = 300.0  # giây


def tabu_search(inst: Instance,
                cfg: TabuSearchConfig = None) -> Tuple[Solution, List[float]]:
    """
    Tabu Search chính.

    Trả về (best_solution, history) trong đó history là list makespan theo iter.
    """
    if cfg is None:
        cfg = TabuSearchConfig()

    # ── Khởi tạo ─────────────────────────────────────────────────────────
    sol = greedy_construction(inst, alpha=cfg.alpha,
                              beta=cfg.beta, gamma=cfg.gamma)
    sol.lambda_tw  = cfg.lambda_tw
    sol.lambda_cap = cfg.lambda_cap

    best_sol  = sol.copy()
    best_obj  = best_sol.objective(inst)

    tabu      = TabuList(cfg.tenure_base, cfg.tenure_delta)
    history   = [best_sol.makespan()]
    no_improve = 0
    start_time = time.time()

    if cfg.verbose:
        print(f"[Init] makespan={best_sol.makespan():.2f}  "
              f"obj={best_obj:.2f}  "
              f"feasible={best_sol.is_feasible(inst)}")

    # ── Vòng lặp chính ───────────────────────────────────────────────────
    for iteration in range(cfg.max_iter):

        # Điều kiện dừng
        if no_improve >= cfg.max_no_improve:
            if cfg.verbose:
                print(f"[Stop] no_improve={no_improve} >= {cfg.max_no_improve}")
            break
        if time.time() - start_time > cfg.time_limit:
            if cfg.verbose:
                print(f"[Stop] time limit {cfg.time_limit:.0f}s reached")
            break

        # Sinh lân cận
        neighbors = generate_neighborhood(sol, inst, cfg.max_neighbors)

        if not neighbors:
            if cfg.verbose:
                print(f"[{iteration}] No neighbors found, stopping.")
            break

        # Sắp xếp theo delta (tốt nhất trước)
        neighbors.sort(key=lambda x: x[0])

        # Chọn nghiệm tốt nhất không bị cấm (hoặc aspiration)
        chosen = None
        for delta, mtype, attr, new_sol in neighbors:
            is_tabu    = tabu.is_tabu(mtype, attr, iteration)
            aspiration = new_sol.objective(inst) < best_obj

            if not is_tabu or aspiration:
                chosen = (delta, mtype, attr, new_sol)
                break

        if chosen is None:
            # Tất cả đều bị cấm → lấy cái ít bị cấm nhất (tenure ngắn nhất)
            chosen = neighbors[0]

        delta, mtype, attr, new_sol = chosen

        # Cập nhật Tabu List
        tabu.add(mtype, attr, iteration)
        tabu.cleanup(iteration)

        # Chuyển sang nghiệm mới
        sol = new_sol
        cur_obj = sol.objective(inst)

        if cur_obj < best_obj:
            best_sol   = sol.copy()
            best_obj   = cur_obj
            no_improve = 0
            if cfg.verbose:
                print(f"[{iteration:4d}] + makespan={best_sol.makespan():.2f}  "
                      f"obj={best_obj:.2f}  "
                      f"move={mtype}  feasible={best_sol.is_feasible(inst)}")
        else:
            no_improve += 1

        history.append(best_sol.makespan())

        # Diversification
        if no_improve == cfg.diversify_thresh:
            if cfg.verbose:
                print(f"[{iteration:4d}] ~ Diversification...")
            sol        = diversify(best_sol, inst)
            sol.lambda_tw  = cfg.lambda_tw
            sol.lambda_cap = cfg.lambda_cap
            no_improve = 0
            tabu.increase_tenure()

    if cfg.verbose:
        elapsed = time.time() - start_time
        print(f"\n[Done] iter={iteration+1}, time={elapsed:.1f}s")
        print(best_sol.summary(inst))

    return best_sol, history
