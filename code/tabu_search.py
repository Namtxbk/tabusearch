"""
tabu_search.py — Tabu Search với Phạt Động (Strategic Oscillation) cho MVRPD-TW

FIX:
  - objective() nhất quán dùng trọng số hiện tại (w_tw, w_cap, w_range, w_assign)
  - Tabu key nhất quán (cust_id, vehicle_type, v_idx, t_idx, pos)
  - Dọn trip rỗng đúng chỗ, sau khi recompute
  - Thêm phạt drone_assign (drone phục vụ C1)
  - Các move nhỏ (relocate, swap) bổ sung bên cạnh ruin&recreate để khai thác
"""
from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

from instance import Instance, Customer
from solution import Trip, Vehicle, Solution, precompute_trip, precompute_vehicle


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TabuSearchConfig:
    max_iter: int = 1000
    max_no_improve: int = 200
    tenure_base: int = 7
    time_limit: float = 60.0
    verbose: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Tabu Set
# ─────────────────────────────────────────────────────────────────────────────

class TabuSet:
    def __init__(self, tenure: int = 10):
        self._data: dict = {}
        self.tenure = tenure

    def add(self, key, current_iter: int):
        self._data[key] = current_iter + self.tenure

    def is_tabu(self, key, current_iter: int) -> bool:
        return self._data.get(key, 0) > current_iter

    def update_tenure(self, tenure: int):
        self.tenure = tenure


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tính objective nhất quán
# ─────────────────────────────────────────────────────────────────────────────

def _obj(sol: Solution, inst: Instance,
         w_tw: float, w_cap: float, w_range: float, w_assign: float) -> float:
    return (sol.makespan()
            + w_tw     * sol.penalty_tw(inst)
            + w_cap    * sol.penalty_cap(inst)
            + w_range  * sol.penalty_range(inst)
            + w_assign * sol.penalty_drone_assign(inst))


# ─────────────────────────────────────────────────────────────────────────────
# Các toán tử lân cận
# ─────────────────────────────────────────────────────────────────────────────

def _drone_eligible(c: Customer, inst: Instance) -> bool:
    if c.is_c1:
        return False
    if c.demand > inst.drone_capacity:
        return False
    rt = inst.travel_time(0, c.id, True) + inst.travel_time(c.id, 0, True)
    return rt <= inst.drone_range


def _clean_solution(sol: Solution, inst: Instance):
    """Dọn trip rỗng rồi precompute lại toàn bộ."""
    for v in sol.trucks + sol.drones:
        v.trips = [t for t in v.trips if len(t.customers()) > 0]
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
        precompute_vehicle(v, inst)


def relocate_neighborhood(
    sol: Solution, inst: Instance,
    tabu: TabuSet, it: int, best_obj: float,
    w_tw: float, w_cap: float, w_range: float, w_assign: float
) -> Tuple[Optional[Solution], float]:
    """
    Toán tử Relocate: lấy 1 khách ra khỏi vị trí hiện tại,
    chèn vào vị trí tốt nhất trong toàn hệ thống (kể cả xe khác, trip khác, trip mới).
    """
    all_vehicles = [(v, False, i) for i, v in enumerate(sol.trucks)] + \
                   [(v, True,  i) for i, v in enumerate(sol.drones)]

    best_sol = None
    best_score = float('inf')
    best_key = None

    # Thu thập danh sách (vehicle_type, v_idx, t_idx, pos_in_trip, cust_id)
    candidates = []
    for v, is_drone, v_idx in all_vehicles:
        for t_idx, trip in enumerate(v.trips):
            for pos, cid in enumerate(trip.sequence):
                if cid != 0:
                    candidates.append((is_drone, v_idx, t_idx, pos, cid))

    random.shuffle(candidates)
    # Giới hạn số move để tránh quá chậm
    candidates = candidates[:30]

    for (src_drone, src_vidx, src_tidx, src_pos, cid) in candidates:
        cust = inst.customers[cid - 1]

        # Xây sol tạm với cid đã bị rút ra
        tmp = sol.copy()
        src_v = tmp.drones[src_vidx] if src_drone else tmp.trucks[src_vidx]
        src_v.trips[src_tidx].sequence.pop(src_pos)
        # Nếu trip chỉ còn [0,0] thì giữ lại (sẽ dọn sau)

        # Thử chèn vào tất cả vị trí có thể
        for (dst_v_obj, dst_drone, dst_vidx) in all_vehicles:
            if dst_drone and not _drone_eligible(cust, inst):
                continue

            dst_v = tmp.drones[dst_vidx] if dst_drone else tmp.trucks[dst_vidx]

            # Chèn vào trip hiện có
            for dst_tidx, dst_trip in enumerate(dst_v.trips):
                for insert_pos in range(1, len(dst_trip.sequence)):
                    cand = tmp.copy()
                    cv = cand.drones[dst_vidx] if dst_drone else cand.trucks[dst_vidx]
                    cv.trips[dst_tidx].sequence.insert(insert_pos, cid)
                    cand.recompute_all(inst)
                    score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                    key = (cid, dst_drone, dst_vidx, dst_tidx, insert_pos)
                    if score < best_score:
                        if not tabu.is_tabu(key, it) or score < best_obj:
                            best_score = score
                            best_sol = cand
                            best_key = key

            # Tạo trip mới
            new_trip = Trip(sequence=[0, cid, 0], is_drone=dst_drone)
            cand = tmp.copy()
            cv = cand.drones[dst_vidx] if dst_drone else cand.trucks[dst_vidx]
            cv.trips.append(new_trip)
            cand.recompute_all(inst)
            score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
            key = (cid, dst_drone, dst_vidx, len(dst_v.trips), -1)
            if score < best_score:
                if not tabu.is_tabu(key, it) or score < best_obj:
                    best_score = score
                    best_sol = cand
                    best_key = key

    if best_sol is not None and best_key is not None:
        tabu.add(best_key, it)
        _clean_solution(best_sol, inst)
        return best_sol, best_score

    return None, float('inf')


def ruin_and_recreate(
    sol: Solution, inst: Instance,
    tabu: TabuSet, it: int, best_obj: float,
    w_tw: float, w_cap: float, w_range: float, w_assign: float
) -> Tuple[Optional[Solution], float]:
    """
    Ruin & Recreate: nhấc 15-30% khách, chèn lại theo best-insertion.
    """
    all_cust_ids = [c.id for c in inst.customers]
    num_remove = max(2, int(len(all_cust_ids) * random.uniform(0.15, 0.30)))
    removed = set(random.sample(all_cust_ids, num_remove))

    new_sol = sol.copy()

    # RUIN
    for v in new_sol.trucks + new_sol.drones:
        for t in v.trips:
            t.sequence = [n for n in t.sequence if n not in removed]

    # Recreate theo thứ tự ngẫu nhiên có ưu tiên due sớm
    removed_list = sorted(removed,
                          key=lambda cid: inst.customers[cid - 1].due)
    random.shuffle(removed_list[:max(1, len(removed_list)//3)])  # xáo nhẹ đầu list

    all_vehicles = [(v, False, i) for i, v in enumerate(new_sol.trucks)] + \
                   [(v, True,  i) for i, v in enumerate(new_sol.drones)]

    for cid in removed_list:
        cust = inst.customers[cid - 1]
        insert_best_sol = None
        insert_best_score = float('inf')
        insert_best_key = None

        for (_, dst_drone, dst_vidx) in all_vehicles:
            if dst_drone and not _drone_eligible(cust, inst):
                continue
            dst_v = new_sol.drones[dst_vidx] if dst_drone else new_sol.trucks[dst_vidx]

            # Chèn vào trip hiện có
            for t_idx, trip in enumerate(dst_v.trips):
                for pos in range(1, len(trip.sequence)):
                    cand = new_sol.copy()
                    cv = cand.drones[dst_vidx] if dst_drone else cand.trucks[dst_vidx]
                    cv.trips[t_idx].sequence.insert(pos, cid)
                    cand.recompute_all(inst)
                    score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                    key = (cid, dst_drone, dst_vidx, t_idx, pos)
                    if score < insert_best_score:
                        if not tabu.is_tabu(key, it) or score < best_obj:
                            insert_best_score = score
                            insert_best_sol = cand
                            insert_best_key = key

            # Tạo trip mới
            new_trip_cand = new_sol.copy()
            cv = new_trip_cand.drones[dst_vidx] if dst_drone else new_trip_cand.trucks[dst_vidx]
            cv.trips.append(Trip(sequence=[0, cid, 0], is_drone=dst_drone))
            new_trip_cand.recompute_all(inst)
            score = _obj(new_trip_cand, inst, w_tw, w_cap, w_range, w_assign)
            key = (cid, dst_drone, dst_vidx, len(dst_v.trips), -1)
            if score < insert_best_score:
                if not tabu.is_tabu(key, it) or score < best_obj:
                    insert_best_score = score
                    insert_best_sol = new_trip_cand
                    insert_best_key = key

        if insert_best_sol is not None:
            new_sol = insert_best_sol
            if insert_best_key:
                tabu.add(insert_best_key, it)
        else:
            # Fallback: ép vào truck[0] trip mới
            v0 = new_sol.trucks[0]
            v0.trips.append(Trip(sequence=[0, cid, 0], is_drone=False))
            new_sol.recompute_all(inst)

    _clean_solution(new_sol, inst)
    score = _obj(new_sol, inst, w_tw, w_cap, w_range, w_assign)
    return new_sol, score


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabu Search
# ─────────────────────────────────────────────────────────────────────────────

def advanced_tabu_search(
    init_sol: Solution, inst: Instance, cfg: TabuSearchConfig
) -> Tuple[Solution, List[float]]:
    """
    Tabu Search tích hợp Strategic Oscillation:
      - Relocate (khai thác, move nhỏ) xen kẽ Ruin&Recreate (thám hiểm, move lớn)
      - Trọng số phạt điều chỉnh động theo feasibility
    """
    t_start = time.time()

    current = init_sol.copy()
    current.recompute_all(inst)

    best = current.copy()

    # Trọng số phạt ban đầu
    w_tw, w_cap, w_range, w_assign = 50.0, 200.0, 200.0, 500.0

    best_obj = _obj(best, inst, w_tw, w_cap, w_range, w_assign)
    current_obj = best_obj

    tabu = TabuSet(tenure=cfg.tenure_base)
    history = [best.makespan()]

    no_improve = 0
    feasible_streak = 0
    infeasible_streak = 0

    for it in range(1, cfg.max_iter + 1):
        # Dừng sớm
        if no_improve >= cfg.max_no_improve:
            if cfg.verbose:
                print(f"  -> Dừng sớm tại iter {it} (no_improve={no_improve})")
            break
        if time.time() - t_start > cfg.time_limit:
            if cfg.verbose:
                print(f"  -> Dừng do time_limit tại iter {it}")
            break

        # Chọn toán tử: relocate (khai thác) hoặc ruin&recreate (thám hiểm)
        use_rr = (it % 5 == 0) or (no_improve > cfg.max_no_improve // 2)

        if use_rr:
            nb_sol, nb_obj = ruin_and_recreate(
                current, inst, tabu, it, best_obj,
                w_tw, w_cap, w_range, w_assign
            )
        else:
            nb_sol, nb_obj = relocate_neighborhood(
                current, inst, tabu, it, best_obj,
                w_tw, w_cap, w_range, w_assign
            )

        if nb_sol is None:
            no_improve += 1
            continue

        current = nb_sol
        current_obj = nb_obj

        # ── Strategic Oscillation (điều chỉnh trọng số phạt) ────────────
        if current.is_feasible(inst):
            feasible_streak += 1
            infeasible_streak = 0
            if feasible_streak >= 8:
                # Đang tốt → giảm phạt để tối ưu makespan sâu hơn
                w_cap   = max(20.0, w_cap   * 0.85)
                w_range = max(20.0, w_range * 0.85)
                w_tw    = max(10.0, w_tw    * 0.85)
                feasible_streak = 0
        else:
            infeasible_streak += 1
            feasible_streak = 0
            if infeasible_streak >= 5:
                # Bị vi phạm nhiều → siết phạt
                w_cap   = min(2000.0, w_cap   * 1.5)
                w_range = min(2000.0, w_range * 1.5)
                w_tw    = min(2000.0, w_tw    * 1.5)
                infeasible_streak = 0

        # Cập nhật best (chỉ nhận nghiệm khả thi + phục vụ đủ)
        if current.is_feasible(inst) and current.all_served(inst):
            cur_ms = current.makespan()
            if cur_ms < best.makespan() or not best.is_feasible(inst):
                best = current.copy()
                best_obj = _obj(best, inst, w_tw, w_cap, w_range, w_assign)
                no_improve = 0
                history.append(best.makespan())
                if cfg.verbose:
                    print(f"  [{it:4d}] ⭐ Makespan={best.makespan():.2f}"
                          f"  w_tw={w_tw:.0f} w_cap={w_cap:.0f}")
            else:
                no_improve += 1
        else:
            no_improve += 1

        if cfg.verbose and it % 100 == 0:
            print(f"  [{it:4d}] cur={current.makespan():.2f}"
                  f"  best={best.makespan():.2f}"
                  f"  feasible={current.is_feasible(inst)}"
                  f"  served={current.all_served(inst)}"
                  f"  w_tw={w_tw:.1f}")

    return best, history
