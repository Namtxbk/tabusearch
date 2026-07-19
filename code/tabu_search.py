"""
tabu_search.py — Tabu Search với Phạt Động (Strategic Oscillation) cho MVRPD-TW

Toán tử lân cận:
  1. Relocate      — di chuyển 1 khách sang vị trí tốt nhất trong hệ thống
  2. Or-opt(2)     — di chuyển 2 khách liên tiếp sang vị trí khác
  3. Swap          — hoán đổi 2 khách ở 2 vị trí khác nhau
  4. 2-opt         — đảo ngược đoạn con trong cùng 1 trip
  5. Cross-trip    — hoán đổi đoạn tuyến giữa 2 trip khác nhau
  6. Ruin&Recreate — phá và tái tạo 15-30% khách (diversification)

Chiến lược chạy:
  - Mỗi vòng lặp chạy TẤT CẢ 5 toán tử khai thác, lấy move tốt nhất trong số đó
  - Mỗi 6 vòng hoặc khi gần ngưỡng dừng → thêm Ruin & Recreate vào pool
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
    def __init__(self, tenure: int = 7):
        self._data: dict = {}
        self.tenure = tenure

    def add(self, key, current_iter: int):
        self._data[key] = current_iter + self.tenure

    def is_tabu(self, key, current_iter: int) -> bool:
        return self._data.get(key, 0) > current_iter


# ─────────────────────────────────────────────────────────────────────────────
# Hàm tiện ích
# ─────────────────────────────────────────────────────────────────────────────

def _obj(sol: Solution, inst: Instance,
         w_tw: float, w_cap: float, w_range: float, w_assign: float) -> float:
    assign_penalty = (sol.penalty_drone_assign(inst)
                       if hasattr(sol, 'penalty_drone_assign') else 0.0)
    cap_pen   = sol.penalty_cap(inst)
    range_pen = sol.penalty_range(inst)
    tw_pen    = sol.penalty_tw(inst)
    # Ràng buộc L_w: thời gian chờ tối đa tại điểm khách.
    # Xử lý như ràng buộc CỨNG (hard penalty 1e6) — cùng nhóm với TW/cap/range
    # vì construction.py đã đảm bảo lời giải khởi đầu feasible về L_w, nên
    # mọi move tạo ra vi phạm L_w đều phải "đắt" hơn bất kỳ lợi ích makespan.
    wait_pen  = sol.penalty_wait(inst) if hasattr(sol, 'penalty_wait') else 0.0

    HARD_CAP_MULT = 1e6
    hard_penalty = HARD_CAP_MULT * (cap_pen + range_pen + tw_pen + wait_pen)

    return (sol.makespan()
            + w_cap    * cap_pen
            + w_range  * range_pen
            + w_assign * assign_penalty
            + hard_penalty)


def _drone_eligible(c: Customer, inst: Instance) -> bool:
    if c.is_c1:
        return False
    if c.demand > inst.drone_capacity:
        return False
    rt = inst.travel_time(0, c.id, True) + inst.travel_time(c.id, 0, True)
    return rt <= inst.drone_range


# ─────────────────────────────────────────────────────────────────────────────
# Forward Time Slack — kiểm tra nhanh O(1)/O(k) thay vì recompute O(n) toàn route
#
# Theo tài liệu: F[i] = min_{k>=i}(l[k] - t[k]) đã được precompute_trip tính sẵn
# trong trip.F. Ý nghĩa: delay tối đa có thể "nhồi" thêm vào từ vị trí i trở đi
# mà KHÔNG vi phạm time window bất kỳ node nào phía sau (kể cả depot cuối).
#
# Quy tắc dùng: nếu một thay đổi tại vị trí i làm node i+1 (node theo cũ) đến
# trễ hơn Δ so với trước, thay đổi đó CHẮC CHẮN feasible về TW của phần ĐUÔI
# CŨ (không đổi) khi và chỉ khi Δ <= trip.F[i+1] (slack TẠI THỜI ĐIỂM TRƯỚC
# khi chèn). Đây chỉ là phép lọc nhanh: nếu Δ > F[i+1] → chắc chắn vi phạm,
# bỏ qua ngay không cần recompute. Nếu Δ <= F[i+1] → có khả năng feasible,
# nhưng vẫn cần recompute đầy đủ để lấy giá trị makespan/penalty chính xác
# cho _obj() (slack chỉ trả lời CÓ/KHÔNG vi phạm, không cho biết makespan
# mới hay tổng penalty mới).
# ─────────────────────────────────────────────────────────────────────────────

def _quick_feasible_after_pos(trip: Trip, pos_after: int, delay: float) -> bool:
    """
    Lọc nhanh O(1): trả về False nếu chắc chắn vi phạm TW ở phần đuôi
    [pos_after+1 .. cuối] của `trip` (TRƯỚC khi thay đổi) khi đuôi đó bị
    delay thêm `delay` đơn vị thời gian. Trả về True nghĩa là "có thể
    feasible, cần kiểm tra kỹ hơn" — KHÔNG đảm bảo feasible tuyệt đối,
    chỉ loại bỏ chắc chắn các trường hợp vi phạm.
    """
    if delay <= 1e-9:
        return True  # không trễ hơn cũ => không thể vi phạm thêm
    if pos_after + 1 >= len(trip.F):
        return True  # không có đuôi để kiểm tra (vd. pos_after là node cuối)
    return delay <= trip.F[pos_after + 1] + 1e-9


def _clean(sol: Solution, inst: Instance):
    for v in sol.trucks + sol.drones:
        v.trips = [t for t in v.trips if len(t.customers()) > 0]
        if not v.trips:
            v.trips = [Trip(sequence=[0, 0], is_drone=v.is_drone)]
        precompute_vehicle(v, inst)


def _all_vehicles(sol: Solution):
    return ([(v, False, i) for i, v in enumerate(sol.trucks)] +
            [(v, True,  i) for i, v in enumerate(sol.drones)])


# ─────────────────────────────────────────────────────────────────────────────
# Toán tử 1: Relocate
# Di chuyển 1 khách ra khỏi vị trí hiện tại, chèn vào vị trí tốt nhất
# trong toàn hệ thống (kể cả trip khác, xe khác, trip mới).
# ─────────────────────────────────────────────────────────────────────────────

def _gen_relocate(sol: Solution, inst: Instance,
                  tabu: TabuSet, it: int, best_obj: float,
                  w_tw, w_cap, w_range, w_assign):
    """Sinh tất cả move Relocate, yield (score, sol, key)."""
    avs = _all_vehicles(sol)

    all_custs = []
    for v, is_drone, vi in avs:
        for ti, trip in enumerate(v.trips):
            for pos, cid in enumerate(trip.sequence):
                if cid != 0:
                    all_custs.append((is_drone, vi, ti, pos, cid))

    random.shuffle(all_custs)
    all_custs = all_custs[:30]

    cdata = {c.id: c for c in inst.all_nodes}

    for (src_drone, src_vi, src_ti, src_pos, cid) in all_custs:
        cust = inst.customers[cid - 1]
        tmp = sol.copy()
        sv = tmp.drones[src_vi] if src_drone else tmp.trucks[src_vi]
        sv.trips[src_ti].sequence.pop(src_pos)
        # QUAN TRỌNG: a[]/F[] không tự cập nhật sau khi sửa sequence trực
        # tiếp, và vì multi-trip có tính tuần tự (start_time trip sau phụ
        # thuộc return_time trip trước), phải precompute lại CẢ VEHICLE
        # nguồn (không chỉ 1 trip) trước khi bộ lọc slack đọc dst_trip.a/F
        # — đặc biệt khi dst trùng đúng vehicle này.
        precompute_vehicle(sv, inst)

        for (_, dst_drone, dst_vi) in avs:
            if dst_drone and not _drone_eligible(cust, inst):
                continue
            dv = tmp.drones[dst_vi] if dst_drone else tmp.trucks[dst_vi]

            for dst_ti, dst_trip in enumerate(dv.trips):
                for ins_pos in range(1, len(dst_trip.sequence)):
                    # ── Lọc nhanh O(1) bằng Forward Time Slack (mục 3.1 tài
                    # liệu): tính thử arrival tại cid và tại node kế tiếp CŨ,
                    # so với slack F của dst_trip TRƯỚC khi chèn, để loại
                    # ngay các vị trí chắc chắn vi phạm TW phía đuôi —
                    # không cần cand.copy()+recompute_all() (O(n)) cho chúng.
                    prev_id = dst_trip.sequence[ins_pos - 1]
                    next_id = dst_trip.sequence[ins_pos]
                    t_prev = dst_trip.a[ins_pos - 1]
                    s_prev = cdata[prev_id].service
                    t_new_at_cid = max(cust.ready,
                                        t_prev + s_prev + inst.travel_time(prev_id, cid, dst_drone))
                    if t_new_at_cid > cust.due + 1e-9:
                        continue  # vi phạm TW ngay tại chính node được chèn
                    next_cust = cdata[next_id]
                    t_old_at_next = dst_trip.a[ins_pos]
                    t_new_at_next = max(next_cust.ready,
                                         t_new_at_cid + cust.service
                                         + inst.travel_time(cid, next_id, dst_drone))
                    delay = t_new_at_next - t_old_at_next
                    if not _quick_feasible_after_pos(dst_trip, ins_pos - 1, delay):
                        continue  # chắc chắn vi phạm đuôi -> bỏ qua, không recompute

                    cand = tmp.copy()
                    cv = cand.drones[dst_vi] if dst_drone else cand.trucks[dst_vi]
                    cv.trips[dst_ti].sequence.insert(ins_pos, cid)
                    cand.recompute_all(inst)
                    score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                    key = ('rel', cid, dst_drone, dst_vi, dst_ti, ins_pos)
                    if not tabu.is_tabu(key, it) or score < best_obj:
                        yield score, cand, key

            cand = tmp.copy()
            cv = cand.drones[dst_vi] if dst_drone else cand.trucks[dst_vi]
            cv.trips.append(Trip(sequence=[0, cid, 0], is_drone=dst_drone))
            cand.recompute_all(inst)
            score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
            key = ('rel', cid, dst_drone, dst_vi, -1, -1)
            if not tabu.is_tabu(key, it) or score < best_obj:
                yield score, cand, key


# ─────────────────────────────────────────────────────────────────────────────
# Toán tử 2: Or-opt(2)
# Di chuyển 2 khách LIÊN TIẾP trong cùng 1 trip sang vị trí khác.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_or_opt2(sol: Solution, inst: Instance,
                 tabu: TabuSet, it: int, best_obj: float,
                 w_tw, w_cap, w_range, w_assign):
    avs = _all_vehicles(sol)

    for (v, src_drone, src_vi) in avs:
        for src_ti, trip in enumerate(v.trips):
            seq = trip.sequence
            if len(seq) < 4:
                continue
            for pos in range(1, len(seq) - 2):
                cid1, cid2 = seq[pos], seq[pos + 1]
                if cid1 == 0 or cid2 == 0:
                    continue
                c1 = inst.customers[cid1 - 1]
                c2 = inst.customers[cid2 - 1]

                tmp = sol.copy()
                sv = tmp.drones[src_vi] if src_drone else tmp.trucks[src_vi]
                del sv.trips[src_ti].sequence[pos:pos + 2]

                for (_, dst_drone, dst_vi) in avs:
                    if dst_drone and (not _drone_eligible(c1, inst)
                                      or not _drone_eligible(c2, inst)):
                        continue
                    dv = tmp.drones[dst_vi] if dst_drone else tmp.trucks[dst_vi]

                    for dst_ti, dst_trip in enumerate(dv.trips):
                        for ins_pos in range(1, len(dst_trip.sequence)):
                            cand = tmp.copy()
                            cv = cand.drones[dst_vi] if dst_drone else cand.trucks[dst_vi]
                            cv.trips[dst_ti].sequence.insert(ins_pos, cid2)
                            cv.trips[dst_ti].sequence.insert(ins_pos, cid1)
                            cand.recompute_all(inst)
                            score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                            key = ('or2', cid1, cid2, dst_drone, dst_vi, dst_ti, ins_pos)
                            if not tabu.is_tabu(key, it) or score < best_obj:
                                yield score, cand, key

                    cand = tmp.copy()
                    cv = cand.drones[dst_vi] if dst_drone else cand.trucks[dst_vi]
                    cv.trips.append(Trip(sequence=[0, cid1, cid2, 0], is_drone=dst_drone))
                    cand.recompute_all(inst)
                    score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                    key = ('or2', cid1, cid2, dst_drone, dst_vi, -1, -1)
                    if not tabu.is_tabu(key, it) or score < best_obj:
                        yield score, cand, key


# ─────────────────────────────────────────────────────────────────────────────
# Toán tử 3: Swap
# Hoán đổi 2 khách ở 2 vị trí khác nhau.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_swap(sol: Solution, inst: Instance,
              tabu: TabuSet, it: int, best_obj: float,
              w_tw, w_cap, w_range, w_assign):
    avs = _all_vehicles(sol)

    positions = []
    for v, is_drone, vi in avs:
        for ti, trip in enumerate(v.trips):
            for pos, cid in enumerate(trip.sequence):
                if cid != 0:
                    positions.append((is_drone, vi, ti, pos, cid))

    random.shuffle(positions)
    positions = positions[:20]

    for idx_a in range(len(positions)):
        for idx_b in range(idx_a + 1, len(positions)):
            da, vai, tai, pa, cida = positions[idx_a]
            db, vbi, tbi, pb, cidb = positions[idx_b]
            ca = inst.customers[cida - 1]
            cb = inst.customers[cidb - 1]

            if da and not _drone_eligible(cb, inst):
                continue
            if db and not _drone_eligible(ca, inst):
                continue

            cand = sol.copy()
            va_obj = cand.drones[vai] if da else cand.trucks[vai]
            vb_obj = cand.drones[vbi] if db else cand.trucks[vbi]
            va_obj.trips[tai].sequence[pa] = cidb
            vb_obj.trips[tbi].sequence[pb] = cida
            cand.recompute_all(inst)
            score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
            key = ('swap', min(cida, cidb), max(cida, cidb))
            if not tabu.is_tabu(key, it) or score < best_obj:
                yield score, cand, key


# ─────────────────────────────────────────────────────────────────────────────
# Toán tử 4: 2-opt
# Đảo ngược đoạn con [i..j] bên trong cùng 1 trip.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_2opt(sol: Solution, inst: Instance,
              tabu: TabuSet, it: int, best_obj: float,
              w_tw, w_cap, w_range, w_assign):
    avs = _all_vehicles(sol)

    for v, is_drone, vi in avs:
        for ti, trip in enumerate(v.trips):
            seq = trip.sequence
            n = len(seq)
            if n < 5:
                continue
            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    cand = sol.copy()
                    cv = cand.drones[vi] if is_drone else cand.trucks[vi]
                    cv.trips[ti].sequence[i:j + 1] = \
                        cv.trips[ti].sequence[i:j + 1][::-1]
                    cand.recompute_all(inst)
                    score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                    key = ('2opt', vi, ti, i, j)
                    if not tabu.is_tabu(key, it) or score < best_obj:
                        yield score, cand, key


# ─────────────────────────────────────────────────────────────────────────────
# Toán tử 5: Cross-trip
# Hoán đổi đoạn đuôi giữa 2 trip của cùng 1 xe.
# ─────────────────────────────────────────────────────────────────────────────

def _gen_cross_trip(sol: Solution, inst: Instance,
                    tabu: TabuSet, it: int, best_obj: float,
                    w_tw, w_cap, w_range, w_assign):
    avs = _all_vehicles(sol)

    for v, is_drone, vi in avs:
        if len(v.trips) < 2:
            continue
        for ta in range(len(v.trips)):
            for tb in range(ta + 1, len(v.trips)):
                seq_a = v.trips[ta].sequence
                seq_b = v.trips[tb].sequence
                for cut_a in range(1, len(seq_a) - 1):
                    for cut_b in range(1, len(seq_b) - 1):
                        tail_a = seq_a[cut_a:-1]
                        tail_b = seq_b[cut_b:-1]
                        if is_drone:
                            if any(inst.customers[c-1].is_c1
                                   for c in tail_a + tail_b):
                                continue
                        cand = sol.copy()
                        cv = cand.drones[vi] if is_drone else cand.trucks[vi]
                        cv.trips[ta].sequence = seq_a[:cut_a] + tail_b + [0]
                        cv.trips[tb].sequence = seq_b[:cut_b] + tail_a + [0]
                        if (cv.trips[ta].sequence == [0, 0] or
                                cv.trips[tb].sequence == [0, 0]):
                            continue
                        cand.recompute_all(inst)
                        score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                        key = ('cross', vi, ta, tb, cut_a, cut_b)
                        if not tabu.is_tabu(key, it) or score < best_obj:
                            yield score, cand, key


# ─────────────────────────────────────────────────────────────────────────────
# Toán tử 6: Ruin & Recreate (Diversification)
# ─────────────────────────────────────────────────────────────────────────────

def op_ruin_recreate(sol: Solution, inst: Instance,
                     tabu: TabuSet, it: int, best_obj: float,
                     w_tw, w_cap, w_range, w_assign
                     ) -> Tuple[Optional[Solution], float, Optional[tuple]]:
    all_cust_ids = [c.id for c in inst.customers]
    num_remove = max(2, int(len(all_cust_ids) * random.uniform(0.15, 0.30)))
    removed = set(random.sample(all_cust_ids, num_remove))

    new_sol = sol.copy()
    for v in new_sol.trucks + new_sol.drones:
        for t in v.trips:
            t.sequence = [n for n in t.sequence if n not in removed]

    removed_list = sorted(removed, key=lambda cid: inst.customers[cid-1].due)
    n3 = max(1, len(removed_list) // 3)
    head = removed_list[:n3]
    random.shuffle(head)
    removed_list = head + removed_list[n3:]

    avs = ([(v, False, i) for i, v in enumerate(new_sol.trucks)] +
           [(v, True,  i) for i, v in enumerate(new_sol.drones)])

    last_key = None
    for cid in removed_list:
        cust = inst.customers[cid - 1]
        ibest_sol, ibest_score, ibest_key = None, float('inf'), None

        for (_, dst_drone, dst_vi) in avs:
            if dst_drone and not _drone_eligible(cust, inst):
                continue
            dv = new_sol.drones[dst_vi] if dst_drone else new_sol.trucks[dst_vi]

            for t_idx, trip in enumerate(dv.trips):
                for pos in range(1, len(trip.sequence)):
                    cand = new_sol.copy()
                    cv = cand.drones[dst_vi] if dst_drone else cand.trucks[dst_vi]
                    cv.trips[t_idx].sequence.insert(pos, cid)
                    cand.recompute_all(inst)
                    score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
                    key = ('rr', cid, dst_drone, dst_vi, t_idx, pos)
                    if score < ibest_score:
                        if not tabu.is_tabu(key, it) or score < best_obj:
                            ibest_score, ibest_sol, ibest_key = score, cand, key

            cand = new_sol.copy()
            cv = cand.drones[dst_vi] if dst_drone else cand.trucks[dst_vi]
            cv.trips.append(Trip(sequence=[0, cid, 0], is_drone=dst_drone))
            cand.recompute_all(inst)
            score = _obj(cand, inst, w_tw, w_cap, w_range, w_assign)
            key = ('rr', cid, dst_drone, dst_vi, -1, -1)
            if score < ibest_score:
                if not tabu.is_tabu(key, it) or score < best_obj:
                    ibest_score, ibest_sol, ibest_key = score, cand, key

        if ibest_sol is not None:
            new_sol = ibest_sol
            if ibest_key:
                tabu.add(ibest_key, it)
                last_key = ibest_key
        else:
            v0 = new_sol.trucks[0]
            v0.trips.append(Trip(sequence=[0, cid, 0], is_drone=False))
            new_sol.recompute_all(inst)

    _clean(new_sol, inst)
    score = _obj(new_sol, inst, w_tw, w_cap, w_range, w_assign)
    return new_sol, score, last_key


# ─────────────────────────────────────────────────────────────────────────────
# Vòng lặp chính — best-of-neighborhood
# Mỗi vòng lặp chạy TẤT CẢ 5 generator khai thác song song,
# lấy move có score thấp nhất trong toàn bộ pool.
# ─────────────────────────────────────────────────────────────────────────────

_GENERATORS = [
    ('Relocate',    _gen_relocate),
    ('Or-opt(2)',   _gen_or_opt2),
    ('Swap',        _gen_swap),
    ('2-opt',       _gen_2opt),
    ('Cross-trip',  _gen_cross_trip),
]


def _hard_ok(sol: Solution, inst: Instance) -> bool:
    """True nếu KHÔNG vi phạm ràng buộc cứng (tải trọng, tầm bay drone, time
    window). Từ khi construction.py đảm bảo TW-feasible tuyệt đối ngay từ
    điểm khởi tạo (mượn thêm phương tiện ảo khi cần), time window được coi
    là ràng buộc vật lý cứng giống tải trọng/tầm bay, không còn là ràng
    buộc mềm cần "dò đường qua vùng infeasible" như thiết kế trước đây."""
    wait_pen = sol.penalty_wait(inst) if hasattr(sol, 'penalty_wait') else 0.0
    return (sol.penalty_cap(inst)   <= 1e-9
            and sol.penalty_range(inst) <= 1e-9
            and sol.penalty_tw(inst)    <= 1e-9
            and wait_pen                <= 1e-9)


def _better_overall(cand: Solution, cand_obj: float, cand_hard_ok: bool,
                     incumbent: Solution, incumbent_obj: float, incumbent_hard_ok: bool) -> bool:
    """So sánh phân cấp (lexicographic) cho best_overall:
    1) Ưu tiên tuyệt đối nghiệm không vi phạm ràng buộc cứng.
    2) Trong cùng nhóm (cả 2 đều hard_ok hoặc đều không), so objective."""
    if cand_hard_ok != incumbent_hard_ok:
        return cand_hard_ok  # cand thắng nếu nó hard_ok còn incumbent không
    return cand_obj < incumbent_obj


def advanced_tabu_search(
    init_sol: Solution, inst: Instance, cfg: TabuSearchConfig
) -> Tuple[Solution, List[float]]:
    t_start = time.time()

    current = init_sol.copy()
    current.recompute_all(inst)
    best = current.copy()

    w_tw, w_cap, w_range, w_assign = 50.0, 200.0, 200.0, 500.0
    best_obj = _obj(best, inst, w_tw, w_cap, w_range, w_assign)

    # best_overall: theo dõi nghiệm có objective (gồm phạt) nhỏ nhất từng gặp,
    # BẤT KỂ feasible hay không. Dùng làm phương án dự phòng nếu suốt quá trình
    # không tìm được nghiệm feasible tuyệt đối nào (instance lớn / khó).
    # Dùng trọng số phạt CỐ ĐỊNH (giá trị khởi tạo) để các lần so sánh giữa các
    # vòng lặp có ý nghĩa nhất quán, không bị lệch do w_* thay đổi theo
    # strategic oscillation.
    W_TW0, W_CAP0, W_RANGE0, W_ASSIGN0 = w_tw, w_cap, w_range, w_assign
    best_overall = current.copy()
    best_overall_obj = _obj(best_overall, inst, W_TW0, W_CAP0, W_RANGE0, W_ASSIGN0)
    best_overall_hard_ok = _hard_ok(best_overall, inst)

    tabu = TabuSet(tenure=cfg.tenure_base)
    history = [best.makespan()]

    no_improve = 0
    feasible_streak = 0
    infeasible_streak = 0

    for it in range(1, cfg.max_iter + 1):
        if no_improve >= cfg.max_no_improve:
            if cfg.verbose:
                print(f"  -> Dừng sớm tại iter {it} (no_improve={no_improve})")
            break
        if time.time() - t_start > cfg.time_limit:
            if cfg.verbose:
                print(f"  -> Dừng do time_limit tại iter {it}")
            break

        nb_sol, nb_score, nb_key, nb_op = None, float('inf'), None, None

        # Mỗi 6 vòng hoặc gần ngưỡng dừng → thêm Ruin & Recreate
        use_rr = (it % 6 == 0) or (no_improve > cfg.max_no_improve // 2)

        if use_rr:
            rr_sol, rr_score, rr_key = op_ruin_recreate(
                current, inst, tabu, it, best_obj,
                w_tw, w_cap, w_range, w_assign)
            if rr_sol is not None and rr_score < nb_score:
                nb_sol, nb_score, nb_key, nb_op = rr_sol, rr_score, rr_key, 'Ruin&Recreate'

        # Chạy tất cả 5 generator khai thác, lấy move tốt nhất
        for op_name, gen_fn in _GENERATORS:
            for score, cand, key in gen_fn(
                    current, inst, tabu, it, best_obj,
                    w_tw, w_cap, w_range, w_assign):
                if score < nb_score:
                    nb_sol, nb_score, nb_key, nb_op = cand, score, key, op_name

        if nb_sol is None:
            no_improve += 1
            continue

        # Ghi tabu cho move khai thác được chọn
        if nb_key and nb_op != 'Ruin&Recreate':
            tabu.add(nb_key, it)
            _clean(nb_sol, inst)

        current = nb_sol

        # Cập nhật best_overall theo objective cố định (không điều kiện feasible).
        # Đây là lưới an toàn: nếu current đang cải thiện dần nhưng chưa kịp
        # chạm feasible tuyệt đối, ta vẫn giữ lại trạng thái tốt nhất đã thấy.
        cur_obj_fixed = _obj(current, inst, W_TW0, W_CAP0, W_RANGE0, W_ASSIGN0)
        cur_hard_ok = _hard_ok(current, inst)
        if _better_overall(current, cur_obj_fixed, cur_hard_ok,
                            best_overall, best_overall_obj, best_overall_hard_ok):
            best_overall_obj = cur_obj_fixed
            best_overall_hard_ok = cur_hard_ok
            best_overall = current.copy()

        # Strategic Oscillation
        #
        # LƯU Ý QUAN TRỌNG sau khi khóa cứng TW/Cap/Range (xem _obj): vì
        # construction.py giờ đảm bảo điểm khởi tạo luôn feasible tuyệt đối
        # về cả 3 ràng buộc này, và mọi toán tử lân cận đều áp dụng cùng
        # hard-penalty 1e6 trong _obj, nên trong PHẦN LỚN các vòng lặp,
        # current sẽ luôn feasible (penalty_cap = penalty_range = penalty_tw
        # = 0) — current.is_feasible() do đó gần như luôn True. Hệ quả:
        # nhánh "feasible_streak" dưới đây sẽ được kích hoạt liên tục, làm
        # w_cap/w_range/w_tw giảm dần một chiều về sàn (20.0/20.0/10.0) và
        # không bao giờ tăng lại — vì nhánh "infeasible_streak" hiếm khi xảy
        # ra. Điều này KHÔNG gây sai (vì hard-penalty 1e6 đã đảm nhiệm việc
        # chặn vi phạm, không phụ thuộc w_cap/w_range/w_tw nữa), nhưng phép
        # oscillation qua lại biên feasible/infeasible — vốn có ý nghĩa khi
        # các ràng buộc này còn LÀ MỀM — giờ chỉ còn tác dụng hình thức.
        # Giữ lại đoạn này (không xóa) để không phá vỡ cấu trúc vòng lặp và
        # vì w_assign vẫn là soft-constraint thực sự (drone phục vụ khách
        # C1 vẫn có thể xảy ra và cần phạt mềm), nhưng người đọc code cần
        # hiểu rằng w_cap/w_range/w_tw từ đây CHỈ còn vai trò phụ trợ, không
        # còn là cơ chế chính kiểm soát feasibility nữa.
        if current.is_feasible(inst):
            feasible_streak += 1
            infeasible_streak = 0
            if feasible_streak >= 8:
                w_cap   = max(20.0,   w_cap   * 0.85)
                w_range = max(20.0,   w_range * 0.85)
                w_tw    = max(10.0,   w_tw    * 0.85)
                feasible_streak = 0
        else:
            infeasible_streak += 1
            feasible_streak = 0
            if infeasible_streak >= 5:
                w_cap   = min(2000.0, w_cap   * 1.5)
                w_range = min(2000.0, w_range * 1.5)
                w_tw    = min(2000.0, w_tw    * 1.5)
                infeasible_streak = 0

        # Cập nhật best
        if current.is_feasible(inst) and current.all_served(inst):
            if current.makespan() < best.makespan() or not best.is_feasible(inst):
                best = current.copy()
                best_obj = _obj(best, inst, w_tw, w_cap, w_range, w_assign)
                no_improve = 0
                history.append(best.makespan())
                if cfg.verbose:
                    print(f"  [{it:4d}] ⭐ [{nb_op}] Makespan={best.makespan():.2f}"
                          f"  w_tw={w_tw:.0f}")
            else:
                no_improve += 1
        else:
            no_improve += 1

        if cfg.verbose and it % 100 == 0:
            print(f"  [{it:4d}] cur={current.makespan():.2f}"
                  f"  best={best.makespan():.2f}"
                  f"  feasible={current.is_feasible(inst)}"
                  f"  w_tw={w_tw:.1f}")

    # Nếu suốt quá trình KHÔNG tìm được nghiệm feasible tuyệt đối nào,
    # best vẫn còn nguyên là init_sol (construction). Trong trường hợp đó,
    # trả về best_overall — nghiệm có objective (gồm phạt) nhỏ nhất từng
    # gặp — vì nó luôn tốt hơn hoặc bằng construction, kể cả khi vẫn infeasible.
    if not best.is_feasible(inst) and best_overall_obj < best_obj:
        if cfg.verbose:
            print(f"  -> Không đạt feasible tuyệt đối; trả về best_overall "
                  f"(obj={best_overall_obj:.2f} < construction obj={best_obj:.2f})")
        best = best_overall
        history.append(best.makespan())

    return best, history