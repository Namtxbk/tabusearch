#pragma once
// tabu_search.h — Tabu Search cho MVRPD-TW
// Tương đương tabu_search.py
//
// Toán tử: Relocate, Or-opt(2), Swap, 2-opt, Cross-trip, Ruin&Recreate

#include "solution.h"
#include "instance.h"
#include <unordered_map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <iostream>
#include <functional>
#include <limits>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
struct TabuSearchConfig {
    int    max_iter       = 1000;
    int    max_no_improve = 200;
    int    tenure_base    = 7;
    double time_limit     = 60.0;
    bool   verbose        = true;
};

// ─────────────────────────────────────────────────────────────────────────────
// TabuSet — lưu trữ các move bị cấm
// Key dùng kiểu tuple mã hoá thành string để dùng hash map đơn giản
// ─────────────────────────────────────────────────────────────────────────────
struct TabuSet {
    int tenure;
    std::unordered_map<std::string, int> data;

    explicit TabuSet(int t = 7) : tenure(t) {}

    void add(const std::string& key, int iter) {
        data[key] = iter + tenure;
    }

    bool is_tabu(const std::string& key, int iter) const {
        auto it = data.find(key);
        return it != data.end() && it->second > iter;
    }
};

// Tạo key string từ các tham số
inline std::string make_key(const std::string& op, int a, int b=-1, int c=-1,
                             int d=-1, int e=-1, int f=-1) {
    std::string k = op;
    k += '|'; k += std::to_string(a);
    if (b != -999) { k += '|'; k += std::to_string(b); }
    if (c != -999) { k += '|'; k += std::to_string(c); }
    if (d != -999) { k += '|'; k += std::to_string(d); }
    if (e != -999) { k += '|'; k += std::to_string(e); }
    if (f != -999) { k += '|'; k += std::to_string(f); }
    return k;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hàm tiện ích
// ─────────────────────────────────────────────────────────────────────────────

// Hàm objective — hard penalty cho TW/cap/range, giống _obj() trong Python
inline double obj(const Solution& sol, const Instance& inst,
                  double w_tw, double w_cap, double w_range, double w_assign)
{
    double cap_pen   = sol.penalty_cap(inst);
    double range_pen = sol.penalty_range(inst);
    double tw_pen    = sol.penalty_tw(inst);

    const double HARD = 1e6;
    double hard = HARD * (cap_pen + range_pen + tw_pen);

    return sol.makespan()
         + w_cap   * cap_pen
         + w_range * range_pen
         + hard;
}

// Kiểm tra drone eligibility
inline bool drone_eligible_ts(int cid, const Instance& inst) {
    const auto& c = inst.node(cid);
    if (c.is_c1) return false;
    if (c.demand > inst.drone_capacity) return false;
    double rt = inst.travel_time(0, cid, true)
              + inst.travel_time(cid, 0, true);
    return rt <= inst.drone_range;
}

// Kiểm tra hard constraints (cap + range + TW) tất cả = 0
inline bool hard_ok(const Solution& sol, const Instance& inst) {
    return sol.penalty_cap(inst)   < 1e-9 &&
           sol.penalty_range(inst) < 1e-9 &&
           sol.penalty_tw(inst)    < 1e-9;
}

// So sánh phân cấp: ưu tiên hard_ok trước, rồi mới so objective
inline bool better_overall(double cand_obj, bool cand_ok,
                            double inc_obj,  bool inc_ok) {
    if (cand_ok != inc_ok) return cand_ok;
    return cand_obj < inc_obj;
}

// Cấu trúc lưu 1 candidate move
struct Move {
    Solution    sol;
    double      score = std::numeric_limits<double>::infinity();
    std::string key;
    std::string op;
};

// Tất cả xe (trucks + drones) với index và loại
struct VehicleRef {
    int  vi;        // index trong sol.trucks / sol.drones
    bool is_drone;
};

inline std::vector<VehicleRef> all_vehicles(const Solution& sol) {
    std::vector<VehicleRef> res;
    for (int i = 0; i < (int)sol.trucks.size(); ++i) res.push_back({i, false});
    for (int i = 0; i < (int)sol.drones.size();  ++i) res.push_back({i, true});
    return res;
}

// ─────────────────────────────────────────────────────────────────────────────
// Toán tử 1: Relocate
// ─────────────────────────────────────────────────────────────────────────────
inline void gen_relocate(const Solution& sol, const Instance& inst,
                         TabuSet& tabu, int it, double best_obj,
                         double w_tw, double w_cap, double w_range, double w_assign,
                         Move& best_move,
                         std::mt19937& rng)
{
    auto avs = all_vehicles(sol);

    // Thu thập danh sách tất cả khách
    struct CustPos { bool is_drone; int vi, ti, pos, cid; };
    std::vector<CustPos> all_custs;
    for (auto& vr : avs) {
        const auto& v = vr.is_drone ? sol.drones[vr.vi] : sol.trucks[vr.vi];
        for (int ti = 0; ti < (int)v.trips.size(); ++ti)
            for (int pos = 0; pos < (int)v.trips[ti].sequence.size(); ++pos) {
                int cid = v.trips[ti].sequence[pos];
                if (cid != 0) all_custs.push_back({vr.is_drone, vr.vi, ti, pos, cid});
            }
    }
    std::shuffle(all_custs.begin(), all_custs.end(), rng);
    if ((int)all_custs.size() > 30) all_custs.resize(30);

    for (auto& src : all_custs) {
        Solution tmp = sol.copy();
        Vehicle& sv = src.is_drone ? tmp.drones[src.vi] : tmp.trucks[src.vi];
        sv.trips[src.ti].sequence.erase(
            sv.trips[src.ti].sequence.begin() + src.pos);
        precompute_vehicle(sv, inst);

        const Customer& cust = inst.node(src.cid);

        for (auto& dst : avs) {
            if (dst.is_drone && !drone_eligible_ts(src.cid, inst)) continue;
            Vehicle& dv = dst.is_drone ? tmp.drones[dst.vi] : tmp.trucks[dst.vi];

            for (int dt = 0; dt < (int)dv.trips.size(); ++dt) {
                auto& dtrip = dv.trips[dt];
                int dsz = (int)dtrip.sequence.size();

                for (int ins = 1; ins < dsz; ++ins) {
                    // Forward time slack lọc nhanh
                    int prev_id = dtrip.sequence[ins-1];
                    int next_id = dtrip.sequence[ins];
                    double t_prev = dtrip.a[ins-1];
                    double s_prev = inst.node(prev_id).service;
                    double t_new_at_cid = std::max(cust.ready,
                        t_prev + s_prev + inst.travel_time(prev_id, src.cid, dst.is_drone));
                    if (t_new_at_cid > cust.due + 1e-9) continue;

                    double t_old_next = dtrip.a[ins];
                    double t_new_next = std::max(inst.node(next_id).ready,
                        t_new_at_cid + cust.service
                        + inst.travel_time(src.cid, next_id, dst.is_drone));
                    double delay = t_new_next - t_old_next;
                    // Slack check
                    if (delay > 1e-9 && ins < (int)dtrip.F.size()) {
                        if (delay > dtrip.F[ins] + 1e-9) continue;
                    }

                    Solution cand = tmp.copy();
                    Vehicle& cv = dst.is_drone ? cand.drones[dst.vi] : cand.trucks[dst.vi];
                    cv.trips[dt].sequence.insert(
                        cv.trips[dt].sequence.begin() + ins, src.cid);
                    cand.recompute_all(inst);
                    double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                    auto key = make_key("rel", src.cid, dst.is_drone?1:0,
                                        dst.vi, dt, ins);
                    if ((!tabu.is_tabu(key, it) || score < best_obj)
                            && score < best_move.score) {
                        best_move = {std::move(cand), score, key, "Relocate"};
                    }
                }

                // Mở trip mới cho xe đích
                {
                    Solution cand = tmp.copy();
                    Vehicle& cv = dst.is_drone ? cand.drones[dst.vi] : cand.trucks[dst.vi];
                    Trip nt({0, src.cid, 0}, dst.is_drone);
                    cv.trips.push_back(std::move(nt));
                    cand.recompute_all(inst);
                    double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                    auto key = make_key("rel", src.cid, dst.is_drone?1:0, dst.vi, -1, -1);
                    if ((!tabu.is_tabu(key, it) || score < best_obj)
                            && score < best_move.score) {
                        best_move = {std::move(cand), score, key, "Relocate"};
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toán tử 2: Or-opt(2)
// ─────────────────────────────────────────────────────────────────────────────
inline void gen_or_opt2(const Solution& sol, const Instance& inst,
                        TabuSet& tabu, int it, double best_obj,
                        double w_tw, double w_cap, double w_range, double w_assign,
                        Move& best_move,
                        std::mt19937& rng)
{
    auto avs = all_vehicles(sol);

    for (auto& src : avs) {
        const Vehicle& sv = src.is_drone ? sol.drones[src.vi] : sol.trucks[src.vi];
        for (int sti = 0; sti < (int)sv.trips.size(); ++sti) {
            const auto& seq = sv.trips[sti].sequence;
            int sz = (int)seq.size();
            if (sz < 4) continue;

            for (int pos = 1; pos < sz-2; ++pos) {
                int cid1 = seq[pos], cid2 = seq[pos+1];
                if (cid1 == 0 || cid2 == 0) continue;

                Solution tmp = sol.copy();
                Vehicle& tsv = src.is_drone ? tmp.drones[src.vi] : tmp.trucks[src.vi];
                tsv.trips[sti].sequence.erase(
                    tsv.trips[sti].sequence.begin() + pos,
                    tsv.trips[sti].sequence.begin() + pos + 2);

                for (auto& dst : avs) {
                    if (dst.is_drone && (!drone_eligible_ts(cid1, inst)
                                      || !drone_eligible_ts(cid2, inst))) continue;
                    Vehicle& dv = dst.is_drone ? tmp.drones[dst.vi] : tmp.trucks[dst.vi];

                    for (int dt = 0; dt < (int)dv.trips.size(); ++dt) {
                        int dsz = (int)dv.trips[dt].sequence.size();
                        for (int ins = 1; ins < dsz; ++ins) {
                            Solution cand = tmp.copy();
                            Vehicle& cv = dst.is_drone ? cand.drones[dst.vi]
                                                       : cand.trucks[dst.vi];
                            cv.trips[dt].sequence.insert(
                                cv.trips[dt].sequence.begin() + ins, cid2);
                            cv.trips[dt].sequence.insert(
                                cv.trips[dt].sequence.begin() + ins, cid1);
                            cand.recompute_all(inst);
                            double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                            auto key = make_key("or2", cid1, cid2,
                                                dst.is_drone?1:0, dst.vi, dt, ins);
                            if ((!tabu.is_tabu(key, it) || score < best_obj)
                                    && score < best_move.score)
                                best_move = {std::move(cand), score, key, "Or-opt(2)"};
                        }
                        // Mở trip mới
                        {
                            Solution cand = tmp.copy();
                            Vehicle& cv = dst.is_drone ? cand.drones[dst.vi]
                                                       : cand.trucks[dst.vi];
                            Trip nt({0, cid1, cid2, 0}, dst.is_drone);
                            cv.trips.push_back(std::move(nt));
                            cand.recompute_all(inst);
                            double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                            auto key = make_key("or2", cid1, cid2,
                                                dst.is_drone?1:0, dst.vi, -1, -1);
                            if ((!tabu.is_tabu(key, it) || score < best_obj)
                                    && score < best_move.score)
                                best_move = {std::move(cand), score, key, "Or-opt(2)"};
                        }
                    }
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toán tử 3: Swap
// ─────────────────────────────────────────────────────────────────────────────
inline void gen_swap(const Solution& sol, const Instance& inst,
                     TabuSet& tabu, int it, double best_obj,
                     double w_tw, double w_cap, double w_range, double w_assign,
                     Move& best_move,
                     std::mt19937& rng)
{
    auto avs = all_vehicles(sol);

    struct Pos { bool is_drone; int vi, ti, pos, cid; };
    std::vector<Pos> positions;
    for (auto& vr : avs) {
        const auto& v = vr.is_drone ? sol.drones[vr.vi] : sol.trucks[vr.vi];
        for (int ti = 0; ti < (int)v.trips.size(); ++ti)
            for (int pos = 0; pos < (int)v.trips[ti].sequence.size(); ++pos) {
                int cid = v.trips[ti].sequence[pos];
                if (cid != 0) positions.push_back({vr.is_drone, vr.vi, ti, pos, cid});
            }
    }
    std::shuffle(positions.begin(), positions.end(), rng);
    if ((int)positions.size() > 20) positions.resize(20);

    int n = (int)positions.size();
    for (int ia = 0; ia < n; ++ia)
    for (int ib = ia+1; ib < n; ++ib) {
        auto& a = positions[ia];
        auto& b = positions[ib];
        if (a.is_drone && !drone_eligible_ts(b.cid, inst)) continue;
        if (b.is_drone && !drone_eligible_ts(a.cid, inst)) continue;

        Solution cand = sol.copy();
        Vehicle& va = a.is_drone ? cand.drones[a.vi] : cand.trucks[a.vi];
        Vehicle& vb = b.is_drone ? cand.drones[b.vi] : cand.trucks[b.vi];
        va.trips[a.ti].sequence[a.pos] = b.cid;
        vb.trips[b.ti].sequence[b.pos] = a.cid;
        cand.recompute_all(inst);
        double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
        int ka = std::min(a.cid, b.cid), kb = std::max(a.cid, b.cid);
        auto key = make_key("swap", ka, kb);
        if ((!tabu.is_tabu(key, it) || score < best_obj) && score < best_move.score)
            best_move = {std::move(cand), score, key, "Swap"};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toán tử 4: 2-opt
// ─────────────────────────────────────────────────────────────────────────────
inline void gen_2opt(const Solution& sol, const Instance& inst,
                     TabuSet& tabu, int it, double best_obj,
                     double w_tw, double w_cap, double w_range, double w_assign,
                     Move& best_move,
                     std::mt19937& rng)
{
    auto avs = all_vehicles(sol);
    for (auto& vr : avs) {
        const Vehicle& v = vr.is_drone ? sol.drones[vr.vi] : sol.trucks[vr.vi];
        for (int ti = 0; ti < (int)v.trips.size(); ++ti) {
            int n = (int)v.trips[ti].sequence.size();
            if (n < 5) continue;
            for (int i = 1; i < n-2; ++i)
            for (int j = i+1; j < n-1; ++j) {
                Solution cand = sol.copy();
                Vehicle& cv = vr.is_drone ? cand.drones[vr.vi] : cand.trucks[vr.vi];
                std::reverse(
                    cv.trips[ti].sequence.begin() + i,
                    cv.trips[ti].sequence.begin() + j + 1);
                cand.recompute_all(inst);
                double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                auto key = make_key("2opt", vr.vi, ti, i, j);
                if ((!tabu.is_tabu(key, it) || score < best_obj) && score < best_move.score)
                    best_move = {std::move(cand), score, key, "2-opt"};
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toán tử 5: Cross-trip
// ─────────────────────────────────────────────────────────────────────────────
inline void gen_cross_trip(const Solution& sol, const Instance& inst,
                           TabuSet& tabu, int it, double best_obj,
                           double w_tw, double w_cap, double w_range, double w_assign,
                           Move& best_move,
                           std::mt19937& rng)
{
    auto avs = all_vehicles(sol);
    for (auto& vr : avs) {
        const Vehicle& v = vr.is_drone ? sol.drones[vr.vi] : sol.trucks[vr.vi];
        int nt = (int)v.trips.size();
        if (nt < 2) continue;

        for (int ta = 0; ta < nt; ++ta)
        for (int tb = ta+1; tb < nt; ++tb) {
            const auto& sa = v.trips[ta].sequence;
            const auto& sb = v.trips[tb].sequence;

            for (int ca = 1; ca < (int)sa.size()-1; ++ca)
            for (int cb = 1; cb < (int)sb.size()-1; ++cb) {
                // Kiểm tra nếu drone: không chở C1
                if (vr.is_drone) {
                    bool ok = true;
                    for (int k = ca; k < (int)sa.size()-1 && ok; ++k)
                        if (inst.node(sa[k]).is_c1) ok = false;
                    for (int k = cb; k < (int)sb.size()-1 && ok; ++k)
                        if (inst.node(sb[k]).is_c1) ok = false;
                    if (!ok) continue;
                }

                Solution cand = sol.copy();
                Vehicle& cv = vr.is_drone ? cand.drones[vr.vi] : cand.trucks[vr.vi];
                // tail_a = sa[ca..end-1], tail_b = sb[cb..end-1]
                std::vector<int> tail_a(sa.begin()+ca, sa.end()-1);
                std::vector<int> tail_b(sb.begin()+cb, sb.end()-1);

                // Xây dựng lại sequence
                cv.trips[ta].sequence.resize(ca);
                cv.trips[ta].sequence.insert(
                    cv.trips[ta].sequence.end(), tail_b.begin(), tail_b.end());
                cv.trips[ta].sequence.push_back(0);

                cv.trips[tb].sequence.resize(cb);
                cv.trips[tb].sequence.insert(
                    cv.trips[tb].sequence.end(), tail_a.begin(), tail_a.end());
                cv.trips[tb].sequence.push_back(0);

                if (cv.trips[ta].sequence == std::vector<int>{0,0} ||
                    cv.trips[tb].sequence == std::vector<int>{0,0}) continue;

                cand.recompute_all(inst);
                double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                auto key = make_key("cross", vr.vi, ta, tb, ca, cb);
                if ((!tabu.is_tabu(key, it) || score < best_obj) && score < best_move.score)
                    best_move = {std::move(cand), score, key, "Cross-trip"};
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Toán tử 6: Ruin & Recreate
// ─────────────────────────────────────────────────────────────────────────────
inline Move op_ruin_recreate(const Solution& sol, const Instance& inst,
                             TabuSet& tabu, int it, double best_obj,
                             double w_tw, double w_cap, double w_range, double w_assign,
                             std::mt19937& rng)
{
    std::vector<int> all_ids;
    for (auto& c : inst.customers) all_ids.push_back(c.id);

    std::uniform_real_distribution<> dist_pct(0.15, 0.30);
    int num_remove = std::max(2, (int)(all_ids.size() * dist_pct(rng)));
    num_remove = std::min(num_remove, (int)all_ids.size());

    std::shuffle(all_ids.begin(), all_ids.end(), rng);
    std::unordered_set<int> removed(all_ids.begin(), all_ids.begin() + num_remove);

    Solution new_sol = sol.copy();
    for (auto& v : new_sol.trucks)
        for (auto& t : v.trips) {
            t.sequence.erase(
                std::remove_if(t.sequence.begin(), t.sequence.end(),
                    [&](int n){ return removed.count(n) > 0; }),
                t.sequence.end());
        }
    for (auto& v : new_sol.drones)
        for (auto& t : v.trips) {
            t.sequence.erase(
                std::remove_if(t.sequence.begin(), t.sequence.end(),
                    [&](int n){ return removed.count(n) > 0; }),
                t.sequence.end());
        }

    // Sắp xếp danh sách removed theo due tăng dần, shuffle phần đầu
    std::vector<int> rem_list(removed.begin(), removed.end());
    std::sort(rem_list.begin(), rem_list.end(), [&](int a, int b){
        return inst.node(a).due < inst.node(b).due;
    });
    int n3 = std::max(1, (int)rem_list.size() / 3);
    std::shuffle(rem_list.begin(), rem_list.begin() + n3, rng);

    auto avs_new = all_vehicles(new_sol);
    std::string last_key;

    for (int cid : rem_list) {
        const Customer& cust = inst.node(cid);
        double ibest_score = std::numeric_limits<double>::infinity();
        Solution ibest_sol;
        std::string ibest_key;
        bool found = false;

        for (auto& dst : avs_new) {
            if (dst.is_drone && !drone_eligible_ts(cid, inst)) continue;
            Vehicle& dv = dst.is_drone ? new_sol.drones[dst.vi] : new_sol.trucks[dst.vi];

            for (int dt = 0; dt < (int)dv.trips.size(); ++dt) {
                int dsz = (int)dv.trips[dt].sequence.size();
                for (int ins = 1; ins < dsz; ++ins) {
                    Solution cand = new_sol.copy();
                    Vehicle& cv = dst.is_drone ? cand.drones[dst.vi] : cand.trucks[dst.vi];
                    cv.trips[dt].sequence.insert(
                        cv.trips[dt].sequence.begin() + ins, cid);
                    cand.recompute_all(inst);
                    double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                    auto key = make_key("rr", cid, dst.is_drone?1:0, dst.vi, dt, ins);
                    if (score < ibest_score && (!tabu.is_tabu(key, it) || score < best_obj)) {
                        ibest_score = score;
                        ibest_sol = std::move(cand);
                        ibest_key = key;
                        found = true;
                    }
                }
            }
            // Mở trip mới
            {
                Solution cand = new_sol.copy();
                Vehicle& cv = dst.is_drone ? cand.drones[dst.vi] : cand.trucks[dst.vi];
                Trip nt({0, cid, 0}, dst.is_drone);
                cv.trips.push_back(std::move(nt));
                cand.recompute_all(inst);
                double score = obj(cand, inst, w_tw, w_cap, w_range, w_assign);
                auto key = make_key("rr", cid, dst.is_drone?1:0, dst.vi, -1, -1);
                if (score < ibest_score && (!tabu.is_tabu(key, it) || score < best_obj)) {
                    ibest_score = score;
                    ibest_sol = std::move(cand);
                    ibest_key = key;
                    found = true;
                }
            }
        }

        if (found) {
            new_sol = ibest_sol;
            if (!ibest_key.empty()) { tabu.add(ibest_key, it); last_key = ibest_key; }
        } else {
            new_sol.trucks[0].trips.push_back(Trip({0, cid, 0}, false));
            new_sol.recompute_all(inst);
        }
    }

    new_sol.clean(inst);
    double score = obj(new_sol, inst, w_tw, w_cap, w_range, w_assign);
    return {std::move(new_sol), score, last_key, "Ruin&Recreate"};
}

// ─────────────────────────────────────────────────────────────────────────────
// Vòng lặp chính: advanced_tabu_search
// ─────────────────────────────────────────────────────────────────────────────
struct TSResult {
    Solution          best;
    std::vector<double> history;
};

inline TSResult advanced_tabu_search(Solution init_sol, const Instance& inst,
                                     const TabuSearchConfig& cfg)
{
    auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&](){
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
    };

    std::mt19937 rng(std::random_device{}());

    Solution current = init_sol.copy();
    current.recompute_all(inst);
    Solution best = current.copy();

    double w_tw = 50.0, w_cap = 200.0, w_range = 200.0, w_assign = 500.0;
    double best_obj_val = obj(best, inst, w_tw, w_cap, w_range, w_assign);

    double W_TW0 = w_tw, W_CAP0 = w_cap, W_RANGE0 = w_range, W_ASSIGN0 = w_assign;
    Solution best_overall = current.copy();
    double best_overall_obj = obj(best_overall, inst, W_TW0, W_CAP0, W_RANGE0, W_ASSIGN0);
    bool best_overall_hard_ok = hard_ok(best_overall, inst);

    TabuSet tabu(cfg.tenure_base);
    std::vector<double> history = { best.makespan() };

    int no_improve = 0, feasible_streak = 0, infeasible_streak = 0;

    for (int it = 1; it <= cfg.max_iter; ++it) {
        if (no_improve >= cfg.max_no_improve) {
            if (cfg.verbose) std::cout << "  -> Dừng sớm tại iter " << it << "\n";
            break;
        }
        if (elapsed() > cfg.time_limit) {
            if (cfg.verbose) std::cout << "  -> Dừng do time_limit tại iter " << it << "\n";
            break;
        }

        Move nb_move;
        nb_move.score = std::numeric_limits<double>::infinity();

        // Ruin & Recreate mỗi 6 vòng hoặc gần ngưỡng dừng
        bool use_rr = (it % 6 == 0) || (no_improve > cfg.max_no_improve / 2);
        if (use_rr) {
            Move rr = op_ruin_recreate(current, inst, tabu, it, best_obj_val,
                                       w_tw, w_cap, w_range, w_assign, rng);
            if (rr.score < nb_move.score) nb_move = std::move(rr);
        }

        // 5 toán tử khai thác
        gen_relocate  (current, inst, tabu, it, best_obj_val, w_tw, w_cap, w_range, w_assign, nb_move, rng);
        gen_or_opt2   (current, inst, tabu, it, best_obj_val, w_tw, w_cap, w_range, w_assign, nb_move, rng);
        gen_swap      (current, inst, tabu, it, best_obj_val, w_tw, w_cap, w_range, w_assign, nb_move, rng);
        gen_2opt      (current, inst, tabu, it, best_obj_val, w_tw, w_cap, w_range, w_assign, nb_move, rng);
        gen_cross_trip(current, inst, tabu, it, best_obj_val, w_tw, w_cap, w_range, w_assign, nb_move, rng);

        if (nb_move.sol.trucks.empty() && nb_move.sol.drones.empty()) {
            ++no_improve; continue;
        }

        if (!nb_move.key.empty() && nb_move.op != "Ruin&Recreate") {
            tabu.add(nb_move.key, it);
            nb_move.sol.clean(inst);
        }
        current = std::move(nb_move.sol);

        // Cập nhật best_overall
        double cur_obj_fixed = obj(current, inst, W_TW0, W_CAP0, W_RANGE0, W_ASSIGN0);
        bool cur_hard_ok = hard_ok(current, inst);
        if (better_overall(cur_obj_fixed, cur_hard_ok, best_overall_obj, best_overall_hard_ok)) {
            best_overall_obj = cur_obj_fixed;
            best_overall_hard_ok = cur_hard_ok;
            best_overall = current.copy();
        }

        // Strategic Oscillation (giữ nguyên cấu trúc như Python)
        if (current.is_feasible(inst)) {
            ++feasible_streak; infeasible_streak = 0;
            if (feasible_streak >= 8) {
                w_cap   = std::max(20.0,   w_cap   * 0.85);
                w_range = std::max(20.0,   w_range * 0.85);
                w_tw    = std::max(10.0,   w_tw    * 0.85);
                feasible_streak = 0;
            }
        } else {
            ++infeasible_streak; feasible_streak = 0;
            if (infeasible_streak >= 5) {
                w_cap   = std::min(2000.0, w_cap   * 1.5);
                w_range = std::min(2000.0, w_range * 1.5);
                w_tw    = std::min(2000.0, w_tw    * 1.5);
                infeasible_streak = 0;
            }
        }

        // Cập nhật best
        if (current.is_feasible(inst) && current.all_served(inst)) {
            if (current.makespan() < best.makespan() || !best.is_feasible(inst)) {
                best = current.copy();
                best_obj_val = obj(best, inst, w_tw, w_cap, w_range, w_assign);
                no_improve = 0;
                history.push_back(best.makespan());
                if (cfg.verbose)
                    std::cout << "  [" << it << "] * [" << nb_move.op << "]"
                              << " Makespan=" << best.makespan() << "\n";
            } else ++no_improve;
        } else ++no_improve;

        if (cfg.verbose && it % 100 == 0)
            std::cout << "  [" << it << "] cur=" << current.makespan()
                      << " best=" << best.makespan()
                      << " feasible=" << current.is_feasible(inst) << "\n";
    }

    // Fallback sang best_overall nếu chưa từng đạt feasible
    if (!best.is_feasible(inst) && best_overall_obj < best_obj_val) {
        if (cfg.verbose) std::cout << "  -> Trả về best_overall\n";
        best = best_overall;
        history.push_back(best.makespan());
    }

    return { std::move(best), std::move(history) };
}
