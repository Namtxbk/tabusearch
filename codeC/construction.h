#pragma once
// construction.h — Greedy Insertion cho MVRPD-TW Multi-Trip
// Tương đương construction.py

#include "solution.h"
#include "instance.h"
#include <algorithm>
#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// Kiểm tra 1 trip có hợp lệ không (tải, TW, pin drone, phân công drone)
// ─────────────────────────────────────────────────────────────────────────────
inline bool trip_feasible(const Trip& trip, const Instance& inst, bool is_drone) {
    // 1. Tải trọng
    double cap = is_drone ? inst.drone_capacity : inst.truck_capacity;
    if (trip.total_load > cap + 1e-9) return false;

    // 2. Time windows
    for (int i = 0; i < (int)trip.sequence.size(); ++i) {
        int nid = trip.sequence[i];
        if (nid == 0) continue;
        if (i >= (int)trip.a.size()) return false;
        if (trip.a[i] > inst.node(nid).due + 1e-9) return false;
    }

    // 3. Tầm bay drone
    if (is_drone) {
        double ft = 0.0;
        for (int k = 0; k < (int)trip.sequence.size()-1; ++k)
            ft += inst.travel_time(trip.sequence[k], trip.sequence[k+1], true);
        if (ft > inst.drone_range + 1e-9) return false;
    }

    // 4. Drone không được phục vụ khách C1
    if (is_drone)
        for (int nid : trip.sequence)
            if (nid != 0 && inst.c1_ids.count(nid)) return false;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kiểm tra xem khách cust_id có thể đi drone không (điều kiện sơ bộ)
// ─────────────────────────────────────────────────────────────────────────────
inline bool drone_eligible(int cust_id, const Instance& inst) {
    const auto& c = inst.node(cust_id);
    if (c.is_c1) return false;
    if (c.demand > inst.drone_capacity) return false;
    double rt = inst.travel_time(0, cust_id, true)
              + inst.travel_time(cust_id, 0, true);
    return rt <= inst.drone_range;
}

// ─────────────────────────────────────────────────────────────────────────────
// Thử chèn khách cust_id vào vị trí tốt nhất trong trip (best-insertion).
// Trả về true nếu thành công, trip được cập nhật in-place.
// ─────────────────────────────────────────────────────────────────────────────
inline bool try_insert_into_trip(Trip& trip, int cust_id,
                                  const Instance& inst, bool is_drone) {
    int best_pos  = -1;
    double best_cost = 1e18;

    int sz = (int)trip.sequence.size();
    for (int pos = 1; pos < sz; ++pos) {
        trip.sequence.insert(trip.sequence.begin() + pos, cust_id);
        precompute_trip(trip, inst);
        if (trip_feasible(trip, inst, is_drone)) {
            double cost = trip.return_time;
            if (cost < best_cost) { best_cost = cost; best_pos = pos; }
        }
        trip.sequence.erase(trip.sequence.begin() + pos);
        precompute_trip(trip, inst);
    }

    if (best_pos >= 0) {
        trip.sequence.insert(trip.sequence.begin() + best_pos, cust_id);
        precompute_trip(trip, inst);
        return true;
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_initial_solution — Greedy Insertion
// Tương đương build_initial_solution() trong construction.py
// ─────────────────────────────────────────────────────────────────────────────
inline Solution build_initial_solution(const Instance& inst) {
    // Sắp xếp khách theo due tăng dần ("khách gấp nhất phục vụ trước")
    std::vector<int> order(inst.customers.size());
    std::iota(order.begin(), order.end(), 1);  // id 1..n
    std::sort(order.begin(), order.end(), [&](int a, int b){
        return inst.node(a).due < inst.node(b).due;
    });

    std::vector<Vehicle> trucks(inst.num_trucks, Vehicle(false));
    std::vector<Vehicle> drones(inst.num_drones, Vehicle(true));

    // Khởi tạo 1 trip rỗng [0,0] cho mỗi phương tiện
    for (auto& v : trucks) {
        Trip t({0,0}, false);
        precompute_trip(t, inst);
        v.trips.push_back(std::move(t));
    }
    for (auto& v : drones) {
        Trip t({0,0}, true);
        precompute_trip(t, inst);
        v.trips.push_back(std::move(t));
    }

    for (int cid : order) {
        bool inserted = false;

        // ── Bước 1: Thử drone ─────────────────────────────────────────────
        if (drone_eligible(cid, inst)) {
            for (auto& v : drones) {
                // Chèn vào trip cuối cùng đang có
                if (try_insert_into_trip(v.trips.back(), cid, inst, true)) {
                    inserted = true; break;
                }
                // Mở trip mới cho drone này
                Trip nt({0, cid, 0}, true, v.trips.back().return_time);
                precompute_trip(nt, inst);
                if (trip_feasible(nt, inst, true)) {
                    v.trips.push_back(std::move(nt));
                    inserted = true; break;
                }
            }
        }
        if (inserted) continue;

        // ── Bước 2: Thử truck ─────────────────────────────────────────────
        for (auto& v : trucks) {
            if (try_insert_into_trip(v.trips.back(), cid, inst, false)) {
                inserted = true; break;
            }
            Trip nt({0, cid, 0}, false, v.trips.back().return_time);
            precompute_trip(nt, inst);
            if (trip_feasible(nt, inst, false)) {
                v.trips.push_back(std::move(nt));
                inserted = true; break;
            }
        }
        if (inserted) continue;

        // ── Fallback: đảm bảo TW-feasible tuyệt đối bằng phương tiện ảo ──
        // Ưu tiên drone ảo trước (nhất quán với Bước 1/2, và drone nhanh hơn)
        if (!inserted && drone_eligible(cid, inst)) {
            Vehicle ev(true);
            Trip et({0, cid, 0}, true, 0.0);
            precompute_trip(et, inst);
            if (trip_feasible(et, inst, true)) {
                ev.trips.push_back(std::move(et));
                drones.push_back(std::move(ev));
                inserted = true;
            }
        }

        if (!inserted) {
            Vehicle ev(false);
            Trip et({0, cid, 0}, false, 0.0);
            precompute_trip(et, inst);
            if (!trip_feasible(et, inst, false)) {
                throw std::runtime_error(
                    "Khách hàng id=" + std::to_string(cid) +
                    " (due=" + std::to_string(inst.node(cid).due) +
                    ") không thể phục vụ đúng hạn bởi bất kỳ phương tiện nào "
                    "(giới hạn vật lý của instance).");
            }
            ev.trips.push_back(std::move(et));
            trucks.push_back(std::move(ev));
            inserted = true;
        }
    }

    // Dọn trip rỗng và precompute lại
    for (auto& v : trucks) {
        v.trips.erase(
            std::remove_if(v.trips.begin(), v.trips.end(),
                [](const Trip& t){ return t.num_customers() == 0; }),
            v.trips.end());
        if (v.trips.empty()) v.trips.push_back(Trip({0,0}, false));
        precompute_vehicle(v, inst);
    }
    for (auto& v : drones) {
        v.trips.erase(
            std::remove_if(v.trips.begin(), v.trips.end(),
                [](const Trip& t){ return t.num_customers() == 0; }),
            v.trips.end());
        if (v.trips.empty()) v.trips.push_back(Trip({0,0}, true));
        precompute_vehicle(v, inst);
    }

    Solution sol;
    sol.trucks = std::move(trucks);
    sol.drones = std::move(drones);
    sol.extra_trucks_used = std::max(0, (int)sol.trucks.size() - inst.num_trucks);
    sol.extra_drones_used = std::max(0, (int)sol.drones.size() - inst.num_drones);
    return sol;
}
