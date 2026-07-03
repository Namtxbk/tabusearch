#pragma once
// solution.h — Trip, Vehicle, Solution cho MVRPD-TW Multi-Trip
// Tương đương solution.py

#include "instance.h"
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <sstream>
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Trip — 1 chuyến đi xuất phát và quay về depot
// ─────────────────────────────────────────────────────────────────────────────
struct Trip {
    std::vector<int>    sequence;    // [0, c1, c2, ..., 0]
    bool                is_drone   = false;
    double              start_time = 0.0;

    // Precomputed
    std::vector<double> a;            // arrival time tại mỗi node
    std::vector<double> F;            // Forward Time Slack
    std::vector<double> prefix_load;
    std::vector<double> suffix_load;
    double total_load  = 0.0;
    double total_dist  = 0.0;
    double return_time = 0.0;

    Trip() : sequence({0, 0}) {}

    Trip(std::vector<int> seq, bool drone, double st = 0.0)
        : sequence(std::move(seq)), is_drone(drone), start_time(st) {}

    // Trả về danh sách khách (bỏ depot đầu/cuối)
    std::vector<int> customers() const {
        if (sequence.size() < 2) return {};
        return std::vector<int>(sequence.begin() + 1, sequence.end() - 1);
    }

    int num_customers() const {
        return (int)sequence.size() - 2;
    }

    Trip copy() const { return *this; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Vehicle — 1 phương tiện gồm nhiều trips tuần tự
// ─────────────────────────────────────────────────────────────────────────────
struct Vehicle {
    bool              is_drone = false;
    std::vector<Trip> trips;

    Vehicle() = default;
    explicit Vehicle(bool drone) : is_drone(drone) {}

    double finish_time() const {
        return trips.empty() ? 0.0 : trips.back().return_time;
    }

    std::vector<int> all_customers() const {
        std::vector<int> res;
        for (auto& t : trips)
            for (int c : t.customers())
                res.push_back(c);
        return res;
    }

    Vehicle copy() const { return *this; }
};

// ─────────────────────────────────────────────────────────────────────────────
// precompute_trip — tính a[], F[], load, dist, return_time cho 1 trip
// ─────────────────────────────────────────────────────────────────────────────
inline void precompute_trip(Trip& trip, const Instance& inst) {
    auto& seq = trip.sequence;
    int n = (int)seq.size();
    bool drone = trip.is_drone;

    // ── Arrival time ────────────────────────────────────────────────────────
    trip.a.resize(n);
    trip.a[0] = trip.start_time;
    for (int i = 1; i < n; ++i) {
        int prev = seq[i-1], curr = seq[i];
        double t_travel = inst.travel_time(prev, curr, drone);
        double s_prev   = inst.node(prev).service;
        double arrive   = trip.a[i-1] + s_prev + t_travel;
        trip.a[i] = std::max(arrive, inst.node(curr).ready);
    }

    // ── Forward Time Slack ──────────────────────────────────────────────────
    trip.F.resize(n);
    trip.F[n-1] = inst.node(seq[n-1]).due - trip.a[n-1];
    for (int i = n-2; i >= 0; --i) {
        int curr = seq[i], nxt = seq[i+1];
        double s_i    = inst.node(curr).service;
        double t_nxt  = inst.travel_time(curr, nxt, drone);
        double wait   = std::max(0.0, inst.node(nxt).ready - (trip.a[i] + s_i + t_nxt));
        trip.F[i] = std::min(inst.node(curr).due - trip.a[i], trip.F[i+1] - wait);
    }

    // ── Prefix / Suffix load ────────────────────────────────────────────────
    trip.prefix_load.assign(n, 0.0);
    trip.suffix_load.assign(n, 0.0);
    for (int i = 1; i < n-1; ++i)
        trip.prefix_load[i] = trip.prefix_load[i-1] + inst.node(seq[i]).demand;
    for (int i = n-2; i >= 1; --i)
        trip.suffix_load[i] = trip.suffix_load[i+1] + inst.node(seq[i]).demand;

    trip.total_load  = (n > 2) ? trip.prefix_load[n-2] : 0.0;
    trip.return_time = trip.a[n-1];

    double d = 0.0;
    for (int i = 0; i < n-1; ++i)
        d += inst.dist(seq[i], seq[i+1]);
    trip.total_dist = d;
}

// ─────────────────────────────────────────────────────────────────────────────
// precompute_vehicle — precompute toàn bộ trips tuần tự của 1 xe
// ─────────────────────────────────────────────────────────────────────────────
inline void precompute_vehicle(Vehicle& v, const Instance& inst) {
    double t = 0.0;
    for (auto& trip : v.trips) {
        trip.start_time = t;
        precompute_trip(trip, inst);
        t = trip.return_time;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Solution
// ─────────────────────────────────────────────────────────────────────────────
struct Solution {
    std::vector<Vehicle> trucks;
    std::vector<Vehicle> drones;

    // Thêm thông tin báo cáo số phương tiện ảo (tương đương extra_trucks_used)
    int extra_trucks_used = 0;
    int extra_drones_used = 0;

    Solution() = default;

    Solution copy() const { return *this; }

    // Tất cả trips phẳng
    std::vector<Trip*> truck_routes() {
        std::vector<Trip*> res;
        for (auto& v : trucks) for (auto& t : v.trips) res.push_back(&t);
        return res;
    }
    std::vector<const Trip*> truck_routes() const {
        std::vector<const Trip*> res;
        for (auto& v : trucks) for (auto& t : v.trips) res.push_back(&t);
        return res;
    }
    std::vector<Trip*> drone_routes() {
        std::vector<Trip*> res;
        for (auto& v : drones) for (auto& t : v.trips) res.push_back(&t);
        return res;
    }
    std::vector<const Trip*> drone_routes() const {
        std::vector<const Trip*> res;
        for (auto& v : drones) for (auto& t : v.trips) res.push_back(&t);
        return res;
    }

    double makespan() const {
        double ms = 0.0;
        for (auto& v : trucks) ms = std::max(ms, v.finish_time());
        for (auto& v : drones) ms = std::max(ms, v.finish_time());
        return ms;
    }

    double penalty_tw(const Instance& inst) const {
        double total = 0.0;
        for (auto* t : truck_routes())
            for (int pos = 0; pos < (int)t->sequence.size(); ++pos) {
                int nid = t->sequence[pos];
                if (nid == 0) continue;
                total += std::max(0.0, t->a[pos] - inst.node(nid).due);
            }
        for (auto* t : drone_routes())
            for (int pos = 0; pos < (int)t->sequence.size(); ++pos) {
                int nid = t->sequence[pos];
                if (nid == 0) continue;
                total += std::max(0.0, t->a[pos] - inst.node(nid).due);
            }
        return total;
    }

    double penalty_cap(const Instance& inst) const {
        double total = 0.0;
        for (auto* t : truck_routes())
            total += std::max(0.0, t->total_load - inst.truck_capacity);
        for (auto* t : drone_routes())
            total += std::max(0.0, t->total_load - inst.drone_capacity);
        return total;
    }

    double penalty_range(const Instance& inst) const {
        double total = 0.0;
        for (auto* t : drone_routes()) {
            double ft = 0.0;
            for (int i = 0; i < (int)t->sequence.size()-1; ++i)
                ft += inst.travel_time(t->sequence[i], t->sequence[i+1], true);
            total += std::max(0.0, ft - inst.drone_range);
        }
        return total;
    }

    bool is_feasible(const Instance& inst) const {
        return penalty_tw(inst)    < 1e-9 &&
               penalty_cap(inst)   < 1e-9 &&
               penalty_range(inst) < 1e-9;
    }

    bool all_served(const Instance& inst) const {
        std::unordered_set<int> served;
        for (auto& v : trucks)
            for (auto& t : v.trips)
                for (int nid : t.sequence)
                    if (nid != 0) {
                        if (served.count(nid)) return false;
                        served.insert(nid);
                    }
        for (auto& v : drones)
            for (auto& t : v.trips)
                for (int nid : t.sequence)
                    if (nid != 0) {
                        if (served.count(nid)) return false;
                        served.insert(nid);
                    }
        int n = (int)inst.customers.size();
        for (int i = 1; i <= n; ++i)
            if (!served.count(i)) return false;
        return true;
    }

    // Tính lại toàn bộ sau khi sửa sequence bất kỳ
    void recompute_all(const Instance& inst) {
        for (auto& v : trucks) precompute_vehicle(v, inst);
        for (auto& v : drones) precompute_vehicle(v, inst);
    }

    // Dọn trip rỗng (chỉ còn [0,0]) sau mỗi toán tử
    void clean(const Instance& inst) {
        for (auto& v : trucks) {
            v.trips.erase(
                std::remove_if(v.trips.begin(), v.trips.end(),
                    [](const Trip& t){ return t.num_customers() == 0; }),
                v.trips.end());
            if (v.trips.empty())
                v.trips.push_back(Trip({0,0}, false));
            precompute_vehicle(v, inst);
        }
        for (auto& v : drones) {
            v.trips.erase(
                std::remove_if(v.trips.begin(), v.trips.end(),
                    [](const Trip& t){ return t.num_customers() == 0; }),
                v.trips.end());
            if (v.trips.empty())
                v.trips.push_back(Trip({0,0}, true));
            precompute_vehicle(v, inst);
        }
    }

    std::string summary(const Instance& inst) const {
        std::ostringstream os;
        os << "Instance     : " << inst.name << "\n"
           << "Makespan     : " << makespan() << "\n"
           << "Feasible     : " << (is_feasible(inst) ? "True" : "False") << "\n"
           << "All served   : " << (all_served(inst) ? "True" : "False") << "\n"
           << "Penalty TW   : " << penalty_tw(inst) << "\n"
           << "Penalty cap  : " << penalty_cap(inst) << "\n"
           << "Penalty range: " << penalty_range(inst) << "\n";
        for (int k = 0; k < (int)trucks.size(); ++k)
            for (int ti = 0; ti < (int)trucks[k].trips.size(); ++ti) {
                auto& t = trucks[k].trips[ti];
                if (t.num_customers() == 0) continue;
                os << "  Truck " << k+1 << " Trip " << ti+1 << ": 0";
                for (int c : t.customers()) os << " -> " << c;
                os << " -> 0"
                   << "  (load=" << t.total_load << "/" << inst.truck_capacity
                   << ", start=" << t.start_time
                   << ", return=" << t.return_time << ")\n";
            }
        for (int k = 0; k < (int)drones.size(); ++k)
            for (int ti = 0; ti < (int)drones[k].trips.size(); ++ti) {
                auto& t = drones[k].trips[ti];
                if (t.num_customers() == 0) continue;
                os << "  Drone " << k+1 << " Trip " << ti+1 << ": 0";
                for (int c : t.customers()) os << " -> " << c;
                os << " -> 0"
                   << "  (load=" << t.total_load << "/" << inst.drone_capacity
                   << ", dist=" << t.total_dist << "/" << inst.drone_range
                   << ", start=" << t.start_time
                   << ", return=" << t.return_time << ")\n";
            }
        return os.str();
    }
};
