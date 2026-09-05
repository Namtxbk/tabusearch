// select_components.hpp
// Mục 7: Chọn candidate (khách hàng / trip) để giảm thời gian tính.
#pragma once

#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include "instance.hpp"
#include "solution.hpp"

struct SearchComponents {
    std::vector<int> selectedCustomers;                 // id khách hàng (1-based)
    std::vector<std::pair<int,int>> selectedTrips;       // (vehicleIdx, tripIdx)
};

// Đóng góp của 1 khách vào V_TW, dùng để rank khi chưa có nghiệm khả thi.
inline double customerTWContribution(const Instance& inst, const Solution& s, int custId) {
    for (const auto& v : s.vehicles) {
        for (const auto& t : v.trips) {
            auto it = std::find(t.customers.begin(), t.customers.end(), custId);
            if (it != t.customers.end()) {
                double arrival = t.arrivalTime.at(custId);
                double due = inst.node(custId).due;
                double tw = std::max(0.0, arrival - due);
                double wait = std::max(0.0, (t.returnTime - arrival) - inst.max_wait);
                return tw + wait;
            }
        }
    }
    return 0.0;
}

inline std::vector<std::pair<int,int>> tripsContainingCustomers(const Solution& s, const std::vector<int>& custIds) {
    std::vector<std::pair<int,int>> result;
    std::set<std::pair<int,int>> seen;
    for (int cid : custIds) {
        for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
            const Vehicle& v = s.vehicles[vi];
            for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
                const auto& t = v.trips[ti];
                if (std::find(t.customers.begin(), t.customers.end(), cid) != t.customers.end()) {
                    auto key = std::make_pair(vi, ti);
                    if (!seen.count(key)) { seen.insert(key); result.push_back(key); }
                }
            }
        }
    }
    return result;
}

inline std::vector<std::pair<int,int>> allTrips(const Solution& s) {
    std::vector<std::pair<int,int>> result;
    for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
        for (int ti = 0; ti < static_cast<int>(s.vehicles[vi].trips.size()); ++ti) {
            result.push_back({vi, ti});
        }
    }
    return result;
}

// FUNCTION SELECT_SEARCH_COMPONENTS(solution s, bestFeasible)
// Trả về (selectedCustomers, selectedTrips) theo mục 7.
// randomFraction: tỉ lệ khách/trip ngẫu nhiên bổ sung, để tránh chỉ tập trung 1 vùng.
inline SearchComponents selectSearchComponents(const Instance& inst, const Solution& s, const Solution* bestFeasible,
                                                std::mt19937& rng, double randomFraction = 0.3, int minRandom = 3) {
    SearchComponents comp;

    if (bestFeasible == nullptr) {
        // Rank khách theo đóng góp vào vi phạm TW + W
        std::vector<std::pair<double,int>> scored;
        for (const auto& c : inst.customers) {
            double contrib = customerTWContribution(inst, s, c.id);
            scored.push_back({contrib, c.id});
        }
        std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

        int n = static_cast<int>(scored.size());
        int topCount = std::max(1, n / 3);
        std::set<int> chosen;
        for (int i = 0; i < topCount && i < n; ++i) chosen.insert(scored[i].second);

        // Khách thuộc trip vượt tải hoặc drone vượt tầm bay
        for (const auto& v : s.vehicles) {
            for (const auto& t : v.trips) {
                bool overCap = t.load > v.capacity(inst) + EPS;
                bool overRange = (v.type == VehicleType::DRONE) && (t.travelDistance > inst.drone_range + EPS);
                if (overCap || overRange) {
                    for (int cid : t.customers) chosen.insert(cid);
                }
            }
        }

        // Một tỉ lệ khách ngẫu nhiên
        int randomCount = std::max(minRandom, static_cast<int>(randomFraction * n));
        std::vector<int> allIds;
        for (const auto& c : inst.customers) allIds.push_back(c.id);
        std::shuffle(allIds.begin(), allIds.end(), rng);
        for (int i = 0; i < randomCount && i < static_cast<int>(allIds.size()); ++i) chosen.insert(allIds[i]);

        comp.selectedCustomers.assign(chosen.begin(), chosen.end());
        comp.selectedTrips = tripsContainingCustomers(s, comp.selectedCustomers);
    } else {
        double currentMakespan = s.makespan;
        std::set<int> criticalVehicleIds;
        for (const auto& v : s.vehicles) {
            if (std::fabs(v.completionTime - currentMakespan) <= EPS) {
                criticalVehicleIds.insert(v.id);
            }
        }

        std::set<int> chosenCustomers;
        std::set<std::pair<int,int>> chosenTrips;
        for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
            const Vehicle& v = s.vehicles[vi];
            if (!criticalVehicleIds.count(v.id)) continue;
            for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
                chosenTrips.insert({vi, ti});
                for (int cid : v.trips[ti].customers) chosenCustomers.insert(cid);
            }
        }

        int n = inst.numCustomers();
        int randomCount = std::max(minRandom, static_cast<int>(randomFraction * n));
        std::vector<int> allIds;
        for (const auto& c : inst.customers) allIds.push_back(c.id);
        std::shuffle(allIds.begin(), allIds.end(), rng);
        for (int i = 0; i < randomCount && i < static_cast<int>(allIds.size()); ++i) chosenCustomers.insert(allIds[i]);

        auto trips = allTrips(s);
        std::shuffle(trips.begin(), trips.end(), rng);
        int randomTripCount = std::max(2, static_cast<int>(randomFraction * trips.size()));
        for (int i = 0; i < randomTripCount && i < static_cast<int>(trips.size()); ++i) chosenTrips.insert(trips[i]);

        comp.selectedCustomers.assign(chosenCustomers.begin(), chosenCustomers.end());
        comp.selectedTrips.assign(chosenTrips.begin(), chosenTrips.end());
    }

    return comp;
}
