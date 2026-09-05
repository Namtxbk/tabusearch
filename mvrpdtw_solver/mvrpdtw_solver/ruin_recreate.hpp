// ruin_recreate.hpp
// Mục 14: Ruin & Recreate.
#pragma once

#include <vector>
#include <set>
#include <cmath>
#include <random>
#include <optional>
#include "instance.hpp"
#include "solution.hpp"
#include "evaluate.hpp"
#include "construction.hpp"
#include "select_components.hpp"

// 14.1 Ruin: chọn q_ruin khách để loại bỏ.
// q_ruin = max{1, floor(rho_ruin * n)}
// Hỗn hợp: ~1/3 khách vi phạm lớn, ~1/3 khách trên phương tiện critical, phần còn lại ngẫu nhiên.
// Nếu nghiệm đang khả thi, tăng tỉ lệ lấy khách từ phương tiện critical.
inline std::vector<int> selectRuinCustomers(const Instance& inst, const Solution& s, int q, std::mt19937& rng) {
    std::set<int> chosen;

    bool currentlyFeasible = isFeasible(s);

    // Nhóm 1: khách có đóng góp vi phạm lớn (TW + waiting)
    std::vector<std::pair<double,int>> scored;
    for (const auto& c : inst.customers) {
        double contrib = customerTWContribution(inst, s, c.id);
        scored.push_back({contrib, c.id});
    }
    std::sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    int groupSize = std::max(1, q / 3);
    for (int i = 0; i < groupSize && i < static_cast<int>(scored.size()) && static_cast<int>(chosen.size()) < q; ++i) {
        if (scored[i].first > EPS) chosen.insert(scored[i].second);
    }

    // Nhóm 2: khách trên phương tiện critical (vehicle có completionTime == makespan)
    double makespan = s.makespan;
    std::vector<int> criticalCustomers;
    for (const auto& v : s.vehicles) {
        if (std::fabs(v.completionTime - makespan) <= EPS) {
            for (const auto& t : v.trips) {
                for (int cid : t.customers) criticalCustomers.push_back(cid);
            }
        }
    }
    std::shuffle(criticalCustomers.begin(), criticalCustomers.end(), rng);
    int criticalGroupSize = currentlyFeasible ? std::max(groupSize, q / 2) : groupSize;
    for (int i = 0; i < criticalGroupSize && i < static_cast<int>(criticalCustomers.size()) && static_cast<int>(chosen.size()) < q; ++i) {
        chosen.insert(criticalCustomers[i]);
    }

    // Nhóm 3: ngẫu nhiên, lấp đầy đến đủ q
    std::vector<int> allIds;
    for (const auto& c : inst.customers) allIds.push_back(c.id);
    std::shuffle(allIds.begin(), allIds.end(), rng);
    for (int cid : allIds) {
        if (static_cast<int>(chosen.size()) >= q) break;
        chosen.insert(cid);
    }

    return std::vector<int>(chosen.begin(), chosen.end());
}

inline void removeCustomersFromSolution(Solution& s, const std::vector<int>& customerIds) {
    std::set<int> toRemove(customerIds.begin(), customerIds.end());
    for (auto& v : s.vehicles) {
        for (auto& t : v.trips) {
            std::vector<int> kept;
            kept.reserve(t.customers.size());
            for (int cid : t.customers) {
                if (!toRemove.count(cid)) kept.push_back(cid);
            }
            t.customers = std::move(kept);
        }
    }
    removeEmptyTrips(s);
}

// 14.2 Recreate: sắp khách bị loại theo l_i tăng dần (+ nhiễu ngẫu nhiên nhẹ),
// chèn lại bằng best insertion trên toàn hệ thống. Không ép khách vào trucks[0].
struct RuinRecreateResult {
    bool success = false;
    Solution solution;
};

inline RuinRecreateResult ruinRecreate(const Instance& inst, const Solution& current, const PenaltyWeights& lambda,
                                        const Solution* bestFeasible, double ruinRateInit, int maximumAttempts,
                                        double H, std::mt19937& rng, double minimumRuinRate = 0.05) {
    RuinRecreateResult result;

    int n = inst.numCustomers();
    double ruinRate = ruinRateInit;

    for (int attempt = 1; attempt <= maximumAttempts; ++attempt) {
        int q = std::max(1, static_cast<int>(std::floor(ruinRate * n)));

        Solution partial = current;
        std::vector<int> removedCustomers = selectRuinCustomers(inst, partial, q, rng);

        removeCustomersFromSolution(partial, removedCustomers);
        evaluateSolution(inst, partial, lambda, H);

        // Sắp theo l_i tăng dần + nhiễu ngẫu nhiên nhẹ (hoán đổi ngẫu nhiên vài cặp liền kề)
        std::sort(removedCustomers.begin(), removedCustomers.end(), [&](int a, int b) {
            return inst.node(a).due < inst.node(b).due;
        });
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        for (size_t i = 0; i + 1 < removedCustomers.size(); ++i) {
            if (coin(rng) < 0.1) std::swap(removedCustomers[i], removedCustomers[i + 1]);
        }

        bool success = true;
        for (int custId : removedCustomers) {
            double baselineDistance = partial.totalDistance;
            auto insertions = generateAllInsertions(inst, partial, custId, /*forbidTrucks0FirstTruck=*/true);

            // Lọc bỏ "trucks[0]" nếu còn phương tiện khác khả dụng (không ép khách vào trucks[0]).
            bool hasNonFirstTruckOption = false;
            for (const auto& im : insertions) {
                int vi = findVehicleIndexById(partial, im.targetVehicleId);
                bool isFirstTruck = (vi == 0 && partial.vehicles[vi].type == VehicleType::TRUCK);
                if (!isFirstTruck) { hasNonFirstTruckOption = true; break; }
            }

            std::vector<InsertionCandidate> evaluatedInsertions;
            for (const auto& im : insertions) {
                int vi = findVehicleIndexById(partial, im.targetVehicleId);
                bool isFirstTruck = (vi == 0 && partial.vehicles[vi].type == VehicleType::TRUCK);
                if (hasNonFirstTruckOption && isFirstTruck) continue; // tránh ép vào trucks[0] nếu có lựa chọn khác

                InsertionCandidate cand = evaluateInsertion(inst, partial, im, lambda, H, baselineDistance);
                if (cand.valid) evaluatedInsertions.push_back(cand);
            }

            if (evaluatedInsertions.empty()) {
                success = false;
                break;
            }

            // SELECT_BEST_CANDIDATE(evaluatedInsertions, bestFeasible)
            const InsertionCandidate* best = nullptr;
            if (bestFeasible == nullptr) {
                for (const auto& c : evaluatedInsertions) {
                    if (best == nullptr) { best = &c; continue; }
                    if (c.solution.totalViolation < best->solution.totalViolation - EPS) { best = &c; continue; }
                    if (std::fabs(c.solution.totalViolation - best->solution.totalViolation) <= EPS) {
                        if (c.solution.normalizedMakespan < best->solution.normalizedMakespan - EPS) { best = &c; continue; }
                        if (std::fabs(c.solution.normalizedMakespan - best->solution.normalizedMakespan) <= EPS) {
                            if (c.solution.totalDistance < best->solution.totalDistance - EPS) best = &c;
                        }
                    }
                }
            } else {
                for (const auto& c : evaluatedInsertions) {
                    if (best == nullptr) { best = &c; continue; }
                    if (c.solution.penalizedObjective < best->solution.penalizedObjective - EPS) { best = &c; continue; }
                    if (std::fabs(c.solution.penalizedObjective - best->solution.penalizedObjective) <= EPS) {
                        if (c.solution.totalViolation < best->solution.totalViolation - EPS) best = &c;
                    }
                }
            }

            partial = best->solution;
        }

        if (success) {
            result.success = true;
            result.solution = std::move(partial);
            return result;
        }

        ruinRate = std::max(minimumRuinRate, 0.8 * ruinRate);
    }

    result.success = false;
    return result;
}
