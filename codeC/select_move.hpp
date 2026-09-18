// select_move.hpp
// Mục 11: Chọn move.
#pragma once

#include <vector>
#include <limits>
#include "evaluate_move.hpp"

// FUNCTION SELECT_BEST_CANDIDATE(candidatePool, bestFeasible)
// - Chưa có nghiệm khả thi: minimize (V_Sigma, C_max_norm, Distance)
// - Đã có nghiệm khả thi: minimize (penalizedObjective, V_Sigma, C_max_norm, Distance)
inline const Candidate* selectBestCandidate(const std::vector<Candidate>& pool, const Solution* bestFeasible) {
    if (pool.empty()) return nullptr;

    const Candidate* best = nullptr;

    if (bestFeasible == nullptr) {
        for (const auto& c : pool) {
            if (!c.valid) continue;
            if (best == nullptr) { best = &c; continue; }
            const Solution& a = c.solution;
            const Solution& b = best->solution;
            if (a.totalViolation < b.totalViolation - EPS) { best = &c; continue; }
            if (std::fabs(a.totalViolation - b.totalViolation) <= EPS) {
                if (a.normalizedMakespan < b.normalizedMakespan - EPS) { best = &c; continue; }
                if (std::fabs(a.normalizedMakespan - b.normalizedMakespan) <= EPS) {
                    if (a.totalDistance < b.totalDistance - EPS) { best = &c; continue; }
                }
            }
        }
    } else {
        for (const auto& c : pool) {
            if (!c.valid) continue;
            if (best == nullptr) { best = &c; continue; }
            const Solution& a = c.solution;
            const Solution& b = best->solution;
            if (a.penalizedObjective < b.penalizedObjective - EPS) { best = &c; continue; }
            if (std::fabs(a.penalizedObjective - b.penalizedObjective) <= EPS) {
                if (a.totalViolation < b.totalViolation - EPS) { best = &c; continue; }
                if (std::fabs(a.totalViolation - b.totalViolation) <= EPS) {
                    if (a.normalizedMakespan < b.normalizedMakespan - EPS) { best = &c; continue; }
                    if (std::fabs(a.normalizedMakespan - b.normalizedMakespan) <= EPS) {
                        if (a.totalDistance < b.totalDistance - EPS) { best = &c; continue; }
                    }
                }
            }
        }
    }

    return best;
}
