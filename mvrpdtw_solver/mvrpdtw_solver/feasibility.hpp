// feasibility.hpp
// Mục 5: Các nghiệm cần lưu — kiểm tra khả thi & so sánh hai nghiệm không khả thi.
#pragma once

#include <cmath>
#include "solution.hpp"

static constexpr double EPS = 1e-9;

// Nghiệm khả thi khi V_Sigma(s) <= epsilon.
inline bool isFeasible(const Solution& s, double epsilon = EPS) {
    return s.totalViolation <= epsilon;
}

// FUNCTION BETTER_INFEASIBLE(s1, s2)
// So sánh 2 nghiệm không khả thi theo thứ tự: (V_Sigma, C_max_norm, Distance)
// s2 == nullptr => s1 luôn tốt hơn (chưa có gì để so sánh).
inline bool betterInfeasible(const Solution& s1, const Solution* s2) {
    if (s2 == nullptr) return true;

    if (s1.totalViolation < s2->totalViolation - EPS) return true;
    if (std::fabs(s1.totalViolation - s2->totalViolation) <= EPS) {
        if (s1.makespan < s2->makespan - EPS) return true;
        if (std::fabs(s1.makespan - s2->makespan) <= EPS) {
            return s1.totalDistance < s2->totalDistance;
        }
    }
    return false;
}
