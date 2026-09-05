// best_solutions.hpp
// Mục 13: Cập nhật nghiệm tốt nhất và stagnation (hDiv, hStop).
#pragma once

#include <memory>
#include "solution.hpp"
#include "feasibility.hpp"

struct BestSolutionsUpdate {
    bool improvedRecord = false;
};

// FUNCTION UPDATE_BEST_SOLUTIONS(current, bestFeasible, bestInfeasible)
// bestFeasible / bestInfeasible: con trỏ tới nghiệm hiện có (nullptr nếu chưa có).
// Trả về true nếu cập nhật bestFeasible thành công (đã tạo/ghi đè); các tham số được sửa in-place.
inline bool updateBestSolutions(const Solution& current,
                                 std::unique_ptr<Solution>& bestFeasible,
                                 std::unique_ptr<Solution>& bestInfeasible) {
    bool improvedRecord = false;

    if (isFeasible(current)) {
        if (bestFeasible == nullptr || current.makespan < bestFeasible->makespan - EPS) {
            bestFeasible = std::make_unique<Solution>(current);
            improvedRecord = true;
        }
    } else {
        if (betterInfeasible(current, bestInfeasible.get())) {
            bestInfeasible = std::make_unique<Solution>(current);
        }
        if (bestFeasible == nullptr) {
            improvedRecord = true; // theo pseudocode: cải thiện "record" khi vẫn ở giai đoạn tìm nghiệm khả thi
        }
    }

    return improvedRecord;
}
