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
        // BUG cũ: gán improvedRecord = true VÔ ĐIỀU KIỆN mỗi khi bestFeasible == nullptr, bất kể
        // current có thực sự tốt hơn bestInfeasible hay không. Hệ quả: hStop/hDiv (đếm số iteration
        // liên tiếp KHÔNG cải thiện) không bao giờ tăng trong suốt giai đoạn chưa có nghiệm khả thi
        // -> diversification (ruin-recreate) không bao giờ được kích hoạt cho tới khi tìm được nghiệm
        // khả thi đầu tiên — chính là trường hợp cần diversification nhất (instance lớn, mắc kẹt).
        bool better = betterInfeasible(current, bestInfeasible.get());
        if (better) {
            bestInfeasible = std::make_unique<Solution>(current);
        }
        if (bestFeasible == nullptr) {
            improvedRecord = better; // chỉ coi là cải thiện khi bestInfeasible thực sự tốt hơn
        }
    }

    return improvedRecord;
}
