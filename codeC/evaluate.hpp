// evaluate.hpp
// Mục 4: Đo mức vi phạm + hàm đánh giá nghiệm (EVALUATE_SOLUTION)
#pragma once

#include <algorithm>
#include <limits>
#include <vector>
#include "instance.hpp"
#include "solution.hpp"
#include "schedule.hpp"

inline double posPart(double x) { return std::max(0.0, x); }

// H = max{1, C_max(s0), max_i l_i} — tính MỘT LẦN từ nghiệm ban đầu, không đổi trong quá trình tìm kiếm.
inline double computeH(const Instance& inst, double initialMakespan) {
    double H = std::max(1.0, initialMakespan);
    for (const auto& c : inst.customers) {
        H = std::max(H, c.due);
    }
    return H;
}

// Tổng hợp các đại lượng vi phạm từ số liệu đã cache trong từng Vehicle (do recomputeVehicle cập nhật).
// O(số vehicle) — KHÔNG duyệt khách/trip. Caller phải đảm bảo mọi vehicle bị thay đổi đã được
// recompute (evaluate_move.hpp / construction.hpp làm đúng việc đó cho các vehicle bị move chạm vào).
inline void aggregateSolutionMetrics(const Instance& inst, Solution& s, const PenaltyWeights& lambda, double H) {
    double maxCompletion = 0.0;
    double VQ = 0.0, VD = 0.0, VTW = 0.0, VW = 0.0;
    double totalDistance = 0.0;
    int served = 0;
    for (const auto& vp : s.vehicles) {
        const Vehicle& v = *vp;
        maxCompletion = std::max(maxCompletion, v.completionTime);
        VQ += v.rawVQ;
        VD += v.rawVD;
        VTW += v.rawVTW;
        VW += v.rawVW;
        totalDistance += v.distance;
        served += v.numCustomers;
    }
    s.makespan = maxCompletion;
    s.normalizedMakespan = (H > 0.0) ? (s.makespan / H) : s.makespan;

    int n = inst.numCustomers();
    if (n <= 0) n = 1; // tránh chia 0
    if (H > 0.0) VTW /= H;

    s.violationCapacity = VQ / n;
    s.violationRange = VD / n;
    s.violationTimeWindow = VTW / n;
    s.violationWaiting = VW / n;
    s.totalViolation = s.violationCapacity + s.violationRange + s.violationTimeWindow + s.violationWaiting;
    s.totalDistance = totalDistance;

    // Không có khách trùng (đã được loại bởi kiểm tra cấu trúc) nên số khách chưa phục vụ = n - tổng đã phục vụ.
    int unassigned = std::max(0, inst.numCustomers() - served);
    s.unassignedCount = unassigned;
    // Mỗi khách chưa được phục vụ cũng được cộng vào tổng vi phạm (nhân hệ số lớn để luôn ưu tiên
    // giảm số khách thiếu trước khi tối ưu makespan/other violations) — quy đổi theo cùng thang đo
    // chuẩn hoá /n như các V_* khác để không làm lệch đơn vị của penalizedObjective.
    double violationUnassigned = static_cast<double>(unassigned) / n;
    s.totalViolation += violationUnassigned;

    s.penalizedObjective = s.normalizedMakespan
        + lambda.lambdaQ * s.violationCapacity
        + lambda.lambdaD * s.violationRange
        + lambda.lambdaTW * s.violationTimeWindow
        + lambda.lambdaW * s.violationWaiting
        + 1000.0 * violationUnassigned; // phạt rất nặng khi còn khách chưa được phục vụ
}

// PROCEDURE EVALUATE_SOLUTION(solution s, penaltyWeights lambda, H)
// Tính lại toàn bộ lịch trình (RECOMPUTE_VEHICLE(s, v, 0)) rồi đo các đại lượng vi phạm.
// Dùng khi KHÔNG chắc vehicle nào đã có lịch trình đúng (construction, sau ruin, sau full rebuild).
// Nếu caller đã tự recompute đúng phạm vi (evaluate_move.hpp), dùng thẳng aggregateSolutionMetrics.
inline void evaluateSolution(const Instance& inst, Solution& s, const PenaltyWeights& lambda, double H) {
    // detachVehicle (không dereference thẳng shared_ptr) để không vô tình mutate 1 vehicle đang
    // được Solution khác (vd. current trước khi copy) chia sẻ chung con trỏ.
    for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
        recomputeVehicleFull(inst, s.detachVehicle(vi));
    }
    aggregateSolutionMetrics(inst, s, lambda, H);
}
