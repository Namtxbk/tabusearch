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

// Đo các đại lượng vi phạm dựa trên lịch trình HIỆN CÓ của các vehicle (không recompute).
// Dùng khi caller đã tự recompute đúng những vehicle bị move/insertion tác động — tránh
// recompute lại toàn bộ solution (lãng phí O(n) mỗi candidate khi chỉ 1-2 vehicle thay đổi).
inline void aggregateSolutionMetrics(const Instance& inst, Solution& s, const PenaltyWeights& lambda, double H) {
    double maxCompletion = 0.0;
    for (const auto& vp : s.vehicles) {
        maxCompletion = std::max(maxCompletion, vp->completionTime);
    }
    s.makespan = maxCompletion;
    s.normalizedMakespan = (H > 0.0) ? (s.makespan / H) : s.makespan;

    int n = inst.numCustomers();
    if (n <= 0) n = 1; // tránh chia 0

    double VQ = 0.0, VD = 0.0, VTW = 0.0, VW = 0.0;
    double totalDistance = 0.0;
    std::vector<bool> served(inst.numCustomers() + 1, false); // index theo id 1-based (0 depot không dùng)

    for (const auto& vp : s.vehicles) {
        const Vehicle& v = *vp;
        double capacity = v.capacity(inst);
        bool isDrone = (v.type == VehicleType::DRONE);

        for (const auto& trip : v.trips) {
            // Vi phạm tải trọng
            if (capacity > 0.0) {
                VQ += posPart(trip.load - capacity) / capacity;
            }

            // Vi phạm tầm bay drone: L_D là giới hạn THỜI GIAN BAY (giây) — energy model "endurance",
            // xác nhận từ dữ liệu benchmark thực tế. Không so sánh bằng quãng đường.
            if (isDrone && inst.drone_range > 0.0) {
                VD += posPart(trip.flightTime - inst.drone_range) / inst.drone_range;
            }

            totalDistance += trip.travelDistance;

            // Vi phạm time window + vi phạm thời gian chờ hàng
            for (std::size_t pos = 0; pos < trip.customers.size(); ++pos) {
                int custId = trip.customers[pos];
                if (custId >= 1 && custId <= inst.numCustomers()) served[custId] = true;
                const Customer& cust = inst.node(custId);
                double arrival = trip.arrivalTime[pos];
                if (H > 0.0) {
                    VTW += posPart(arrival - cust.due) / H;
                }
                double wait = trip.returnTime - arrival; // r_sigma(i) - a_i
                if (inst.max_wait > 0.0) {
                    VW += posPart(wait - inst.max_wait) / inst.max_wait;
                }
            }
        }
    }

    s.violationCapacity = VQ / n;
    s.violationRange = VD / n;
    s.violationTimeWindow = VTW / n;
    s.violationWaiting = VW / n;
    s.totalViolation = s.violationCapacity + s.violationRange + s.violationTimeWindow + s.violationWaiting;
    s.totalDistance = totalDistance;

    int unassigned = 0;
    for (int cid = 1; cid <= inst.numCustomers(); ++cid) {
        if (!served[cid]) ++unassigned;
    }
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
