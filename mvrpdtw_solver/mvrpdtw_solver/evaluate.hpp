// evaluate.hpp
// Mục 4: Đo mức vi phạm + hàm đánh giá nghiệm (EVALUATE_SOLUTION)
#pragma once

#include <algorithm>
#include <limits>
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

// PROCEDURE EVALUATE_SOLUTION(solution s, penaltyWeights lambda, H)
// Tính lại toàn bộ lịch trình (RECOMPUTE_VEHICLE(s, v, 0)) rồi đo các đại lượng vi phạm.
inline void evaluateSolution(const Instance& inst, Solution& s, const PenaltyWeights& lambda, double H) {
    for (auto& v : s.vehicles) {
        recomputeVehicleFull(inst, v);
    }

    double maxCompletion = 0.0;
    for (const auto& v : s.vehicles) {
        maxCompletion = std::max(maxCompletion, v.completionTime);
    }
    s.makespan = maxCompletion;
    s.normalizedMakespan = (H > 0.0) ? (s.makespan / H) : s.makespan;

    int n = inst.numCustomers();
    if (n <= 0) n = 1; // tránh chia 0

    double VQ = 0.0, VD = 0.0, VTW = 0.0, VW = 0.0;
    double totalDistance = 0.0;

    for (const auto& v : s.vehicles) {
        double capacity = v.capacity(inst);
        bool isDrone = (v.type == VehicleType::DRONE);

        for (const auto& trip : v.trips) {
            // Vi phạm tải trọng
            if (capacity > 0.0) {
                VQ += posPart(trip.load - capacity) / capacity;
            }

            // Vi phạm tầm bay drone (tổng quãng đường chuyến, cả đi lẫn về)
            if (isDrone && inst.drone_range > 0.0) {
                VD += posPart(trip.travelDistance - inst.drone_range) / inst.drone_range;
            }

            totalDistance += trip.travelDistance;

            // Vi phạm time window + vi phạm thời gian chờ hàng
            for (int custId : trip.customers) {
                const Customer& cust = inst.node(custId);
                double arrival = trip.arrivalTime.at(custId);
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

    s.penalizedObjective = s.normalizedMakespan
        + lambda.lambdaQ * s.violationCapacity
        + lambda.lambdaD * s.violationRange
        + lambda.lambdaTW * s.violationTimeWindow
        + lambda.lambdaW * s.violationWaiting;
}
