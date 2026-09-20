// solution.hpp
// Cấu trúc dữ liệu: Trip, Vehicle, Solution (mục 1.2, 1.3 của tài liệu).
#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <memory>
#include "instance.hpp"

enum class VehicleType { TRUCK, DRONE };

// Một chuyến (trip) sigma_vk = [0, i1, i2, ..., im, 0]
struct Trip {
    std::uint64_t uid = 0;          // định danh duy nhất toàn cục của trip (ổn định qua các move)
    int vehicleId = -1;             // id của vehicle sở hữu trip này (tại thời điểm hiện tại)

    std::vector<int> customers;     // danh sách id khách hàng (1-based), theo thứ tự phục vụ

    double startTime = 0.0;
    double returnTime = 0.0;
    double load = 0.0;
    double travelDistance = 0.0;
    double flightTime = 0.0;        // Tổng thời gian di chuyển thực tế của trip (giây) — dùng để check "Endurance"
                                     // (drone_range là giới hạn THỜI GIAN BAY, không phải quãng đường).
                                     // Với truck, trường này không dùng để kiểm tra ràng buộc gì nhưng vẫn được tính cho đầy đủ.

    // arrivalTime / waitingTime song song với customers theo VỊ TRÍ (không phải customer id) —
    // dùng vector thay vì unordered_map để tránh cấp phát heap theo từng phần tử khi copy Trip
    // (mỗi candidate move đánh giá đều copy Solution, nên chi phí cấp phát này lặp lại rất nhiều lần).
    std::vector<double> arrivalTime;
    std::vector<double> waitingTime;

    // Tra cứu thời điểm đến của customerId trong trip này (O(độ dài trip), trip thường ngắn).
    double arrivalOf(int customerId) const {
        for (std::size_t i = 0; i < customers.size(); ++i) {
            if (customers[i] == customerId) return arrivalTime[i];
        }
        return 0.0;
    }

    bool empty() const { return customers.empty(); }
};

struct Vehicle {
    int id = -1;
    VehicleType type = VehicleType::TRUCK;
    std::vector<Trip> trips;        // R_v = (sigma_v1, ..., sigma_vm) — thứ tự có ý nghĩa
    double completionTime = 0.0;    // C_v(s)

    double capacity(const Instance& inst) const {
        return (type == VehicleType::TRUCK) ? inst.truck_capacity : inst.drone_capacity;
    }
};

struct Solution {
    // shared_ptr<Vehicle> thay vì Vehicle theo giá trị: copy Solution (Solution sPrime = s;) khi đó
    // chỉ copy các con trỏ (O(số vehicle)) thay vì deep-copy TOÀN BỘ khách của mọi vehicle mỗi lần —
    // mỗi candidate move được đánh giá đều copy cả Solution nên chi phí này lặp lại rất nhiều lần.
    // BẮT BUỘC: trước khi mutate 1 vehicle, phải gọi detachVehicle(idx) để "tách riêng" (clone) nó
    // ra khỏi các Solution khác đang chia sẻ chung con trỏ — nếu không sẽ làm hỏng solution khác.
    std::vector<std::shared_ptr<Vehicle>> vehicles;

    // Đảm bảo vehicles[idx] không bị chia sẻ với Solution nào khác rồi trả về tham chiếu mutable.
    // Chỉ thực sự clone khi cần (use_count() > 1); gọi lại nhiều lần trên cùng idx không tốn thêm.
    Vehicle& detachVehicle(int idx) {
        if (vehicles[idx].use_count() > 1) {
            vehicles[idx] = std::make_shared<Vehicle>(*vehicles[idx]);
        }
        return *vehicles[idx];
    }

    double makespan = 0.0;             // C_max(s)
    double normalizedMakespan = 0.0;   // C_max(s) / H

    double violationCapacity = 0.0;    // V_Q
    double violationRange = 0.0;       // V_D
    double violationTimeWindow = 0.0;  // V_TW
    double violationWaiting = 0.0;     // V_W
    double totalViolation = 0.0;       // V_Sigma

    double penalizedObjective = 0.0;   // F_lambda(s)
    double totalDistance = 0.0;

    int unassignedCount = 0;           // Số khách CHƯA được phục vụ (0 nếu đã phục vụ hết)
                                        // Nghiệm chỉ thực sự khả thi khi unassignedCount == 0 VÀ totalViolation <= epsilon.

    bool isFeasible(double epsilon = 1e-9) const {
        return totalViolation <= epsilon && unassignedCount == 0;
    }
};

// Bộ trọng số phạt lambda = (lambda_Q, lambda_D, lambda_TW, lambda_W)
struct PenaltyWeights {
    double lambdaQ = 1.0;
    double lambdaD = 1.0;
    double lambdaTW = 1.0;
    double lambdaW = 1.0;

    double lambdaMinQ = 0.001, lambdaMaxQ = 1000.0;
    double lambdaMinD = 0.001, lambdaMaxD = 1000.0;
    double lambdaMinTW = 0.001, lambdaMaxTW = 1000.0;
    double lambdaMinW = 0.001, lambdaMaxW = 1000.0;
};

// Bộ sinh uid duy nhất cho trip
struct TripUidGenerator {
    std::uint64_t next = 1;
    std::uint64_t generate() { return next++; }
};
