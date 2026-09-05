// solution.hpp
// Cấu trúc dữ liệu: Trip, Vehicle, Solution (mục 1.2, 1.3 của tài liệu).
#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
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

    // arrivalTime / waitingTime theo customer id (1-based) chỉ cho khách trong trip này
    std::unordered_map<int, double> arrivalTime;
    std::unordered_map<int, double> waitingTime;

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
    std::vector<Vehicle> vehicles;

    double makespan = 0.0;             // C_max(s)
    double normalizedMakespan = 0.0;   // C_max(s) / H

    double violationCapacity = 0.0;    // V_Q
    double violationRange = 0.0;       // V_D
    double violationTimeWindow = 0.0;  // V_TW
    double violationWaiting = 0.0;     // V_W
    double totalViolation = 0.0;       // V_Sigma

    double penalizedObjective = 0.0;   // F_lambda(s)
    double totalDistance = 0.0;

    bool isFeasible(double epsilon = 1e-9) const {
        return totalViolation <= epsilon;
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
