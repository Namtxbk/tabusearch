// schedule.hpp
// Mục 2: Kiểm tra tính tương thích tĩnh (STATIC_COMPATIBLE)
// Mục 3: Tính lịch cho một phương tiện (RECOMPUTE_VEHICLE)
#pragma once

#include <algorithm>
#include "instance.hpp"
#include "solution.hpp"

// ============================================================
// Mục 2. Kiểm tra tính tương thích tĩnh
// ============================================================
// Khách i chỉ nên được xem là tương thích với drone nếu:
//   i in C2, q_i <= M_D, d_0i + d_i0 <= L_D, tau^D_i0 <= L_w
// Với truck: q_i <= M_T, tau^T_i0 <= L_w  (không có ràng buộc quãng đường)
inline bool staticCompatible(const Instance& inst, int customerId, const Vehicle& v) {
    const Customer& c = inst.node(customerId);

    if (v.type == VehicleType::DRONE) {
        if (c.is_c1) return false;                                   // phải thuộc C2
        if (c.demand > inst.drone_capacity) return false;             // q_i <= M_D
        double roundTripDist = inst.dist(0, customerId) + inst.dist(customerId, 0);
        if (roundTripDist > inst.drone_range) return false;           // d_0i + d_i0 <= L_D
        double travelBack = inst.travelTime(customerId, 0, true);     // tau^D_i0
        if (travelBack > inst.max_wait) return false;                 // <= L_w
        return true;
    } else { // TRUCK
        if (c.demand > inst.truck_capacity) return false;             // q_i <= M_T
        double travelBack = inst.travelTime(customerId, 0, false);    // tau^T_i0
        if (travelBack > inst.max_wait) return false;                 // <= L_w
        return true;
    }
}

// ============================================================
// Mục 3. Tính lịch cho một phương tiện — RECOMPUTE_VEHICLE
// ============================================================
// Tính lại startTime/arrivalTime/returnTime/load/travelDistance/waitingTime
// cho các trip của vehicle v, bắt đầu từ chỉ số firstAffectedTrip.
// Tất cả toán tử PHẢI gọi hàm này sau khi thay đổi nghiệm.
inline void recomputeVehicle(const Instance& inst, Vehicle& v, int firstAffectedTrip) {
    double currentTime;
    if (firstAffectedTrip <= 0) {
        currentTime = 0.0;
        firstAffectedTrip = 0;
    } else {
        currentTime = v.trips[firstAffectedTrip - 1].returnTime;
    }

    bool isDrone = (v.type == VehicleType::DRONE);

    for (int k = firstAffectedTrip; k < static_cast<int>(v.trips.size()); ++k) {
        Trip& trip = v.trips[k];
        trip.startTime = currentTime;
        trip.load = 0.0;
        trip.travelDistance = 0.0;
        trip.arrivalTime.clear();
        trip.waitingTime.clear();

        int previousNode = 0; // depot
        double t = trip.startTime;

        for (int custId : trip.customers) {
            const Customer& cust = inst.node(custId);
            t += inst.travelTime(previousNode, custId, isDrone);
            t = std::max(t, cust.ready);           // a_i = max{e_i, prev + tau}
            trip.arrivalTime[custId] = t;
            trip.load += cust.demand;
            trip.travelDistance += inst.dist(previousNode, custId);
            previousNode = custId;
        }

        t += inst.travelTime(previousNode, 0, isDrone);
        trip.travelDistance += inst.dist(previousNode, 0);
        trip.returnTime = t;

        for (int custId : trip.customers) {
            trip.waitingTime[custId] = trip.returnTime - trip.arrivalTime[custId];
        }

        currentTime = trip.returnTime;
    }

    if (v.trips.empty()) {
        v.completionTime = 0.0;
    } else {
        v.completionTime = v.trips.back().returnTime;
    }
}

// Tính lại toàn bộ các trip của 1 vehicle từ đầu (tiện dùng ở init / evaluate toàn cục).
inline void recomputeVehicleFull(const Instance& inst, Vehicle& v) {
    recomputeVehicle(inst, v, 0);
}
