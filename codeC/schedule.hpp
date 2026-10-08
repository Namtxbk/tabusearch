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
//   i in C2, q_i <= M_D, flightTime(0,i)+flightTime(i,0) <= L_D, tau^D_i0 <= L_w
// LƯU Ý: L_D (drone_range / "Endurance fixed time") là GIỚI HẠN THỜI GIAN BAY (giây),
// KHÔNG PHẢI giới hạn quãng đường — xác nhận từ dữ liệu benchmark thực tế (energy model "endurance").
// Vì vậy ta so sánh theo travelTime (quãng đường / vận tốc), không so sánh trực tiếp khoảng cách.
inline bool staticCompatible(const Instance& inst, int customerId, const Vehicle& v) {
    const Customer& c = inst.node(customerId);

    if (v.type == VehicleType::DRONE) {
        if (c.is_c1) return false;                                   // phải thuộc C2
        if (c.demand > inst.drone_capacity) return false;             // q_i <= M_D
        double roundTripFlightTime = inst.travelTime(0, customerId, true) + inst.travelTime(customerId, 0, true);
        if (roundTripFlightTime > inst.drone_range) return false;     // thời gian bay khứ hồi <= L_D (giây)
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
        trip.flightTime = 0.0;
        trip.arrivalTime.clear();
        trip.arrivalTime.reserve(trip.customers.size());
        trip.waitingTime.clear();
        trip.waitingTime.reserve(trip.customers.size());

        int previousNode = 0; // depot
        double t = trip.startTime;

        for (int custId : trip.customers) {
            const Customer& cust = inst.node(custId);
            double legTime = inst.travelTime(previousNode, custId, isDrone);
            t += legTime;
            t = std::max(t, cust.ready);           // a_i = max{e_i, prev + tau}
            trip.arrivalTime.push_back(t);
            trip.load += cust.demand;
            trip.travelDistance += inst.dist(previousNode, custId);
            trip.flightTime += legTime;
            previousNode = custId;
        }

        {
            double lastLegTime = inst.travelTime(previousNode, 0, isDrone);
            t += lastLegTime;
            trip.flightTime += lastLegTime;
        }
        trip.travelDistance += inst.dist(previousNode, 0);
        trip.returnTime = t;

        for (double arrival : trip.arrivalTime) {
            trip.waitingTime.push_back(trip.returnTime - arrival);
        }

        currentTime = trip.returnTime;
    }

    if (v.trips.empty()) {
        v.completionTime = 0.0;
    } else {
        v.completionTime = v.trips.back().returnTime;
    }

    // Cập nhật số liệu thô của vehicle (xem Vehicle::rawVQ...) — O(độ dài vehicle), chỉ chạy cho
    // vehicle vừa bị recompute nên các vehicle không đổi giữ nguyên số liệu cũ.
    double capacity = v.capacity(inst);
    v.rawVQ = v.rawVD = v.rawVTW = v.rawVW = v.distance = 0.0;
    v.numCustomers = 0;
    for (const auto& trip : v.trips) {
        if (capacity > 0.0) v.rawVQ += std::max(0.0, trip.load - capacity) / capacity;
        if (isDrone && inst.drone_range > 0.0) {
            v.rawVD += std::max(0.0, trip.flightTime - inst.drone_range) / inst.drone_range;
        }
        v.distance += trip.travelDistance;
        for (std::size_t pos = 0; pos < trip.customers.size(); ++pos) {
            v.rawVTW += std::max(0.0, trip.arrivalTime[pos] - inst.node(trip.customers[pos]).due);
            if (inst.max_wait > 0.0) {
                v.rawVW += std::max(0.0, trip.returnTime - trip.arrivalTime[pos] - inst.max_wait) / inst.max_wait;
            }
        }
        v.numCustomers += static_cast<int>(trip.customers.size());
    }
}

// Tính lại toàn bộ các trip của 1 vehicle từ đầu (tiện dùng ở init / evaluate toàn cục).
inline void recomputeVehicleFull(const Instance& inst, Vehicle& v) {
    recomputeVehicle(inst, v, 0);
}
