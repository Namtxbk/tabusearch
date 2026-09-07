// move.hpp
// Định nghĩa các loại move (mục 6) và thuộc tính tabu (mục 9).
#pragma once

#include <string>
#include <vector>
#include <set>
#include <cstdint>
#include <sstream>

enum class MoveType {
    Relocate,
    OrOpt2,
    Swap,
    TwoOpt,
    CrossTrip,
    TripRelocate
};

// Vị trí đích: (vehicleId, tripIndex trong R_v, insertionPosition trong trip.customers)
// tripIndex == -1  => tạo trip mới tại newTripPosition (dùng insertionPosition làm newTripPosition)
struct MoveTarget {
    int vehicleId = -1;
    int tripIndex = -1;        // -1 nếu là trip mới
    int insertionPosition = 0; // vị trí chèn trong trip (hoặc vị trí trip mới trong chuỗi trip)
};

// Move tổng quát — chứa đủ thông tin để APPLY_MOVE tái tạo lại hành động.
struct Move {
    MoveType type;

    // Relocate: 1 khách hàng
    int customerId = -1;

    // OrOpt2: block 2 khách liên tiếp
    int customerId2 = -1;

    // Nguồn (đối với Relocate / OrOpt2 / TripRelocate)
    int sourceVehicleId = -1;
    int sourceTripIndex = -1;
    int sourcePosition = -1; // vị trí bắt đầu block trong trip nguồn

    // Đích
    MoveTarget target;

    // Swap: khách i, j (dùng customerId, customerId2 làm i, j)

    // TwoOpt: đảo đoạn [p, q] trong 1 trip
    int tripVehicleId = -1;
    int tripIndexForTwoOpt = -1;
    int p = -1, q = -1;

    // CrossTrip: (vehicleA, tripA, cutA) <-> (vehicleB, tripB, cutB)
    int vehicleA = -1, tripIndexA = -1, cutA = -1;
    int vehicleB = -1, tripIndexB = -1, cutB = -1;

    // TripRelocate
    std::uint64_t tripUid = 0;
};

// ============================================================
// Mục 9. Thuộc tính tabu
// ============================================================
// Thuộc tính cung: (vehicleId, fromNode, toNode)  — fromNode/toNode = 0 cho depot.
// Thuộc tính phân công: ("ASSIGN", customerId, forbiddenVehicleId)
// Thuộc tính thứ tự trip: ("TRIP_RETURN", tripUid, sourceVehicleId, sourcePredecessorTripUid)
//
// Ta biểu diễn mỗi thuộc tính dưới dạng 1 chuỗi (string key) để dễ dùng làm khóa hash.
struct TabuAttribute {
    std::string key;
    bool operator==(const TabuAttribute& o) const { return key == o.key; }
    bool operator<(const TabuAttribute& o) const { return key < o.key; }
};

inline TabuAttribute arcAttribute(int vehicleId, int fromNode, int toNode) {
    std::ostringstream oss;
    oss << "ARC|" << vehicleId << "|" << fromNode << "|" << toNode;
    return TabuAttribute{oss.str()};
}

inline TabuAttribute assignAttribute(int customerId, int forbiddenVehicleId) {
    std::ostringstream oss;
    oss << "ASSIGN|" << customerId << "|" << forbiddenVehicleId;
    return TabuAttribute{oss.str()};
}

inline TabuAttribute tripReturnAttribute(std::uint64_t tripUid, int sourceVehicleId, std::uint64_t sourcePredecessorTripUid) {
    std::ostringstream oss;
    oss << "TRIP_RETURN|" << tripUid << "|" << sourceVehicleId << "|" << sourcePredecessorTripUid;
    return TabuAttribute{oss.str()};
}

using AttributeSet = std::set<TabuAttribute>;

inline AttributeSet setDifference(const AttributeSet& a, const AttributeSet& b) {
    AttributeSet result;
    for (const auto& x : a) {
        if (b.find(x) == b.end()) result.insert(x);
    }
    return result;
}
