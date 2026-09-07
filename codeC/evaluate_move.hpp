// evaluate_move.hpp
// Mục 8: Đánh giá một move — EVALUATE_MOVE.
#pragma once

#include <optional>
#include <set>
#include <algorithm>
#include "instance.hpp"
#include "solution.hpp"
#include "schedule.hpp"
#include "evaluate.hpp"
#include "move.hpp"
#include "operators.hpp"

struct Candidate {
    bool valid = false;
    Solution solution;
    Move move;
    AttributeSet removedAttributes;
    AttributeSet addedAttributes;
};

// Ràng buộc cấu trúc phải loại ngay (mục 8, danh sách "Các ràng buộc cấu trúc phải loại ngay"):
//  - Khách xuất hiện 0 lần hoặc nhiều hơn 1 lần
//  - C1 nằm trên drone
//  - Sử dụng phương tiện ngoài tập K+D  (không áp dụng vì ta chỉ thao tác trên các vehicle có sẵn)
//  - Trip không bắt đầu/kết thúc tại depot (ngầm định đúng vì cấu trúc Trip luôn có depot 2 đầu)
//  - Khách được giao cho loại phương tiện không tương thích tĩnh
// forceAllCustomersPresent: true khi kiểm tra solution HOÀN CHỈNH (dùng trong EVALUATE_MOVE trên
// solution đã đủ mọi khách); false khi kiểm tra solution BỘ PHẬN trong quá trình construction/insertion
// (khi đó ta chỉ cần đảm bảo KHÔNG xuất hiện >1 lần và tương thích tĩnh, không cần đã đủ mọi khách).
inline bool violatesStructuralConstraint(const Instance& inst, const Solution& s, bool forceAllCustomersPresent = true) {
    std::set<int> seen;
    int totalCustomers = inst.numCustomers();

    for (const auto& v : s.vehicles) {
        for (const auto& t : v.trips) {
            for (int custId : t.customers) {
                if (seen.count(custId)) return true; // xuất hiện > 1 lần
                seen.insert(custId);

                if (!staticCompatible(inst, custId, v)) return true; // không tương thích tĩnh
            }
        }
    }

    if (forceAllCustomersPresent && static_cast<int>(seen.size()) != totalCustomers) return true; // thiếu khách (0 lần)

    return false;
}

// EXTRACT_TABU_ATTRIBUTES(s, move):
// Trả về tập thuộc tính CUNG hiện diện trong solution s liên quan tới các vehicle mà move chạm vào,
// hợp với thuộc tính phân công (ASSIGN) và thuộc tính thứ tự trip (TRIP_RETURN) khi áp dụng.
inline AttributeSet extractTabuAttributes(const Instance& /*inst*/, const Solution& s, const Move& m) {
    AttributeSet attrs;

    std::vector<int> vehicleIds;
    if (m.type == MoveType::Swap) {
        CustomerLocation locI = locateCustomer(s, m.customerId);
        CustomerLocation locJ = locateCustomer(s, m.customerId2);
        if (locI.found) vehicleIds.push_back(s.vehicles[locI.vehicleIdx].id);
        if (locJ.found) vehicleIds.push_back(s.vehicles[locJ.vehicleIdx].id);
    } else {
        vehicleIds = affectedVehicleIds(m);
    }

    attrs = extractTabuAttributesForVehicles(s, vehicleIds);

    // Thuộc tính phân công: nếu move đổi vehicle của 1 khách, thêm ASSIGN(customerId, oldVehicle)
    if (m.type == MoveType::Relocate || m.type == MoveType::OrOpt2) {
        if (m.sourceVehicleId != m.target.vehicleId) {
            attrs.insert(assignAttribute(m.customerId, m.sourceVehicleId));
            if (m.type == MoveType::OrOpt2) {
                attrs.insert(assignAttribute(m.customerId2, m.sourceVehicleId));
            }
        }
    } else if (m.type == MoveType::Swap) {
        CustomerLocation locI = locateCustomer(s, m.customerId);
        CustomerLocation locJ = locateCustomer(s, m.customerId2);
        if (locI.found && locJ.found) {
            int vehI = s.vehicles[locI.vehicleIdx].id;
            int vehJ = s.vehicles[locJ.vehicleIdx].id;
            if (vehI != vehJ) {
                attrs.insert(assignAttribute(m.customerId, vehI));
                attrs.insert(assignAttribute(m.customerId2, vehJ));
            }
        }
    } else if (m.type == MoveType::TripRelocate) {
        if (m.sourceVehicleId != m.target.vehicleId) {
            // Thuộc tính thứ tự trip: (TRIP_RETURN, tripUid, sourceVehicleId, sourcePredecessorTripUid)
            int srcVi = findVehicleIndexById(s, m.sourceVehicleId);
            std::uint64_t predUid = 0;
            if (srcVi >= 0 && m.sourceTripIndex > 0 && m.sourceTripIndex - 1 < static_cast<int>(s.vehicles[srcVi].trips.size())) {
                predUid = s.vehicles[srcVi].trips[m.sourceTripIndex - 1].uid;
            }
            attrs.insert(tripReturnAttribute(m.tripUid, m.sourceVehicleId, predUid));
        }
    }

    return attrs;
}

// FUNCTION EVALUATE_MOVE(solution s, move m, penaltyWeights lambda, H)
inline Candidate evaluateMove(const Instance& inst, const Solution& s, const Move& m,
                               const PenaltyWeights& lambda, double H,
                               double maxAllowedInfeasibility = std::numeric_limits<double>::infinity()) {
    Candidate result;

    Solution sPrime = s; // deep copy (Solution/Vehicle/Trip đều copy-able theo giá trị)

    AttributeSet oldAttributes = extractTabuAttributes(inst, s, m);

    applyMove(sPrime, m);
    removeEmptyTrips(sPrime);

    if (violatesStructuralConstraint(inst, sPrime)) {
        result.valid = false;
        return result;
    }

    std::vector<int> touched;
    if (m.type == MoveType::Swap) {
        // với swap, các vehicle liên quan xác định trên sPrime SAU khi áp dụng (vị trí không đổi vehicle)
        CustomerLocation locI = locateCustomer(sPrime, m.customerId);
        CustomerLocation locJ = locateCustomer(sPrime, m.customerId2);
        if (locI.found) touched.push_back(sPrime.vehicles[locI.vehicleIdx].id);
        if (locJ.found) touched.push_back(sPrime.vehicles[locJ.vehicleIdx].id);
    } else {
        touched = affectedVehicleIds(m);
    }

    for (int vid : touched) {
        int vi = findVehicleIndexById(sPrime, vid);
        if (vi < 0) continue;
        recomputeVehicle(inst, sPrime.vehicles[vi], 0); // an toàn: tính lại toàn bộ trip của vehicle bị ảnh hưởng
    }

    evaluateSolution(inst, sPrime, lambda, H);

    if (sPrime.totalViolation > maxAllowedInfeasibility) {
        result.valid = false;
        return result;
    }

    AttributeSet newAttributes = extractTabuAttributes(inst, sPrime, m);

    result.valid = true;
    result.solution = std::move(sPrime);
    result.move = m;
    result.removedAttributes = setDifference(oldAttributes, newAttributes);
    result.addedAttributes = setDifference(newAttributes, oldAttributes);
    return result;
}
