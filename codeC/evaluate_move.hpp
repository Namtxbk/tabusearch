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
    int totalCustomers = inst.numCustomers();

    // Buffer tái sử dụng giữa các lần gọi (hàm này chạy trên MỌI candidate) — vector<char> đánh dấu
    // theo id thay cho std::set<int> tránh cấp phát 1 node cây cho mỗi khách mỗi candidate.
    static std::vector<char> seen;
    if (static_cast<int>(seen.size()) < totalCustomers + 1) {
        seen.assign(totalCustomers + 1, 0);
    } else {
        std::fill(seen.begin(), seen.begin() + totalCustomers + 1, 0);
    }

    int seenCount = 0;
    for (const auto& vp : s.vehicles) {
        const Vehicle& v = *vp;
        for (const auto& t : v.trips) {
            for (int custId : t.customers) {
                if (custId < 0 || custId > totalCustomers || seen[custId]) return true; // xuất hiện > 1 lần
                seen[custId] = 1;
                ++seenCount;

                if (!staticCompatible(inst, custId, v)) return true; // không tương thích tĩnh
            }
        }
    }

    if (forceAllCustomersPresent && seenCount != totalCustomers) return true; // thiếu khách (0 lần)

    return false;
}

// Phiên bản rẻ của violatesStructuralConstraint cho delta evaluation: chỉ kiểm tra các vehicle bị move
// chạm vào (khách trùng lặp trong chúng, id hợp lệ, tương thích tĩnh) — O(kích thước các vehicle đó).
// Các vehicle còn lại giữ nguyên từ solution cha (vốn đã hợp lệ). Việc "thiếu/thừa khách" được kiểm
// tra riêng bằng customerCountMismatch (sau khi vehicle đã recompute) thay vì đếm lại toàn bộ solution.
inline bool violatesStructuralConstraintTouched(const Instance& inst, const Solution& s, const std::vector<int>& touchedIdx) {
    int totalCustomers = inst.numCustomers();
    static std::vector<char> seen;
    if (static_cast<int>(seen.size()) < totalCustomers + 1) seen.assign(totalCustomers + 1, 0);

    bool bad = false;
    std::vector<int> marked;
    for (int vi : touchedIdx) {
        const Vehicle& v = *s.vehicles[vi];
        for (const auto& t : v.trips) {
            for (int custId : t.customers) {
                if (custId < 1 || custId > totalCustomers || seen[custId] || !staticCompatible(inst, custId, v)) {
                    bad = true; break;
                }
                seen[custId] = 1;
                marked.push_back(custId);
            }
            if (bad) break;
        }
        if (bad) break;
    }
    for (int id : marked) seen[id] = 0; // chỉ reset các ô đã đánh dấu (không fill cả mảng)
    return bad;
}

// Tổng số khách các vehicle đang phục vụ (dùng số liệu cache, O(số vehicle)) phải đúng bằng n.
inline bool customerCountMismatch(const Instance& inst, const Solution& s) {
    int total = 0;
    for (const auto& vp : s.vehicles) total += vp->numCustomers;
    return total != inst.numCustomers();
}

// Thuộc tính phân công (ASSIGN) và thứ tự trip (TRIP_RETURN) của move trên solution s.
inline AttributeSet extractExtraAttributes(const Solution& s, const Move& m) {
    AttributeSet attrs;

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
            int vehI = s.vehicles[locI.vehicleIdx]->id;
            int vehJ = s.vehicles[locJ.vehicleIdx]->id;
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
            if (srcVi >= 0 && m.sourceTripIndex > 0 && m.sourceTripIndex - 1 < static_cast<int>(s.vehicles[srcVi]->trips.size())) {
                predUid = s.vehicles[srcVi]->trips[m.sourceTripIndex - 1].uid;
            }
            attrs.insert(tripReturnAttribute(m.tripUid, m.sourceVehicleId, predUid));
        }
    }

    return attrs;
}

// EXTRACT_TABU_ATTRIBUTES(s, move) — cài đặt THAM CHIẾU (đầy đủ): cung của mọi trip của các vehicle bị
// chạm + ASSIGN + TRIP_RETURN. Chậm (O(kích thước vehicle) phần tử, insert/contains tuyến tính) nên
// evaluateMove KHÔNG dùng mà dùng diffTabuAttributes (cho cùng removed/added); giữ lại để đối chiếu/kiểm thử.
inline AttributeSet extractTabuAttributes(const Instance& /*inst*/, const Solution& s, const Move& m) {
    std::vector<int> vehicleIds;
    if (m.type == MoveType::Swap) {
        CustomerLocation locI = locateCustomer(s, m.customerId);
        CustomerLocation locJ = locateCustomer(s, m.customerId2);
        if (locI.found) vehicleIds.push_back(s.vehicles[locI.vehicleIdx]->id);
        if (locJ.found) vehicleIds.push_back(s.vehicles[locJ.vehicleIdx]->id);
    } else {
        vehicleIds = affectedVehicleIds(m);
    }

    AttributeSet attrs = extractTabuAttributesForVehicles(s, vehicleIds);
    for (const auto& a : extractExtraAttributes(s, m)) attrs.insert(a);
    return attrs;
}

// Chỉ mục cung theo id khách để hỏi "cung x->y có trong tập không" trong O(1) mà không hash: mỗi khách
// xuất hiện 1 lần trong 1 vehicle nên có nhiều nhất 1 cung ra (succ) và có/không cung depot->x (start).
// Chỉ các ô đã ghi mới được reset (theo usedList) nên clear() rẻ.
struct ArcIndex {
    std::vector<int> succ;      // succ[x] = khách kế tiếp sau x (0 = depot), -1 = chưa có cung ra
    std::vector<char> start;    // start[x]: có cung depot -> x
    std::vector<char> used;     // ô nào đã ghi (để reset)
    std::vector<int> usedList;

    void ensure(int n) {
        if (static_cast<int>(succ.size()) < n + 1) {
            succ.assign(n + 1, -1); start.assign(n + 1, 0); used.assign(n + 1, 0);
        }
    }
    void touch(int x) { if (!used[x]) { used[x] = 1; usedList.push_back(x); } }
    void addTrip(const Trip& t) {
        int prev = 0;
        for (int c : t.customers) {
            touch(c);
            if (prev == 0) start[c] = 1; else succ[prev] = c;
            prev = c;
        }
        if (prev != 0) succ[prev] = 0; // cung prev -> depot
    }
    bool has(int from, int to) const {
        if (from == 0) return start[to] != 0;
        return succ[from] == to;
    }
    void clear() {
        for (int x : usedList) { succ[x] = -1; start[x] = 0; used[x] = 0; }
        usedList.clear();
    }
};

// Trip t còn nguyên (cùng uid + cùng danh sách khách) ở vehicle `other` không?
inline bool tripUnchangedIn(const Trip& t, const Vehicle* other) {
    if (!other) return false;
    for (const auto& ot : other->trips) {
        if (ot.uid == t.uid) return ot.customers == t.customers;
    }
    return false;
}

// Thêm vào `out` các cung (vehicle vid) của `changedTrips` mà KHÔNG có trong `ref` (chỉ mục cung phía kia).
inline void collectArcsMissingIn(int vid, const ArcIndex& ref, const std::vector<const Trip*>& changedTrips,
                                 AttributeSet& out) {
    for (const Trip* t : changedTrips) {
        if (t->customers.empty()) continue;
        int prev = 0;
        for (int c : t->customers) {
            if (!ref.has(prev, c)) out.insertUnique(arcAttribute(vid, prev, c));
            prev = c;
        }
        if (!ref.has(prev, 0)) out.insertUnique(arcAttribute(vid, prev, 0));
    }
}

// removed = attrs(s) \ attrs(sPrime), added = attrs(sPrime) \ attrs(s) — tính trực tiếp thay vì dựng 2 tập
// đầy đủ rồi trừ nhau. Trip không đổi ở cả 2 phía triệt tiêu nhau nên bị bỏ; với trip đổi, so cung qua ArcIndex.
// Kết quả (theo tập hợp) bằng setDifference(extractTabuAttributes(s), extractTabuAttributes(sPrime)).
inline void diffTabuAttributes(const Instance& inst, const Solution& s, const Solution& sPrime, const Move& m,
                               const std::vector<int>& touchedIds, AttributeSet& removed, AttributeSet& added) {
    static ArcIndex oldIdx, newIdx;
    oldIdx.ensure(inst.numCustomers());
    newIdx.ensure(inst.numCustomers());

    std::vector<int> done; // vehicle đã xử lý (touched có thể lặp id)
    for (int vid : touchedIds) {
        if (std::find(done.begin(), done.end(), vid) != done.end()) continue;
        done.push_back(vid);

        int oi = findVehicleIndexById(s, vid), ni = findVehicleIndexById(sPrime, vid);
        const Vehicle* ov = (oi >= 0) ? s.vehicles[oi].get() : nullptr;
        const Vehicle* nv = (ni >= 0) ? sPrime.vehicles[ni].get() : nullptr;

        std::vector<const Trip*> oldChanged, newChanged;
        if (ov) for (const auto& t : ov->trips) if (!tripUnchangedIn(t, nv)) oldChanged.push_back(&t);
        if (nv) for (const auto& t : nv->trips) if (!tripUnchangedIn(t, ov)) newChanged.push_back(&t);

        for (const Trip* t : oldChanged) oldIdx.addTrip(*t);
        for (const Trip* t : newChanged) newIdx.addTrip(*t);
        collectArcsMissingIn(vid, newIdx, oldChanged, removed);
        collectArcsMissingIn(vid, oldIdx, newChanged, added);
        oldIdx.clear();
        newIdx.clear();
    }

    // ASSIGN / TRIP_RETURN: rất ít phần tử, giữ nguyên cách tính cũ.
    AttributeSet oldExtra = extractExtraAttributes(s, m);
    AttributeSet newExtra = extractExtraAttributes(sPrime, m);
    for (const auto& a : setDifference(oldExtra, newExtra)) removed.insert(a);
    for (const auto& a : setDifference(newExtra, oldExtra)) added.insert(a);
}

// FUNCTION EVALUATE_MOVE(solution s, move m, penaltyWeights lambda, H)
inline Candidate evaluateMove(const Instance& inst, const Solution& s, const Move& m,
                               const PenaltyWeights& lambda, double H,
                               double maxAllowedInfeasibility = std::numeric_limits<double>::infinity()) {
    Candidate result;

    Solution sPrime = s; // deep copy (Solution/Vehicle/Trip đều copy-able theo giá trị)

    applyMove(sPrime, m);
    removeEmptyTrips(sPrime);

    std::vector<int> touched;
    if (m.type == MoveType::Swap) {
        // với swap, các vehicle liên quan xác định trên sPrime SAU khi áp dụng (vị trí không đổi vehicle)
        CustomerLocation locI = locateCustomer(sPrime, m.customerId);
        CustomerLocation locJ = locateCustomer(sPrime, m.customerId2);
        if (locI.found) touched.push_back(sPrime.vehicles[locI.vehicleIdx]->id);
        if (locJ.found) touched.push_back(sPrime.vehicles[locJ.vehicleIdx]->id);
    } else {
        touched = affectedVehicleIds(m);
    }

    // Chỉ số vehicle (không trùng) bị move chạm vào.
    std::vector<int> touchedIdx;
    for (int vid : touched) {
        int vi = findVehicleIndexById(sPrime, vid);
        if (vi < 0) continue;
        if (std::find(touchedIdx.begin(), touchedIdx.end(), vi) == touchedIdx.end()) touchedIdx.push_back(vi);
    }

    // Loại cấu trúc sai TRƯỚC khi recompute (recompute tra inst.node(id) nên id phải hợp lệ).
    if (violatesStructuralConstraintTouched(inst, sPrime, touchedIdx)) {
        result.valid = false;
        return result;
    }

    for (int vi : touchedIdx) {
        recomputeVehicle(inst, sPrime.detachVehicle(vi), 0); // tính lại toàn bộ trip của vehicle bị ảnh hưởng
    }

    if (customerCountMismatch(inst, sPrime)) { // thiếu/thừa khách
        result.valid = false;
        return result;
    }

    // Delta evaluation: chỉ các vehicle trong touched vừa được recompute, các vehicle còn lại giữ số
    // liệu cache cũ -> tổng hợp O(số vehicle), không duyệt lại toàn bộ khách.
    aggregateSolutionMetrics(inst, sPrime, lambda, H);

    if (sPrime.totalViolation > maxAllowedInfeasibility) {
        result.valid = false;
        return result;
    }

    // removed/added tính trực tiếp từ phần thay đổi của các vehicle bị chạm (xem diffTabuAttributes).
    result.valid = true;
    diffTabuAttributes(inst, s, sPrime, m, touched, result.removedAttributes, result.addedAttributes);
    result.solution = std::move(sPrime);
    result.move = m;
    return result;
}
