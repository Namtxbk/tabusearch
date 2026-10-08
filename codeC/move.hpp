// move.hpp
// Định nghĩa các loại move (mục 6) và thuộc tính tabu (mục 9).
#pragma once

#include <vector>
#include <set>
#include <cstdint>
#include <functional>

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
// Trước đây dựng key dạng string qua ostringstream (formatting/locale khá chậm, gọi trên mọi
// candidate). Thay bằng key số nguyên thuần (kind + tối đa 4 trường) — cùng ngữ nghĩa phân biệt
// thuộc tính, chỉ khác cách biểu diễn để so sánh/hash rẻ hơn.
enum class TabuAttrKind { Arc, Assign, TripReturn };

struct TabuAttribute {
    TabuAttrKind kind = TabuAttrKind::Arc;
    long long a = 0, b = 0, c = 0;

    bool operator==(const TabuAttribute& o) const {
        return kind == o.kind && a == o.a && b == o.b && c == o.c;
    }
    bool operator<(const TabuAttribute& o) const {
        if (kind != o.kind) return kind < o.kind;
        if (a != o.a) return a < o.a;
        if (b != o.b) return b < o.b;
        return c < o.c;
    }
};

struct TabuAttributeHash {
    std::size_t operator()(const TabuAttribute& t) const noexcept {
        std::size_t h = std::hash<int>{}(static_cast<int>(t.kind));
        auto mix = [&h](long long v) {
            std::size_t hv = std::hash<long long>{}(v);
            h ^= hv + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(t.a); mix(t.b); mix(t.c);
        return h;
    }
};

inline TabuAttribute arcAttribute(int vehicleId, int fromNode, int toNode) {
    return TabuAttribute{TabuAttrKind::Arc, vehicleId, fromNode, toNode};
}

inline TabuAttribute assignAttribute(int customerId, int forbiddenVehicleId) {
    return TabuAttribute{TabuAttrKind::Assign, customerId, forbiddenVehicleId, 0};
}

inline TabuAttribute tripReturnAttribute(std::uint64_t tripUid, int sourceVehicleId, std::uint64_t sourcePredecessorTripUid) {
    return TabuAttribute{TabuAttrKind::TripReturn, static_cast<long long>(tripUid), sourceVehicleId,
                          static_cast<long long>(sourcePredecessorTripUid)};
}

// AttributeSet trước đây là std::set<TabuAttribute> (cây đỏ-đen, cấp phát 1 node/phần tử) —
// nhưng mỗi tập chỉ có vài chục phần tử (arc của 1-2 trip bị move đụng tới) và được dựng LẶP LẠI
// (2 lần: old + new attribute) trên MỌI candidate. Với kích thước nhỏ như vậy, vector + scan tuyến
// tính nhanh hơn nhiều (không cấp phát heap cho từng phần tử, cache-friendly hơn cây).
class AttributeSet {
public:
    void insert(const TabuAttribute& a) {
        if (!contains(a)) items_.push_back(a);
    }
    // Chèn KHÔNG kiểm tra trùng — chỉ dùng khi caller đảm bảo phần tử là duy nhất (vd. các cung của 1 solution).
    void insertUnique(const TabuAttribute& a) { items_.push_back(a); }
    bool contains(const TabuAttribute& a) const {
        for (const auto& x : items_) {
            if (x == a) return true;
        }
        return false;
    }
    bool empty() const { return items_.empty(); }
    std::size_t size() const { return items_.size(); }

    std::vector<TabuAttribute>::const_iterator begin() const { return items_.begin(); }
    std::vector<TabuAttribute>::const_iterator end() const { return items_.end(); }

private:
    std::vector<TabuAttribute> items_;
};

inline AttributeSet setDifference(const AttributeSet& a, const AttributeSet& b) {
    AttributeSet result;
    for (const auto& x : a) {
        if (!b.contains(x)) result.insert(x);
    }
    return result;
}
