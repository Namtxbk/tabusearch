// operators.hpp
// Mục 6: Các toán tử lân cận — Relocate, Or-opt(2), Swap, 2-opt, Cross-trip, Trip-relocate.
// Cung cấp: APPLY_MOVE, EXTRACT_TABU_ATTRIBUTES, và GENERATE_* cho từng toán tử.
#pragma once

#include <algorithm>
#include <vector>
#include <random>
#include <set>
#include <tuple>
#include "instance.hpp"
#include "solution.hpp"
#include "schedule.hpp"
#include "move.hpp"

// Bộ sinh uid toàn cục cho trip mới được tạo bởi các toán tử (Relocate/OrOpt2 khi tạo trip mới).
// Đơn giản hoá: dùng biến static thay vì truyền TripUidGenerator qua mọi hàm apply.
inline std::uint64_t nextGlobalTripUid() {
    static std::uint64_t counter = 1000000; // tách biệt khỏi uid do buildInitialSolution/construction cấp phát
    return counter++;
}

// ------------------------------------------------------------
// Tiện ích tìm vị trí 1 khách hàng trong Solution
// ------------------------------------------------------------
struct CustomerLocation {
    bool found = false;
    int vehicleIdx = -1;
    int tripIdx = -1;
    int positionInTrip = -1;
};

inline CustomerLocation locateCustomer(const Solution& s, int customerId) {
    CustomerLocation loc;
    for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
        const Vehicle& v = *s.vehicles[vi];
        for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
            const Trip& t = v.trips[ti];
            for (int pi = 0; pi < static_cast<int>(t.customers.size()); ++pi) {
                if (t.customers[pi] == customerId) {
                    loc.found = true;
                    loc.vehicleIdx = vi;
                    loc.tripIdx = ti;
                    loc.positionInTrip = pi;
                    return loc;
                }
            }
        }
    }
    return loc;
}

inline int findVehicleIndexById(const Solution& s, int vehicleId) {
    for (int i = 0; i < static_cast<int>(s.vehicles.size()); ++i) {
        if (s.vehicles[i]->id == vehicleId) return i;
    }
    return -1;
}

// Xóa các trip rỗng khỏi solution (giữ đúng thứ tự các trip còn lại).
// Bỏ qua (không detach/copy) những vehicle không có trip rỗng nào — tránh clone lãng phí
// mỗi khi hàm này chạy sau MỌI move (kể cả move không tạo ra trip rỗng).
inline void removeEmptyTrips(Solution& s) {
    for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
        const Vehicle& cv = *s.vehicles[vi];
        bool hasEmpty = false;
        for (const auto& t : cv.trips) {
            if (t.empty()) { hasEmpty = true; break; }
        }
        if (!hasEmpty) continue;

        Vehicle& v = s.detachVehicle(vi);
        std::vector<Trip> kept;
        kept.reserve(v.trips.size());
        for (auto& t : v.trips) {
            if (!t.empty()) kept.push_back(std::move(t));
        }
        v.trips = std::move(kept);
    }
}

// Node trước / sau 1 vị trí trong trip (0 = depot nếu ở đầu/cuối).
inline int nodeAt(const Trip& t, int pos) {
    // pos == -1 => trước phần tử đầu tiên (depot)
    // pos == size => sau phần tử cuối (depot)
    if (pos < 0 || pos >= static_cast<int>(t.customers.size())) return 0;
    return t.customers[pos];
}

// ============================================================
// EXTRACT_TABU_ATTRIBUTES(s, move)
// ============================================================
// Trích các thuộc tính cung (ARC) hiện diện trong nghiệm liên quan trực tiếp tới move.
// Với mỗi trip bị ảnh hưởng, ta lấy toàn bộ các cung (depot->i1, i_h->i_{h+1}, im->depot).
inline void collectArcsOfTrip(int vehicleId, const Trip& t, AttributeSet& out) {
    if (t.customers.empty()) return;
    int prev = 0;
    for (int c : t.customers) {
        out.insert(arcAttribute(vehicleId, prev, c));
        prev = c;
    }
    out.insert(arcAttribute(vehicleId, prev, 0));
}

// Lấy các cung của TẤT CẢ các trip bị move "chạm vào" (nguồn + đích).
// Đây là một sự đơn giản hoá an toàn: ta luôn thu thập cung tại các trip nguồn/đích
// TRƯỚC khi áp dụng move (oldAttributes) và SAU khi áp dụng move (newAttributes).
inline AttributeSet extractTabuAttributesForVehicles(const Solution& s, const std::vector<int>& vehicleIds) {
    AttributeSet out;
    for (int vid : vehicleIds) {
        int vi = findVehicleIndexById(s, vid);
        if (vi < 0) continue;
        for (const auto& t : s.vehicles[vi]->trips) {
            collectArcsOfTrip(vid, t, out);
        }
    }
    return out;
}

// ============================================================
// 6.1 Relocate
// ============================================================
// Lấy 1 khách i khỏi vị trí hiện tại, chèn vào:
//  - mọi vị trí trong mọi trip tương thích
//  - 1 trip mới tại mọi vị trí trong chuỗi trip của phương tiện tương thích
// Số target cho 1 khách là O(số vehicle tương thích × tổng độ dài trip của chúng) — với instance
// lớn (nhiều vehicle, trip dài) đây là nguồn bùng nổ candidate pool lớn nhất trong thực nghiệm
// (lớn hơn cả Swap/CrossTrip). Đếm tổng target trước (rẻ — O(số trip), không phải O(số vị trí));
// nếu vượt maxTargetsPerCustomer thì lấy mẫu ngẫu nhiên (vehicle rồi trip/vị trí trong đó) thay vì
// duyệt hết. Instance nhỏ (tổng target <= cap) giữ NGUYÊN hành vi duyệt đủ như trước.
inline std::vector<Move> generateRelocateMoves(const Instance& inst, const Solution& s,
                                                const std::vector<int>& sourceCustomers,
                                                std::mt19937& rng, int maxTargetsPerCustomer = 80) {
    std::vector<Move> moves;

    for (int custId : sourceCustomers) {
        CustomerLocation loc = locateCustomer(s, custId);
        if (!loc.found) continue;

        int sourceVehicleId = s.vehicles[loc.vehicleIdx]->id;
        int sourceTripIdx = loc.tripIdx;
        int sourcePos = loc.positionInTrip;

        std::vector<int> compatVi;
        long long totalTargets = 0;
        for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
            const Vehicle& v = *s.vehicles[vi];
            if (!staticCompatible(inst, custId, v)) continue;
            compatVi.push_back(vi);
            long long c = static_cast<long long>(v.trips.size()) + 1; // + vị trí trip mới
            for (const auto& t : v.trips) c += static_cast<long long>(t.customers.size()) + 1;
            totalTargets += c;
        }
        if (compatVi.empty()) continue;

        // insPos == -1 (dùng nội bộ) sẽ không xảy ra ở đây; ti == -1 nghĩa là trip mới.
        auto emitTarget = [&](int vi, int ti, int insPos) -> bool {
            const Vehicle& v = *s.vehicles[vi];
            if (v.id == sourceVehicleId && ti == sourceTripIdx && insPos == sourcePos) {
                return false; // đúng vị trí ban đầu -> bỏ qua
            }

            Move m;
            m.type = MoveType::Relocate;
            m.customerId = custId;
            m.sourceVehicleId = sourceVehicleId;
            m.sourceTripIndex = sourceTripIdx;
            m.sourcePosition = sourcePos;
            m.target.vehicleId = v.id;
            m.target.tripIndex = ti;
            m.target.insertionPosition = insPos;
            moves.push_back(m);
            return true;
        };

        if (totalTargets <= maxTargetsPerCustomer) {
            for (int vi : compatVi) {
                const Vehicle& v = *s.vehicles[vi];
                for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
                    int limit = static_cast<int>(v.trips[ti].customers.size());
                    for (int insPos = 0; insPos <= limit; ++insPos) emitTarget(vi, ti, insPos);
                }
                int numTrips = static_cast<int>(v.trips.size());
                for (int newTripPos = 0; newTripPos <= numTrips; ++newTripPos) emitTarget(vi, -1, newTripPos);
            }
        } else {
            std::uniform_int_distribution<std::size_t> pickVehIdx(0, compatVi.size() - 1);
            std::set<std::tuple<int,int,int>> seen;
            int added = 0, attempts = 0;
            int maxAttempts = maxTargetsPerCustomer * 4;
            while (added < maxTargetsPerCustomer && attempts < maxAttempts) {
                ++attempts;
                int vi = compatVi[pickVehIdx(rng)];
                const Vehicle& v = *s.vehicles[vi];
                int numTrips = static_cast<int>(v.trips.size());
                std::uniform_int_distribution<int> pickSlot(0, numTrips); // numTrips == "trip mới"
                int slot = pickSlot(rng);
                int ti, insPos;
                if (slot == numTrips) {
                    ti = -1;
                    std::uniform_int_distribution<int> pickNewPos(0, numTrips);
                    insPos = pickNewPos(rng);
                } else {
                    ti = slot;
                    int limit = static_cast<int>(v.trips[ti].customers.size());
                    std::uniform_int_distribution<int> pickPos(0, limit);
                    insPos = pickPos(rng);
                }
                if (!seen.insert({vi, ti, insPos}).second) continue;
                if (emitTarget(vi, ti, insPos)) ++added;
            }
        }
    }

    return moves;
}

// Áp dụng Relocate lên solution (sửa trực tiếp sPrime).
inline void applyRelocate(Solution& s, const Move& m) {
    int srcVi = findVehicleIndexById(s, m.sourceVehicleId);
    Vehicle& srcVeh = s.detachVehicle(srcVi);
    Trip& srcTrip = srcVeh.trips[m.sourceTripIndex];

    // Gỡ khách khỏi vị trí nguồn (tìm lại vị trí hiện tại theo id để an toàn)
    auto it = std::find(srcTrip.customers.begin(), srcTrip.customers.end(), m.customerId);
    srcTrip.customers.erase(it);

    int dstVi = findVehicleIndexById(s, m.target.vehicleId);
    Vehicle& dstVeh = s.detachVehicle(dstVi);

    if (m.target.tripIndex == -1) {
        // Tạo trip mới
        Trip newTrip;
        newTrip.uid = nextGlobalTripUid();
        newTrip.vehicleId = dstVeh.id;
        newTrip.customers.push_back(m.customerId);
        int insertAt = std::min(m.target.insertionPosition, static_cast<int>(dstVeh.trips.size()));
        dstVeh.trips.insert(dstVeh.trips.begin() + insertAt, std::move(newTrip));
    } else {
        int dstTripIdx = m.target.tripIndex;
        // Nếu source và dest là cùng 1 trip vật lý và ta vừa xoá 1 phần tử trước insertPos,
        // dstTripIdx vẫn đúng vì ta xác định theo index trip (không đổi do cùng trip).
        Trip& dstTrip = dstVeh.trips[dstTripIdx];
        int insPos = std::min(m.target.insertionPosition, static_cast<int>(dstTrip.customers.size()));
        dstTrip.customers.insert(dstTrip.customers.begin() + insPos, m.customerId);
    }
}

// ============================================================
// 6.2 Or-opt(2)
// ============================================================
// Giống Relocate: số target cho 1 block (p, p+1) là O(số vehicle tương thích × tổng độ dài trip) —
// đếm trước (rẻ) rồi enumerate đủ (instance nhỏ, giữ NGUYÊN hành vi cũ) hoặc lấy mẫu (instance lớn).
inline std::vector<Move> generateOrOpt2Moves(const Instance& inst, const Solution& s,
                                              const std::vector<std::pair<int,int>>& selectedTrips,
                                              std::mt19937& rng, int maxTargetsPerBlock = 80) {
    // selectedTrips: danh sách (vehicleIdx, tripIdx) cần xét làm nguồn
    std::vector<Move> moves;

    for (const auto& st : selectedTrips) {
        int srcVi = st.first, srcTi = st.second;
        if (srcVi < 0 || srcVi >= static_cast<int>(s.vehicles.size())) continue;
        const Vehicle& srcVeh = *s.vehicles[srcVi];
        if (srcTi < 0 || srcTi >= static_cast<int>(srcVeh.trips.size())) continue;
        const Trip& srcTrip = srcVeh.trips[srcTi];
        int m = static_cast<int>(srcTrip.customers.size());

        for (int p = 0; p <= m - 2; ++p) {
            int c1 = srcTrip.customers[p];
            int c2 = srcTrip.customers[p + 1];

            std::vector<int> compatVi;
            long long totalTargets = 0;
            for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
                const Vehicle& v = *s.vehicles[vi];
                if (!staticCompatible(inst, c1, v) || !staticCompatible(inst, c2, v)) continue;
                compatVi.push_back(vi);
                long long c = static_cast<long long>(v.trips.size()) + 1;
                for (const auto& t : v.trips) c += static_cast<long long>(t.customers.size()) + 1;
                totalTargets += c;
            }
            if (compatVi.empty()) continue;

            auto emitTarget = [&](int vi, int ti, int insPos) -> bool {
                const Vehicle& v = *s.vehicles[vi];
                bool sameTrip = (v.id == srcVeh.id && ti == srcTi);
                if (sameTrip && insPos >= p && insPos <= p + 2) return false; // trùng/chồng vị trí gốc

                Move mv;
                mv.type = MoveType::OrOpt2;
                mv.customerId = c1;
                mv.customerId2 = c2;
                mv.sourceVehicleId = srcVeh.id;
                mv.sourceTripIndex = srcTi;
                mv.sourcePosition = p;
                mv.target.vehicleId = v.id;
                mv.target.tripIndex = ti;
                mv.target.insertionPosition = insPos;
                moves.push_back(mv);
                return true;
            };

            if (totalTargets <= maxTargetsPerBlock) {
                for (int vi : compatVi) {
                    const Vehicle& v = *s.vehicles[vi];
                    for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
                        int limit = static_cast<int>(v.trips[ti].customers.size());
                        for (int insPos = 0; insPos <= limit; ++insPos) emitTarget(vi, ti, insPos);
                    }
                    int numTrips = static_cast<int>(v.trips.size());
                    for (int newTripPos = 0; newTripPos <= numTrips; ++newTripPos) emitTarget(vi, -1, newTripPos);
                }
            } else {
                std::uniform_int_distribution<std::size_t> pickVehIdx(0, compatVi.size() - 1);
                std::set<std::tuple<int,int,int>> seen;
                int added = 0, attempts = 0;
                int maxAttempts = maxTargetsPerBlock * 4;
                while (added < maxTargetsPerBlock && attempts < maxAttempts) {
                    ++attempts;
                    int vi = compatVi[pickVehIdx(rng)];
                    const Vehicle& v = *s.vehicles[vi];
                    int numTrips = static_cast<int>(v.trips.size());
                    std::uniform_int_distribution<int> pickSlot(0, numTrips);
                    int slot = pickSlot(rng);
                    int ti, insPos;
                    if (slot == numTrips) {
                        ti = -1;
                        std::uniform_int_distribution<int> pickNewPos(0, numTrips);
                        insPos = pickNewPos(rng);
                    } else {
                        ti = slot;
                        int limit = static_cast<int>(v.trips[ti].customers.size());
                        std::uniform_int_distribution<int> pickPos(0, limit);
                        insPos = pickPos(rng);
                    }
                    if (!seen.insert({vi, ti, insPos}).second) continue;
                    if (emitTarget(vi, ti, insPos)) ++added;
                }
            }
        }
    }

    return moves;
}

inline void applyOrOpt2(Solution& s, const Move& m) {
    int srcVi = findVehicleIndexById(s, m.sourceVehicleId);
    Vehicle& srcVeh = s.detachVehicle(srcVi);
    Trip& srcTrip = srcVeh.trips[m.sourceTripIndex];

    // Gỡ block 2 khách (giữ thứ tự) khỏi vị trí nguồn — tìm theo id để an toàn.
    auto it1 = std::find(srcTrip.customers.begin(), srcTrip.customers.end(), m.customerId);
    int p1 = static_cast<int>(it1 - srcTrip.customers.begin());
    // block liền kề: phần tử tiếp theo phải là customerId2
    std::vector<int> block = { m.customerId, m.customerId2 };
    srcTrip.customers.erase(srcTrip.customers.begin() + p1, srcTrip.customers.begin() + p1 + 2);

    int dstVi = findVehicleIndexById(s, m.target.vehicleId);
    Vehicle& dstVeh = s.detachVehicle(dstVi);

    if (m.target.tripIndex == -1) {
        Trip newTrip;
        newTrip.uid = nextGlobalTripUid();
        newTrip.vehicleId = dstVeh.id;
        newTrip.customers = block;
        int insertAt = std::min(m.target.insertionPosition, static_cast<int>(dstVeh.trips.size()));
        dstVeh.trips.insert(dstVeh.trips.begin() + insertAt, std::move(newTrip));
    } else {
        Trip& dstTrip = dstVeh.trips[m.target.tripIndex];
        int insPos = std::min(m.target.insertionPosition, static_cast<int>(dstTrip.customers.size()));
        dstTrip.customers.insert(dstTrip.customers.begin() + insPos, block.begin(), block.end());
    }
}

// ============================================================
// 6.3 Swap
// ============================================================
// Sinh move Swap cho 1 cặp (i, j) đã chọn theo vị trí trong selectedCustomers, trả về true nếu
// cặp hợp lệ (đã push vào moves).
inline bool trySwapPair(const Instance& inst, const Solution& s, int i, int j, std::vector<Move>& moves) {
    if (i == j) return false;

    CustomerLocation locI = locateCustomer(s, i);
    CustomerLocation locJ = locateCustomer(s, j);
    if (!locI.found || !locJ.found) return false;

    const Vehicle& vehicleI = *s.vehicles[locI.vehicleIdx];
    const Vehicle& vehicleJ = *s.vehicles[locJ.vehicleIdx];

    if (!staticCompatible(inst, i, vehicleJ)) return false;
    if (!staticCompatible(inst, j, vehicleI)) return false;
    if (vehicleI.id == vehicleJ.id && locI.tripIdx == locJ.tripIdx &&
        locI.positionInTrip == locJ.positionInTrip) return false; // không đổi gì

    Move m;
    m.type = MoveType::Swap;
    m.customerId = i;
    m.customerId2 = j;
    moves.push_back(m);
    return true;
}

// Số cặp (i,j) trong selectedCustomers là O(k^2) — với instance lớn (k ~ vài chục),
// việc duyệt hết mọi cặp làm bùng nổ candidate pool (mỗi candidate lại tốn 1 lần evaluateMove).
// Nếu số cặp vượt quá maxPairs, chuyển sang lấy mẫu ngẫu nhiên thay vì duyệt hết — với instance
// nhỏ (số cặp <= maxPairs) hành vi giữ NGUYÊN như duyệt đầy đủ trước đây.
inline std::vector<Move> generateSwapMoves(const Instance& inst, const Solution& s,
                                            const std::vector<int>& selectedCustomers,
                                            std::mt19937& rng, int maxPairs = 600) {
    std::vector<Move> moves;
    std::size_t k = selectedCustomers.size();
    long long totalPairs = (k >= 2) ? (static_cast<long long>(k) * static_cast<long long>(k - 1) / 2) : 0;

    if (totalPairs <= maxPairs) {
        for (std::size_t a = 0; a < k; ++a) {
            for (std::size_t b = a + 1; b < k; ++b) {
                trySwapPair(inst, s, selectedCustomers[a], selectedCustomers[b], moves);
            }
        }
        return moves;
    }

    // Lấy mẫu ngẫu nhiên maxPairs cặp chỉ số phân biệt (không lặp lại trong 1 lần gọi).
    std::uniform_int_distribution<std::size_t> pick(0, k - 1);
    std::set<std::pair<std::size_t,std::size_t>> seenIdx;
    int added = 0;
    int attempts = 0;
    int maxAttempts = maxPairs * 4;
    while (added < maxPairs && attempts < maxAttempts) {
        ++attempts;
        std::size_t ia = pick(rng), ib = pick(rng);
        if (ia == ib) continue;
        if (ia > ib) std::swap(ia, ib);
        if (!seenIdx.insert({ia, ib}).second) continue;
        if (trySwapPair(inst, s, selectedCustomers[ia], selectedCustomers[ib], moves)) ++added;
    }
    return moves;
}

inline void applySwap(Solution& s, const Move& m) {
    CustomerLocation locI = locateCustomer(s, m.customerId);
    CustomerLocation locJ = locateCustomer(s, m.customerId2);

    Vehicle& vehI = s.detachVehicle(locI.vehicleIdx);
    Vehicle& vehJ = s.detachVehicle(locJ.vehicleIdx);

    Trip& tripI = vehI.trips[locI.tripIdx];
    Trip& tripJ = vehJ.trips[locJ.tripIdx];

    std::swap(tripI.customers[locI.positionInTrip], tripJ.customers[locJ.positionInTrip]);
}

// ============================================================
// 6.4 2-opt
// ============================================================
inline std::vector<Move> generateTwoOptMoves(const Instance& /*inst*/, const Solution& s, const std::vector<std::pair<int,int>>& selectedTrips) {
    std::vector<Move> moves;

    for (const auto& st : selectedTrips) {
        int vi = st.first, ti = st.second;
        if (vi < 0 || vi >= static_cast<int>(s.vehicles.size())) continue;
        const Vehicle& v = *s.vehicles[vi];
        if (ti < 0 || ti >= static_cast<int>(v.trips.size())) continue;
        const Trip& t = v.trips[ti];
        int m = static_cast<int>(t.customers.size());

        for (int p = 0; p <= m - 2; ++p) {
            for (int q = p + 1; q <= m - 1; ++q) {
                if (q == p + 1) continue; // đảo 1 phần tử = không đổi

                Move mv;
                mv.type = MoveType::TwoOpt;
                mv.tripVehicleId = v.id;
                mv.tripIndexForTwoOpt = ti;
                mv.p = p;
                mv.q = q;
                moves.push_back(mv);
            }
        }
    }

    return moves;
}

inline void applyTwoOpt(Solution& s, const Move& m) {
    int vi = findVehicleIndexById(s, m.tripVehicleId);
    Vehicle& v = s.detachVehicle(vi);
    Trip& t = v.trips[m.tripIndexForTwoOpt];
    std::reverse(t.customers.begin() + m.p, t.customers.begin() + m.q + 1);
}

// ============================================================
// 6.5 Cross-trip
// ============================================================
// Với 1 cặp trip (lenA, lenB customer), số tổ hợp (cutA, cutB) là O(lenA*lenB) — với trip dài
// (instance lớn, nhiều khách/trip) và nhiều cặp trip được chọn, tổng số move CrossTrip bùng nổ
// rất nhanh (mỗi move lại tốn 1 lần evaluateMove với deep-copy). Nếu tổ hợp vượt maxCombosPerTripPair
// thì lấy mẫu ngẫu nhiên thay vì duyệt hết; instance nhỏ (tổ hợp <= cap) giữ NGUYÊN hành vi duyệt đủ.
// maxTotalMoves là giới hạn tổng (an toàn thêm khi có nhiều cặp trip được chọn).
inline std::vector<Move> generateCrossTripMoves(const Instance& inst, const Solution& s,
                                                 const std::vector<std::pair<int,int>>& selectedTrips,
                                                 std::mt19937& rng,
                                                 int maxCombosPerTripPair = 40, int maxTotalMoves = 800) {
    std::vector<Move> moves;

    for (size_t a = 0; a < selectedTrips.size() && static_cast<int>(moves.size()) < maxTotalMoves; ++a) {
        for (size_t b = a + 1; b < selectedTrips.size() && static_cast<int>(moves.size()) < maxTotalMoves; ++b) {
            int viA = selectedTrips[a].first, tiA = selectedTrips[a].second;
            int viB = selectedTrips[b].first, tiB = selectedTrips[b].second;
            if (viA < 0 || viA >= static_cast<int>(s.vehicles.size())) continue;
            if (viB < 0 || viB >= static_cast<int>(s.vehicles.size())) continue;

            const Vehicle& vehA = *s.vehicles[viA];
            const Vehicle& vehB = *s.vehicles[viB];
            if (tiA < 0 || tiA >= static_cast<int>(vehA.trips.size())) continue;
            if (tiB < 0 || tiB >= static_cast<int>(vehB.trips.size())) continue;

            const Trip& ta = vehA.trips[tiA];
            const Trip& tb = vehB.trips[tiB];
            int lenA = static_cast<int>(ta.customers.size());
            int lenB = static_cast<int>(tb.customers.size());

            auto tryCombo = [&](int cutA, int cutB) -> bool {
                bool tailAEmpty = (cutA == lenA);
                bool tailBEmpty = (cutB == lenB);
                if (tailAEmpty && tailBEmpty) return false;

                bool ok = true;
                for (int k = cutA; k < lenA && ok; ++k) {
                    if (!staticCompatible(inst, ta.customers[k], vehB)) ok = false;
                }
                for (int k = cutB; k < lenB && ok; ++k) {
                    if (!staticCompatible(inst, tb.customers[k], vehA)) ok = false;
                }
                if (!ok) return false;

                Move mv;
                mv.type = MoveType::CrossTrip;
                mv.vehicleA = vehA.id; mv.tripIndexA = tiA; mv.cutA = cutA;
                mv.vehicleB = vehB.id; mv.tripIndexB = tiB; mv.cutB = cutB;
                moves.push_back(mv);
                return true;
            };

            long long totalCombos = static_cast<long long>(lenA + 1) * static_cast<long long>(lenB + 1);
            if (totalCombos <= maxCombosPerTripPair) {
                for (int cutA = 0; cutA <= lenA; ++cutA) {
                    for (int cutB = 0; cutB <= lenB; ++cutB) {
                        tryCombo(cutA, cutB);
                    }
                }
            } else {
                std::uniform_int_distribution<int> pickA(0, lenA);
                std::uniform_int_distribution<int> pickB(0, lenB);
                std::set<std::pair<int,int>> seenCombo;
                int added = 0, attempts = 0;
                int maxAttempts = maxCombosPerTripPair * 4;
                while (added < maxCombosPerTripPair && attempts < maxAttempts) {
                    ++attempts;
                    int cutA = pickA(rng), cutB = pickB(rng);
                    if (!seenCombo.insert({cutA, cutB}).second) continue;
                    if (tryCombo(cutA, cutB)) ++added;
                }
            }
        }
    }

    return moves;
}

inline void applyCrossTrip(Solution& s, const Move& m) {
    int viA = findVehicleIndexById(s, m.vehicleA);
    int viB = findVehicleIndexById(s, m.vehicleB);
    Vehicle& vehA = s.detachVehicle(viA);
    Vehicle& vehB = s.detachVehicle(viB);
    Trip& ta = vehA.trips[m.tripIndexA];
    Trip& tb = vehB.trips[m.tripIndexB];

    std::vector<int> tailA(ta.customers.begin() + m.cutA, ta.customers.end());
    std::vector<int> tailB(tb.customers.begin() + m.cutB, tb.customers.end());

    ta.customers.erase(ta.customers.begin() + m.cutA, ta.customers.end());
    tb.customers.erase(tb.customers.begin() + m.cutB, tb.customers.end());

    ta.customers.insert(ta.customers.end(), tailB.begin(), tailB.end());
    tb.customers.insert(tb.customers.end(), tailA.begin(), tailA.end());
}

// ============================================================
// 6.6 Trip-relocate
// ============================================================
inline std::vector<Move> generateTripRelocateMoves(const Instance& inst, const Solution& s, const std::vector<std::pair<int,int>>& selectedTrips) {
    std::vector<Move> moves;

    for (const auto& st : selectedTrips) {
        int srcVi = st.first, srcTi = st.second;
        if (srcVi < 0 || srcVi >= static_cast<int>(s.vehicles.size())) continue;
        const Vehicle& srcVeh = *s.vehicles[srcVi];
        if (srcTi < 0 || srcTi >= static_cast<int>(srcVeh.trips.size())) continue;
        const Trip& t = srcVeh.trips[srcTi];

        for (const auto& vp : s.vehicles) {
            const Vehicle& v = *vp;
            bool compatible = true;
            for (int custId : t.customers) {
                if (!staticCompatible(inst, custId, v)) { compatible = false; break; }
            }
            if (!compatible) continue;

            int numTrips = static_cast<int>(v.trips.size());
            for (int destPos = 0; destPos <= numTrips; ++destPos) {
                bool isOriginalPos = (v.id == srcVeh.id && (destPos == srcTi || destPos == srcTi + 1));
                if (v.id == srcVeh.id && destPos == srcTi) continue; // vị trí gốc
                (void)isOriginalPos;

                Move mv;
                mv.type = MoveType::TripRelocate;
                mv.tripUid = t.uid;
                mv.sourceVehicleId = srcVeh.id;
                mv.sourceTripIndex = srcTi;
                mv.target.vehicleId = v.id;
                mv.target.insertionPosition = destPos;
                moves.push_back(mv);
            }
        }
    }

    return moves;
}

inline void applyTripRelocate(Solution& s, const Move& m) {
    int srcVi = findVehicleIndexById(s, m.sourceVehicleId);
    Vehicle& srcVeh = s.detachVehicle(srcVi);

    // Tìm trip theo uid (an toàn hơn index vì index có thể đã lệch)
    int srcTi = -1;
    for (int i = 0; i < static_cast<int>(srcVeh.trips.size()); ++i) {
        if (srcVeh.trips[i].uid == m.tripUid) { srcTi = i; break; }
    }
    Trip moved = std::move(srcVeh.trips[srcTi]);
    srcVeh.trips.erase(srcVeh.trips.begin() + srcTi);

    int dstVi = findVehicleIndexById(s, m.target.vehicleId);
    Vehicle& dstVeh = s.detachVehicle(dstVi);
    int insertAt = std::min(m.target.insertionPosition, static_cast<int>(dstVeh.trips.size()));
    moved.vehicleId = dstVeh.id;
    dstVeh.trips.insert(dstVeh.trips.begin() + insertAt, std::move(moved));
}

// ============================================================
// APPLY_MOVE tổng hợp
// ============================================================
inline void applyMove(Solution& s, const Move& m) {
    switch (m.type) {
        case MoveType::Relocate:     applyRelocate(s, m); break;
        case MoveType::OrOpt2:       applyOrOpt2(s, m); break;
        case MoveType::Swap:         applySwap(s, m); break;
        case MoveType::TwoOpt:       applyTwoOpt(s, m); break;
        case MoveType::CrossTrip:    applyCrossTrip(s, m); break;
        case MoveType::TripRelocate: applyTripRelocate(s, m); break;
    }
}

// Danh sách các vehicleId bị move "chạm vào" (dùng để recompute + extract tabu attributes).
inline std::vector<int> affectedVehicleIds(const Move& m) {
    std::vector<int> ids;
    switch (m.type) {
        case MoveType::Relocate:
        case MoveType::OrOpt2:
            ids.push_back(m.sourceVehicleId);
            ids.push_back(m.target.vehicleId);
            break;
        case MoveType::Swap:
            // sẽ được điền bởi caller dựa trên vị trí thực tế trước khi move (xem evaluate_move.hpp)
            break;
        case MoveType::TwoOpt:
            ids.push_back(m.tripVehicleId);
            break;
        case MoveType::CrossTrip:
            ids.push_back(m.vehicleA);
            ids.push_back(m.vehicleB);
            break;
        case MoveType::TripRelocate:
            ids.push_back(m.sourceVehicleId);
            ids.push_back(m.target.vehicleId);
            break;
    }
    return ids;
}
