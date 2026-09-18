// construction.hpp
// "Init solution" pseudocode ở đầu tài liệu: chèn tuần tự khách hàng theo deadline tăng dần,
// chọn move tốt nhất trong feasibleMoves (không tăng vi phạm TW) hoặc, nếu rỗng,
// penalizedMoves theo thứ tự từ điển (ΔTW, makespan, khoảng cách thêm vào).
// Cũng dùng lại được cho bước Recreate của Ruin & Recreate (mục 14.2).
#pragma once

#include <algorithm>
#include <vector>
#include <limits>
#include <random>
#include "instance.hpp"
#include "solution.hpp"
#include "schedule.hpp"
#include "move.hpp"
#include "operators.hpp"
#include "evaluate_move.hpp"
#include "feasibility.hpp"

// GENERATE_ALL_INSERTIONS(partial, i): mọi vị trí trong mọi trip tương thích của customer i,
// và 1 trip mới tại mọi vị trí trong chuỗi trip của mỗi phương tiện tương thích.
// Đây thực chất là move Relocate với "customer i chưa nằm trong nghiệm" — ta mô phỏng bằng cách
// thêm khách vào 1 trip tạm rỗng ("virtual holding trip") trước, rồi coi là Relocate.
// Để đơn giản & đúng ngữ nghĩa APPLY_MOVE hiện có, ta cung cấp hàm chuyên biệt sinh & áp dụng
// "insertion move" (không cần trip nguồn).
struct InsertionMove {
    int customerId = -1;
    int targetVehicleId = -1;
    int targetTripIndex = -1;   // -1 = trip mới
    int insertionPosition = 0;
    bool forbidTrucks0 = false; // dùng ở Recreate: không ép khách vào trucks[0]
};

inline std::vector<InsertionMove> generateAllInsertions(const Instance& inst, const Solution& s, int custId, bool forbidTrucks0FirstTruck = false) {
    std::vector<InsertionMove> moves;

    for (int vi = 0; vi < static_cast<int>(s.vehicles.size()); ++vi) {
        const Vehicle& v = s.vehicles[vi];
        if (!staticCompatible(inst, custId, v)) continue;

        bool isFirstTruck = (v.type == VehicleType::TRUCK) && (vi == 0);
        if (forbidTrucks0FirstTruck && isFirstTruck) {
            // "Không ép khách vào trucks[0]" — vẫn cho phép nếu không còn lựa chọn nào khác;
            // ở đây ta chỉ *hạ thấp ưu tiên* bằng cách vẫn sinh move nhưng đánh dấu lại,
            // caller (Recreate) có thể lọc bớt nếu có phương tiện khác khả dụng.
        }

        for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
            int limit = static_cast<int>(v.trips[ti].customers.size());
            for (int pos = 0; pos <= limit; ++pos) {
                InsertionMove im;
                im.customerId = custId;
                im.targetVehicleId = v.id;
                im.targetTripIndex = ti;
                im.insertionPosition = pos;
                moves.push_back(im);
            }
        }

        int numTrips = static_cast<int>(v.trips.size());
        for (int newPos = 0; newPos <= numTrips; ++newPos) {
            InsertionMove im;
            im.customerId = custId;
            im.targetVehicleId = v.id;
            im.targetTripIndex = -1;
            im.insertionPosition = newPos;
            moves.push_back(im);
        }
    }

    return moves;
}

inline void applyInsertion(Solution& s, const InsertionMove& im) {
    int vi = findVehicleIndexById(s, im.targetVehicleId);
    Vehicle& v = s.vehicles[vi];

    if (im.targetTripIndex == -1) {
        Trip newTrip;
        newTrip.uid = nextGlobalTripUid();
        newTrip.vehicleId = v.id;
        newTrip.customers.push_back(im.customerId);
        int insertAt = std::min(im.insertionPosition, static_cast<int>(v.trips.size()));
        v.trips.insert(v.trips.begin() + insertAt, std::move(newTrip));
    } else {
        Trip& t = v.trips[im.targetTripIndex];
        int pos = std::min(im.insertionPosition, static_cast<int>(t.customers.size()));
        t.customers.insert(t.customers.begin() + pos, im.customerId);
    }
}

// Đánh giá 1 insertion move (tương tự EVALUATE_MOVE nhưng cho việc chèn khách chưa có trong nghiệm).
struct InsertionCandidate {
    bool valid = false;
    bool feasibleNoAdditionalTW = false; // "creates no additional TW violation"
    double deltaTW = 0.0;                // penalty tăng thêm nếu không feasible-TW
    double resultingMakespan = 0.0;
    double additionalDistance = 0.0;
    Solution solution;
    InsertionMove move;
};

inline InsertionCandidate evaluateInsertion(const Instance& inst, const Solution& s, const InsertionMove& im,
                                             const PenaltyWeights& lambda, double H,
                                             double baselineDistance) {
    InsertionCandidate result;
    Solution sPrime = s;
    applyInsertion(sPrime, im);

    int vi = findVehicleIndexById(sPrime, im.targetVehicleId);
    if (vi < 0) { result.valid = false; return result; }
    recomputeVehicle(inst, sPrime.vehicles[vi], 0);

    if (violatesStructuralConstraint(inst, sPrime, /*forceAllCustomersPresent=*/false)) {
        result.valid = false;
        return result;
    }

    evaluateSolution(inst, sPrime, lambda, H);

    result.valid = true;
    result.solution = std::move(sPrime);
    result.move = im;
    result.resultingMakespan = result.solution.makespan;
    result.additionalDistance = result.solution.totalDistance - baselineDistance;

    // ΔTW(a) = tổng penalty(t_j - l_j) cho các khách đã chèn trong solution x của move a.
    // Ở construction, "solution x của move a" = toàn bộ solution sau khi chèn -> ta tính theo
    // định nghĩa compute ΔTW(a) trong tài liệu: chỉ xét khách ĐÃ ĐƯỢC CHÈN (tức là khách vừa thêm).
    const Customer& c = inst.node(im.customerId);
    const Trip& targetTrip = (im.targetTripIndex == -1)
        ? result.solution.vehicles[vi].trips[std::min(im.insertionPosition, (int)result.solution.vehicles[vi].trips.size() - 1)]
        : result.solution.vehicles[vi].trips[im.targetTripIndex];
    double arrival = targetTrip.arrivalTime.count(im.customerId) ? targetTrip.arrivalTime.at(im.customerId) : 0.0;
    double penalty = std::max(0.0, arrival - c.due);
    result.deltaTW = penalty;
    result.feasibleNoAdditionalTW = (penalty <= EPS);

    return result;
}

// Xây dựng nghiệm khởi tạo rỗng: mỗi vehicle không có trip nào.
inline Solution buildEmptySolution(const Instance& inst) {
    Solution s;
    int nextVehicleId = 0;
    for (int i = 0; i < inst.num_trucks; ++i) {
        Vehicle v;
        v.id = nextVehicleId++;
        v.type = VehicleType::TRUCK;
        s.vehicles.push_back(v);
    }
    for (int i = 0; i < inst.num_drones; ++i) {
        Vehicle v;
        v.id = nextVehicleId++;
        v.type = VehicleType::DRONE;
        s.vehicles.push_back(v);
    }
    return s;
}

// PROCEDURE Init solution (đầu tài liệu):
// Sắp khách theo deadline tăng dần; với mỗi khách, sinh mọi move khả dĩ trên mọi vehicle tương thích;
// nếu có move không tăng vi phạm TW -> chọn move có makespan kết quả nhỏ nhất;
// nếu không -> chọn theo thứ tự từ điển (deltaTW tăng, makespan, khoảng cách thêm vào).
inline Solution buildInitialSolution(const Instance& inst, const PenaltyWeights& lambda, double& outH, TripUidGenerator& uidGen) {
    (void)uidGen; // uid được cấp phát tự động bởi nextGlobalTripUid() ngay khi tạo trip mới
    Solution s = buildEmptySolution(inst);

    std::vector<Customer> sorted = inst.customers;
    std::sort(sorted.begin(), sorted.end(), [](const Customer& a, const Customer& b) { return a.due < b.due; });

    // H tạm thời để đánh giá trong quá trình construction; sẽ tính lại chính xác sau khi có s0 đầy đủ.
    double HTemp = 1.0;
    for (const auto& c : inst.customers) HTemp = std::max(HTemp, c.due);

    for (const auto& cust : sorted) {
        double baselineDistance = s.totalDistance;
        auto insertions = generateAllInsertions(inst, s, cust.id);

        std::vector<InsertionCandidate> feasibleMoves;
        std::vector<InsertionCandidate> penalizedMoves;

        for (const auto& im : insertions) {
            InsertionCandidate cand = evaluateInsertion(inst, s, im, lambda, HTemp, baselineDistance);
            if (!cand.valid) continue;
            if (cand.feasibleNoAdditionalTW) feasibleMoves.push_back(cand);
            else penalizedMoves.push_back(cand);
        }

        const InsertionCandidate* chosen = nullptr;
        if (!feasibleMoves.empty()) {
            chosen = &feasibleMoves[0];
            for (const auto& c : feasibleMoves) {
                if (c.resultingMakespan < chosen->resultingMakespan - EPS) chosen = &c;
            }
        } else if (!penalizedMoves.empty()) {
            chosen = &penalizedMoves[0];
            for (const auto& c : penalizedMoves) {
                if (c.deltaTW < chosen->deltaTW - EPS) { chosen = &c; continue; }
                if (std::fabs(c.deltaTW - chosen->deltaTW) <= EPS) {
                    if (c.resultingMakespan < chosen->resultingMakespan - EPS) { chosen = &c; continue; }
                    if (std::fabs(c.resultingMakespan - chosen->resultingMakespan) <= EPS) {
                        if (c.additionalDistance < chosen->additionalDistance - EPS) { chosen = &c; }
                    }
                }
            }
        }

        if (chosen != nullptr) {
            s = chosen->solution;
        }
        // Nếu không có move nào hợp lệ (rất hiếm với move rỗng luôn khả dụng), bỏ qua khách này.
    }

    for (auto& v : s.vehicles) recomputeVehicleFull(inst, v);

    double maxCompletion = 0.0;
    for (const auto& v : s.vehicles) maxCompletion = std::max(maxCompletion, v.completionTime);
    outH = computeH(inst, maxCompletion);

    return s;
}
