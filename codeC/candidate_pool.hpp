// candidate_pool.hpp
// Mục 15: Sinh candidate pool tổng hợp - BUILD_CANDIDATE_POOL.
//
// Có 3 chế độ sinh lân cận (NeighborhoodMode):
//   ALL        : mỗi vòng lặp chạy cả 6 toán tử (hành vi gốc).
//   RANDOM_OP  : mỗi vòng lặp chọn NGẪU NHIÊN 1 toán tử, rồi lấy mẫu ngẫu nhiên tối đa
//                sampleSize move của toán tử đó.
//   SEQUENTIAL : mỗi vòng lặp dùng 1 toán tử theo thứ tự xoay vòng
//                Relocate -> Or-opt(2) -> Swap -> 2-opt -> Cross-trip -> Trip-relocate -> Relocate ...
//                rồi lấy mẫu ngẫu nhiên tối đa sampleSize move của toán tử đó (giống RANDOM_OP,
//                chỉ khác cách chọn toán tử -> so sánh công bằng).
//   ADAPTIVE   : chọn toán tử theo trọng số (roulette wheel), trọng số được cập nhật dần theo hiệu quả
//                của từng toán tử (operator_weights.hpp); lấy mẫu sampleSize move như RANDOM_OP.
// Với RANDOM_OP / SEQUENTIAL / ADAPTIVE: nếu toán tử được chọn không sinh được candidate hợp lệ nào thì
// chuyển sang toán tử khác (ngẫu nhiên / kế tiếp) để vòng lặp không bị lãng phí.
#pragma once

#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <numeric>
#include "instance.hpp"
#include "solution.hpp"
#include "move.hpp"
#include "operators.hpp"
#include "evaluate_move.hpp"
#include "select_components.hpp"
#include "operator_weights.hpp"

enum class NeighborhoodMode { ALL, RANDOM_OP, SEQUENTIAL, ADAPTIVE };

inline const char* neighborhoodModeName(NeighborhoodMode m) {
    switch (m) {
        case NeighborhoodMode::ALL:        return "all";
        case NeighborhoodMode::RANDOM_OP:  return "random";
        case NeighborhoodMode::SEQUENTIAL: return "seq";
        case NeighborhoodMode::ADAPTIVE:   return "adaptive";
    }
    return "?";
}

constexpr int NUM_OPERATORS = 6;
static_assert(NUM_OPERATORS == AOS_NUM_OPERATORS, "so toan tu phai khop operator_weights.hpp");
inline const char* const* operatorNames() {
    static const char* names[NUM_OPERATORS] = {"Relocate", "OrOpt2", "Swap", "TwoOpt", "CrossTrip", "TripRelocate"};
    return names;
}
inline const char* operatorName(int op) {
    return (op >= 0 && op < NUM_OPERATORS) ? operatorNames()[op] : "?";
}

// Sinh move của 1 toán tử (0..5) trên các thành phần đã chọn.
inline std::vector<Move> generateMovesForOperator(int op, const Instance& inst, const Solution& current,
                                                   const SearchComponents& comp, std::mt19937& rng) {
    switch (op) {
        case 0: return generateRelocateMoves(inst, current, comp.selectedCustomers, rng);
        case 1: return generateOrOpt2Moves(inst, current, comp.selectedTrips, rng);
        case 2: return generateSwapMoves(inst, current, comp.selectedCustomers, rng);
        case 3: return generateTwoOptMoves(inst, current, comp.selectedTrips);
        case 4: return generateCrossTripMoves(inst, current, comp.selectedTrips, rng);
        case 5: return generateTripRelocateMoves(inst, current, comp.selectedTrips);
    }
    return {};
}

inline void evaluateMovesInto(const Instance& inst, const Solution& current, const std::vector<Move>& moves,
                              const PenaltyWeights& lambda, double H, double maxAllowedInfeasibility,
                              std::vector<Candidate>& pool) {
    for (const auto& m : moves) {
        Candidate c = evaluateMove(inst, current, m, lambda, H, maxAllowedInfeasibility);
        if (c.valid) pool.push_back(std::move(c));
    }
}

struct NeighborhoodConfig {
    NeighborhoodMode mode = NeighborhoodMode::ALL;
    int sampleSize = 100;   // RANDOM_OP / SEQUENTIAL / ADAPTIVE: số move tối đa lấy mẫu từ toán tử được chọn (<=0: lấy hết)
    OperatorWeightParams aos; // tham số cập nhật trọng số (chỉ dùng cho ADAPTIVE)
};

// FUNCTION BUILD_CANDIDATE_POOL(current, bestFeasible, penaltyWeights lambda, H)
// iteration: dùng để xác định toán tử ở chế độ SEQUENTIAL.
// usedOperator (out, tuỳ chọn): toán tử thực sự sinh ra pool (-1 nếu ALL hoặc pool rỗng).
inline std::vector<Candidate> buildCandidatePool(const Instance& inst, const Solution& current,
                                                  const Solution* bestFeasible, const PenaltyWeights& lambda,
                                                  double H, std::mt19937& rng,
                                                  double maxAllowedInfeasibility = std::numeric_limits<double>::infinity(),
                                                  const NeighborhoodConfig& cfg = NeighborhoodConfig(),
                                                  long long iteration = 0, int* usedOperator = nullptr,
                                                  const OperatorWeights* opWeights = nullptr) {
    std::vector<Candidate> candidatePool;
    if (usedOperator) *usedOperator = -1;

    SearchComponents comp = selectSearchComponents(inst, current, bestFeasible, rng);

    if (cfg.mode == NeighborhoodMode::ALL) {
        std::vector<Move> rawMoves;
        for (int op = 0; op < NUM_OPERATORS; ++op) {
            auto moves = generateMovesForOperator(op, inst, current, comp, rng);
            rawMoves.insert(rawMoves.end(), moves.begin(), moves.end());
        }
        evaluateMovesInto(inst, current, rawMoves, lambda, H, maxAllowedInfeasibility, candidatePool);
        return candidatePool;
    }

    // Thứ tự thử các toán tử trong vòng lặp này.
    std::vector<int> order(NUM_OPERATORS);
    if (cfg.mode == NeighborhoodMode::RANDOM_OP) {
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);   // order[0] = toán tử chọn ngẫu nhiên
    } else if (cfg.mode == NeighborhoodMode::ADAPTIVE && opWeights != nullptr) {
        auto ord = opWeights->selectionOrder(rng);  // order[0] = toán tử chọn theo trọng số
        std::copy(ord.begin(), ord.end(), order.begin());
    } else { // SEQUENTIAL
        int start = static_cast<int>((iteration > 0 ? iteration - 1 : 0) % NUM_OPERATORS);
        for (int k = 0; k < NUM_OPERATORS; ++k) order[k] = (start + k) % NUM_OPERATORS;
    }

    for (int op : order) {
        std::vector<Move> moves = generateMovesForOperator(op, inst, current, comp, rng);
        if (moves.empty()) continue;
        // Sinh lân cận ngẫu nhiên từ toán tử đã chọn: lấy mẫu sampleSize move.
        if (cfg.sampleSize > 0 && static_cast<int>(moves.size()) > cfg.sampleSize) {
            std::shuffle(moves.begin(), moves.end(), rng);
            moves.resize(cfg.sampleSize);
        }
        evaluateMovesInto(inst, current, moves, lambda, H, maxAllowedInfeasibility, candidatePool);
        if (!candidatePool.empty()) {
            if (usedOperator) *usedOperator = op;
            break;
        }
    }

    return candidatePool;
}
