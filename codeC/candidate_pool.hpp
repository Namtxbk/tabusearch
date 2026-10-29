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
// Với RANDOM_OP / SEQUENTIAL: nếu toán tử được chọn không sinh được candidate hợp lệ nào thì
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

enum class NeighborhoodMode { ALL, RANDOM_OP, SEQUENTIAL };

inline const char* neighborhoodModeName(NeighborhoodMode m) {
    switch (m) {
        case NeighborhoodMode::ALL:        return "all";
        case NeighborhoodMode::RANDOM_OP:  return "random";
        case NeighborhoodMode::SEQUENTIAL: return "seq";
    }
    return "?";
}

constexpr int NUM_OPERATORS = 6;
inline const char* operatorName(int op) {
    static const char* names[NUM_OPERATORS] = {"Relocate", "OrOpt2", "Swap", "TwoOpt", "CrossTrip", "TripRelocate"};
    return (op >= 0 && op < NUM_OPERATORS) ? names[op] : "?";
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
    int sampleSize = 100;   // RANDOM_OP / SEQUENTIAL: số move tối đa lấy mẫu từ toán tử được chọn (<=0: lấy hết)
};

// FUNCTION BUILD_CANDIDATE_POOL(current, bestFeasible, penaltyWeights lambda, H)
// iteration: dùng để xác định toán tử ở chế độ SEQUENTIAL.
// usedOperator (out, tuỳ chọn): toán tử thực sự sinh ra pool (-1 nếu ALL hoặc pool rỗng).
inline std::vector<Candidate> buildCandidatePool(const Instance& inst, const Solution& current,
                                                  const Solution* bestFeasible, const PenaltyWeights& lambda,
                                                  double H, std::mt19937& rng,
                                                  double maxAllowedInfeasibility = std::numeric_limits<double>::infinity(),
                                                  const NeighborhoodConfig& cfg = NeighborhoodConfig(),
                                                  long long iteration = 0, int* usedOperator = nullptr) {
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
