// candidate_pool.hpp
// Mục 15: Sinh candidate pool tổng hợp - BUILD_CANDIDATE_POOL.
#pragma once

#include <vector>
#include <random>
#include "instance.hpp"
#include "solution.hpp"
#include "move.hpp"
#include "operators.hpp"
#include "evaluate_move.hpp"
#include "select_components.hpp"

// FUNCTION BUILD_CANDIDATE_POOL(current, bestFeasible, penaltyWeights lambda, H)
inline std::vector<Candidate> buildCandidatePool(const Instance& inst, const Solution& current,
                                                  const Solution* bestFeasible, const PenaltyWeights& lambda,
                                                  double H, std::mt19937& rng,
                                                  double maxAllowedInfeasibility = std::numeric_limits<double>::infinity()) {
    std::vector<Candidate> candidatePool;

    SearchComponents comp = selectSearchComponents(inst, current, bestFeasible, rng);

    std::vector<Move> rawMoves;

    auto relocateMoves = generateRelocateMoves(inst, current, comp.selectedCustomers);
    rawMoves.insert(rawMoves.end(), relocateMoves.begin(), relocateMoves.end());

    auto orOpt2Moves = generateOrOpt2Moves(inst, current, comp.selectedTrips);
    rawMoves.insert(rawMoves.end(), orOpt2Moves.begin(), orOpt2Moves.end());

    auto swapMoves = generateSwapMoves(inst, current, comp.selectedCustomers);
    rawMoves.insert(rawMoves.end(), swapMoves.begin(), swapMoves.end());

    auto twoOptMoves = generateTwoOptMoves(inst, current, comp.selectedTrips);
    rawMoves.insert(rawMoves.end(), twoOptMoves.begin(), twoOptMoves.end());

    auto crossTripMoves = generateCrossTripMoves(inst, current, comp.selectedTrips);
    rawMoves.insert(rawMoves.end(), crossTripMoves.begin(), crossTripMoves.end());

    auto tripRelocateMoves = generateTripRelocateMoves(inst, current, comp.selectedTrips);
    rawMoves.insert(rawMoves.end(), tripRelocateMoves.begin(), tripRelocateMoves.end());

    for (const auto& m : rawMoves) {
        Candidate c = evaluateMove(inst, current, m, lambda, H, maxAllowedInfeasibility);
        if (c.valid) candidatePool.push_back(std::move(c));
    }

    return candidatePool;
}
