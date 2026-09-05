// strategic_oscillation.hpp
// Mục 12: Strategic Oscillation - cập nhật trọng số phạt lambda theo tỉ lệ nghiệm khả thi từng ràng buộc.
#pragma once

#include <algorithm>
#include "solution.hpp"

struct FeasibleCounts {
    long long Q = 0, D = 0, TW = 0, W = 0;
    void reset() { Q = D = TW = W = 0; }
};

struct OscillationParams {
    int segmentLength = 20;      // L_lambda
    double rhoMin = 0.20;
    double rhoMax = 0.80;
    double gammaIncrease = 1.50;
    double gammaDecrease = 0.85;
};

// PROCEDURE UPDATE_PENALTIES(lambda, feasibleCounts, segmentLength)
inline void updatePenalties(PenaltyWeights& lambda, FeasibleCounts& feasibleCounts, const OscillationParams& params) {
    auto updateOne = [&](double& lam, long long count, double lamMin, double lamMax) {
        double ratio = static_cast<double>(count) / static_cast<double>(params.segmentLength);
        if (ratio < params.rhoMin) {
            lam = std::min(lamMax, params.gammaIncrease * lam);
        } else if (ratio > params.rhoMax) {
            lam = std::max(lamMin, params.gammaDecrease * lam);
        }
    };

    updateOne(lambda.lambdaQ, feasibleCounts.Q, lambda.lambdaMinQ, lambda.lambdaMaxQ);
    updateOne(lambda.lambdaD, feasibleCounts.D, lambda.lambdaMinD, lambda.lambdaMaxD);
    updateOne(lambda.lambdaTW, feasibleCounts.TW, lambda.lambdaMinTW, lambda.lambdaMaxTW);
    updateOne(lambda.lambdaW, feasibleCounts.W, lambda.lambdaMinW, lambda.lambdaMaxW);

    feasibleCounts.reset();
}

// Cập nhật bộ đếm sau mỗi vòng lặp (gọi trước updatePenalties, mỗi iteration).
inline void collectFeasibilityStats(const Solution& current, FeasibleCounts& feasibleCounts, double epsilon = EPS) {
    if (current.violationCapacity <= epsilon) feasibleCounts.Q += 1;
    if (current.violationRange <= epsilon) feasibleCounts.D += 1;
    if (current.violationTimeWindow <= epsilon) feasibleCounts.TW += 1;
    if (current.violationWaiting <= epsilon) feasibleCounts.W += 1;
}
