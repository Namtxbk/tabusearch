// tabu_search.hpp
// Mục 16: ALGORITHM ADAPTIVE_TABU_SEARCH
#pragma once

#include <chrono>
#include <memory>
#include <random>
#include <iostream>
#include "instance.hpp"
#include "solution.hpp"
#include "evaluate.hpp"
#include "feasibility.hpp"
#include "tabu.hpp"
#include "select_move.hpp"
#include "strategic_oscillation.hpp"
#include "best_solutions.hpp"
#include "candidate_pool.hpp"
#include "ruin_recreate.hpp"
#include "construction.hpp"

struct TabuSearchParams {
    long long maxIterations = 20000;      // N_max
    double timeLimitSeconds = 60.0;       // T_lim
    long long stoppingStagnation = 500;   // H_stop
    long long diversificationStagnation = 100; // H_div
    int baseTabuTenure = 7;               // tau0
    int penaltySegmentLength = 20;        // L_lambda
    double ruinRate = 0.15;               // rho_ruin
    int ruinMaxAttempts = 5;
    double maxAllowedInfeasibility = 5.0; // ngưỡng chặn candidate quá tệ (an toàn thực thi)
    unsigned int randomSeed = 42;
};

struct TabuSearchResult {
    bool foundFeasible = false;
    Solution best;      // bestFeasible nếu tìm được, ngược lại bestInfeasible (diagnostic)
    long long iterations = 0;
    double elapsedSeconds = 0.0;
};

inline TabuSearchResult adaptiveTabuSearch(const Instance& inst, const TabuSearchParams& params) {
    using clock = std::chrono::steady_clock;
    auto startTime = clock::now();

    std::mt19937 rng(params.randomSeed);
    TripUidGenerator uidGen;

    PenaltyWeights lambda; // lambdaQ=D=TW=W=1 mặc định

    // 1-2. current <- s0 ; H
    double H = 1.0;
    Solution current = buildInitialSolution(inst, lambda, H, uidGen);

    // 3-4. lambda khởi tạo = 1 ; EVALUATE_SOLUTION(current, lambda)
    evaluateSolution(inst, current, lambda, H);

    // 5. bestFeasible / bestInfeasible
    std::unique_ptr<Solution> bestFeasible;
    std::unique_ptr<Solution> bestInfeasible;
    if (isFeasible(current)) {
        bestFeasible = std::make_unique<Solution>(current);
    } else {
        bestInfeasible = std::make_unique<Solution>(current);
    }

    // 6. tabuUntil
    TabuList tabuList;

    // 7. feasibleCounts
    FeasibleCounts feasibleCounts;
    OscillationParams oscParams;
    oscParams.segmentLength = params.penaltySegmentLength;

    // 8. iteration, hDiv, hStop
    long long iteration = 0;
    long long hDiv = 0;
    long long hStop = 0;

    auto elapsedSeconds = [&]() {
        return std::chrono::duration<double>(clock::now() - startTime).count();
    };

    // 9. Vòng lặp chính
    while (iteration < params.maxIterations &&
           elapsedSeconds() < params.timeLimitSeconds &&
           hStop < params.stoppingStagnation) {

        ++iteration; // 10

        // 11-17. Diversification
        if (hDiv >= params.diversificationStagnation) {
            RuinRecreateResult rr = ruinRecreate(inst, current, lambda, bestFeasible.get(),
                                                  params.ruinRate, params.ruinMaxAttempts, H, rng);
            if (rr.success) {
                current = rr.solution;
                tabuList.clear();
                hDiv = 0;
            }

            bool improved = updateBestSolutions(current, bestFeasible, bestInfeasible);
            if (improved) hStop = 0; else hStop += 1;

            continue; // 17
        }

        // 18. candidatePool
        std::vector<Candidate> candidatePool = buildCandidatePool(
            inst, current, bestFeasible.get(), lambda, H, rng, params.maxAllowedInfeasibility);

        // 19-23. admissiblePool
        std::vector<const Candidate*> admissiblePool;
        for (const auto& c : candidatePool) {
            bool tabu = isTabu(c, tabuList, iteration);
            bool aspiration = satisfiesAspiration(c, bestFeasible.get(), bestInfeasible.get());
            if (!tabu || aspiration) admissiblePool.push_back(&c);
        }

        // 24-27. Tránh search bị chặn hoàn toàn bởi tabu
        if (admissiblePool.empty()) {
            if (!candidatePool.empty()) {
                // Nới lỏng: giải phóng thuộc tính tabu hết hạn sớm nhất rồi cho phép mọi candidate.
                long long earliest = -1;
                for (const auto& kv : tabuList.tabuUntil) {
                    if (earliest == -1 || kv.second < earliest) earliest = kv.second;
                }
                if (earliest != -1) {
                    for (auto& kv : tabuList.tabuUntil) {
                        if (kv.second == earliest) kv.second = iteration - 1; // hết hạn ngay
                    }
                }
                for (const auto& c : candidatePool) admissiblePool.push_back(&c);
            }

            if (admissiblePool.empty()) {
                // Không có move hợp lệ nào (candidatePool rỗng) -> ép diversification ở vòng sau
                hDiv = params.diversificationStagnation;
                continue;
            }
        }

        // 28-30. Chọn & thực hiện move tốt nhất
        std::vector<Candidate> admissibleCopies;
        admissibleCopies.reserve(admissiblePool.size());
        for (const auto* p : admissiblePool) admissibleCopies.push_back(*p);

        const Candidate* selected = selectBestCandidate(admissibleCopies, bestFeasible.get());
        if (selected == nullptr) {
            hDiv = params.diversificationStagnation;
            continue;
        }

        current = selected->solution;
        registerTabu(*selected, tabuList, iteration, params.baseTabuTenure, rng);

        // 31-32. Cập nhật nghiệm tốt nhất
        bool improved = updateBestSolutions(current, bestFeasible, bestInfeasible);
        if (improved) { hDiv = 0; hStop = 0; }
        else { hDiv += 1; hStop += 1; }

        // 33-36. Thống kê khả thi từng ràng buộc
        collectFeasibilityStats(current, feasibleCounts);

        // 37-38. Strategic Oscillation
        if (iteration % oscParams.segmentLength == 0) {
            updatePenalties(lambda, feasibleCounts, oscParams);
        }
    }

    // 39-40. Kết quả
    TabuSearchResult result;
    result.iterations = iteration;
    result.elapsedSeconds = elapsedSeconds();

    if (bestFeasible != nullptr) {
        result.foundFeasible = true;
        result.best = *bestFeasible;
    } else {
        result.foundFeasible = false;
        if (bestInfeasible != nullptr) {
            result.best = *bestInfeasible;
        } else {
            result.best = current;
        }
        std::cerr << "No feasible solution found (report per pseudocode)." << std::endl;
    }

    return result;
}
