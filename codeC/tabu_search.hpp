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

// ====================== [DEBUG LOG - có thể xoá cả khối này] ======================
// Log giá trị nghiệm hiện tại (current) ở mỗi vòng lặp + kiểm tra có khả thi không.
// Không ảnh hưởng thuật toán, chỉ in ra std::cout. Muốn tắt: xoá lời gọi logIterationDebug(...)
// ở trong vòng lặp bên dưới, hoặc xoá luôn hàm này.
inline void logIterationDebug(long long iteration, const Solution& current) {
    std::cout << "[debug][iter " << iteration << "] "
              << "penalizedObjective=" << current.penalizedObjective
              << " makespan=" << current.makespan
              << " totalViolation=" << current.totalViolation
              << " (Q=" << current.violationCapacity
              << ", D=" << current.violationRange
              << ", TW=" << current.violationTimeWindow
              << ", W=" << current.violationWaiting << ")"
              << " unassigned=" << current.unassignedCount
              << " feasible=" << (isFeasible(current) ? "YES" : "NO")
              << "\n";
}
// ====================== [/DEBUG LOG] ======================

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
    NeighborhoodConfig neighborhood;      // chế độ sinh lân cận: all / random / seq
};

struct TabuSearchResult {
    bool foundFeasible = false;
    Solution best;      // bestFeasible nếu tìm được, ngược lại bestInfeasible (diagnostic)
    long long iterations = 0;
    double elapsedSeconds = 0.0;

    // Chỉ để báo cáo (không ảnh hưởng thuật toán):
    bool hasBestFeasible = false;
    Solution bestFeasible;   // lời giải hợp lệ tốt nhất (makespan nhỏ nhất)
    bool hasBestAny = false;
    Solution bestAny;        // lời giải makespan nhỏ nhất đã duyệt (phục vụ đủ khách), có thể vi phạm
    long long operatorUsage[NUM_OPERATORS] = {0}; // số vòng lặp mỗi toán tử sinh pool (chế độ random/seq/adaptive)
    OperatorWeights operatorWeights;              // trọng số cuối cùng (chế độ adaptive)

    // Thời điểm ra kết quả (giây, tính từ lúc bắt đầu, gồm cả dựng nghiệm ban đầu); -1 nếu không có.
    double timeFirstFeasible = -1.0;   // lần đầu có nghiệm khả thi
    double timeBestFeasible = -1.0;    // lần cuối cải thiện bestFeasible = lúc ra kết quả cuối cùng
    long long iterBestFeasible = -1;
};

// Theo dõi lời giải có makespan nhỏ nhất trong mọi lời giải đã duyệt (kể cả không khả thi).
// Chỉ xét lời giải đã phục vụ hết khách, nếu không makespan không có ý nghĩa so sánh.
inline void trackBestAny(const Solution& s, std::unique_ptr<Solution>& bestAny) {
    if (s.unassignedCount != 0) return;
    if (bestAny == nullptr || s.makespan < bestAny->makespan - EPS ||
        (std::fabs(s.makespan - bestAny->makespan) <= EPS && s.totalViolation < bestAny->totalViolation)) {
        bestAny = std::make_unique<Solution>(s);
    }
}

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
    std::unique_ptr<Solution> bestAny; // chỉ để báo cáo
    trackBestAny(current, bestAny);

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
    long long operatorUsage[NUM_OPERATORS] = {0};
    OperatorWeights opWeights(params.neighborhood.aos); // chỉ dùng khi mode = ADAPTIVE
    const bool adaptive = (params.neighborhood.mode == NeighborhoodMode::ADAPTIVE);

    auto elapsedSeconds = [&]() {
        return std::chrono::duration<double>(clock::now() - startTime).count();
    };

    // Ghi lại thời điểm bestFeasible được tạo/cải thiện (bestFeasible chỉ đổi con trỏ khi cải thiện).
    double timeFirstFeasible = -1.0, timeBestFeasible = -1.0;
    long long iterBestFeasible = -1;
    const Solution* lastBestFeasiblePtr = nullptr;
    auto noteFeasibleImproved = [&]() {
        if (bestFeasible != nullptr && bestFeasible.get() != lastBestFeasiblePtr) {
            lastBestFeasiblePtr = bestFeasible.get();
            timeBestFeasible = elapsedSeconds();
            iterBestFeasible = iteration;
            if (timeFirstFeasible < 0) timeFirstFeasible = timeBestFeasible;
        }
    };
    noteFeasibleImproved(); // nghiệm ban đầu có thể đã khả thi

    // 9. Vòng lặp chính
    // Điều kiện dừng theo stagnation (H_stop) CHỈ áp dụng khi đã có nghiệm khả thi. Nếu chưa có,
    // search tiếp tục đến khi hết thời gian / hết vòng lặp — tránh trường hợp bị kẹt ở 1 nghiệm gần
    // khả thi rồi dừng sớm mà chưa dùng hết ngân sách (vd 50.10.4, seed 42 dừng ở 15s / 3000s).
    while (iteration < params.maxIterations &&
           elapsedSeconds() < params.timeLimitSeconds &&
           (bestFeasible == nullptr || hStop < params.stoppingStagnation)) {

        ++iteration; // 10

        // 11-17. Diversification
        if (hDiv >= params.diversificationStagnation) {
            RuinRecreateResult rr = ruinRecreate(inst, current, lambda, bestFeasible.get(),
                                                  params.ruinRate, params.ruinMaxAttempts, H, rng);
            if (rr.success) {
                current = rr.solution;
                tabuList.clear();
            }
            // Luôn reset hDiv sau 1 lần thử (kể cả thất bại) — nếu không, khi ruinRecreate thất bại
            // hDiv vẫn >= ngưỡng nên vòng lặp SAU sẽ lại rơi vào đúng nhánh này, thử lại mãi mãi và
            // không bao giờ quay lại tìm kiếm bình thường (buildCandidatePool) nữa trong suốt phần
            // thời gian còn lại — bug khiến search bị "kẹt" khi hạ diversificationStagnation xuống thấp.
            hDiv = 0;

            bool improved = updateBestSolutions(current, bestFeasible, bestInfeasible);
            noteFeasibleImproved();
            trackBestAny(current, bestAny);
            if (improved) hStop = 0; else hStop += 1;

            logIterationDebug(iteration, current); // [DEBUG LOG - có thể xoá]

            continue; // 17
        }

        // 18. candidatePool
        int usedOperator = -1;
        std::vector<Candidate> candidatePool = buildCandidatePool(
            inst, current, bestFeasible.get(), lambda, H, rng, params.maxAllowedInfeasibility,
            params.neighborhood, iteration, &usedOperator, &opWeights);
        if (usedOperator >= 0) ++operatorUsage[usedOperator];

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
                logIterationDebug(iteration, current); // [DEBUG LOG - có thể xoá]
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
            logIterationDebug(iteration, current); // [DEBUG LOG - có thể xoá]
            continue;
        }

        const double prevViolation = current.totalViolation;
        const double prevMakespan = current.makespan;
        current = selected->solution;
        registerTabu(*selected, tabuList, iteration, params.baseTabuTenure, rng);

        // 31-32. Cập nhật nghiệm tốt nhất
        bool improved = updateBestSolutions(current, bestFeasible, bestInfeasible);
        noteFeasibleImproved();
        trackBestAny(current, bestAny);
        if (improved) { hDiv = 0; hStop = 0; }
        else { hDiv += 1; hStop += 1; }

        // Adaptive operator selection: thưởng điểm cho toán tử đã sinh ra move vừa thực hiện.
        if (adaptive && usedOperator >= 0) {
            OperatorOutcome outcome = OperatorOutcome::ACCEPTED;
            if (improved) {
                outcome = OperatorOutcome::NEW_BEST;
            } else if (current.totalViolation < prevViolation - EPS ||
                       (std::fabs(current.totalViolation - prevViolation) <= EPS && current.makespan < prevMakespan - EPS)) {
                outcome = OperatorOutcome::IMPROVED;
            }
            opWeights.recordOutcome(usedOperator, outcome);
        }

        // 33-36. Thống kê khả thi từng ràng buộc
        collectFeasibilityStats(current, feasibleCounts);

        // 37-38. Strategic Oscillation
        if (iteration % oscParams.segmentLength == 0) {
            updatePenalties(lambda, feasibleCounts, oscParams);
        }

        logIterationDebug(iteration, current); // [DEBUG LOG - có thể xoá]
    }

    // 39-40. Kết quả
    TabuSearchResult result;
    result.iterations = iteration;
    result.elapsedSeconds = elapsedSeconds();
    for (int op = 0; op < NUM_OPERATORS; ++op) result.operatorUsage[op] = operatorUsage[op];
    result.operatorWeights = opWeights;
    result.timeFirstFeasible = timeFirstFeasible;
    result.timeBestFeasible = timeBestFeasible;
    result.iterBestFeasible = iterBestFeasible;
    if (bestFeasible != nullptr) { result.hasBestFeasible = true; result.bestFeasible = *bestFeasible; }
    if (bestAny != nullptr) { result.hasBestAny = true; result.bestAny = *bestAny; }

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
