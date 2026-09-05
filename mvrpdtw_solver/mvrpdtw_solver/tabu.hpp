// tabu.hpp
// Mục 9: Danh sách Tabu (tenure, IS_TABU, REGISTER_TABU)
// Mục 10: Aspiration criterion
#pragma once

#include <unordered_map>
#include <random>
#include "move.hpp"
#include "evaluate_move.hpp"
#include "feasibility.hpp"

// ============================================================
// Mục 9.4-9.6 Tabu tenure, kiểm tra & ghi tabu
// ============================================================
struct TabuList {
    std::unordered_map<std::string, long long> tabuUntil; // key = TabuAttribute.key -> iteration hết hạn

    void clear() { tabuUntil.clear(); }

    long long getUntil(const TabuAttribute& a) const {
        auto it = tabuUntil.find(a.key);
        return (it == tabuUntil.end()) ? -1 : it->second;
    }

    void setUntil(const TabuAttribute& a, long long iter) {
        tabuUntil[a.key] = iter;
    }
};

// tau ~ U[tau0, 2*tau0]
inline int sampleTenure(int tau0, std::mt19937& rng) {
    if (tau0 <= 0) tau0 = 1;
    std::uniform_int_distribution<int> dist(tau0, 2 * tau0);
    return dist(rng);
}

// FUNCTION IS_TABU(candidate c, tabuUntil, iteration)
inline bool isTabu(const Candidate& c, const TabuList& tabuList, long long iteration) {
    for (const auto& a : c.addedAttributes) {
        long long until = tabuList.getUntil(a);
        if (until > iteration) return true;
    }
    return false;
}

// PROCEDURE REGISTER_TABU(selectedCandidate, tabuUntil, iteration, tau0)
inline void registerTabu(const Candidate& selected, TabuList& tabuList, long long iteration, int tau0, std::mt19937& rng) {
    int tenure = sampleTenure(tau0, rng);
    for (const auto& a : selected.removedAttributes) {
        tabuList.setUntil(a, iteration + tenure);
    }
}

// ============================================================
// Mục 10. Aspiration criterion
// ============================================================
// 10.1 Đã có nghiệm khả thi: candidate khả thi và cải thiện makespan tốt nhất.
// 10.2 Chưa có nghiệm khả thi: candidate cải thiện nghiệm không khả thi tốt nhất
//      theo thứ tự (V_Sigma, C_max_norm, Distance).
inline bool satisfiesAspiration(const Candidate& c, const Solution* bestFeasible, const Solution* bestInfeasible) {
    const Solution& sPrime = c.solution;

    if (isFeasible(sPrime)) {
        if (bestFeasible == nullptr) return true;
        return sPrime.makespan < bestFeasible->makespan - EPS;
    }

    if (bestFeasible == nullptr) {
        return betterInfeasible(sPrime, bestInfeasible);
    }

    return false;
}
