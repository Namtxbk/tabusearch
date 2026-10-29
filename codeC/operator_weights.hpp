// operator_weights.hpp
// Adaptive Operator Selection (AOS) — chọn toán tử lân cận theo trọng số và cập nhật trọng số
// theo hiệu quả (kiểu ALNS, Ropke & Pisinger 2006). Dùng cho chế độ --mode=adaptive.
//
// Mỗi vòng lặp:
//   1. selectOperator()  : chọn toán tử theo roulette wheel, xác suất p_i = w_i / sum(w).
//   2. recordOutcome()   : cộng điểm cho toán tử vừa dùng theo kết quả của move được chọn:
//        NEW_BEST  (sigma1): tạo ra nghiệm tốt nhất mới (bestFeasible / bestInfeasible cải thiện)
//        IMPROVED  (sigma2): tốt hơn nghiệm hiện tại (ít vi phạm hơn, hoặc cùng vi phạm mà makespan nhỏ hơn)
//        ACCEPTED  (sigma3): move được thực hiện nhưng không cải thiện
//   3. Hết mỗi đoạn segmentLength vòng -> updateWeights():
//        w_i <- (1 - r) * w_i + r * (score_i / uses_i)     (nếu toán tử i được dùng trong đoạn)
//        w_i <- max(w_i, minWeight)                          (tránh toán tử bị "chết" hẳn)
//      rồi reset score/uses của đoạn.
#pragma once

#include <array>
#include <random>
#include <ostream>
#include <iomanip>
#include <algorithm>

constexpr int AOS_NUM_OPERATORS = 6;

enum class OperatorOutcome { NEW_BEST, IMPROVED, ACCEPTED };

struct OperatorWeightParams {
    int segmentLength = 100;     // số vòng lặp mỗi đoạn trước khi cập nhật trọng số
    double reaction = 0.1;       // r trong [0,1]: 0 = không học, 1 = chỉ dùng điểm của đoạn vừa rồi
    double sigmaNewBest = 33.0;  // sigma1
    double sigmaImproved = 9.0;  // sigma2
    double sigmaAccepted = 1.0;  // sigma3
    double initialWeight = 1.0;
    double minWeight = 0.05;
};

struct OperatorWeights {
    OperatorWeightParams params;
    std::array<double, AOS_NUM_OPERATORS> weights{};
    std::array<double, AOS_NUM_OPERATORS> segmentScore{};
    std::array<long long, AOS_NUM_OPERATORS> segmentUses{};
    std::array<long long, AOS_NUM_OPERATORS> totalUses{};
    std::array<long long, AOS_NUM_OPERATORS> totalNewBest{};
    int iterationsInSegment = 0;
    int segmentsDone = 0;

    explicit OperatorWeights(const OperatorWeightParams& p = OperatorWeightParams()) : params(p) {
        weights.fill(p.initialWeight);
    }

    // Roulette wheel: chọn toán tử i với xác suất w_i / sum(w), bỏ qua các toán tử có excluded[i] = true.
    // Trả về -1 nếu mọi toán tử đều bị loại.
    int selectOperator(std::mt19937& rng, const std::array<bool, AOS_NUM_OPERATORS>& excluded = {}) const {
        double total = 0.0;
        for (int i = 0; i < AOS_NUM_OPERATORS; ++i) if (!excluded[i]) total += weights[i];
        if (total <= 0.0) return -1;
        std::uniform_real_distribution<double> dist(0.0, total);
        double x = dist(rng);
        int last = -1;
        for (int i = 0; i < AOS_NUM_OPERATORS; ++i) {
            if (excluded[i]) continue;
            last = i;
            if (x < weights[i]) return i;
            x -= weights[i];
        }
        return last; // sai số làm tròn
    }

    // Thứ tự thử toán tử cho 1 vòng lặp: lần lượt rút roulette không hoàn lại.
    // Phần tử đầu là toán tử được chọn; các phần tử sau dùng khi toán tử trước không sinh được candidate.
    std::array<int, AOS_NUM_OPERATORS> selectionOrder(std::mt19937& rng) const {
        std::array<int, AOS_NUM_OPERATORS> order{};
        std::array<bool, AOS_NUM_OPERATORS> excluded{};
        for (int k = 0; k < AOS_NUM_OPERATORS; ++k) {
            int op = selectOperator(rng, excluded);
            if (op < 0) { // mọi trọng số còn lại = 0 -> lấy theo thứ tự
                for (int i = 0; i < AOS_NUM_OPERATORS; ++i) if (!excluded[i]) { op = i; break; }
            }
            order[k] = op;
            excluded[op] = true;
        }
        return order;
    }

    // Ghi nhận kết quả của toán tử op ở vòng lặp hiện tại; tự cập nhật trọng số khi hết đoạn.
    void recordOutcome(int op, OperatorOutcome outcome) {
        if (op < 0 || op >= AOS_NUM_OPERATORS) return;
        double s = params.sigmaAccepted;
        if (outcome == OperatorOutcome::NEW_BEST) { s = params.sigmaNewBest; ++totalNewBest[op]; }
        else if (outcome == OperatorOutcome::IMPROVED) s = params.sigmaImproved;
        segmentScore[op] += s;
        ++segmentUses[op];
        ++totalUses[op];
        if (++iterationsInSegment >= params.segmentLength) updateWeights();
    }

    // Cập nhật trọng số cuối đoạn rồi reset điểm của đoạn.
    void updateWeights() {
        for (int i = 0; i < AOS_NUM_OPERATORS; ++i) {
            if (segmentUses[i] > 0) {
                double avg = segmentScore[i] / static_cast<double>(segmentUses[i]);
                weights[i] = (1.0 - params.reaction) * weights[i] + params.reaction * avg;
            }
            weights[i] = std::max(weights[i], params.minWeight);
        }
        segmentScore.fill(0.0);
        segmentUses.fill(0);
        iterationsInSegment = 0;
        ++segmentsDone;
    }

    void print(std::ostream& os, const char* const* names) const {
        double total = 0.0;
        for (double w : weights) total += w;
        os << "AOS (" << segmentsDone << " lan cap nhat trong so):\n";
        for (int i = 0; i < AOS_NUM_OPERATORS; ++i) {
            os << "  " << std::left << std::setw(13) << names[i] << std::right << std::fixed
               << " w=" << std::setprecision(4) << std::setw(8) << weights[i]
               << " p=" << std::setprecision(1) << std::setw(5) << (total > 0 ? 100.0 * weights[i] / total : 0.0) << "%"
               << " | dung=" << totalUses[i] << " | tao best moi=" << totalNewBest[i] << "\n";
        }
    }
};
