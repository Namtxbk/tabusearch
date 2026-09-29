# Tóm tắt codebase MVRPD-TW — tham chiếu nhanh cho cập nhật LaTeX

> Cập nhật: 2026-09-29. Đọc file này thay vì đọc lại toàn bộ `.hpp`.

---

## 1. Cấu trúc file

| File | Nội dung | Mục LaTeX |
|---|---|---|
| `instance.hpp` | Struct Customer, Instance, đọc JSON | Ký hiệu bài toán |
| `solution.hpp` | Struct Trip, Vehicle, Solution, PenaltyWeights | Cấu trúc lời giải |
| `schedule.hpp` | `staticCompatible`, `recomputeVehicle` | Mục 2-3 |
| `feasibility.hpp` | `isFeasible`, `betterInfeasible`, `EPS=1e-9` | Mục 5 |
| `evaluate.hpp` | `evaluateSolution`, `computeH` | Hàm mục tiêu |
| `move.hpp` | Enum MoveType, struct Move, TabuAttribute | Mục 6, 9 |
| `operators.hpp` | generate*/apply* cho 6 toán tử | Mục 6 |
| `evaluate_move.hpp` | `evaluateMove`, `violatesStructuralConstraint` | Mục 8 |
| `tabu.hpp` | `isTabu`, `registerTabu`, `satisfiesAspiration` | Mục 9-10 |
| `select_components.hpp` | `selectSearchComponents`, `customerTWContribution` | Mục 7 |
| `construction.hpp` | `buildInitialSolution`, `generateAllInsertions` | Greedy Insertion |
| `candidate_pool.hpp` | `buildCandidatePool` | Mục 15 |
| `select_move.hpp` | `selectBestCandidate` | Mục 11 |
| `best_solutions.hpp` | `updateBestSolutions` | Mục 13 |
| `strategic_oscillation.hpp` | `updatePenalties`, `collectFeasibilityStats` | Strategic Oscillation |
| `ruin_recreate.hpp` | `ruinRecreate`, `selectRuinCustomers` | Mục 14 |
| `tabu_search.hpp` | `adaptiveTabuSearch` — vòng lặp chính | Pseudocode TS |
| `main.cpp` | Tham số thực tế chạy chương trình | Điều kiện dừng |

---

## 2. Tham số thực tế (main.cpp)

```
maxIterations          = 2000
timeLimitSeconds       = 20.0 s
stoppingStagnation     = 500   (H_stop)
diversificationStagnation = 360 (H_div)
baseTabuTenure         = 7     (tau0)
penaltySegmentLength   = 20    (L_lambda)
ruinRate               = 0.15  (rho_ruin)
ruinMaxAttempts        = 5
randomSeed             = 42
```

---

## 3. Hàm mục tiêu (evaluate.hpp)

```
F_lambda(s) = C_max(s)/H
            + lambda_Q  * V_Q
            + lambda_D  * V_D
            + lambda_TW * V_TW
            + lambda_W  * V_W
            + 1000 * V_unassigned
```

**H** = max{1, C_max(s0), max_i l_i} — tính MỘT LẦN từ nghiệm ban đầu, không đổi.

**Vi phạm — tất cả chia /n:**
- V_Q  = (1/n) * Σ_trip  max(0, (load - M_v) / M_v)
- V_D  = (1/n) * Σ_drone_trip  max(0, (flightTime - L_D) / L_D)
- V_TW = (1/n) * Σ_cust  max(0, (a_i - l_i) / H)
- V_W  = (1/n) * Σ_cust  max(0, (returnTime_trip - a_i - L_w) / L_w)
- V_unassigned = unassigned_count / n  (× 1000 trong hàm)

> **Lưu ý**: KHÔNG có M_phy=1e9 hay M_tmp=1e6. Tất cả vi phạm dùng lambda động.

**Khả thi**: totalViolation <= 1e-9 AND unassignedCount == 0

**Lambda**: bắt đầu = 1.0, khoảng [0.001, 1000], điều chỉnh bởi Strategic Oscillation.

---

## 4. Greedy Insertion (construction.hpp)

- Xe khởi tạo **không có trip nào** (buildEmptySolution: trips = [])
- Sắp khách theo l_i tăng dần (EDD)
- Với mỗi khách: sinh tất cả moves → phân loại F (ΔTW≤0) / P (ΔTW>0)
  - ΔTW(a) = max(0, a_c - l_c) cho khách vừa chèn (KHÔNG phải toàn solution)
- Chọn: F ≠ ∅ → min Makespan; F = ∅ → lex(ΔTW, Makespan, Dist)
- Sau greedy: tính H từ C_max(s0)

---

## 5. Tabu List (tabu.hpp, move.hpp)

**Tenure**: τ ~ U[τ₀, 2τ₀] ngẫu nhiên (không cố định)

**Ba loại thuộc tính**:
1. Arc:        `ARC|vehicleId|fromNode|toNode`
2. Assignment: `ASSIGN|customerId|forbiddenVehicleId`
3. Trip return:`TRIP_RETURN|tripUid|sourceVehicleId|predTripUid`

**Cơ chế**:
- extractTabuAttributes(s_before, move) → oldAttrs
- extractTabuAttributes(s_after, move) → newAttrs
- removedAttributes = oldAttrs \ newAttrs  → ghi vào T khi REGISTER
- addedAttributes   = newAttrs \ oldAttrs  → kiểm tra khi IS_TABU

**IS_TABU**: candidate bị tabu nếu ANY addedAttribute trong T còn hiệu lực
**REGISTER**: ghi removedAttributes vào T với expiry = iter + τ

---

## 6. Aspiration Criterion (tabu.hpp)

- Case 10.1 (có s*_F): candidate khả thi VÀ C_max(candidate) < C_max(s*_F)
- Case 10.2 (chưa có s*_F): candidate tốt hơn s*_I theo (V_Sigma, C_max_norm, Distance)

---

## 7. Sáu toán tử lân cận (operators.hpp)

| # | Toán tử | Move | Phạm vi |
|---|---|---|---|
| 6.1 | Relocate | 1 khách | Toàn hệ thống |
| 6.2 | Or-opt(2) | 2 khách liên tiếp | Toàn hệ thống |
| 6.3 | Swap | Hoán đổi 2 khách | Toàn hệ thống |
| 6.4 | 2-opt | Đảo đoạn [i..j] | Trong 1 chuyến |
| 6.5 | Cross-trip | Hoán đổi đuôi | **Bất kỳ 2 chuyến** (khác xe được) |
| 6.6 | Trip-Relocate | Di chuyển cả chuyến | Toàn hệ thống |

> Cross-trip: khác với mô tả cũ ("cùng xe"), code cho phép 2 xe khác nhau.
> Trip-Relocate: toàn bộ khách trong trip di chuyển sang vị trí/xe khác.

---

## 8. SelectSearchComponents (select_components.hpp)

**Chưa có s*_F**:
- Top n/3 khách có đóng góp V_TW + V_W cao nhất
- Khách từ trip vượt tải / drone vượt tầm bay
- Thêm 30% khách ngẫu nhiên (min 3)

**Đã có s*_F**:
- Toàn bộ khách + trip từ phương tiện critical (C_v == C_max)
- Thêm 30% khách ngẫu nhiên + 30% trip ngẫu nhiên

---

## 9. Strategic Oscillation (strategic_oscillation.hpp)

- Mỗi L_lambda = 20 vòng: đếm feasible per-constraint (Q, D, TW, W riêng)
- ratio = count / L_lambda
  - ratio < rho_min (0.20) → lambda *= 1.50  (tăng phạt, cap 1000)
  - ratio > rho_max (0.80) → lambda *= 0.85  (giảm phạt, floor 0.001)
  - otherwise → giữ nguyên
- Reset count về 0 sau mỗi lần update

---

## 10. Diversification — Ruin & Recreate (ruin_recreate.hpp)

**Kích hoạt**: h_div >= H_div (= 360 trong main.cpp)
**Sau R&R**: nếu success → current = new_sol, tabuList.clear(), h_div = 0

**Ruin**: q = max(1, floor(rho * n)), rho_init = 0.15
- Nhóm 1 (~q/3): khách TW+W violation cao
- Nhóm 2 (~q/3, hoặc q/2 nếu feasible): khách từ critical vehicle
- Nhóm 3: random lấp đầy

**Recreate**:
- Sort theo l_i + random swaps (prob 0.1 mỗi cặp liền kề)
- Best-insertion trên toàn hệ thống
- Tránh trucks[0] nếu còn phương tiện khác

**Retry**: nếu fail → rho = max(0.05, 0.8*rho), thử lại ≤ 5 lần

---

## 11. Vòng lặp chính (tabu_search.hpp)

```
Hai bộ đếm tách biệt:
  h_stop: không cải thiện s*_F → dừng khi >= H_stop
  h_div:  không cải thiện bất kỳ best → kích hoạt R&R khi >= H_div

Flow mỗi vòng:
  1. Check h_div >= H_div → R&R, clear tabu, reset h_div, continue
  2. buildCandidatePool (6 operators + random selection)
  3. filter admissible (not tabu OR aspiration)
  4. if admissible empty → release earliest tabu entry
  5. selectBestCandidate (F_lambda hoặc lex tùy có s*_F chưa)
  6. registerTabu (removedAttributes)
  7. updateBestSolutions → cập nhật h_div, h_stop
  8. collectFeasibilityStats → sau L_lambda vòng: updatePenalties
```

**SelectBestCandidate**:
- Chưa có s*_F: lex(V_Sigma, C_max_norm, Distance)
- Đã có s*_F: lex(penalizedObjective, V_Sigma, C_max_norm, Distance)

**Kết quả**: trả về s*_F nếu tìm được, ngược lại trả về s*_I (hoặc current)

---

## 12. Định nghĩa thời gian chờ hàng (schedule.hpp)

```cpp
waitingTime[custId] = trip.returnTime - trip.arrivalTime[custId]
// = thời gian hàng nằm trên xe từ khi lấy đến khi xe về depot
// Constraint: wait <= L_w = 3600 giây (60 phút)
```

staticCompatible cũng kiểm tra: travelTime(cust → depot) <= L_w

---

## 13. Ánh xạ section LaTeX ↔ file code

| Section LaTeX | File(s) |
|---|---|
| Tổng quan | tabu_search.hpp (overview) |
| Greedy Insertion | construction.hpp |
| Hàm mục tiêu | evaluate.hpp, solution.hpp |
| Chọn thành phần | select_components.hpp |
| Danh sách Tabu | tabu.hpp, move.hpp, evaluate_move.hpp |
| Aspiration | tabu.hpp: satisfiesAspiration |
| Toán tử | operators.hpp |
| Strategic Oscillation | strategic_oscillation.hpp |
| Diversification/R&R | ruin_recreate.hpp |
| Điều kiện dừng | tabu_search.hpp + main.cpp |
| Pseudocode TS | tabu_search.hpp: adaptiveTabuSearch |
