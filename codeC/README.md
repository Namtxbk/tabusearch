# MVRPD-TW Tabu Search Solver (C++)

Giải bài toán Multi-Vehicle Routing Problem with Drones and Time Windows (Multi-Trip),
theo đúng pseudocode trong tài liệu `thuat_toan_tabu_search_NAM.pdf`.

## Cấu trúc file (theo 16 phần trong tài liệu)

| File                        | Nội dung                                                              |
|------------------------------|-------------------------------------------------------------------------|
| `instance.hpp`               | Đọc Instance từ JSON (tương thích `instance.py`)                       |
| `solution.hpp`                | Customer/Trip/Vehicle/Solution/PenaltyWeights (mục 1)                  |
| `schedule.hpp`                | STATIC_COMPATIBLE (mục 2) + RECOMPUTE_VEHICLE (mục 3)                  |
| `evaluate.hpp`                | EVALUATE_SOLUTION — đo vi phạm, makespan, penalized objective (mục 4)  |
| `feasibility.hpp`             | isFeasible + BETTER_INFEASIBLE (mục 5)                                 |
| `move.hpp`                    | Định nghĩa Move + thuộc tính tabu (mục 6, 9)                           |
| `operators.hpp`               | 6 toán tử lân cận: Relocate, Or-opt(2), Swap, 2-opt, Cross-trip, Trip-relocate (mục 6) + APPLY_MOVE |
| `evaluate_move.hpp`           | EVALUATE_MOVE — áp dụng tạm thời, kiểm tra cấu trúc, trích tabu attrs (mục 8) |
| `select_components.hpp`       | SELECT_SEARCH_COMPONENTS (mục 7)                                       |
| `tabu.hpp`                    | Tabu tenure, IS_TABU, REGISTER_TABU (mục 9) + Aspiration (mục 10)      |
| `select_move.hpp`             | SELECT_BEST_CANDIDATE (mục 11)                                         |
| `strategic_oscillation.hpp`   | UPDATE_PENALTIES — Strategic Oscillation (mục 12)                      |
| `best_solutions.hpp`          | UPDATE_BEST_SOLUTIONS + stagnation counters (mục 13)                   |
| `construction.hpp`            | Init solution (construction heuristic) + hàm insertion dùng chung      |
| `ruin_recreate.hpp`           | Ruin & Recreate (mục 14)                                                |
| `candidate_pool.hpp`          | BUILD_CANDIDATE_POOL (mục 15)                                          |
| `tabu_search.hpp`             | ADAPTIVE_TABU_SEARCH — vòng lặp chính (mục 16)                         |
| `main.cpp`                    | Entry point: đọc instance, chạy solver, in kết quả                     |
| `json.hpp`                    | Thư viện nlohmann/json (single header, MIT license)                    |

## Build (MSYS2/MinGW hoặc Linux g++)

```bash
g++ -std=c++17 -O2 -Wall -Wextra -o solver main.cpp
```

## Chạy

```bash
./solver <path_to_instance.json> [override_max_wait]
```

- `override_max_wait` (tuỳ chọn, số thực): override tạm giá trị L_w để test/debug —
  bỏ qua nếu không truyền, solver sẽ dùng `max_wait` mặc định = 60 (theo `instance.py`),
  hoặc trường `"max_wait"` nếu có trong JSON.

Ví dụ:
```bash
./solver easy_test.json           # dùng L_w mặc định = 60
./solver 6_5_1.json 400           # test với L_w = 400
```

## Lưu ý quan trọng về instance mẫu `6_5_1.json`

Đã verify với `result.csv` (benchmark thực tế): solver cho `Makespan = 699.6657`,
khớp gần khít với đáp án chuẩn `Cost = 699.6656534991815` (route truck giống hệt
`[0,2,0]`; route drone khác cách chia trip nhưng phục vụ đúng tập khách và cho
makespan tương đương).

## Các bug quan trọng đã phát hiện & sửa trong quá trình test với dữ liệu thực tế

1. **`drone_range` ("Endurance fixed time") là GIỚI HẠN THỜI GIAN BAY (giây),
   KHÔNG PHẢI giới hạn quãng đường.** Tài liệu PDF mô tả mô hình "range" đơn giản
   (so khoảng cách), nhưng dữ liệu benchmark thực tế dùng mô hình "endurance"
   (so thời gian bay = quãng đường / vận tốc). Đã sửa `staticCompatible`
   (schedule.hpp) và vi phạm `V_D` (evaluate.hpp) để so sánh `flightTime`
   thay vì `travelDistance`. Trip có thêm trường `flightTime` (solution.hpp).

2. **Đơn vị thời gian toàn hệ thống là GIÂY, không phải phút.**
   `max_wait` (L_w) mặc định phải là **3600** (= 60 phút), không phải 60 như
   `instance.py` gợi ý (file đó tính bằng phút cho mục đích khác). Xác nhận từ
   cột `Waiting time limit = 3600` trong `result.csv`. Đã sửa mặc định trong
   `instance.hpp`. Nếu JSON của em không có trường `"max_wait"`, solver dùng
   3600 — đúng theo mọi instance trong bộ benchmark (README dự án ghi rõ
   "L_w = 60 phút cho tất cả các instance").

3. **`evaluateSolution` không hề kiểm tra thiếu khách hàng.** Trước đây, nếu
   quá trình construction/insertion chỉ chèn được 1/6 khách (do static
   incompatibility ở bug #1+#2 gây ra), solver vẫn báo "Feasible: YES" với
   `Total violation: 0` vì hàm này chỉ tính V_Q/V_D/V_TW/V_W, không đếm số
   khách còn thiếu. Đã thêm `Solution::unassignedCount` — `isFeasible()` giờ
   yêu cầu cả `totalViolation <= epsilon` VÀ `unassignedCount == 0`. Số khách
   thiếu cũng được cộng vào `totalViolation` (đã chuẩn hoá theo n) và phạt rất
   nặng (hệ số 1000) trong `penalizedObjective` để Tabu Search luôn ưu tiên
   phục vụ đủ khách trước khi tối ưu makespan.

**File đính kèm `result.csv`**: bảng benchmark đầy đủ (nhiều instance khác nhau,
mỗi instance có thể có nhiều dòng lời giải tối ưu tương đương) — dùng để so
sánh `Makespan` solver tìm được với cột `Cost [minute]` cho cùng `Problem`.
Lưu ý: cột 13 = `Truck paths`, cột 14 = `Drone paths` (dễ đọc nhầm ngược).

Với instance này, khoảng cách giữa depot và khách hàng rất lớn (hàng nghìn đơn vị)
so với vận tốc (~15-31 đơn vị/giây) — sau khi sửa 2 bug trên, mọi khách hàng đều
tìm được phương tiện tương thích và solver đạt nghiệm khả thi hoàn toàn.

## Trạng thái implementation

Đã hoàn thành đầy đủ 16 phần theo "Thứ tự code cần hoàn thành" ở cuối tài liệu.
Cách tiếp cận hiện tại là **deep-copy + tính lại toàn bộ nghiệm sau mỗi move**
(đúng khuyến nghị "Ở phiên bản đầu tiên" của tài liệu) — CHƯA tối ưu bằng
incremental evaluation (chỉ tính lại phương tiện/trip bị ảnh hưởng ở mức move
generation, dù RECOMPUTE_VEHICLE đã hỗ trợ `firstAffectedTrip` để làm việc này
khi cần tối ưu tốc độ sau).

### Các điểm em nên tự kiểm tra / tinh chỉnh thêm

1. **Tham số** `TabuSearchParams` trong `tabu_search.hpp` (Nmax, Tlim, HStop,
   HDiv, tau0, segment length, ruin rate) đang để giá trị thử nghiệm ban đầu
   theo mục 12 tài liệu — em có thể chỉnh qua `struct TabuSearchParams` hoặc
   thêm parser tham số dòng lệnh.
2. **Hiệu năng**: với instance lớn, độ phức tạp sinh move (đặc biệt Swap —
   O(n²) cặp khách, và Cross-trip — O(số trip² × độ dài trip²)) có thể chậm.
   Nên áp dụng candidate limiting mạnh hơn (mục 7) khi scale lên.
3. **SELECT_SEARCH_COMPONENTS / Ruin selection** dùng ngẫu nhiên (`std::mt19937`)
   — seed cố định trong `TabuSearchParams::randomSeed` để tái lập kết quả khi debug.
