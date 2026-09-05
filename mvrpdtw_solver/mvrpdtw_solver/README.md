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

Với instance này, khoảng cách giữa depot và khách hàng rất lớn (hàng nghìn đơn vị)
so với vận tốc (~15-31 đơn vị/phút) và `drone_lim=700`. Kết quả:
- **Không khách nào tương thích với drone** (khứ hồi luôn > 700).
- Với `L_w=60` (mặc định), **không khách nào tương thích ngay cả với truck**
  (STATIC_COMPATIBLE mục 2 loại hết vì τ_i0 > 60).

Đây là đặc điểm của instance test, không phải lỗi solver — đã verify bằng
`easy_test.json` (thông số tỉ lệ thực tế hơn) cho ra nghiệm khả thi hoàn toàn
(`Total violation: 0.0000`) chỉ sau 300 vòng lặp / ~0.5s.

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
