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
| `candidate_pool.hpp`          | BUILD_CANDIDATE_POOL (mục 15) + 3 chế độ sinh lân cận all/random/seq   |
| `tabu_search.hpp`             | ADAPTIVE_TABU_SEARCH — vòng lặp chính (mục 16)                         |
| `main.cpp`                    | Entry point: đọc tham số dòng lệnh, chạy solver, ghi record/report     |
| `run_all.bat`                 | Build + chạy tất cả `data/` (chế độ all)                              |
| `run_compare.bat`             | Build + chạy so sánh 3 chế độ all / random / seq                       |
| `json.hpp`                    | Thư viện nlohmann/json (single header, MIT license)                    |

## Build (MSYS2/MinGW hoặc Linux g++)

PowerShell (Windows, MSYS2 ucrt64), chạy trong thư mục `codeC`:
```powershell
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"
g++ -std=c++17 -O2 -static main.cpp -o solver.exe
```

Linux / Git Bash:
```bash
g++ -std=c++17 -O2 -Wall -Wextra -o solver main.cpp
```

## Chạy

```bash
./solver [path_to_instance.json] [override_max_wait] [--mode=...] [--sample=N] [--seed=N] [--iter=N] [--time=S]
```

- Không truyền file: chạy **tất cả** file `*.json` trong thư mục `data/`.
- `override_max_wait` (tuỳ chọn, số thực): override tạm giá trị L_w để test/debug —
  bỏ qua nếu không truyền, solver dùng trường `"max_wait"` trong JSON (mặc định 3600).
- Các tuỳ chọn `--...` đặt ở vị trí nào cũng được:

| Tuỳ chọn | Ý nghĩa | Mặc định |
|---|---|---|
| `--mode=all\|random\|seq` | Cách sinh lân cận (xem mục bên dưới) | `all` |
| `--sample=N` | Số move tối đa lấy mẫu ngẫu nhiên từ toán tử được chọn (chỉ dùng cho `random`/`seq`; `0` = lấy hết) | `100` |
| `--seed=N` | Seed ngẫu nhiên | `42` |
| `--iter=N` | Số vòng lặp tối đa | `2000` |
| `--time=S` | Giới hạn thời gian cho mỗi instance (giây) | không giới hạn |

Kết quả:
- `record/<instance>_<tag>_<ngay-gio>.txt`: chi tiết từng instance (debug log, lời giải, validator).
  Ở chế độ `random`/`seq` có thêm dòng `Operator usage`, tức số vòng lặp mà mỗi toán tử được dùng.
- `report/report_<tag>_<ngay-gio>.txt`: bảng tổng hợp tất cả instance của lần chạy đó
  (dòng đầu ghi chế độ, sample, seed, iter, time).
- `<tag>` gồm chế độ, sample và seed, ví dụ `all_seed42` hay `seq_sample100_seed3`.
  Nhờ vậy các lần chạy khác seed/sample trong cùng 1 phút không ghi đè file của nhau.

## Lệnh chạy mẫu (PowerShell, trong thư mục `codeC`)

Tất cả lệnh dưới đây cần build `solver.exe` trước (xem mục Build). Muốn dùng trong Git Bash
thì thay `.\solver.exe` bằng `./solver.exe` và `data\` bằng `data/`.

### 1. Chạy cơ bản (chế độ gốc `all`)

```powershell
# Chạy tất cả file trong data\
.\solver.exe

# Chạy 1 file
.\solver.exe data\10.10.1.json

# Chạy 1 file với L_w = 400 (override max_wait)
.\solver.exe data\10.10.1.json 400

# Đặt số vòng lặp và giới hạn thời gian
.\solver.exe data\50.10.1.json --iter=5000 --time=60
```

### 2. Chế độ chọn toán tử ngẫu nhiên (`random`)

```powershell
# 1 file
.\solver.exe data\50.10.1.json --mode=random --sample=1000 --time=3000 --iter=1000000000

# Tất cả data\
.\solver.exe --mode=random --sample=10000 --time=3000 --iter=1000000000
```

### 3. Chế độ tuần tự (`seq`)

```powershell
# 1 file
.\solver.exe data\50.10.1.json --mode=seq --sample=1000 --time=3000 --iter=1000000000

# Tất cả data\
.\solver.exe --mode=seq --sample=1000 --time=3000 --iter=1000000000
```

### 4. So sánh 3 chế độ với cùng ngân sách thời gian

```powershell
# Cách 1: dùng script có sẵn (tự build + chạy cả 3 chế độ trên tất cả data\)
.\run_compare.bat

# Cách 2: chạy tay trên 1 file
foreach ($m in "all", "random", "seq") {
    .\solver.exe data\50.10.1.json --mode=$m --sample=1000 --time=3000 --iter=1000000000
}
```

### 5. Chạy nhiều seed (để lấy trung bình / tốt nhất)

```powershell
# seq với seed 1..5 trên tất cả data\
foreach ($s in 1..5) {
    .\solver.exe --mode=seq --sample=1000 --seed=$s --time=3000 --iter=1000000000
}

# Cả random và seq, seed 1..5, trên 1 file
foreach ($m in "random", "seq") {
    foreach ($s in 1..5) {
        .\solver.exe data\50.10.1.json --mode=$m --sample=1000 --seed=$s --time=3000 --iter=1000000000
    }
}
```

### 6. Thử các kích thước mẫu `--sample` khác nhau

```powershell
foreach ($n in 50, 100, 200, 0) {   # 0 = lấy hết move của toán tử được chọn
    .\solver.exe data\50.10.1.json --mode=seq --sample=$n --time=3000 --iter=1000000000
}
```

## Các chế độ sinh lân cận (`--mode`)

Ở cả 3 chế độ, tập khách/trip được xét vẫn chọn bằng `SELECT_SEARCH_COMPONENTS`. Các bước sau đó
cũng giữ nguyên: lọc tabu + aspiration, chọn move tốt nhất, Strategic Oscillation và Ruin & Recreate.
3 chế độ chỉ khác nhau ở bước sinh candidate pool (`candidate_pool.hpp`):

| Chế độ | Mỗi vòng lặp |
|---|---|
| `all` | Chạy **cả 6 toán tử** (Relocate, Or-opt(2), Swap, 2-opt, Cross-trip, Trip-relocate), đánh giá toàn bộ move sinh ra (phiên bản gốc). |
| `random` | Chọn **ngẫu nhiên 1 toán tử**, sinh move của toán tử đó rồi lấy mẫu ngẫu nhiên tối đa `--sample` move để đánh giá. |
| `seq` | Chọn toán tử **tuần tự xoay vòng** theo số vòng lặp: Relocate → Or-opt(2) → Swap → 2-opt → Cross-trip → Trip-relocate → Relocate…, rồi lấy mẫu ngẫu nhiên tối đa `--sample` move như `random`. |

- `random` và `seq` dùng chung cách lấy mẫu, chỉ khác cách chọn toán tử, nên so sánh được trực tiếp với nhau.
- Nếu toán tử được chọn không sinh được candidate hợp lệ nào, solver chuyển sang toán tử khác
  ngay trong vòng đó: ngẫu nhiên với `random`, toán tử kế tiếp với `seq`.

### So sánh 3 chế độ: `run_compare.bat`

Nhấp đúp `run_compare.bat`, hoặc gõ `run_compare.bat` trong terminal. Script sẽ build `solver.exe`
rồi lần lượt chạy `all`, `random`, `seq` trên toàn bộ `data/`, mỗi chế độ ra 1 file
`report/report_<tag>_...txt`. Các tham số chỉnh ở đầu file:

```bat
set TIME_LIMIT=30
set ITER=1000000000
set sample=1000
set SEED=42
```

- `TIME_LIMIT`: số giây cho mỗi instance, dùng chung cho cả 3 chế độ.
- `ITER`: đặt rất lớn để solver dừng theo thời gian.
- `SAMPLE`, `SEED`: giống `--sample`, `--seed`.

Lưu ý khi đánh giá kết quả:
- **So sánh theo cùng thời gian, không theo cùng số vòng lặp.** Mỗi vòng của `all` đánh giá nhiều move hơn hẳn,
  nên cùng số vòng thì `all` sẽ tốn thời gian gấp nhiều lần. Vì vậy script dùng `--time` chung và `--iter` rất lớn.
- `random`/`seq` có thể dừng trước `TIME_LIMIT` vì điều kiện dừng: 5000 vòng liên tiếp không cải thiện
  (`stoppingStagnation` trong `main.cpp`). Điều kiện này **chỉ áp dụng khi đã tìm được nghiệm hợp lệ**.
  Khi chưa có nghiệm hợp lệ, solver chạy tiếp đến khi hết `--time` hoặc `--iter`, nên instance khó
  có thể chạy hết toàn bộ `TIME_LIMIT`.
- `result.csv` (benchmark) là lời giải của bài toán **không có khung thời gian**: thời gian xe chỉ tính
  quãng đường / vận tốc, không xét [open, close]. Vì vậy makespan của solver (có TW) bằng hoặc lớn hơn
  benchmark là bình thường; benchmark chỉ dùng làm mốc tham khảo.
- Kết quả phụ thuộc seed. Nên chạy vài seed (ví dụ `--seed=1` … `--seed=5`) rồi lấy trung bình / tốt nhất.
- Có thể thử thêm vài giá trị `--sample` (50, 100, 200) để xem ảnh hưởng của kích thước mẫu.

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
