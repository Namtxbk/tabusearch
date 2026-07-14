# MVRPD-TW Solver (C++)

Bộ giải bài toán MVRPD-TW (Multi-Vehicle Routing Problem with Drones and Time Windows) bằng thuật toán Tabu Search, viết bằng C++ (tương đương `batch_compare.py`).

## 1. Cấu trúc thư mục

```
codeC/
├── main.cpp          # entry point, đọc dữ liệu, chạy batch, xuất kết quả
├── instance.h         # đọc/parse instance bài toán
├── construction.h      # thuật toán khởi tạo nghiệm ban đầu
├── solution.h          # cấu trúc nghiệm, tính makespan
├── tabu_search.h       # thuật toán Tabu Search chính
├── include/json.hpp    # thư viện nlohmann/json (single header)
└── solver.exe          # file thực thi sau khi biên dịch
```

## 2. Yêu cầu môi trường

- Trình biên dịch C++17 trở lên. Khuyến nghị: **MinGW-w64 GCC** (bản WinLibs, POSIX threads, UCRT runtime).
- Nếu máy chưa có, cài bằng winget:

```powershell
winget install --source winget --id BrechtSanders.WinLibs.POSIX.UCRT -e
```

Sau khi cài xong, **mở terminal mới** để PATH được cập nhật, rồi kiểm tra:

```powershell
g++ --version
```

## 3. Biên dịch

Từ thư mục `codeC`:

```powershell
g++ -std=c++17 -O3 -static -static-libgcc -static-libstdc++ -o solver.exe main.cpp -I.
```

Lưu ý: bắt buộc dùng `-static -static-libgcc -static-libstdc++` để `solver.exe` chạy độc lập, không cần cài thêm DLL runtime (libstdc++-6.dll, libgcc_s_seh-1.dll...) trên máy chạy.

Không có lỗi in ra → biên dịch thành công, sinh ra file `solver.exe`.

## 4. Chạy chương trình

### Cú pháp

```
solver.exe --data_dir <thư_mục_dữ_liệu> --baseline <file_baseline.csv> [--output <file_kết_quả.csv>] [--max_iter N] [--max_no_improve N] [--tenure N] [--time_limit T] [--verbose]
```

| Tham số | Bắt buộc | Mặc định | Ý nghĩa |
|---|---|---|---|
| `--data_dir` | Có | — | Thư mục chứa các file `.json` instance đầu vào |
| `--baseline` | Có | — | File CSV chứa kết quả nền (best-known) để so sánh Gap |
| `--output` | Không | `ket_qua_so_sanh.csv` | File CSV ghi kết quả chi tiết từng instance |
| `--max_iter` | Không | 1000 | Số vòng lặp tối đa của Tabu Search |
| `--max_no_improve` | Không | 200 | Dừng sớm nếu không cải thiện sau N vòng |
| `--tenure` | Không | 7 | Độ dài tabu tenure |
| `--time_limit` | Không | 60 (giây) | Giới hạn thời gian chạy cho mỗi instance |
| `--verbose` | Không | tắt | In chi tiết tiến trình từng iteration ra màn hình |

### Ví dụ: chạy toàn bộ bộ dữ liệu, xuất log ra file txt

```powershell
.\solver.exe --data_dir ..\code\WithTimeWindows --baseline ..\code\result.csv --output ket_qua.csv --verbose > run_log.txt 2>&1
```

- `> run_log.txt 2>&1` chuyển toàn bộ output (kể cả lỗi) vào file `run_log.txt` trong thư mục `codeC` thay vì hiện trên màn hình.

### Ví dụ: tuỳ chỉnh tham số Tabu Search

```powershell
.\solver.exe --data_dir ..\code\WithTimeWindows --baseline ..\code\result.csv --output ket_qua.csv --max_iter 2000 --max_no_improve 300 --tenure 7 --time_limit 120 --verbose > run_log.txt 2>&1
```

### Ví dụ: chạy nhanh thử 1 instance

Copy 1 file `.json` cần test sang một thư mục riêng, ví dụ `testdata/`, rồi trỏ `--data_dir` vào đó:

```powershell
mkdir testdata
copy ..\code\WithTimeWindows\10.10.1.json testdata\
.\solver.exe --data_dir testdata --baseline ..\code\result.csv --output test_out.csv --verbose
```

## 5. Kết quả đầu ra

- **File `--output` (CSV)**: mỗi dòng là 1 instance với Makespan tìm được, Makespan baseline, và % Gap (âm = tốt hơn baseline, dương = kém hơn).
- **Log console / file redirect**: in tiến trình từng iteration khi dùng `--verbose`, kết thúc bằng phần "TỔNG KẾT" gồm số nghiệm khả thi, gap trung bình, gap nhỏ nhất/lớn nhất.

## 6. Lưu ý

- Dữ liệu mẫu hiện có tại `..\code\WithTimeWindows` (đầu vào) và `..\code\result.csv` (baseline).
- Với bộ dữ liệu đầy đủ (~80 instance), thời gian chạy có thể mất vài phút tuỳ `--time_limit` và `--max_iter`.
- Nếu sửa code (`.h`/`.cpp`), phải biên dịch lại bước 3 trước khi chạy lại.
