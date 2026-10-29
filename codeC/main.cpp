// main.cpp
// Chạy Adaptive Tabu Search cho bài toán MVRPD-TW.
//   solver.exe                 -> chạy TẤT CẢ file *.json trong thư mục data/
//   solver.exe <file.json>     -> chỉ chạy 1 file
//   solver.exe <file.json> <max_wait>
// Tuỳ chọn (đặt ở bất kỳ vị trí nào):
//   --mode=all|random|seq|adaptive   cách sinh lân cận (mặc định all = cả 6 toán tử mỗi vòng)
//                           random  : chọn ngẫu nhiên 1 toán tử / vòng, lấy mẫu ngẫu nhiên move của nó
//                           seq     : xoay vòng tuần tự 6 toán tử, lấy mẫu ngẫu nhiên move của nó
//                           adaptive: chọn toán tử theo trọng số (roulette), trọng số tự cập nhật (operator_weights.hpp)
//   --sample=N              số move tối đa lấy mẫu từ toán tử được chọn (random/seq/adaptive, mặc định 100, 0 = lấy hết)
//   --aos-seg=N             adaptive: số vòng lặp mỗi đoạn trước khi cập nhật trọng số (mặc định 100)
//   --aos-r=R               adaptive: hệ số phản ứng r trong [0,1] (mặc định 0.1)
//   --aos-scores=a,b,c      adaptive: điểm thưởng nghiệm tốt nhất mới / cải thiện nghiệm hiện tại / chỉ được chấp nhận (mặc định 33,9,1)
//   --seed=N                seed ngẫu nhiên (mặc định 42)
//   --iter=N                số vòng lặp tối đa (mặc định 2000)
//   --time=S                giới hạn thời gian mỗi instance, giây (mặc định không giới hạn)
//                           -> so sánh công bằng giữa các mode: dùng cùng --time và --iter lớn
//   vd: solver.exe --mode=random --sample=100
// Mỗi instance ghi chi tiết (kèm debug log) vào record/<ten>_<YYYY-MM-DD_HH-MM>.txt,
// tổng hợp lời giải của tất cả instance ghi vào 1 file duy nhất report/report_<YYYY-MM-DD_HH-MM>.txt.
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <filesystem>
#include "instance.hpp"
#include "tabu_search.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;

static const char* DATA_DIR = "data";
static const char* RECORD_DIR = "record";
static const char* REPORT_DIR = "report";

// Cấu hình chạy lấy từ dòng lệnh.
struct RunConfig {
    double maxWaitOverride = -1.0;
    NeighborhoodConfig neighborhood;
    unsigned int seed = 42;
    long long maxIterations = 2000;
    double timeLimitSeconds = 200000.0;
};

static const char* vehicleTypeName(VehicleType t) {
    return (t == VehicleType::TRUCK) ? "TRUCK" : "DRONE";
}

// fileSafe = true -> "YYYY-MM-DD_HH-MM" (dùng cho tên file), false -> "DD/MM/YYYY HH:MM"
static std::string nowString(bool fileSafe) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), fileSafe ? "%Y-%m-%d_%H-%M" : "%d/%m/%Y %H:%M", &tm);
    return buf;
}

static void printViolations(const Solution& s, std::ostream& os) {
    os << "Tong vi pham: " << s.totalViolation
       << " | Q (tai trong)=" << s.violationCapacity
       << " | D (tam bay drone)=" << s.violationRange
       << " | TW (khung thoi gian)=" << s.violationTimeWindow
       << " | W (thoi gian cho)=" << s.violationWaiting
       << " | Khach chua phuc vu=" << s.unassignedCount << "\n";
}

static void printSolution(const Solution& s, std::ostream& os) {
    os << std::fixed << std::setprecision(4);
    os << "Makespan: " << s.makespan << "\n";
    os << "Total distance: " << s.totalDistance << "\n";
    printViolations(s, os);
    os << "Feasible: " << (s.isFeasible() ? "YES" : "NO") << "\n";

    for (const auto& vp : s.vehicles) {
        const auto& v = *vp;
        os << "Vehicle " << v.id << " [" << vehicleTypeName(v.type)
           << "] completion=" << v.completionTime << "\n";
        for (size_t ti = 0; ti < v.trips.size(); ++ti) {
            const auto& t = v.trips[ti];
            os << "  Trip " << ti << " (uid=" << t.uid << "): 0";
            for (int cid : t.customers) os << " -> " << cid;
            os << " -> 0 | start=" << t.startTime
               << " return=" << t.returnTime
               << " load=" << t.load
               << " dist=" << t.travelDistance << "\n";
        }
    }
}

// Một lời giải kèm kết quả validator (dùng cho báo cáo).
struct ReportedSolution {
    bool has = false;
    Solution sol;
    ValidationReport validation;
};

// Kết quả chạy 1 instance.
struct RunSummary {
    std::string name;
    std::string path;
    int customers = 0, trucks = 0, drones = 0;
    bool ok = false;            // false -> lỗi khi đọc / chạy
    std::string errorMsg;
    long long iterations = 0;
    double seconds = 0.0;
    long long operatorUsage[NUM_OPERATORS] = {0};
    double timeFirstFeasible = -1.0;   // giây: lúc có nghiệm hợp lệ đầu tiên (-1 = không có)
    double timeBestFeasible = -1.0;    // giây: lúc tìm ra nghiệm hợp lệ tốt nhất = thời gian ra kết quả
    long long iterBestFeasible = -1;
    std::string aosSummary;            // trọng số toán tử cuối cùng (chế độ adaptive)
    std::string finishedAt;
    ReportedSolution bestFeasible;  // lời giải tốt nhất hợp lệ
    ReportedSolution bestAny;       // lời giải tốt nhất tìm được (có thể không hợp lệ)
};

static void printReported(const char* title, const ReportedSolution& r, std::ostream& os) {
    os << "[" << title << "]\n";
    if (!r.has) {
        os << "  (khong tim duoc)\n";
        return;
    }
    printSolution(r.sol, os);
    printValidationReport(r.validation, os);
}

static RunSummary runInstance(const std::string& path, const RunConfig& cfg, std::ostream& os) {
    const double maxWaitOverride = cfg.maxWaitOverride;
    RunSummary sum;
    sum.path = path;
    try {
        Instance inst = readJsonInstance(path);
        sum.name = inst.name;
        if (maxWaitOverride > 0.0) {
            inst.max_wait = maxWaitOverride;
            os << "[debug] override max_wait = " << maxWaitOverride << "\n";
        }
        sum.customers = inst.numCustomers();
        sum.trucks = inst.num_trucks;
        sum.drones = inst.num_drones;

        os << "File: " << path << "\n";
        os << "L_D (drone_range) = " << inst.drone_range << " | drone_speed = " << inst.drone_speed << "\n";
        os << "Instance: " << inst.name
           << " | customers=" << inst.numCustomers()
           << " | trucks=" << inst.num_trucks
           << " | drones=" << inst.num_drones << "\n\n";

        TabuSearchParams params;
        params.maxIterations = cfg.maxIterations;
        params.timeLimitSeconds = cfg.timeLimitSeconds;
        params.stoppingStagnation = 5000;
        // 360 quá cao so với tổng số iteration khả dụng trong ngân sách thời gian (n lớn chỉ chạy
        // được ~100-400 iteration/20s) -> ruin-recreate gần như không bao giờ được kích hoạt.
        // Hạ xuống để search có cơ hội thoát local optimum trong cùng ngân sách thời gian.
        params.diversificationStagnation = 30;
        params.neighborhood = cfg.neighborhood;
        params.randomSeed = cfg.seed;
        os << "Neighborhood mode = " << neighborhoodModeName(cfg.neighborhood.mode)
           << " | sample = " << cfg.neighborhood.sampleSize << " | seed = " << cfg.seed << "\n\n";

        // adaptiveTabuSearch in debug log ra std::cout -> chuyển hướng sang file record.
        std::streambuf* oldBuf = std::cout.rdbuf(os.rdbuf());
        TabuSearchResult result;
        try {
            result = adaptiveTabuSearch(inst, params);
        } catch (...) {
            std::cout.rdbuf(oldBuf);
            throw;
        }
        std::cout.rdbuf(oldBuf);

        sum.iterations = result.iterations;
        sum.seconds = result.elapsedSeconds;
        for (int op = 0; op < NUM_OPERATORS; ++op) sum.operatorUsage[op] = result.operatorUsage[op];
        sum.timeFirstFeasible = result.timeFirstFeasible;
        sum.timeBestFeasible = result.timeBestFeasible;
        sum.iterBestFeasible = result.iterBestFeasible;
        if (result.hasBestFeasible) {
            sum.bestFeasible = {true, result.bestFeasible, validateSolution(inst, result.bestFeasible)};
        }
        // Chưa từng có lời giải phục vụ đủ khách -> dùng lời giải trả về của thuật toán.
        const Solution& any = result.hasBestAny ? result.bestAny : result.best;
        sum.bestAny = {true, any, validateSolution(inst, any)};
        sum.ok = true;

        os << "\nIterations: " << result.iterations
           << " | Time: " << result.elapsedSeconds << "s"
           << " | Feasible found: " << (result.foundFeasible ? "YES" : "NO") << "\n";
        if (result.foundFeasible) {
            os << "Thoi gian ra loi giai hop le dau tien: " << result.timeFirstFeasible << "s"
               << " | Thoi gian ra ket qua (loi giai hop le tot nhat): " << result.timeBestFeasible << "s"
               << " (iteration " << result.iterBestFeasible << ")\n";
        }
        if (cfg.neighborhood.mode != NeighborhoodMode::ALL) {
            os << "Operator usage:";
            for (int op = 0; op < NUM_OPERATORS; ++op) os << " " << operatorName(op) << "=" << sum.operatorUsage[op];
            os << "\n";
        }
        if (cfg.neighborhood.mode == NeighborhoodMode::ADAPTIVE) {
            std::ostringstream w;
            result.operatorWeights.print(w, operatorNames());
            sum.aosSummary = w.str();
            os << sum.aosSummary;
        }
        os << "\n";
        printReported("LOI GIAI TOT NHAT HOP LE", sum.bestFeasible, os);
        os << "\n";
        printReported("LOI GIAI TOT NHAT TIM DUOC (co the khong hop le)", sum.bestAny, os);
    } catch (const std::exception& ex) {
        sum.ok = false;
        sum.errorMsg = ex.what();
        os << "Error: " << ex.what() << "\n";
    }
    sum.finishedAt = nowString(false);
    os << "\nHoan thanh luc: " << sum.finishedAt << "\n";
    return sum;
}

// Sắp xếp tên theo số: "6.5.1" < "10.5.1" < "100.10.1".
static std::vector<int> numericKey(const std::string& name) {
    std::vector<int> key;
    std::stringstream ss(name);
    std::string part;
    while (std::getline(ss, part, '.')) {
        try { key.push_back(std::stoi(part)); } catch (...) { key.push_back(0); }
    }
    return key;
}

static bool ensureDir(const char* dir) {
    std::error_code ec;
    if (fs::exists(dir) && !fs::is_directory(dir)) {
        std::cerr << "Error: '" << dir << "' ton tai nhung khong phai thu muc.\n";
        return false;
    }
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "Error: khong tao duoc thu muc '" << dir << "': " << ec.message() << "\n";
        return false;
    }
    return true;
}

// Ghi file báo cáo tổng hợp duy nhất: bảng tóm tắt + chi tiết lời giải từng file.
static void writeReport(const std::string& reportPath, const std::vector<RunSummary>& runs,
                        const std::string& startedAt, const std::string& finishedAt, const RunConfig& cfg,
                        double wallSeconds) {
    std::ofstream os(reportPath);
    const std::string line(182, '=');
    const std::string dash(182, '-');

    int nFeasible = 0, nError = 0;
    double totalTime = 0.0;
    for (const auto& r : runs) {
        if (!r.ok) { ++nError; continue; }
        nFeasible += r.bestFeasible.has;
        totalTime += r.seconds;
    }

    os << line << "\n";
    os << "BAO CAO TONG HOP KET QUA ADAPTIVE TABU SEARCH (MVRPD-TW)\n";
    os << "Che do sinh lan can: " << neighborhoodModeName(cfg.neighborhood.mode)
       << " | sample=" << cfg.neighborhood.sampleSize << " | seed=" << cfg.seed
       << " | iter=" << cfg.maxIterations << " | time=" << cfg.timeLimitSeconds << "s\n";
    if (cfg.neighborhood.mode == NeighborhoodMode::ADAPTIVE) {
        const auto& a = cfg.neighborhood.aos;
        os << "Tham so AOS: seg=" << a.segmentLength << " | r=" << a.reaction
           << " | scores=" << a.sigmaNewBest << "," << a.sigmaImproved << "," << a.sigmaAccepted << "\n";
    }
    os << "Bat dau chay : " << startedAt << "\n";
    os << "Hoan thanh   : " << finishedAt << "\n";
    os << "So file test : " << runs.size()
       << " | Co loi giai hop le: " << nFeasible
       << " | Khong co loi giai hop le: " << (static_cast<int>(runs.size()) - nFeasible - nError)
       << " | Loi khi chay: " << nError << "\n";
    os << std::fixed << std::setprecision(2) << "Tong thoi gian chay: " << totalTime << "s\n";
    os << line << "\n\n";

    // ---- Thống kê thời gian ----
    // Time      : thời gian chạy thuật toán của instance (đến khi dừng).
    // T_first   : lúc có lời giải hợp lệ đầu tiên.
    // T_best    : lúc tìm ra lời giải hợp lệ tốt nhất (= thời gian ra kết quả cuối cùng).
    struct Stat {
        int count = 0; double sum = 0.0, mn = 0.0, mx = 0.0; std::string mnName, mxName;
        void add(double v, const std::string& name) {
            if (count == 0 || v < mn) { mn = v; mnName = name; }
            if (count == 0 || v > mx) { mx = v; mxName = name; }
            sum += v; ++count;
        }
        void print(std::ostream& o, const char* label) const {
            o << "  " << std::left << std::setw(42) << label << std::right;
            if (count == 0) { o << "(khong co)\n"; return; }
            o << "Tong=" << std::setw(10) << sum << "s | TB=" << std::setw(9) << sum / count
              << "s | Min=" << std::setw(9) << mn << "s (" << mnName << ")"
              << " | Max=" << std::setw(9) << mx << "s (" << mxName << ") | so instance=" << count << "\n";
        }
    };
    Stat stRun, stFirst, stBest;
    for (const auto& r : runs) {
        if (!r.ok) continue;
        stRun.add(r.seconds, r.name);
        if (r.bestFeasible.has) {
            stFirst.add(r.timeFirstFeasible, r.name);
            stBest.add(r.timeBestFeasible, r.name);
        }
    }
    os << "THONG KE THOI GIAN" << std::setprecision(2) << "\n";
    stRun.print(os, "Thoi gian chay (Time, moi instance):");
    stFirst.print(os, "Ra loi giai hop le dau tien (T_first):");
    stBest.print(os, "Ra ket qua - hop le tot nhat (T_best):");
    os << "  " << std::left << std::setw(42) << "Thoi gian thuc te ca lan chay:" << std::right
       << wallSeconds << "s (wall-clock, gom ca doc du lieu, ghi record)\n";
    os << "  Ghi chu: T_first/T_best chi tinh tren cac instance co loi giai hop le.\n";
    os << line << "\n\n";

    // ---- Bảng tóm tắt ----
    os << "BANG TONG HOP\n";
    os << "  - Hop le tot nhat   : makespan cua loi giai tot nhat hop le ('-' neu khong tim duoc)\n";
    os << "  - Tot nhat tim duoc : makespan nho nhat trong cac loi giai da duyet (co the vi pham);\n";
    os << "                        cac cot vi pham (Tong VP, Q, D, TW, W, Chua PV) la cua loi giai nay\n";
    os << "  - T_first / T_best  : giay luc co loi giai hop le dau tien / luc ra loi giai hop le tot nhat ('-' neu khong co)\n\n";
    os << std::left << std::setw(10) << "File" << std::right
       << std::setw(5) << "n" << std::setw(4) << "T" << std::setw(4) << "D"
       << std::setw(10) << "Time(s)" << std::setw(7) << "Iter"
       << std::setw(11) << "T_first(s)" << std::setw(11) << "T_best(s)" << std::setw(10) << "Iter_best"
       << std::setw(17) << "Hop le tot nhat"
       << std::setw(19) << "Tot nhat tim duoc"
       << std::setw(12) << "Tong VP"
       << std::setw(11) << "Q" << std::setw(11) << "D"
       << std::setw(11) << "TW" << std::setw(11) << "W"
       << std::setw(9) << "Chua PV" << "\n";
    os << dash << "\n";
    for (const auto& r : runs) {
        os << std::left << std::setw(10) << r.name << std::right;
        if (!r.ok) {
            os << "  LOI: " << r.errorMsg << "\n";
            continue;
        }
        const Solution& a = r.bestAny.sol;
        os << std::setw(5) << r.customers << std::setw(4) << r.trucks << std::setw(4) << r.drones
           << std::setprecision(2) << std::setw(10) << r.seconds << std::setw(7) << r.iterations;
        if (r.bestFeasible.has)
            os << std::setw(11) << r.timeFirstFeasible << std::setw(11) << r.timeBestFeasible
               << std::setw(10) << r.iterBestFeasible;
        else
            os << std::setw(11) << "-" << std::setw(11) << "-" << std::setw(10) << "-";
        os << std::setprecision(4);
        if (r.bestFeasible.has) os << std::setw(17) << r.bestFeasible.sol.makespan;
        else os << std::setw(17) << "-";
        os << std::setw(19) << a.makespan
           << std::setw(12) << a.totalViolation
           << std::setw(11) << a.violationCapacity << std::setw(11) << a.violationRange
           << std::setw(11) << a.violationTimeWindow << std::setw(11) << a.violationWaiting
           << std::setw(9) << a.unassignedCount << "\n";
    }
    os << "\n\n";

    // ---- Chi tiết từng file ----
    os << "CHI TIET LOI GIAI TUNG FILE\n";
    for (const auto& r : runs) {
        os << "\n" << line << "\n";
        os << "File: " << r.path << " | Instance: " << r.name << "\n";
        if (!r.ok) {
            os << "LOI: " << r.errorMsg << "\n";
            continue;
        }
        os << "Khach hang=" << r.customers << " | Truck=" << r.trucks << " | Drone=" << r.drones << "\n";
        os << std::setprecision(2) << "Thoi gian chay: " << r.seconds << "s | Iterations: " << r.iterations;
        if (r.bestFeasible.has)
            os << " | T_first=" << r.timeFirstFeasible << "s | T_best=" << r.timeBestFeasible
               << "s (iter " << r.iterBestFeasible << ")";
        os
           << " | Hoan thanh luc: " << r.finishedAt << "\n";
        os << dash << "\n";
        if (!r.aosSummary.empty()) os << r.aosSummary << "\n";
        printReported("LOI GIAI TOT NHAT HOP LE", r.bestFeasible, os);
        os << "\n";
        printReported("LOI GIAI TOT NHAT TIM DUOC (co the khong hop le)", r.bestAny, os);
    }
    os << "\n" << line << "\n";
    os << "Hoan thanh luc: " << finishedAt << "\n";
}

int main(int argc, char** argv) {
    RunConfig cfg;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind("--mode=", 0) == 0) {
            std::string m = a.substr(7);
            if (m == "all") cfg.neighborhood.mode = NeighborhoodMode::ALL;
            else if (m == "random") cfg.neighborhood.mode = NeighborhoodMode::RANDOM_OP;
            else if (m == "seq") cfg.neighborhood.mode = NeighborhoodMode::SEQUENTIAL;
            else if (m == "adaptive") cfg.neighborhood.mode = NeighborhoodMode::ADAPTIVE;
            else { std::cerr << "Error: --mode phai la all|random|seq|adaptive\n"; return 1; }
        } else if (a.rfind("--aos-seg=", 0) == 0) {
            cfg.neighborhood.aos.segmentLength = std::max(1, std::stoi(a.substr(10)));
        } else if (a.rfind("--aos-r=", 0) == 0) {
            cfg.neighborhood.aos.reaction = std::stod(a.substr(8));
        } else if (a.rfind("--aos-scores=", 0) == 0) {
            // dạng s1,s2,s3 (new best, improved, accepted)
            std::stringstream ss(a.substr(13));
            std::string part;
            double v[3] = {cfg.neighborhood.aos.sigmaNewBest, cfg.neighborhood.aos.sigmaImproved,
                           cfg.neighborhood.aos.sigmaAccepted};
            for (int k = 0; k < 3 && std::getline(ss, part, ','); ++k) v[k] = std::stod(part);
            cfg.neighborhood.aos.sigmaNewBest = v[0];
            cfg.neighborhood.aos.sigmaImproved = v[1];
            cfg.neighborhood.aos.sigmaAccepted = v[2];
        } else if (a.rfind("--sample=", 0) == 0) {
            cfg.neighborhood.sampleSize = std::stoi(a.substr(9));
        } else if (a.rfind("--seed=", 0) == 0) {
            cfg.seed = static_cast<unsigned int>(std::stoul(a.substr(7)));
        } else if (a.rfind("--iter=", 0) == 0) {
            cfg.maxIterations = std::stoll(a.substr(7));
        } else if (a.rfind("--time=", 0) == 0) {
            cfg.timeLimitSeconds = std::stod(a.substr(7));
        } else {
            positional.push_back(a);
        }
    }
    if (positional.size() > 1) cfg.maxWaitOverride = std::stod(positional[1]);
    // Tên file kèm mode/sample/seed để các lần chạy nhiều seed trong cùng 1 phút không ghi đè nhau.
    std::string modeTag = neighborhoodModeName(cfg.neighborhood.mode);
    if (cfg.neighborhood.mode != NeighborhoodMode::ALL)
        modeTag += "_sample" + std::to_string(cfg.neighborhood.sampleSize);
    modeTag += "_seed" + std::to_string(cfg.seed);

    std::vector<std::string> files;
    if (!positional.empty()) {
        files.push_back(positional[0]);
    } else {
        if (!fs::is_directory(DATA_DIR)) {
            std::cerr << "Error: khong tim thay thu muc '" << DATA_DIR << "'.\n";
            return 1;
        }
        for (const auto& e : fs::directory_iterator(DATA_DIR)) {
            if (e.is_regular_file() && e.path().extension() == ".json")
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end(), [](const std::string& a, const std::string& b) {
            return numericKey(fs::path(a).stem().string()) < numericKey(fs::path(b).stem().string());
        });
    }
    if (files.empty()) {
        std::cerr << "Error: khong co file .json nao de chay.\n";
        return 1;
    }
    if (!ensureDir(RECORD_DIR) || !ensureDir(REPORT_DIR)) return 1;

    std::string startedAt = nowString(false);
    auto wallStart = std::chrono::steady_clock::now();
    std::vector<RunSummary> summaries;
    for (size_t i = 0; i < files.size(); ++i) {
        const std::string& path = files[i];
        std::cerr << "[" << (i + 1) << "/" << files.size() << "] " << path << " ... " << std::flush;

        std::ostringstream buf;
        RunSummary sum = runInstance(path, cfg, buf);
        if (sum.name.empty()) sum.name = fs::path(path).stem().string();

        std::string recordPath = (fs::path(RECORD_DIR) /
            (sum.name + "_" + modeTag + "_" + nowString(true) + ".txt")).string();
        std::ofstream rec(recordPath);
        rec << buf.str();

        if (sum.ok) {
            std::cerr << std::fixed << std::setprecision(4) << "hop le tot nhat=";
            if (sum.bestFeasible.has)
                std::cerr << sum.bestFeasible.sol.makespan << " (T_best=" << std::setprecision(2)
                          << sum.timeBestFeasible << "s)" << std::setprecision(4);
            else std::cerr << "-";
            std::cerr << " | tot nhat tim duoc=" << sum.bestAny.sol.makespan
                      << " (VP=" << sum.bestAny.sol.totalViolation << ")"
                      << " (" << std::setprecision(2) << sum.seconds << "s) -> " << recordPath << "\n";
        } else {
            std::cerr << "LOI: " << sum.errorMsg << "\n";
        }
        summaries.push_back(sum);
    }

    std::string finished = nowString(false);
    std::string reportPath = (fs::path(REPORT_DIR) / ("report_" + modeTag + "_" + nowString(true) + ".txt")).string();
    double wallSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    writeReport(reportPath, summaries, startedAt, finished, cfg, wallSeconds);

    int nError = 0;
    for (const auto& r : summaries) nError += !r.ok;
    std::cerr << "\nDa ghi bao cao tong hop: " << reportPath << " (hoan thanh luc " << finished << ")\n";
    return nError == 0 ? 0 : 1;
}
