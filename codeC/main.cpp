// main.cpp
// Chạy Adaptive Tabu Search cho bài toán MVRPD-TW.
//   solver.exe                 -> chạy TẤT CẢ file *.json trong thư mục data/
//   solver.exe <file.json>     -> chỉ chạy 1 file
//   solver.exe <file.json> <max_wait>
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
#include <filesystem>
#include "instance.hpp"
#include "tabu_search.hpp"
#include "validator.hpp"

namespace fs = std::filesystem;

static const char* DATA_DIR = "data";
static const char* RECORD_DIR = "record";
static const char* REPORT_DIR = "report";

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

static RunSummary runInstance(const std::string& path, double maxWaitOverride, std::ostream& os) {
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
        params.maxIterations = 2000;
        params.timeLimitSeconds = 200000.0;
        params.stoppingStagnation = 5000;
        // 360 quá cao so với tổng số iteration khả dụng trong ngân sách thời gian (n lớn chỉ chạy
        // được ~100-400 iteration/20s) -> ruin-recreate gần như không bao giờ được kích hoạt.
        // Hạ xuống để search có cơ hội thoát local optimum trong cùng ngân sách thời gian.
        params.diversificationStagnation = 30;

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
        if (result.hasBestFeasible) {
            sum.bestFeasible = {true, result.bestFeasible, validateSolution(inst, result.bestFeasible)};
        }
        // Chưa từng có lời giải phục vụ đủ khách -> dùng lời giải trả về của thuật toán.
        const Solution& any = result.hasBestAny ? result.bestAny : result.best;
        sum.bestAny = {true, any, validateSolution(inst, any)};
        sum.ok = true;

        os << "\nIterations: " << result.iterations
           << " | Time: " << result.elapsedSeconds << "s"
           << " | Feasible found: " << (result.foundFeasible ? "YES" : "NO") << "\n\n";
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
                        const std::string& startedAt, const std::string& finishedAt) {
    std::ofstream os(reportPath);
    const std::string line(150, '=');
    const std::string dash(150, '-');

    int nFeasible = 0, nError = 0;
    double totalTime = 0.0;
    for (const auto& r : runs) {
        if (!r.ok) { ++nError; continue; }
        nFeasible += r.bestFeasible.has;
        totalTime += r.seconds;
    }

    os << line << "\n";
    os << "BAO CAO TONG HOP KET QUA ADAPTIVE TABU SEARCH (MVRPD-TW)\n";
    os << "Bat dau chay : " << startedAt << "\n";
    os << "Hoan thanh   : " << finishedAt << "\n";
    os << "So file test : " << runs.size()
       << " | Co loi giai hop le: " << nFeasible
       << " | Khong co loi giai hop le: " << (static_cast<int>(runs.size()) - nFeasible - nError)
       << " | Loi khi chay: " << nError << "\n";
    os << std::fixed << std::setprecision(2) << "Tong thoi gian chay: " << totalTime << "s\n";
    os << line << "\n\n";

    // ---- Bảng tóm tắt ----
    os << "BANG TONG HOP\n";
    os << "  - Hop le tot nhat   : makespan cua loi giai tot nhat hop le ('-' neu khong tim duoc)\n";
    os << "  - Tot nhat tim duoc : makespan nho nhat trong cac loi giai da duyet (co the vi pham);\n";
    os << "                        cac cot vi pham (Tong VP, Q, D, TW, W, Chua PV) la cua loi giai nay\n\n";
    os << std::left << std::setw(10) << "File" << std::right
       << std::setw(5) << "n" << std::setw(4) << "T" << std::setw(4) << "D"
       << std::setw(10) << "Time(s)" << std::setw(7) << "Iter"
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
           << std::setprecision(2) << std::setw(10) << r.seconds << std::setw(7) << r.iterations
           << std::setprecision(4);
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
        os << std::setprecision(2) << "Thoi gian chay: " << r.seconds << "s | Iterations: " << r.iterations
           << " | Hoan thanh luc: " << r.finishedAt << "\n";
        os << dash << "\n";
        printReported("LOI GIAI TOT NHAT HOP LE", r.bestFeasible, os);
        os << "\n";
        printReported("LOI GIAI TOT NHAT TIM DUOC (co the khong hop le)", r.bestAny, os);
    }
    os << "\n" << line << "\n";
    os << "Hoan thanh luc: " << finishedAt << "\n";
}

int main(int argc, char** argv) {
    double maxWaitOverride = (argc > 2) ? std::stod(argv[2]) : -1.0;

    std::vector<std::string> files;
    if (argc > 1) {
        files.push_back(argv[1]);
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
    std::vector<RunSummary> summaries;
    for (size_t i = 0; i < files.size(); ++i) {
        const std::string& path = files[i];
        std::cerr << "[" << (i + 1) << "/" << files.size() << "] " << path << " ... " << std::flush;

        std::ostringstream buf;
        RunSummary sum = runInstance(path, maxWaitOverride, buf);
        if (sum.name.empty()) sum.name = fs::path(path).stem().string();

        std::string recordPath = (fs::path(RECORD_DIR) /
            (sum.name + "_" + nowString(true) + ".txt")).string();
        std::ofstream rec(recordPath);
        rec << buf.str();

        if (sum.ok) {
            std::cerr << std::fixed << std::setprecision(4) << "hop le tot nhat=";
            if (sum.bestFeasible.has) std::cerr << sum.bestFeasible.sol.makespan;
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
    std::string reportPath = (fs::path(REPORT_DIR) / ("report_" + nowString(true) + ".txt")).string();
    writeReport(reportPath, summaries, startedAt, finished);

    int nError = 0;
    for (const auto& r : summaries) nError += !r.ok;
    std::cerr << "\nDa ghi bao cao tong hop: " << reportPath << " (hoan thanh luc " << finished << ")\n";
    return nError == 0 ? 0 : 1;
}
