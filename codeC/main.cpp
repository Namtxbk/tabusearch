// main.cpp — MVRPD-TW Solver (C++)
// Tương đương batch_compare.py
//
// Biên dịch:
//   g++ -std=c++17 -O3 -o solver main.cpp -I.
//
// Chạy:
//   ./solver --data_dir WithTimeWindows --baseline result.csv --output ket_qua.csv
//   ./solver --data_dir WithTimeWindows --baseline result.csv --output ket_qua.csv \
//             --max_iter 2000 --max_no_improve 300 --tenure 7 --time_limit 120 --verbose

#include "instance.h"
#include "solution.h"
#include "construction.h"
#include "tabu_search.h"
#include "include/json.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <cstring>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
// Đọc baseline CSV, lấy makespan tốt nhất cho mỗi Problem
// (tương đương load_baseline() trong Python)
// ─────────────────────────────────────────────────────────────────────────────
static double parse_list_max(const std::string& s) {
    // Parse "[1234.56]" hoặc "1234.56" thành double, trả về max
    std::string t = s;
    for (char& c : t) if (c == '[' || c == ']') c = ' ';
    std::istringstream ss(t);
    double val = 0.0, mx = 0.0;
    bool first = true;
    while (ss >> val) { mx = first ? val : std::max(mx, val); first = false; }
    return mx;
}

static std::map<std::string, double> load_baseline(const std::string& path) {
    std::map<std::string, double> best;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Không mở được baseline: " + path);

    std::string line;
    std::getline(f, line);  // header
    // Tìm index cột "Problem", "Truck working time", "Drone working time"
    std::vector<std::string> headers;
    std::istringstream hss(line);
    std::string col;
    while (std::getline(hss, col, ',')) headers.push_back(col);

    int idx_prob = -1, idx_truck = -1, idx_drone = -1;
    for (int i = 0; i < (int)headers.size(); ++i) {
        if (headers[i] == "Problem")           idx_prob  = i;
        if (headers[i] == "Truck working time") idx_truck = i;
        if (headers[i] == "Drone working time") idx_drone = i;
    }
    if (idx_prob < 0) throw std::runtime_error("Baseline CSV thiếu cột 'Problem'");

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cols;
        std::string tok;
        bool in_q = false;
        for (char c : line) {
            if (c == '"') { in_q = !in_q; continue; }
            if (c == ',' && !in_q) { cols.push_back(tok); tok.clear(); }
            else tok += c;
        }
        cols.push_back(tok);

        if (idx_prob >= (int)cols.size()) continue;
        std::string prob = cols[idx_prob];
        // Trim spaces
        prob.erase(0, prob.find_first_not_of(" \t\r\n"));
        prob.erase(prob.find_last_not_of(" \t\r\n") + 1);

        double truck_t = (idx_truck >= 0 && idx_truck < (int)cols.size())
                          ? parse_list_max(cols[idx_truck]) : 0.0;
        double drone_t = (idx_drone >= 0 && idx_drone < (int)cols.size())
                          ? parse_list_max(cols[idx_drone]) : 0.0;
        double ms = std::max(truck_t, drone_t);

        if (!best.count(prob) || ms < best[prob]) best[prob] = ms;
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sắp xếp tên instance kiểu "6.5.1", "10.5.2" theo thứ tự số tự nhiên
// ─────────────────────────────────────────────────────────────────────────────
static std::vector<int> natural_key(const std::string& s) {
    std::vector<int> keys;
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, '.')) {
        bool all_digit = !tok.empty();
        for (char c : tok) if (!std::isdigit(c)) { all_digit = false; break; }
        keys.push_back(all_digit ? std::stoi(tok) : 0);
    }
    return keys;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Parse arguments
    std::string data_dir, baseline_path, output_path = "ket_qua_so_sanh.csv";
    TabuSearchConfig cfg;
    cfg.verbose = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--data_dir") && i+1 < argc)
            data_dir = argv[++i];
        else if (!strcmp(argv[i], "--baseline") && i+1 < argc)
            baseline_path = argv[++i];
        else if (!strcmp(argv[i], "--output") && i+1 < argc)
            output_path = argv[++i];
        else if (!strcmp(argv[i], "--max_iter") && i+1 < argc)
            cfg.max_iter = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--max_no_improve") && i+1 < argc)
            cfg.max_no_improve = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--tenure") && i+1 < argc)
            cfg.tenure_base = std::stoi(argv[++i]);
        else if (!strcmp(argv[i], "--time_limit") && i+1 < argc)
            cfg.time_limit = std::stod(argv[++i]);
        else if (!strcmp(argv[i], "--verbose"))
            cfg.verbose = true;
    }

    if (data_dir.empty() || baseline_path.empty()) {
        std::cerr << "Cách dùng: solver --data_dir <dir> --baseline <csv> "
                     "[--output <csv>] [--max_iter N] [--max_no_improve N] "
                     "[--tenure N] [--time_limit T] [--verbose]\n";
        return 1;
    }

    // Đọc baseline
    std::cout << "Đang đọc baseline từ: " << baseline_path << "\n";
    auto baseline = load_baseline(baseline_path);
    std::cout << "  -> Tìm thấy " << baseline.size() << " instance trong baseline.\n\n";

    // Liệt kê và sắp xếp file JSON
    std::vector<std::string> json_files;
    for (auto& entry : fs::directory_iterator(data_dir))
        if (entry.path().extension() == ".json")
            json_files.push_back(entry.path().filename().string());
    std::sort(json_files.begin(), json_files.end(), [](const std::string& a, const std::string& b){
        auto ka = natural_key(a.substr(0, a.size()-5));
        auto kb = natural_key(b.substr(0, b.size()-5));
        return ka < kb;
    });

    if (json_files.empty()) {
        std::cerr << "[LỖI] Không tìm thấy file .json trong " << data_dir << "\n";
        return 1;
    }

    // Mở file output và ghi header
    std::ofstream out(output_path);
    out << "Problem,Makespan_baseline,Makespan_algo,Gap_%,"
        << "Feasible,AllServed,Construction_s,TS_s,"
        << "Penalty_TW,Penalty_Cap,Penalty_Range,"
        << "Num_trucks_goc,Num_drones_goc,"
        << "Extra_trucks_construction,Extra_drones_construction,"
        << "Num_trucks_used_final,Num_drones_used_final\n";

    int n_total = (int)json_files.size();
    int n_feasible = 0, n_served = 0, n_valid_gap = 0;
    double sum_gap = 0.0, min_gap = 1e18, max_gap = -1e18;

    for (int idx = 0; idx < n_total; ++idx) {
        std::string fname = json_files[idx];
        std::string problem = fname.substr(0, fname.size()-5);
        std::string fpath = data_dir + "/" + fname;

        std::cout << "[" << idx+1 << "/" << n_total << "] "
                  << "Đang chạy " << problem << " ... " << std::flush;

        try {
            Instance inst = read_json_instance(fpath);

            // Construction
            auto t0 = std::chrono::steady_clock::now();
            Solution init_sol = build_initial_solution(inst);
            double t_con = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            int extra_trucks_con = init_sol.extra_trucks_used;
            int extra_drones_con = init_sol.extra_drones_used;

            // Tabu Search
            t0 = std::chrono::steady_clock::now();
            auto [best_sol, history] = advanced_tabu_search(init_sol, inst, cfg);
            double t_ts = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();

            double ms_algo = best_sol.makespan();
            bool feasible  = best_sol.is_feasible(inst);
            bool served    = best_sol.all_served(inst);
            double pen_tw  = best_sol.penalty_tw(inst);
            double pen_cap = best_sol.penalty_cap(inst);
            double pen_rng = best_sol.penalty_range(inst);

            int num_trucks_final = 0, num_drones_final = 0;
            for (auto& v : best_sol.trucks)
                if (std::any_of(v.trips.begin(), v.trips.end(),
                    [](const Trip& t){ return t.num_customers() > 0; }))
                    ++num_trucks_final;
            for (auto& v : best_sol.drones)
                if (std::any_of(v.trips.begin(), v.trips.end(),
                    [](const Trip& t){ return t.num_customers() > 0; }))
                    ++num_drones_final;

            double ms_base = -1.0;
            if (baseline.count(problem)) ms_base = baseline[problem];

            double gap = 0.0;
            bool has_gap = (ms_base > 0.0);
            if (has_gap) gap = (ms_algo - ms_base) / ms_base * 100.0;

            // Print
            std::cout << std::fixed << std::setprecision(2)
                      << "Makespan=" << ms_algo
                      << "  Baseline=" << (has_gap ? std::to_string(ms_base) : "N/A")
                      << "  Gap=" << (has_gap ? std::to_string(gap) + "%" : "N/A")
                      << "\n";

            // Ghi CSV
            out << std::fixed << std::setprecision(4)
                << problem << ","
                << (has_gap ? std::to_string(ms_base) : "") << ","
                << ms_algo << ","
                << (has_gap ? std::to_string(gap) : "") << ","
                << (feasible ? "True" : "False") << ","
                << (served   ? "True" : "False") << ","
                << t_con << "," << t_ts << ","
                << pen_tw << "," << pen_cap << "," << pen_rng << ","
                << inst.num_trucks << "," << inst.num_drones << ","
                << extra_trucks_con << "," << extra_drones_con << ","
                << num_trucks_final << "," << num_drones_final << "\n";

            if (feasible) ++n_feasible;
            if (served)   ++n_served;
            if (has_gap) {
                sum_gap += gap; ++n_valid_gap;
                min_gap = std::min(min_gap, gap);
                max_gap = std::max(max_gap, gap);
            }

        } catch (const std::exception& e) {
            std::cerr << "LỖI: " << e.what() << "\n";
            out << problem << ",,,,ERROR,,,,,,,,,,,\n";
        }
    }

    out.close();

    // Tổng kết
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TỔNG KẾT\n" << std::string(60, '=') << "\n";
    std::cout << "  Tổng số instance       : " << n_total << "\n";
    std::cout << "  Số nghiệm khả thi      : " << n_feasible << "/" << n_total << "\n";
    std::cout << "  Số nghiệm phục vụ đủ   : " << n_served   << "/" << n_total << "\n";
    if (n_valid_gap > 0) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Gap trung bình         : " << sum_gap / n_valid_gap << "%\n";
        std::cout << "  Gap nhỏ nhất           : " << min_gap << "%\n";
        std::cout << "  Gap lớn nhất           : " << max_gap << "%\n";
    }
    std::cout << "\n  Kết quả chi tiết đã lưu tại: " << output_path << "\n";
    return 0;
}
