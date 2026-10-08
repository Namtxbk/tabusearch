// main.cpp
// Chạy Adaptive Tabu Search cho bài toán MVRPD-TW từ file instance JSON.
#include <iostream>
#include <iomanip>
#include <string>
#include "instance.hpp"
#include "tabu_search.hpp"
#include "validator.hpp"

static const char* vehicleTypeName(VehicleType t) {
    return (t == VehicleType::TRUCK) ? "TRUCK" : "DRONE";
}

static void printSolution(const Instance& inst, const Solution& s) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Makespan: " << s.makespan << "\n";
    std::cout << "Total distance: " << s.totalDistance << "\n";
    std::cout << "Total violation: " << s.totalViolation
               << " (Q=" << s.violationCapacity
               << ", D=" << s.violationRange
               << ", TW=" << s.violationTimeWindow
               << ", W=" << s.violationWaiting << ")\n";
    std::cout << "Feasible: " << (s.isFeasible() ? "YES" : "NO") << "\n\n";

    for (const auto& vp : s.vehicles) {
        const auto& v = *vp;
        std::cout << "Vehicle " << v.id << " [" << vehicleTypeName(v.type)
                   << "] completion=" << v.completionTime << "\n";
        for (size_t ti = 0; ti < v.trips.size(); ++ti) {
            const auto& t = v.trips[ti];
            std::cout << "  Trip " << ti << " (uid=" << t.uid << "): 0";
            for (int cid : t.customers) std::cout << " -> " << cid;
            std::cout << " -> 0 | start=" << t.startTime
                       << " return=" << t.returnTime
                       << " load=" << t.load
                       << " dist=" << t.travelDistance << "\n";
        }
    }
    (void)inst;
}

int main(int argc, char** argv) {
    std::string path = (argc > 1) ? argv[1] : "6_5_1.json";
    double maxWaitOverride = (argc > 2) ? std::stod(argv[2]) : -1.0; 

    try {
        Instance inst = readJsonInstance(path);
        if (maxWaitOverride > 0.0) {
            inst.max_wait = maxWaitOverride;
            std::cout << "[debug] override max_wait = " << maxWaitOverride << "\n";
        }
        std::cout << "Instance: " << inst.name
                  << " | customers=" << inst.numCustomers()
                  << " | trucks=" << inst.num_trucks
                  << " | drones=" << inst.num_drones << "\n\n";

        TabuSearchParams params;
        params.maxIterations = 2000;
        params.timeLimitSeconds = 200000.0;
        params.stoppingStagnation = 500;
        // 360 quá cao so với tổng số iteration khả dụng trong ngân sách thời gian (n lớn chỉ chạy
        // được ~100-400 iteration/20s) -> ruin-recreate gần như không bao giờ được kích hoạt.
        // Hạ xuống để search có cơ hội thoát local optimum trong cùng ngân sách thời gian.
        params.diversificationStagnation = 30;

        TabuSearchResult result = adaptiveTabuSearch(inst, params);

        std::cout << "Iterations: " << result.iterations
                  << " | Time: " << result.elapsedSeconds << "s"
                  << " | Feasible found: " << (result.foundFeasible ? "YES" : "NO") << "\n\n";

        printSolution(inst, result.best);

        std::cout << "\n";
        ValidationReport report = validateSolution(inst, result.best);
        printValidationReport(report);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
