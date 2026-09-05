// main.cpp
// Chạy Adaptive Tabu Search cho bài toán MVRPD-TW từ file instance JSON.
#include <iostream>
#include <iomanip>
#include <string>
#include "instance.hpp"
#include "tabu_search.hpp"

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

    for (const auto& v : s.vehicles) {
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
    double maxWaitOverride = (argc > 2) ? std::stod(argv[2]) : -1.0; // tham số debug: override L_w nếu > 0

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
        params.timeLimitSeconds = 20.0;
        params.stoppingStagnation = 300;
        params.diversificationStagnation = 60;

        TabuSearchResult result = adaptiveTabuSearch(inst, params);

        std::cout << "Iterations: " << result.iterations
                  << " | Time: " << result.elapsedSeconds << "s"
                  << " | Feasible found: " << (result.foundFeasible ? "YES" : "NO") << "\n\n";

        printSolution(inst, result.best);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
