// validator.hpp
// Bộ kiểm tra lời giải ĐỘC LẬP — chỉ dùng dữ liệu thô của Instance (toạ độ, demand, e_i, l_i, tham số xe)
// và cấu trúc tuyến (vehicles -> trips -> customers). KHÔNG gọi recomputeVehicle / evaluateSolution /
// distMat, mà tự mô phỏng lại lịch trình từ đầu, để phát hiện lỗi trong chính các hàm đó.
//
// Kiểm tra:
//   [Cấu trúc]  số truck/drone đúng với instance, id xe không trùng, id khách hợp lệ,
//               mọi khách được phục vụ ĐÚNG 1 lần, khách C1 không đi drone.
//   [Ràng buộc] tải trọng mỗi chuyến <= M_v, thời gian bay chuyến drone <= L_D,
//               a_i <= l_i, (return_trip - a_i) <= L_w.
//   [Nhất quán] startTime/arrivalTime/returnTime/load/travelDistance/flightTime đã lưu trong Trip,
//               completionTime của Vehicle, makespan/totalDistance/unassignedCount của Solution
//               khớp với giá trị mô phỏng lại; cờ isFeasible() khớp với kết luận của validator.
#pragma once

#include <cmath>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "instance.hpp"
#include "solution.hpp"

struct ValidationReport {
    bool valid = true;                    // true <=> không có lỗi (errors rỗng)
    std::vector<std::string> errors;      // vi phạm ràng buộc / sai cấu trúc / số liệu lưu sai
    std::vector<std::string> warnings;    // bất thường nhưng không làm lời giải sai (trip rỗng, uid trùng...)

    double makespan = 0.0;                // C_max mô phỏng lại độc lập
    double totalDistance = 0.0;           // tổng quãng đường mô phỏng lại độc lập

    void error(const std::string& msg) { valid = false; errors.push_back(msg); }
    void warn(const std::string& msg) { warnings.push_back(msg); }
};

namespace validator_detail {

// Sai số cho phép: tuyệt đối cho ràng buộc, tương đối cho so khớp số liệu đã lưu.
static constexpr double CONSTRAINT_TOL = 1e-6;
static constexpr double MATCH_TOL = 1e-6;

inline bool nearlyEqual(double a, double b) {
    return std::fabs(a - b) <= MATCH_TOL * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

// Khoảng cách Euclid tính thẳng từ toạ độ (không dùng inst.distMat).
inline double euclid(const Instance& inst, int i, int j) {
    const Customer& a = (i == 0) ? inst.depot : inst.customers[i - 1];
    const Customer& b = (j == 0) ? inst.depot : inst.customers[j - 1];
    double dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline double speedOf(const Instance& inst, bool isDrone) {
    double sp = isDrone ? inst.drone_speed : inst.truck_speed;
    return (sp <= 0.0) ? 1.0 : sp;
}

inline std::string tripTag(const Vehicle& v, int ti) {
    std::ostringstream os;
    os << (v.type == VehicleType::DRONE ? "DRONE " : "TRUCK ") << v.id << " / trip " << ti;
    return os.str();
}

} // namespace validator_detail

inline ValidationReport validateSolution(const Instance& inst, const Solution& s) {
    using namespace validator_detail;
    ValidationReport rep;
    const int n = inst.numCustomers();

    // ---------- Đội xe ----------
    int truckCount = 0, droneCount = 0;
    std::set<int> vehicleIds;
    for (const auto& vp : s.vehicles) {
        if (!vp) { rep.error("Solution chua con tro vehicle null"); continue; }
        (vp->type == VehicleType::TRUCK ? truckCount : droneCount)++;
        if (!vehicleIds.insert(vp->id).second) {
            rep.error("Trung id vehicle: " + std::to_string(vp->id));
        }
    }
    if (truckCount != inst.num_trucks) {
        rep.error("So truck = " + std::to_string(truckCount) + ", instance yeu cau " + std::to_string(inst.num_trucks));
    }
    if (droneCount != inst.num_drones) {
        rep.error("So drone = " + std::to_string(droneCount) + ", instance yeu cau " + std::to_string(inst.num_drones));
    }

    // ---------- Mô phỏng từng xe ----------
    std::vector<int> visitCount(n + 1, 0);
    std::set<std::uint64_t> tripUids;
    double makespan = 0.0, totalDistance = 0.0;

    for (const auto& vp : s.vehicles) {
        if (!vp) continue;
        const Vehicle& v = *vp;
        const bool isDrone = (v.type == VehicleType::DRONE);
        const double speed = speedOf(inst, isDrone);
        const double capacity = isDrone ? inst.drone_capacity : inst.truck_capacity;

        double clock = 0.0; // mọi xe xuất phát tại depot lúc 0, các chuyến nối tiếp nhau

        for (int ti = 0; ti < static_cast<int>(v.trips.size()); ++ti) {
            const Trip& trip = v.trips[ti];
            const std::string tag = tripTag(v, ti);

            if (trip.customers.empty()) rep.warn(tag + ": trip rong");
            if (!tripUids.insert(trip.uid).second) rep.warn(tag + ": trung uid " + std::to_string(trip.uid));
            if (trip.vehicleId != -1 && trip.vehicleId != v.id) {
                rep.warn(tag + ": trip.vehicleId=" + std::to_string(trip.vehicleId) + " khac id xe so huu");
            }

            const double start = clock;
            double t = start, load = 0.0, dist = 0.0, flight = 0.0;
            int prev = 0;
            std::vector<double> arrivals;
            arrivals.reserve(trip.customers.size());
            bool tripStructOk = true;

            for (int cid : trip.customers) {
                if (cid < 1 || cid > n) {
                    rep.error(tag + ": id khach khong hop le " + std::to_string(cid));
                    tripStructOk = false;
                    break;
                }
                ++visitCount[cid];
                const Customer& c = inst.customers[cid - 1];
                if (isDrone && c.is_c1) {
                    rep.error(tag + ": khach " + std::to_string(cid) + " thuoc C1 (chi truck) nhung di bang drone");
                }

                double d = euclid(inst, prev, cid);
                t += d / speed;
                t = std::max(t, c.ready);              // a_i = max{e_i, a_prev + tau}
                arrivals.push_back(t);
                load += c.demand;
                dist += d;
                flight += d / speed;
                prev = cid;
            }
            if (!tripStructOk) continue;

            double dBack = euclid(inst, prev, 0);
            t += dBack / speed;
            dist += dBack;
            flight += dBack / speed;
            const double ret = t;

            // ---- Ràng buộc ----
            if (load > capacity + CONSTRAINT_TOL) {
                rep.error(tag + ": vuot tai " + std::to_string(load) + " > " + std::to_string(capacity));
            }
            if (isDrone && flight > inst.drone_range + CONSTRAINT_TOL) {
                rep.error(tag + ": vuot thoi gian bay " + std::to_string(flight) + " > " + std::to_string(inst.drone_range));
            }
            for (std::size_t p = 0; p < trip.customers.size(); ++p) {
                int cid = trip.customers[p];
                const Customer& c = inst.customers[cid - 1];
                if (arrivals[p] > c.due + CONSTRAINT_TOL) {
                    rep.error(tag + ": khach " + std::to_string(cid) + " den tre a=" + std::to_string(arrivals[p])
                              + " > l=" + std::to_string(c.due));
                }
                double wait = ret - arrivals[p];
                if (wait > inst.max_wait + CONSTRAINT_TOL) {
                    rep.error(tag + ": khach " + std::to_string(cid) + " cho hang " + std::to_string(wait)
                              + " > L_w=" + std::to_string(inst.max_wait));
                }
            }
            if (ret > inst.depot.due + CONSTRAINT_TOL) {
                rep.warn(tag + ": ve depot luc " + std::to_string(ret) + " sau gio dong cua " + std::to_string(inst.depot.due));
            }

            // ---- Nhất quán với số liệu đã lưu ----
            if (!nearlyEqual(trip.startTime, start))
                rep.error(tag + ": startTime luu=" + std::to_string(trip.startTime) + " thuc=" + std::to_string(start));
            if (!nearlyEqual(trip.returnTime, ret))
                rep.error(tag + ": returnTime luu=" + std::to_string(trip.returnTime) + " thuc=" + std::to_string(ret));
            if (!nearlyEqual(trip.load, load))
                rep.error(tag + ": load luu=" + std::to_string(trip.load) + " thuc=" + std::to_string(load));
            if (!nearlyEqual(trip.travelDistance, dist))
                rep.error(tag + ": travelDistance luu=" + std::to_string(trip.travelDistance) + " thuc=" + std::to_string(dist));
            if (!nearlyEqual(trip.flightTime, flight))
                rep.error(tag + ": flightTime luu=" + std::to_string(trip.flightTime) + " thuc=" + std::to_string(flight));
            if (trip.arrivalTime.size() != trip.customers.size()) {
                rep.error(tag + ": arrivalTime co " + std::to_string(trip.arrivalTime.size())
                          + " phan tu, trip co " + std::to_string(trip.customers.size()) + " khach");
            } else {
                for (std::size_t p = 0; p < arrivals.size(); ++p) {
                    if (!nearlyEqual(trip.arrivalTime[p], arrivals[p])) {
                        rep.error(tag + ": arrivalTime[khach " + std::to_string(trip.customers[p]) + "] luu="
                                  + std::to_string(trip.arrivalTime[p]) + " thuc=" + std::to_string(arrivals[p]));
                    }
                }
            }

            totalDistance += dist;
            clock = ret;
        }

        if (!nearlyEqual(v.completionTime, clock)) {
            rep.error("Vehicle " + std::to_string(v.id) + ": completionTime luu=" + std::to_string(v.completionTime)
                      + " thuc=" + std::to_string(clock));
        }
        makespan = std::max(makespan, clock);
    }

    // ---------- Mỗi khách đúng 1 lần ----------
    int unassigned = 0;
    for (int cid = 1; cid <= n; ++cid) {
        if (visitCount[cid] == 0) {
            ++unassigned;
            rep.error("Khach " + std::to_string(cid) + " chua duoc phuc vu");
        } else if (visitCount[cid] > 1) {
            rep.error("Khach " + std::to_string(cid) + " duoc phuc vu " + std::to_string(visitCount[cid]) + " lan");
        }
    }

    // ---------- Nhất quán cấp Solution ----------
    if (!nearlyEqual(s.makespan, makespan))
        rep.error("makespan luu=" + std::to_string(s.makespan) + " thuc=" + std::to_string(makespan));
    if (!nearlyEqual(s.totalDistance, totalDistance))
        rep.error("totalDistance luu=" + std::to_string(s.totalDistance) + " thuc=" + std::to_string(totalDistance));
    if (s.unassignedCount != unassigned)
        rep.error("unassignedCount luu=" + std::to_string(s.unassignedCount) + " thuc=" + std::to_string(unassigned));

    rep.makespan = makespan;
    rep.totalDistance = totalDistance;

    // Cờ khả thi của solver phải trùng kết luận của validator (cả hai chiều).
    if (s.isFeasible() && !rep.valid)
        rep.errors.push_back("Solver bao isFeasible()=YES nhung validator phat hien loi o tren");
    if (!s.isFeasible() && rep.valid)
        rep.warn("Validator khong thay loi nhung solver bao isFeasible()=NO (totalViolation="
                 + std::to_string(s.totalViolation) + ")");

    return rep;
}

inline void printValidationReport(const ValidationReport& rep, std::ostream& os = std::cout) {
    os << "=== VALIDATOR (doc lap) ===\n";
    os << "Ket qua: " << (rep.valid ? "HOP LE" : "KHONG HOP LE")
       << " | makespan=" << rep.makespan << " | distance=" << rep.totalDistance << "\n";
    for (const auto& e : rep.errors) os << "  [LOI]    " << e << "\n";
    for (const auto& w : rep.warnings) os << "  [CANH BAO] " << w << "\n";
}
