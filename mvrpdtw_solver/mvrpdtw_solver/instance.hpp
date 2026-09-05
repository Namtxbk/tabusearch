// instance.hpp
// Đọc dữ liệu bài toán MVRPD-TW (Multi-Vehicle Routing Problem with Drones and Time Windows)
// Format JSON tương thích với instance.py do người dùng cung cấp.
#pragma once

#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include "json.hpp"

using json = nlohmann::json;

// Service time tại điểm khách coi như = 0 (bốc/dỡ tức thì).
static constexpr double L_W_SERVICE = 0.0;

struct Customer {
    int id = 0;            // 0 = depot, 1..n = khách hàng
    double x = 0.0, y = 0.0;
    double demand = 0.0;    // q_i
    double ready = 0.0;     // e_i
    double due = 0.0;       // l_i
    double service = 0.0;   // thời gian phục vụ tại khách (coi = 0)
    bool is_c1 = false;     // true => C1 (chỉ truck), false => C2 (truck hoặc drone)
};

struct Instance {
    std::string name;

    int num_trucks = 1;
    int num_drones = 1;

    double truck_capacity = 400.0;  // M_T
    double drone_capacity = 2.27;   // M_D
    double drone_range = 700.0;     // L_D (quãng đường tối đa 1 chuyến drone)
    double max_wait = 60.0;         // L_w (thời gian chờ tối đa của hàng tại kho sau khi lấy)

    double truck_speed = 1.0;
    double drone_speed = 1.5;

    Customer depot;
    std::vector<Customer> customers;   // index 0..n-1 <-> id 1..n
    std::set<int> c1_ids;              // id (1-based) thuộc C1
    std::set<int> c2_ids;              // id (1-based) thuộc C2

    std::vector<std::vector<double>> distMat; // [0..n][0..n], 0 = depot

    int numCustomers() const { return static_cast<int>(customers.size()); }

    // node index: 0 = depot, i = customers[i-1] với id = i
    const Customer& node(int nodeIdx) const {
        if (nodeIdx == 0) return depot;
        return customers[nodeIdx - 1];
    }

    double dist(int i, int j) const { return distMat[i][j]; }

    double travelTime(int i, int j, bool isDrone) const {
        double speed = isDrone ? drone_speed : truck_speed;
        if (speed <= 0.0) speed = 1.0;
        return distMat[i][j] / speed;
    }

    void buildDistanceMatrix() {
        int n = numCustomers() + 1; // + depot
        distMat.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            const Customer& a = node(i);
            for (int j = 0; j < n; ++j) {
                const Customer& b = node(j);
                double dx = a.x - b.x;
                double dy = a.y - b.y;
                distMat[i][j] = std::sqrt(dx * dx + dy * dy);
            }
        }
    }
};

inline Instance readJsonInstance(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f.is_open()) {
        throw std::runtime_error("Khong mo duoc file: " + filepath);
    }
    json data;
    f >> data;

    if (!data.contains("requests") || data["requests"].empty()) {
        throw std::runtime_error("File " + filepath + " khong chua du lieu requests.");
    }

    Instance inst;
    // instance name = tên file không có đường dẫn / phần mở rộng
    {
        std::string base = filepath;
        auto slashPos = base.find_last_of("/\\");
        if (slashPos != std::string::npos) base = base.substr(slashPos + 1);
        auto dotPos = base.find_last_of('.');
        if (dotPos != std::string::npos) base = base.substr(0, dotPos);
        inst.name = base;
    }

    inst.num_trucks     = data.value("truck_num", 1);
    inst.num_drones     = data.value("drone_num", 1);
    inst.truck_capacity = data.value("truck_cap", 400.0);
    inst.drone_capacity = data.value("drone_cap", 2.27);
    inst.drone_range    = data.value("drone_lim", 700.0);
    inst.truck_speed    = data.value("truck_vel", 1.0);
    inst.drone_speed    = data.value("drone_vel", 1.5);
    double depotClose   = data.value("close", 9999.0);
    // max_wait (L_w): không có trong JSON mẫu -> giữ giá trị mặc định của Instance
    // (60.0, khớp với L_W_MAX trong instance.py); có thể override nếu JSON có trường "max_wait".
    inst.max_wait = data.value("max_wait", inst.max_wait);

    inst.depot = Customer{};
    inst.depot.id = 0;
    inst.depot.x = 0.0;
    inst.depot.y = 0.0;
    inst.depot.demand = 0.0;
    inst.depot.ready = 0.0;
    inst.depot.due = depotClose;
    inst.depot.service = 0.0;
    inst.depot.is_c1 = false;

    const auto& requests = data["requests"];
    int idx = 1;
    for (const auto& r : requests) {
        // [x, y, demand, ableServiceByDrone, r_i(bỏ qua), opentime, closetime]
        Customer c;
        c.id = idx;
        c.x = r.at(0).get<double>();
        c.y = r.at(1).get<double>();
        c.demand = r.at(2).get<double>();
        int ableDrone = r.at(3).get<int>();
        c.is_c1 = (ableDrone == 0);   // 0 => chỉ truck (C1); 1 => cho phép drone (C2)
        // r.at(4) = r_i, bỏ qua
        c.ready = r.at(5).get<double>();
        c.due = r.at(6).get<double>();
        c.service = L_W_SERVICE;

        inst.customers.push_back(c);
        if (c.is_c1) inst.c1_ids.insert(idx);
        else inst.c2_ids.insert(idx);
        ++idx;
    }

    inst.buildDistanceMatrix();
    return inst;
}
