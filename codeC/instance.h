#pragma once
// instance.h — Instance và Customer cho MVRPD-TW
// Tương đương instance.py

#include <string>
#include <vector>
#include <unordered_set>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include "include/json.hpp"

using json = nlohmann::json;

static const double L_W = 0.0;  // Service time cố định (phút)

// ─────────────────────────────────────────────────────────────────────────────
// Customer
// ─────────────────────────────────────────────────────────────────────────────
struct Customer {
    int    id      = 0;
    double x       = 0.0;
    double y       = 0.0;
    double demand  = 0.0;
    double ready   = 0.0;  // opentime
    double due     = 0.0;  // closetime
    double service = 0.0;  // L_W
    bool   is_c1   = false; // true => chỉ truck, false => drone được phép
};

// ─────────────────────────────────────────────────────────────────────────────
// Instance
// ─────────────────────────────────────────────────────────────────────────────
struct Instance {
    std::string name;
    int    num_trucks     = 1;
    int    num_drones     = 1;
    double truck_capacity = 400.0;
    double drone_capacity = 2.27;
    double drone_range    = 700.0;
    double truck_speed    = 1.0;
    double drone_speed    = 1.5;

    Customer               depot;      // id=0
    std::vector<Customer>  customers;  // id 1..n, index = id-1
    std::unordered_set<int> c1_ids;
    std::unordered_set<int> c2_ids;

    // Ma trận khoảng cách (depot=0, khách 1..n)
    std::vector<std::vector<double>> dist_mat;

    // Tính ma trận khoảng cách Euclid
    void build_dist() {
        int n = (int)customers.size() + 1;  // 0=depot, 1..n=khách
        dist_mat.assign(n, std::vector<double>(n, 0.0));
        // all_nodes: [depot, customers[0], ..., customers[n-2]]
        // all_nodes[0] = depot, all_nodes[k] = customers[k-1] với k>=1
        auto& d = dist_mat;
        auto node = [&](int id) -> const Customer& {
            return id == 0 ? depot : customers[id - 1];
        };
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                double dx = node(i).x - node(j).x;
                double dy = node(i).y - node(j).y;
                d[i][j] = std::hypot(dx, dy);
            }
    }

    inline double dist(int i, int j) const {
        return dist_mat[i][j];
    }

    inline double travel_time(int i, int j, bool is_drone) const {
        double speed = is_drone ? drone_speed : truck_speed;
        return dist_mat[i][j] / speed;
    }

    // Truy cập customer theo id (0 = depot)
    inline const Customer& node(int id) const {
        return id == 0 ? depot : customers[id - 1];
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Đọc file JSON instance
// Tương đương read_json_instance() trong instance.py
// ─────────────────────────────────────────────────────────────────────────────
inline Instance read_json_instance(const std::string& filepath) {
    std::ifstream f(filepath);
    if (!f) throw std::runtime_error("Không mở được file: " + filepath);
    json data = json::parse(f);

    // Tên instance = basename không có đuôi .json
    std::string name = filepath;
    auto slash = name.find_last_of("/\\");
    if (slash != std::string::npos) name = name.substr(slash + 1);
    if (name.size() > 5 && name.substr(name.size()-5) == ".json")
        name = name.substr(0, name.size()-5);

    auto& requests = data["requests"];
    if (requests.empty())
        throw std::runtime_error("File không chứa requests: " + filepath);

    Instance inst;
    inst.name = name;
    inst.num_trucks     = data.value("truck_num", 1);
    inst.num_drones     = data.value("drone_num", 1);
    inst.truck_capacity = data.value("truck_cap", 400.0);
    inst.drone_capacity = data.value("drone_cap", 2.27);
    inst.drone_range    = data.value("drone_lim", 700.0);
    inst.truck_speed    = data.value("truck_vel", 1.0);
    inst.drone_speed    = data.value("drone_vel", 1.5);
    double depot_close  = data.value("close", 9999.0);

    // Depot: tọa độ (0,0) cố định
    inst.depot = {0, 0.0, 0.0, 0.0, 0.0, depot_close, 0.0, false};

    for (int idx = 1; idx <= (int)requests.size(); ++idx) {
        auto& r = requests[idx - 1];
        // [x, y, demand, ableServiceByDrone, r_i(bỏ qua), opentime, closetime]
        int able_drone = r[3].get<int>();
        bool is_c1 = (able_drone == 0);

        Customer c;
        c.id      = idx;
        c.x       = r[0].get<double>();
        c.y       = r[1].get<double>();
        c.demand  = r[2].get<double>();
        c.ready   = r[5].get<double>();
        c.due     = r[6].get<double>();
        c.service = L_W;
        c.is_c1   = is_c1;

        inst.customers.push_back(c);
        if (is_c1) inst.c1_ids.insert(idx);
        else        inst.c2_ids.insert(idx);
    }

    inst.build_dist();
    return inst;
}
