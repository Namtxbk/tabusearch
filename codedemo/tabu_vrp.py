import os
import time
import vrplib
import pandas as pd
import math
from random import shuffle, randint

# =================================================================
# 1. CẤU HÌNH THAM SỐ THUẬT TOÁN (Hyperparameters)
# =================================================================
MAX_TABU_SIZE = 50       # Tabu Tenure: Số lượt một lời giải bị cấm trong danh sách Tabu
NEIGHBORHOOD_SIZE = 40   # Neighborhood Size: Số lượng láng giềng tạo ra ở mỗi vòng lặp
STOPPING_TURN = 100      # Stopping Condition: Dừng nếu sau 100 lượt không tìm thấy kết quả tốt hơn
FOLDER_A = "A"          

def calculate_euclidean(p1, p2):
    """Tính khoảng cách đường chim bay giữa 2 điểm (Toạ độ X, Y)"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_best_solution(sol_path):
    """Đọc kết quả tối ưu từ file .sol để làm mốc so sánh (Benchmark)"""
    if os.path.exists(sol_path):
        try:
            sol = vrplib.read_solution(sol_path)
            return sol['cost']
        except:
            return None
    return None

# =================================================================
# 2. HÀM FITNESS (Hàm thích nghi)
# =================================================================
def fitness_vrp(route, instance):
    """
    Tính tổng quãng đường của toàn bộ đội xe.
    Cơ chế: Duyệt qua danh sách khách hàng, nếu xe đầy (vượt Capacity) 
    thì xe phải quay về Depot (nút 0) rồi mới đi tiếp khách hàng đó.
    """
    coords = instance['node_coord']
    demands = instance['demand']
    capacity = instance['capacity']
    
    total_distance = 0
    current_load = 0
    current_node = 0  # Bắt đầu tại Depot
    depot_coord = coords[0]
    
    for customer in route:
        demand = demands[customer]
        customer_coord = coords[customer]
        
        # Kiểm tra ràng buộc tải trọng (Capacity Constraint)
        if current_load + demand > capacity:
            # Nếu quá tải: Xe hiện tại quay về Depot
            total_distance += calculate_euclidean(coords[current_node], depot_coord)
            current_node = 0
            current_load = 0
            
        # Đi từ vị trí hiện tại đến khách hàng tiếp theo
        total_distance += calculate_euclidean(coords[current_node], customer_coord)
        current_load += demand
        current_node = customer
        
    # Sau khi phục vụ khách cuối cùng, xe phải quay về Depot
    total_distance += calculate_euclidean(coords[current_node], depot_coord)
    return total_distance

# =================================================================
# 3. NEIGHBORHOOD (Không gian láng giềng)
# =================================================================
def get_neighbors(state):
    """
    Tạo ra tập hợp các lời giải láng giềng bằng phép biến đổi 2-opt Swap.
    Chọn ngẫu nhiên 2 vị trí và đảo ngược đoạn khách hàng ở giữa.
    """
    neighbors = []
    n = len(state)
    if n < 2: return [state]
    
    for _ in range(NEIGHBORHOOD_SIZE):
        i, j = randint(0, n - 1), randint(0, n - 1)
        if i > j: i, j = j, i
        # Phép biến đổi 2-opt
        neighbor = state[:i] + state[i:j+1][::-1] + state[j+1:]
        neighbors.append(neighbor)
    return neighbors


def tabu_search_vrp(instance):
    """Giải bài toán VRP cho 1 instance cụ thể bằng Tabu Search"""
    num_nodes = len(instance['node_coord'])
    # Search Space: Hoán vị của tất cả các mã khách hàng (từ 1 đến n)
    customers = list(range(1, num_nodes))
    
    # Khởi tạo lời giải ban đầu ngẫu nhiên
    s_best = customers[:]
    shuffle(s_best) 
    v_best = fitness_vrp(s_best, instance)
    
    best_candidate = s_best[:]
    tabu_list = [s_best[:]] # Cấu trúc Tabu: Lưu các lời giải đã đi qua
    
    best_keep_turn = 0 
    
    while best_keep_turn < STOPPING_TURN:
        neighbors = get_neighbors(best_candidate)
        
        # Giả định láng giềng đầu tiên là tốt nhất để bắt đầu so sánh
        best_candidate_in_neighbors = neighbors[0]
        v_candidate = fitness_vrp(best_candidate_in_neighbors, instance)
        
        for s_candidate in neighbors:
            v_s = fitness_vrp(s_candidate, instance)
            is_tabu = s_candidate in tabu_list
            
            # --- ASPIRATION CONDITION (Điều kiện khát vọng) ---
            # Chấp nhận lời giải kể cả khi đang bị cấm (Tabu) nếu nó tốt hơn kỷ lục hiện tại (v_best)
            if (not is_tabu or v_s < v_best) and (v_s < v_candidate):
                best_candidate_in_neighbors = s_candidate
                v_candidate = v_s

        best_candidate = best_candidate_in_neighbors

        # Cập nhật kỷ lục mới (Best So Far)
        if v_candidate < v_best:
            s_best = best_candidate[:]
            v_best = v_candidate
            best_keep_turn = 0 # Reset bộ đếm nếu tìm thấy kết quả tốt hơn
        else:
            best_keep_turn += 1 # Tăng bộ đếm nếu không có cải thiện

        # Cập nhật danh sách Tabu (Tabu Tenure)
        tabu_list.append(best_candidate[:])
        if len(tabu_list) > MAX_TABU_SIZE:
            tabu_list.pop(0) # Xóa lời giải cũ nhất khi danh sách đầy
            
    return v_best

# =================================================================
# 5. ĐIỀU KHIỂN THỰC NGHIỆM (Experiment Runner)
# =================================================================
def run_experiment():
    """Quét thư mục, chạy thuật toán và tổng hợp kết quả vào bảng"""
    if not os.path.exists(FOLDER_A):
        print(f"Lỗi: Không tìm thấy thư mục '{FOLDER_A}'")
        return

    results = []
    # Lọc lấy tất cả các file có đuôi .vrp
    files = sorted([f for f in os.listdir(FOLDER_A) if f.endswith('.vrp')])
    
    print(f"Đang xử lý {len(files)} file trong thư mục '{FOLDER_A}'...")

    for file_name in files:
        vrp_path = os.path.join(FOLDER_A, file_name)
        sol_path = vrp_path.replace('.vrp', '.sol')
        
        try:
            # Đọc dữ liệu file VRP
            instance = vrplib.read_instance(vrp_path)
            optimal_value = get_best_solution(sol_path)
            
            # Thực thi và đo thời gian
            start_t = time.time()
            best_found = tabu_search_vrp(instance)
            exec_time = time.time() - start_t
            
            # Tính toán độ lệch % so với kết quả tối ưu của tác giả (Gap)
            gap_str = "N/A"
            if optimal_value and optimal_value > 0:
                gap = ((best_found - optimal_value) / optimal_value) * 100
                gap_str = f"{gap:.2f}%"
            
            # Lưu kết quả của file hiện tại
            results.append({
                "Ten Instance": file_name,
                "Ket qua toi uu  (Optimal)": optimal_value if optimal_value else "N/A",
                "Ket qua Tabu": round(best_found, 2),
                "Do lech (%)": gap_str,
                "Thoi gian chay (s)": round(exec_time, 4)
            })
            print(f"Đã xử lý xong: {file_name}")
            
        except Exception as e:
            print(f"Lỗi khi xử lý {file_name}: {e}")
    
    # Xuất dữ liệu ra bảng DataFrame và file CSV
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("BANG KET QUA TABU SEARCH CHO TSP VRP")
    print("="*60)
    print(df.to_string(index=False))
    
    df.to_csv("Ket_qua_TabuSearch_VRP.csv", index=False)
    print("\n[THÔNG BÁO] Đã xuất bảng kết quả ra file: Ket_qua_TabuSearch_VRP.csv")

if __name__ == "__main__":
    run_experiment()