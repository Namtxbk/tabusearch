import os
import time
import pandas as pd
import math
from random import shuffle, randint


FOLDER_TSP = "tsp" 

def manual_read_tsp(path):

    coords = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        
    start_reading = False
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            break
        # Bắt đầu đọc khi gặp từ khóa tọa độ
        if "NODE_COORD_SECTION" in line:
            start_reading = True
            continue
        # Kết thúc đọc nếu gặp các Section khác (ví dụ: TOUR_SECTION)
        if start_reading and (line[0].isalpha()):
            break
        
        if start_reading:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[node_id] = (x, y)
                except ValueError:
                    continue
    return coords

def get_opt_value(file_name):
    """
    Giá trị tối ưu thực tế của một số instance phổ biến (Benchmark).
    Bạn có thể thêm các giá trị khác vào đây.
    """
    benchmarks = {
        "lin318.tsp": 42029,
        "pcb442.tsp": 50778,
        "pa561.tsp": 2763,
        "p654.tsp": 34643,
        "pcb1173.tsp": 56892
    }
    return benchmarks.get(file_name, None)

# =================================================================
# 2. HÀM FITNESS VÀ TABU SEARCH
# =================================================================
def calculate_euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def fitness_tsp(route, coords):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += calculate_euclidean(coords[route[i]], coords[route[i+1]])
    # Quay về điểm xuất phát
    total_distance += calculate_euclidean(coords[route[-1]], coords[route[0]])
    return total_distance

def get_neighbors(state, size):
    neighbors = []
    n = len(state)
    for _ in range(size):
        i, j = randint(0, n - 1), randint(0, n - 1)
        if i > j: i, j = j, i
        if i == j: continue
        # Phép biến đổi 2-opt láng giềng
        neighbor = state[:i] + state[i:j+1][::-1] + state[j+1:]
        neighbors.append(neighbor)
    return neighbors

def tabu_search_tsp(coords):
    nodes = list(coords.keys())
    s_best = nodes[:]
    shuffle(s_best)
    v_best = fitness_tsp(s_best, coords)
    
    best_candidate = s_best[:]
    tabu_list = [s_best[:]]
    
    best_keep_turn = 0
    # Tham số cấu hình (Tabu Tenure, Neighborhood Size, Stopping Condition)
    max_tabu = 50
    nh_size = 40
    stop_limit = 100
    
    while best_keep_turn < stop_limit:
        neighbors = get_neighbors(best_candidate, nh_size)
        if not neighbors: break
        
        # Chọn láng giềng tốt nhất
        best_candidate = neighbors[0]
        v_candidate = fitness_tsp(best_candidate, coords)
        
        for s_candidate in neighbors:
            v_s = fitness_tsp(s_candidate, coords)
            # Aspiration Condition: Phá lệ nếu tốt hơn kỷ lục hiện tại
            if (s_candidate not in tabu_list or v_s < v_best) and v_s < v_candidate:
                best_candidate = s_candidate
                v_candidate = v_s

        if v_candidate < v_best:
            s_best, v_best = best_candidate[:], v_candidate
            best_keep_turn = 0
        else:
            best_keep_turn += 1

        tabu_list.append(best_candidate[:])
        if len(tabu_list) > max_tabu:
            tabu_list.pop(0)
            
    return v_best

# =================================================================
# 3. THỰC THI THỬ NGHIỆM
# =================================================================
def run_tsp_experiment():
    if not os.path.exists(FOLDER_TSP):
        print(f"Lỗi: Không tìm thấy thư mục '{FOLDER_TSP}'")
        return

    results = []
    files = sorted([f for f in os.listdir(FOLDER_TSP) if f.endswith('.tsp')])

    for file_name in files:
        path = os.path.join(FOLDER_TSP, file_name)
        
        try:
            # Đọc tọa độ thủ công để tránh lỗi thư viện
            coords = manual_read_tsp(path)
            if not coords:
                print(f"Bỏ qua {file_name}: Không tìm thấy tọa độ hoặc định dạng ma trận (Explicit).")
                continue
                
            opt_val = get_opt_value(file_name)
            
            start_t = time.time()
            best_found = tabu_search_tsp(coords)
            exec_time = time.time() - start_t
            
            gap = f"{((best_found - opt_val) / opt_val * 100):.2f}%" if opt_val else "N/A"
            
            results.append({
                "File": file_name,
                "Optimal (Target)": opt_val if opt_val else "N/A",
                "Tabu Result": round(best_found, 2),
                "Gap (%)": gap,
                "Time (s)": round(exec_time, 2)
            })
            print(f"Xong: {file_name}")
        except Exception as e:
            print(f"Lỗi khi xử lý {file_name}: {e}")

    # Xuất kết quả
    if results:
        print("\n" + "="*70)
        print("KẾT QUẢ THỰC NGHIỆM TABU SEARCH TSP")
        print("="*70)
        print(pd.DataFrame(results).to_string(index=False))
    else:
        print("Không có kết quả nào được tạo ra.")

if __name__ == "__main__":
    run_tsp_experiment()