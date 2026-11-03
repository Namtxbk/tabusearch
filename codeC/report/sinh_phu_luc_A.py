"""Sinh latex/11_03_2026/Chuong/Phu_luc_A.tex tu report/so_sanh_seq_random_bang1.csv va result.csv.

Cach dung: python report/sinh_phu_luc_A.py  (chay lai sau khi chay tong_hop_seq_random.py)
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "latex", "11_03_2026", "Chuong", "Phu_luc_A.tex")

ref = {}
with open(os.path.join(ROOT, "result.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        c = float(r["Cost [minute]"]) * 60
        ref[r["Problem"]] = min(ref.get(r["Problem"], float("inf")), c)

with open(os.path.join(HERE, "so_sanh_seq_random_bang1.csv"), encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))


def num(x, d=2):
    s = f"{float(x):.{d}f}".replace(".", ",")
    return f"${s.replace(',', '{,}')}$" if s.startswith("-") else s


def runs_table(prefix, title, label):
    cols = [f"{prefix}_run{i}" for i in range(1, 6)] + [f"{prefix}_best", f"{prefix}_avg"]
    out = [r"\begin{small}", r"\begin{longtable}{l|rrrrr|rr}",
           rf"\caption{{{title}}}\label{{{label}}}\\",
           r"\toprule",
           r"\textbf{Instance} & Seed 1 & Seed 2 & Seed 3 & Seed 4 & Seed 5 & \textbf{Tốt nhất} & \textbf{TB} \\",
           r"\midrule", r"\endfirsthead",
           r"\toprule",
           r"\textbf{Instance} & Seed 1 & Seed 2 & Seed 3 & Seed 4 & Seed 5 & \textbf{Tốt nhất} & \textbf{TB} \\",
           r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for r in rows:
        out.append(r["Instance"] + " & " + " & ".join(num(r[c]) for c in cols) + r" \\")
    out += [r"\end{longtable}", r"\end{small}", ""]
    return "\n".join(out)


def gap_table():
    out = [r"\begin{small}", r"\begin{longtable}{l|r|rr|rr|rr}",
           r"\caption{So sánh từng instance với lời giải tham chiếu và giữa hai chiến lược}\label{tab:appA_gap}\\",
           r"\toprule",
           r" & & \multicolumn{2}{c|}{\textbf{Tuần tự}} & \multicolumn{2}{c|}{\textbf{Ngẫu nhiên}} & \multicolumn{2}{c}{\textbf{Gap (\%)}} \\",
           r"\textbf{Instance} & \textbf{Tham chiếu} & Tốt nhất & TB & Tốt nhất & TB & best & avg \\",
           r"\midrule", r"\endfirsthead",
           r"\toprule",
           r"\textbf{Instance} & \textbf{Tham chiếu} & Tốt nhất & TB & Tốt nhất & TB & best & avg \\",
           r"\midrule", r"\endhead", r"\bottomrule", r"\endfoot"]
    for r in rows:
        cells = [num(ref[r["Instance"]]), num(r["Seq_best"]), num(r["Seq_avg"]), num(r["Rnd_best"]),
                 num(r["Rnd_avg"]), num(r["Gap_best(%)"], 3), num(r["Gap_avg(%)"], 3)]
        out.append(r["Instance"] + " & " + " & ".join(cells) + r" \\")
    out += [r"\end{longtable}", r"\end{small}", ""]
    return "\n".join(out)


body = [r"\documentclass[../DoAn.tex]{subfiles}", r"\begin{document}", "",
        "% Tệp này được sinh tự động bởi codeC/report/sinh_phu_luc_A.py. Không sửa tay.", "",
        r"Phụ lục này trình bày kết quả chi tiết của thí nghiệm ở Mục~\ref{sec:exp_seqrand}. "
        r"Bảng~\ref{tab:appA_seq} và Bảng~\ref{tab:appA_rnd} liệt kê makespan của lời giải khả thi tốt nhất "
        r"trong từng lần chạy của chiến lược tuần tự và chiến lược ngẫu nhiên, cùng giá trị tốt nhất và trung bình "
        r"qua năm lần chạy. Bảng~\ref{tab:appA_gap} so sánh các giá trị này với lời giải tham chiếu và trình bày "
        r"gap giữa hai chiến lược theo công thức~\eqref{eq:gap}, trong đó gap dương nghĩa là chiến lược ngẫu nhiên tốt hơn. "
        r"Tất cả các lời giải trong phụ lục đều khả thi và đã được bộ kiểm tra độc lập xác nhận.", "",
        r"\section{Kết quả của chiến lược tuần tự}", "",
        runs_table("Seq", "Makespan của chiến lược tuần tự qua năm lần chạy", "tab:appA_seq"),
        r"\section{Kết quả của chiến lược ngẫu nhiên}", "",
        runs_table("Rnd", "Makespan của chiến lược ngẫu nhiên qua năm lần chạy", "tab:appA_rnd"),
        r"\section{So sánh với lời giải tham chiếu và giữa hai chiến lược}", "",
        gap_table(),
        r"\end{document}", ""]

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(body))
print("Da ghi", OUT)
