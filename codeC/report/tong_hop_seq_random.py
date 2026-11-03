"""Tong hop 5 lan chay seq va 5 lan chay random tu cac file report/report_<mode>_sample<N>_seed<S>_*.txt.

Cach dung: python report/tong_hop_seq_random.py [sample] [seed1,seed2,...]
Ket qua: report/so_sanh_seq_random_bang1.csv, report/so_sanh_seq_random_bang2.csv, report/so_sanh_seq_random.md
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "1000"
SEEDS = (sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5").split(",")
MODES = ["seq", "random"]
ROW = re.compile(r"^\d+\.\d+\.\d+\s")


def parse(path):
    rows, order, total = {}, [], None
    in_table = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("Tong thoi gian chay:"):
                total = float(line.split(":")[1].strip().rstrip("s"))
            if line.startswith("BANG TONG HOP"):
                in_table = True
            elif line.startswith("CHI TIET"):
                break
            m = ROW.match(line) if in_table else None
            if m:
                # Lan chay khong co loi giai hop le: lay gia tri ham vi pham (Tong VP cua bestInfeasible)
                # Doc cot tu cuoi dong: ... Hop_le_tot_nhat Tot_nhat_tim_duoc Tong_VP Q D TW W Chua_PV
                tok = line.split()
                name, t, feas, viol = tok[0], tok[4], tok[-8], tok[-6]
                rows[name] = (float(viol) if feas == "-" else float(feas), feas != "-", float(t))
                order.append(name)
    return rows, order, total


def latest(mode, seed):
    files = sorted(glob.glob(os.path.join(HERE, f"report_{mode}_sample{SAMPLE}_seed{seed}_*.txt")))
    if not files:
        sys.exit(f"Khong tim thay report cho mode={mode} seed={seed}")
    return files[-1]


data = {m: [] for m in MODES}   # data[mode][run] = rows
totals = {m: [] for m in MODES}  # wall-clock tong cua tung lan chay
order = None
for m in MODES:
    for s in SEEDS:
        rows, o, total = parse(latest(m, s))
        data[m].append(rows)
        totals[m].append(total)
        order = order or o


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def fmt(x, d=4):
    return "-" if x is None else f"{x:.{d}f}"


# ---------------- Bang 1 ----------------
infeasible_notes = []
b1 = []
for name in order:
    rec = {"name": name}
    for m in MODES:
        vals = [r[name][0] for r in data[m]]
        ok = [r[name][1] for r in data[m]]
        feas = [v for v, f in zip(vals, ok) if f]
        if len(feas) < len(vals):
            infeasible_notes.append(f"{name} ({m}): {len(vals) - len(feas)}/{len(vals)} lan khong co loi giai hop le "
                                    f"-> dung gia tri ham vi pham (danh dau *)")
        rec[m] = vals
        rec[m + "_ok"] = ok
        # Nghiem hop le luon tot hon nghiem vi pham; neu ca 5 lan deu vi pham thi lay vi pham nho nhat
        rec[m + "_best"] = min(feas) if feas else min(vals)
        rec[m + "_avg"] = mean(vals)
        rec[m + "_time"] = mean([r[name][2] for r in data[m]])
    sb, rb, sa, ra = rec["seq_best"], rec["random_best"], rec["seq_avg"], rec["random_avg"]
    rec["gap_best"] = (sb - rb) / sb * 100 if sb and rb else None
    rec["gap_avg"] = (sa - ra) / sa * 100 if sa and ra else None
    b1.append(rec)

valid = [r for r in b1 if r["gap_best"] is not None and r["gap_avg"] is not None]

# ---------------- Bang 2 ----------------
def run_sums(m):
    # tong gia tri (makespan, hoac ham vi pham neu khong hop le) cua tung lan chay tren 80 instance
    names = [r["name"] for r in valid]
    return [sum(run[n][0] for n in names) for run in data[m]], names


seq_sums, common = run_sums("seq")
rnd_sums, _ = run_sums("random")
col1, col2 = min(seq_sums), mean(seq_sums)
col3, col4 = min(rnd_sums), mean(rnd_sums)
col5 = sum(r["gap_best"] for r in valid)
col6 = sum(r["gap_avg"] for r in valid)
time_seq = [sum(r[n][2] for n in order) for r in data["seq"]]
time_rnd = [sum(r[n][2] for n in order) for r in data["random"]]

# ---------------- Ghi file ----------------
hdr1 = (["Instance"] + [f"Seq_run{i + 1}" for i in range(len(SEEDS))] + ["Seq_best", "Seq_avg"]
        + [f"Rnd_run{i + 1}" for i in range(len(SEEDS))] + ["Rnd_best", "Rnd_avg", "Gap_best(%)", "Gap_avg(%)"])
with open(os.path.join(HERE, "so_sanh_seq_random_bang1.csv"), "w", encoding="utf-8-sig") as f:
    f.write(",".join(hdr1) + "\n")
    for r in b1:
        cells = ([r["name"]] + [fmt(v) + ("" if f else "*") for v, f in zip(r["seq"], r["seq_ok"])] + [fmt(r["seq_best"]), fmt(r["seq_avg"])]
                 + [fmt(v) + ("" if f else "*") for v, f in zip(r["random"], r["random_ok"])] + [fmt(r["random_best"]), fmt(r["random_avg"]),
                                                     fmt(r["gap_best"]), fmt(r["gap_avg"])])
        f.write(",".join(cells) + "\n")

hdr2 = ["Seq_tong_lan_tot_nhat", "Seq_tong_trung_binh", "Rnd_tong_lan_tot_nhat", "Rnd_tong_trung_binh",
        "Tong_gap_best(%)", "Tong_gap_avg(%)", "Thoi_gian_seq_TB(s)", "Thoi_gian_rnd_TB(s)"]
row2 = [fmt(col1), fmt(col2), fmt(col3), fmt(col4), fmt(col5), fmt(col6),
        fmt(mean(time_seq), 2), fmt(mean(time_rnd), 2)]
with open(os.path.join(HERE, "so_sanh_seq_random_bang2.csv"), "w", encoding="utf-8-sig") as f:
    f.write(",".join(hdr2) + "\n" + ",".join(row2) + "\n")

with open(os.path.join(HERE, "so_sanh_seq_random.md"), "w", encoding="utf-8") as f:
    f.write(f"# So sanh seq vs random (sample={SAMPLE}, seeds={','.join(SEEDS)})\n\n")
    f.write("## Bang 1. Makespan hop le tot nhat cua tung lan chay\n\n")
    f.write("| " + " | ".join(hdr1) + " |\n|" + "---|" * len(hdr1) + "\n")
    for r in b1:
        cells = ([r["name"]] + [fmt(v) + ("" if f else "*") for v, f in zip(r["seq"], r["seq_ok"])] + [fmt(r["seq_best"]), fmt(r["seq_avg"])]
                 + [fmt(v) + ("" if f else "*") for v, f in zip(r["random"], r["random_ok"])] + [fmt(r["random_best"]), fmt(r["random_avg"]),
                                                     fmt(r["gap_best"]), fmt(r["gap_avg"])])
        f.write("| " + " | ".join(cells) + " |\n")
    f.write("\n## Bang 2. Tong hop\n\n")
    f.write("| " + " | ".join(hdr2) + " |\n|" + "---|" * len(hdr2) + "\n")
    f.write("| " + " | ".join(row2) + " |\n\n")
    f.write(f"- So instance tinh tong: {len(common)}/{len(order)}; o '*' = lan chay khong co loi giai hop le, "
            f"gia tri la ham vi pham (Tong VP) thay cho makespan\n")
    f.write(f"- So instance tinh gap: {len(valid)}/{len(order)}\n")
    f.write(f"- Tong makespan tung lan seq: {', '.join(fmt(x, 2) for x in seq_sums)}\n")
    f.write(f"- Tong makespan tung lan random: {', '.join(fmt(x, 2) for x in rnd_sums)}\n")
    f.write(f"- Tong Time(s) 80 instance tung lan seq: {', '.join(fmt(x, 2) for x in time_seq)}\n")
    f.write(f"- Tong Time(s) 80 instance tung lan random: {', '.join(fmt(x, 2) for x in time_rnd)}\n")
    f.write(f"- Gap trung binh / instance: best={fmt(col5 / len(valid))}%, avg={fmt(col6 / len(valid))}%\n")
    for n in infeasible_notes:
        f.write(f"- {n}\n")

print(open(os.path.join(HERE, "so_sanh_seq_random.md"), encoding="utf-8").read())
