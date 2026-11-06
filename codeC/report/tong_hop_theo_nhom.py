"""Tong hop seq vs random THEO NHOM KICH THUOC n (6, 10, 12, 20, 50, 100) - ban chi tiet cua Bang 2.

Cach dung:
  python report/tong_hop_theo_nhom.py <tag> [date_prefix] [sample] [seeds]
    tag         : phan them vao ten report, vd "_tc0.002_hs0" (gioi han thoi gian C*n^2), "" = chay day du
    date_prefix : chi lay report co ngay gio bat dau bang chuoi nay, vd "2026-11-03_10" (mac dinh: report moi nhat)
Ket qua: report/theo_nhom<tag>.md va report/theo_nhom<tag>.csv

Quy uoc:
  - Mot instance chi duoc dua vao tong makespan / gap neu CA 10 lan chay (2 che do x 5 seed) deu co loi giai hop le,
    de hai che do duoc so tren cung tap instance. So instance bi loai va so lan chay khong hop le duoc bao cao rieng.
  - (1)/(3): tong makespan cua lan chay co tong nho nhat trong nhom; (2)/(4): trung binh tong qua 5 lan.
  - gap_best = (b' - b)/b' * 100 voi b', b la makespan tot nhat 5 lan cua seq, random; gap_avg tuong tu voi trung binh.
  - Time: tong thoi gian chay cua nhom trong 1 lan chay, trung binh qua 5 lan.
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else ""
DATE = sys.argv[2] if len(sys.argv) > 2 else ""
SAMPLE = sys.argv[3] if len(sys.argv) > 3 else "1000"
SEEDS = (sys.argv[4] if len(sys.argv) > 4 else "1,2,3,4,5").split(",")
MODES = ["seq", "random"]
ROW = re.compile(r"^\d+\.\d+\.\d+\s")


def parse(path):
    rows = {}
    in_table = False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("BANG TONG HOP"):
                in_table = True
            elif line.startswith("CHI TIET"):
                break
            if in_table and ROW.match(line):
                tok = line.split()
                feas = tok[-8]
                rows[tok[0]] = {"n": int(tok[1]), "time": float(tok[4]), "iter": int(tok[5]),
                                "ok": feas != "-", "z": float(feas) if feas != "-" else None}
    return rows


def latest(mode, seed):
    files = sorted(glob.glob(os.path.join(HERE, f"report_{mode}_sample{SAMPLE}{TAG}_seed{seed}_{DATE}*.txt")))
    if not files:
        sys.exit(f"Khong tim thay report cho mode={mode} seed={seed} tag={TAG!r} date={DATE!r}")
    return files[-1]


data = {m: [parse(latest(m, s)) for s in SEEDS] for m in MODES}
names = list(data["seq"][0].keys())
groups = sorted({data["seq"][0][x]["n"] for x in names})


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(insts):
    """Tra ve dict cac cot cho mot tap instance (mot nhom hoac toan bo)."""
    R = len(SEEDS)
    valid = [x for x in insts if all(data[m][r][x]["ok"] for m in MODES for r in range(R))]
    out = {"inst": len(insts), "valid": len(valid)}
    for m in MODES:
        sums = [sum(data[m][r][x]["z"] for x in valid) for r in range(R)]
        out[m + "_runbest"] = min(sums) if valid else None
        out[m + "_runavg"] = mean(sums) if valid else None
        out[m + "_infeas"] = sum(not data[m][r][x]["ok"] for x in insts for r in range(R))
        out[m + "_time"] = mean([sum(data[m][r][x]["time"] for x in insts) for r in range(R)])
        out[m + "_iter"] = mean([data[m][r][x]["iter"] for x in insts for r in range(R)])
    gb = ga = 0.0
    win_b = [0, 0, 0]
    win_a = [0, 0, 0]
    for x in valid:
        sb = min(data["seq"][r][x]["z"] for r in range(R))
        rb = min(data["random"][r][x]["z"] for r in range(R))
        sa = mean([data["seq"][r][x]["z"] for r in range(R)])
        ra = mean([data["random"][r][x]["z"] for r in range(R)])
        gb += (sb - rb) / sb * 100
        ga += (sa - ra) / sa * 100
        for (s, r_, w) in ((sb, rb, win_b), (sa, ra, win_a)):
            d = (s - r_) / s * 100
            w[0 if d > 1e-4 else (2 if d < -1e-4 else 1)] += 1
    out["gap_best"], out["gap_avg"] = gb, ga
    out["wtl_best"], out["wtl_avg"] = win_b, win_a
    return out


rows = [(str(n), summarize([x for x in names if data["seq"][0][x]["n"] == n])) for n in groups]
rows.append(("Tong", summarize(names)))


def f(x, d=2):
    return "-" if x is None else f"{x:.{d}f}"


hdr = ["n", "So_inst", "Inst_tinh_tong", "Seq_lan_tot_nhat", "Seq_trung_binh", "Rnd_lan_tot_nhat", "Rnd_trung_binh",
       "Tong_gap_best(%)", "Tong_gap_avg(%)", "Seq_time(s)", "Rnd_time(s)", "Seq_iter_TB", "Rnd_iter_TB",
       "Seq_lan_khong_hop_le", "Rnd_lan_khong_hop_le", "Rnd_T/H/Th_best", "Rnd_T/H/Th_avg"]
lines = []
for n, o in rows:
    lines.append([n, str(o["inst"]), str(o["valid"]), f(o["seq_runbest"]), f(o["seq_runavg"]),
                  f(o["random_runbest"]), f(o["random_runavg"]), f(o["gap_best"]), f(o["gap_avg"]),
                  f(o["seq_time"]), f(o["random_time"]), f(o["seq_iter"], 0), f(o["random_iter"], 0),
                  str(o["seq_infeas"]), str(o["random_infeas"]),
                  "/".join(map(str, o["wtl_best"])), "/".join(map(str, o["wtl_avg"]))])

base = os.path.join(HERE, f"theo_nhom{TAG or '_day_du'}")
with open(base + ".csv", "w", encoding="utf-8-sig") as fp:
    fp.write(",".join(hdr) + "\n")
    for l in lines:
        fp.write(",".join(l) + "\n")
with open(base + ".md", "w", encoding="utf-8") as fp:
    fp.write(f"# Seq vs random theo nhom n (tag={TAG!r}, date={DATE!r}, sample={SAMPLE}, seeds={','.join(SEEDS)})\n\n")
    fp.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for l in lines:
        fp.write("| " + " | ".join(l) + " |\n")
    fp.write("\nInstance bi loai khoi tong (co lan chay khong hop le):\n")
    R = len(SEEDS)
    for x in names:
        bad = {m: sum(not data[m][r][x]["ok"] for r in range(R)) for m in MODES}
        if bad["seq"] or bad["random"]:
            fp.write(f"- {x}: seq {bad['seq']}/{R}, random {bad['random']}/{R} lan khong hop le\n")
print(open(base + ".md", encoding="utf-8").read())
