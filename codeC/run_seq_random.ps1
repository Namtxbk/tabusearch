# So sanh 2 che do chon toan tu: tuan tu (seq) va ngau nhien (random), moi che do chay 5 seed
# tren TAT CA file trong data\, roi tong hop thanh 2 bang + sinh lai Phu luc A cua luan van.
#
# Cach dung (PowerShell, trong thu muc codeC):
#   .\run_seq_random.ps1                 # chay song song 10 tien trinh (nhanh, ~15-20 phut)
#   .\run_seq_random.ps1 -Sequential     # chay lan luot tung lan (cham hon, thoi gian do chinh xac hon)
#   .\run_seq_random.ps1 -Sample 500 -Seeds 1,2,3
# Neu bi chan chay script: Set-ExecutionPolicy -Scope Process Bypass

param(
    [int]$Sample = 1000,
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    [int]$TimeLimit = 20,
    [switch]$Sequential
)

Set-Location $PSScriptRoot
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"

Write-Host "[1/3] Build solver.exe ..."
g++ -std=c++17 -O2 -static main.cpp -o solver.exe
if ($LASTEXITCODE -ne 0) { Write-Host "Build THAT BAI."; exit 1 }

Write-Host "[2/3] Chay seq va random, sample=$Sample, seeds=$($Seeds -join ',') ..."
New-Item -ItemType Directory -Force logs | Out-Null
$procs = @()
foreach ($m in "seq", "random") {
    foreach ($s in $Seeds) {
        $solverArgs = "--mode=$m --sample=$Sample --seed=$s --time=$TimeLimit --iter=1000000000"
        $log = "logs\${m}_seed$s.log"
        if ($Sequential) {
            Write-Host "  - $m seed $s"
            Start-Process -FilePath .\solver.exe -ArgumentList $solverArgs -NoNewWindow -Wait -RedirectStandardOutput $log
        } else {
            $procs += Start-Process -FilePath .\solver.exe -ArgumentList $solverArgs -NoNewWindow -PassThru -RedirectStandardOutput $log
        }
    }
}
if (-not $Sequential) {
    Write-Host "  Dang chay $($procs.Count) tien trinh song song, xem tien do trong thu muc logs\ ..."
    $procs | Wait-Process
}

Write-Host "[3/3] Tong hop ket qua ..."
python report\tong_hop_seq_random.py $Sample ($Seeds -join ',')
if ($Seeds.Count -eq 5) { python report\sinh_phu_luc_A.py } else { Write-Host "  (Bo qua sinh Phu luc A vi bang phu luc can dung 5 seed)" }
Write-Host "Xong. Ket qua: report\so_sanh_seq_random.md, report\so_sanh_seq_random_bang1.csv, report\so_sanh_seq_random_bang2.csv"
