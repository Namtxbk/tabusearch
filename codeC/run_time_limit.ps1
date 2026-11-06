# So sanh seq va random voi ngan sach thoi gian CO DINH, ti le voi kich thuoc bai: T_lim = C * n^2 (giay).
# Tat H_stop (--hstop=0) de hai che do dung dung cung mot ngan sach thoi gian cho moi instance.
# Moi tien trinh duoc gan vao 1 nhan P rieng (logical 0,2,...,10 cua i5-13500) de thoi gian do on dinh,
# tranh truong hop tien trinh roi vao nhan E (cham hon) lam lech ket qua giua cac lan chay.
#
# Cach dung (PowerShell, trong thu muc codeC):
#   .\run_time_limit.ps1                       # C = 0.002 va 0.001, 5 seed
#   .\run_time_limit.ps1 -Coefs 0.002 -Seeds 1,2,3
# Tong hop: python report\tong_hop_theo_nhom.py <tag>   (vd tag = _tc0.002_hs0)

param(
    [int]$Sample = 1000,
    [int[]]$Seeds = @(1, 2, 3, 4, 5),
    [string[]]$Coefs = @("0.002", "0.001"),
    [int[]]$Cores = @(0, 2, 4, 6, 8, 10)
)

Set-Location $PSScriptRoot
$env:PATH = "C:\msys64\ucrt64\bin;$env:PATH"

Write-Host "[1/2] Build solver.exe ..."
g++ -std=c++17 -O2 -static main.cpp -o solver.exe
if ($LASTEXITCODE -ne 0) { Write-Host "Build THAT BAI."; exit 1 }
New-Item -ItemType Directory -Force logs | Out-Null

foreach ($c in $Coefs) {
    Write-Host "[2/2] C = $c ..."
    $jobs = @()
    foreach ($s in $Seeds) { foreach ($m in "seq", "random") { $jobs += , @($m, $s) } }
    $running = @{}   # core -> process
    foreach ($j in $jobs) {
        # cho den khi co nhan trong
        while ($true) {
            $free = $Cores | Where-Object { -not $running.ContainsKey($_) -or $running[$_].HasExited }
            if ($free) { $core = @($free)[0]; break }
            Start-Sleep -Milliseconds 500
        }
        $m, $s = $j
        $solverArgs = "--mode=$m --sample=$Sample --seed=$s --time-coef=$c --hstop=0 --iter=1000000000"
        $log = "logs\${m}_tc${c}_seed$s.log"
        $p = Start-Process -FilePath .\solver.exe -ArgumentList $solverArgs -NoNewWindow -PassThru -RedirectStandardError $log
        $p.ProcessorAffinity = [IntPtr](1 -shl $core)
        $running[$core] = $p
        Write-Host "  - $m seed $s -> core $core"
    }
    $running.Values | Wait-Process
}
Write-Host "Xong."
