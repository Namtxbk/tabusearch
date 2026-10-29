@echo off
REM So sanh 3 che do sinh lan can tren TAT CA file trong data\
REM   all    : ca 6 toan tu moi vong lap (goc)
REM   random : chon ngau nhien 1 toan tu / vong, lay mau ngau nhien SAMPLE move
REM   seq    : xoay vong tuan tu 6 toan tu, lay mau ngau nhien SAMPLE move
REM Ca 3 che do dung chung ngan sach TIME (giay/instance) de so sanh cong bang.
REM Ket qua: report\report_<mode>_<ngay-gio>.txt cho tung che do.

cd /d "%~dp0"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"
set TIME_LIMIT=30
set ITER=1000000000
set SAMPLE=100
set SEED=42

echo [build] Dang build solver.exe ...
g++ -std=c++17 -O2 -static main.cpp -o solver.exe
if errorlevel 1 (
    echo Build THAT BAI.
    pause
    exit /b 1
)

for %%M in (all random seq) do (
    echo.
    echo ===== Che do %%M =====
    solver.exe --mode=%%M --sample=%SAMPLE% --seed=%SEED% --time=%TIME_LIMIT% --iter=%ITER%
)
echo.
echo Xong. Bao cao nam trong thu muc report\
pause
