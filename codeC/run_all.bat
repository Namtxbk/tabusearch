@echo off
REM Build lai solver.exe roi chay TAT CA file test trong data\
REM Ket qua: record\<ten>_<ngay-gio>.txt (tung file) va report\report_<ngay-gio>.txt (tong hop)
REM Cach dung: nhap dup file nay, hoac go run_all.bat trong terminal.

cd /d "%~dp0"
set "PATH=C:\msys64\ucrt64\bin;%PATH%"

echo [1/2] Dang build solver.exe ...
g++ -std=c++17 -O2 -static main.cpp -o solver.exe
if errorlevel 1 (
    echo Build THAT BAI.
    pause
    exit /b 1
)

echo [2/2] Dang chay tat ca file trong data\ ...
solver.exe
echo.
echo Xong. Bao cao tong hop nam trong thu muc report\
pause
