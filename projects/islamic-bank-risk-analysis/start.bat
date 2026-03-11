@echo off
chcp 65001 > nul
echo.
echo ========================================================
echo   🕌 Islomiy Bank Risk Tizimi - Ishga tushirilmoqda...
echo ========================================================
echo.
echo [1/2] Backend server ishga tushmoqda...
echo       http://localhost:8000 da ochiladi
echo.
echo [2/2] ML modellar yuklanmoqda (~30-60 soniya)...
echo       Brauzer avtomatik ochiladi
echo.
timeout /t 2 /nobreak > nul
start "" "http://localhost:8000"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
pause


