@echo off
setlocal enabledelayedexpansion

echo ========================================
echo  C++ Armor Detector Build Script
echo ========================================
echo.

cd /d "%~dp0%"

if exist build_vs rd /s /q build_vs
mkdir build_vs
cd build_vs

echo [1/3] Configuring with CMake...
echo.

cmake .. -G "Visual Studio 17 2022" -A x64
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    echo Please make sure CMake is in your PATH and OpenCV is installed.
    pause
    exit /b 1
)

echo.
echo [2/3] Building Release version...
echo.

cmake --build . --config Release
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo  BUILD SUCCESSFUL!
echo ========================================
echo.
echo Executable: %cd%\Release\armor_detector.exe
echo.
echo Usage:
echo   armor_detector.exe --camera 0       (Camera test)
echo   armor_detector.exe --video xxx.mp4  (Video test)
echo   armor_detector.exe --image xxx.png  (Image test)
echo.
pause
