@echo off
pushd "%~dp0"
setlocal enabledelayedexpansion

:: Try to auto-detect Boost
set BOOST_PATH=
for /d %%D in (C:\local\boost_*) do (
    set BOOST_PATH=%%D
)

if "%BOOST_PATH%"=="" (
    echo Could not auto-detect Boost installation.
    set /p BOOST_PATH=Enter the path to Boost library: 
) else (
    echo Found Boost at %BOOST_PATH%
)

if not exist plots mkdir plots

echo Compiling...
cl /nologo /EHsc /O2 /std:c++17 /I "%BOOST_PATH%" /I. gyro_vert_sim.cpp /Fe:sim_gv.exe
if errorlevel 1 exit /b
echo Running normal simulation...
sim_gv.exe --vel_meas_error OFF -o sim_gv.csv
if errorlevel 1 exit /b
python gyro_vert_sim_plot.py sim_gv.csv plots\sim_gv
if errorlevel 1 exit /b

echo Running simulation without Earth's rate correction and velocity measurement error...
sim_gv.exe --earth_rate_correction OFF --vel_meas_error ON -o sim_gv.csv
if errorlevel 1 exit /b
python gyro_vert_sim_plot.py sim_gv.csv plots\sim_gv_earth_and_v
if errorlevel 1 exit /b

echo Running simulation without gyro pendulum dynamic tuning...
sim_gv.exe --dynamic_tuning OFF -o sim_gv.csv
if errorlevel 1 exit /b
python gyro_vert_sim_plot.py sim_gv.csv plots\sim_gv_dyn_tun
if errorlevel 1 exit /b

echo Running simulation without pitch angle correction...
sim_gv.exe --pitch_correction OFF -o sim_gv.csv
if errorlevel 1 exit /b
python gyro_vert_sim_plot.py sim_gv.csv plots\sim_gv_pitch
if errorlevel 1 exit /b

echo Running simulation with velocity measurement error...
sim_gv.exe --vel_meas_error ON -o sim_gv.csv
if errorlevel 1 exit /b
python gyro_vert_sim_plot.py sim_gv.csv plots\sim_gv_v
if errorlevel 1 exit /b

:: Cleanup
del sim_gv.exe
del sim_gv.csv

endlocal
popd
