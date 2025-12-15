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
cl /nologo /EHsc /O2 /std:c++17 /I "%BOOST_PATH%" /I. prior_gyro_vert_sim.cpp /Fe:prior_sim_gv.exe
if errorlevel 1 exit /b
echo Running normal simulation...
prior_sim_gv.exe -o prior_sim_gv.csv
if errorlevel 1 exit /b
python prior_gyro_vert_sim_plot.py prior_sim_gv.csv plots\prior_sim_gv
if errorlevel 1 exit /b

echo Running simulation without a turn...
prior_sim_gv.exe --turn OFF --pitch OFF -o prior_sim_gv.csv
if errorlevel 1 exit /b
python prior_gyro_vert_sim_plot.py prior_sim_gv.csv plots\prior_sim_gv_no_turn
if errorlevel 1 exit /b

echo Running simulation without an inclination...
prior_sim_gv.exe --pitch OFF -o prior_sim_gv.csv
if errorlevel 1 exit /b
python  prior_gyro_vert_sim_plot.py prior_sim_gv.csv plots\prior_sim_gv_no_incl --highlight
if errorlevel 1 exit /b

echo Running simulation without turning error correction and withot inclination...
prior_sim_gv.exe --correction OFF --pitch OFF --turn OFF -o prior_sim_gv.csv
if errorlevel 1 exit /b
python prior_gyro_vert_sim_plot.py prior_sim_gv.csv plots\prior_sim_gv_no_corr --highlight
if errorlevel 1 exit /b

:: Cleanup
del prior_prior_sim_gv.exe
del prior_sim_gv.csv

endlocal
popd
