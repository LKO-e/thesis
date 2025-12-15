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
cl /nologo /EHsc /O2 /std:c++17 /I "%BOOST_PATH%" gyro_pend_sim.cpp /Fe:sim_gp.exe
if errorlevel 1 exit /b
echo Running simulation...
sim_gp.exe -o sim_gp.csv
if errorlevel 1 exit /b
python gyro_pend_sim_plot.py sim_gp.csv plots
if errorlevel 1 exit /b

:: Cleanup
del sim_gp.exe
del sim_gp.csv
del gyro_pend_sim.obj

endlocal
popd
