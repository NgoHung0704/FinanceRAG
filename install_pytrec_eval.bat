@echo off
echo Setting up Visual Studio Build Environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Installing pytrec_eval...
"D:\Anaconda\envs\financerag\python.exe" -m pip install pytrec_eval

echo.
echo Done!
pause
