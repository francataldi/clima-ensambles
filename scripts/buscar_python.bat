@echo off
if defined PY goto :eof
if exist "%USERPROFILE%\.conda\environments.txt" for /f "usebackq delims=" %%E in ("%USERPROFILE%\.conda\environments.txt") do call :probar "%%E\python.exe"
for %%R in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do call :raiz "%%~R"
for /d %%P in ("%LOCALAPPDATA%\Programs\Python\Python3*") do call :probar "%%~P\python.exe"
for /d %%P in ("C:\Python3*") do call :probar "%%~P\python.exe"
for /f "delims=" %%P in ('where python 2^>nul') do call :probar "%%P"
if not defined PY goto :nada
for %%I in ("%PY%") do set "D=%%~dpI"
set "PATH=%D%;%D%Library\mingw-w64\bin;%D%Library\usr\bin;%D%Library\bin;%D%Scripts;%D%bin;%PATH%"
echo Python: %PY%
goto :eof

:nada
echo No encontre un Python con torch con CUDA, pytest, pyyaml, xarray y zarr.
goto :eof

:raiz
call :probar "%~1\python.exe"
for /d %%E in ("%~1\envs\*") do call :probar "%%~E\python.exe"
goto :eof

:probar
if defined PY goto :eof
if not exist "%~1" goto :eof
setlocal
set "D=%~dp1"
set "PATH=%D%;%D%Library\mingw-w64\bin;%D%Library\usr\bin;%D%Library\bin;%D%Scripts;%D%bin;%PATH%"
"%~1" -c "import torch, pytest, yaml, xarray, zarr; assert torch.cuda.is_available()" >nul 2>nul
if errorlevel 1 (
    endlocal
    goto :eof
)
endlocal & set "PY=%~1"
goto :eof
