@echo off
cd /d "%~dp0"
title Manteniendo la PC despierta
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
"%PY%" scripts\mantener_despierta.py %*
pause
