@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
call scripts\buscar_python.bat
if not defined PY (pause & exit /b 1)
"%PY%" -m pytest -q || (pause & exit /b 1)
if not exist runs mkdir runs
if exist runs\listo.txt del runs\listo.txt
start "Manteniendo la PC despierta" /min cmd /c mantener_despierta.bat --hasta runs\listo.txt --horas 24
title 1/8 Bajando ERA5 de WeatherBench 2
"%PY%" scripts\download_era5.py --out data\era5_64x32 --start 1990 --end 2022 || (pause & exit /b 1)
title 2/8 Calculando estadisticas y climatologia
if not exist data\era5_64x32\stats.json "%PY%" scripts\compute_stats.py --data data\era5_64x32 --start 1990-01-01 --end 2017-12-31T18
title 3/8 Entrenando la red
"%PY%" scripts\train.py --config configs\principal.yaml || (pause & exit /b 1)
title 4/8 Evaluando 2020 a 10 dias con 50 escenarios
if not exist runs\evaluacion\listo_v2.txt ("%PY%" scripts\evaluate.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --start 2020-01-01 --end 2020-12-31T18 --steps 40 --members 50 --out runs\evaluacion && echo listo> runs\evaluacion\listo_v2.txt)
title 5/8 Comparando estrategias de ensamble
if not exist runs\comparacion\listo_v2.txt ("%PY%" scripts\compare_ensembles.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --start 2020-01-01 --end 2020-12-31T18 --every 12 --members 30 --out runs\comparacion && echo listo> runs\comparacion\listo_v2.txt)
title 6/8 Calibrando la perturbacion inicial con 2019
if not exist runs\calibracion\listo_v2.txt ("%PY%" scripts\calibrate_ensemble.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --start 2019-01-01 --end 2019-12-31T18 --every 12 --members 30 --out runs\calibracion && echo listo> runs\calibracion\listo_v2.txt)
title 7/8 Pronostico para Buenos Aires
if not exist runs\pronostico\listo_v2.txt ("%PY%" scripts\forecast.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --init 2020-12-28T00 --steps 40 --members 50 --out runs\pronostico && echo listo> runs\pronostico\listo_v2.txt)
title 8/8 Armando el visor
if not exist runs\visor\listo_v2.txt ("%PY%" scripts\viewer.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --init 2020-12-28T00 --members 20 --out runs\visor\visor.html && echo listo> runs\visor\listo_v2.txt)
echo listo> runs\listo.txt
title Todo terminado
pause
