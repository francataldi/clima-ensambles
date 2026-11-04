@echo off
python -m pytest -q || exit /b 1
python scripts\download_era5.py --out data\era5_64x32 --start 1990 --end 2022 || exit /b 1
python scripts\compute_stats.py --data data\era5_64x32 --start 1990-01-01 --end 2017-12-31T18 || exit /b 1
python scripts\train.py --config configs\rtx3060.yaml || exit /b 1
python scripts\evaluate.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --start 2020-01-01 --end 2020-12-31T18 --members 50 --out runs\evaluacion
python scripts\compare_ensembles.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --start 2020-01-01 --end 2020-12-31T18 --every 12 --members 30 --out runs\comparacion
python scripts\forecast.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --init 2020-12-28T00 --members 50 --out runs\pronostico
python scripts\viewer.py --checkpoints runs\afno\best.pt --data data\era5_64x32 --init 2020-12-28T00 --members 20 --out runs\visor\visor.html
