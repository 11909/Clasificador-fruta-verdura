@echo off
setlocal
set EXTRA=
rem Complemento: verduras de colores y los 20 alimentos que no tienen ninguna foto. Baja con el filtro CLIP
rem y arma los zips para Drive. Se puede volver a correr: sigue donde se quedo.
cd /d "%~dp0"
if not exist venv\Scripts\python.exe (
  py -3.12 -m venv venv
  if errorlevel 1 (
    echo Falta Python 3.12. Instalalo con:  winget install Python.Python.3.12   y vuelve a correr esto.
    pause
    exit /b 1
  )
)
venv\Scripts\python -c "import torch, transformers, icrawler" 2>nul
if errorlevel 1 (
  venv\Scripts\python -m pip install --upgrade pip
  venv\Scripts\python -m pip install pillow numpy icrawler transformers
  venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
if exist salida_bajo\fotos set EXTRA=%EXTRA% --excluir salida_bajo\fotos
venv\Scripts\python web_scrapping.py --plan plan_complemento.csv --salida salida_complemento --lote k1 %EXTRA% %*
venv\Scripts\python web_scrapping.py --salida salida_complemento --lote k1 --empaquetar
echo.
echo Listo. Los zips quedan en salida_complemento\para_drive\^<alimento^>\. Sube cada zip dentro de la
echo carpeta del mismo alimento en Drive (si no existe, crea la carpeta con ese nombre).
pause
