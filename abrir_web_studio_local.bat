@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND=%ROOT%web-studio-v2\backend"
set "FRONTEND=%ROOT%web-studio-v2\frontend"
set "LANDING_URL=http://127.0.0.1:5180/"
set "TOOL_URL=http://127.0.0.1:5180/tool"
set "API_URL=http://127.0.0.1:8010/health"

if not exist "%BACKEND%\main.py" (
  echo No se encuentra el backend en:
  echo %BACKEND%
  pause
  exit /b 1
)

if not exist "%FRONTEND%\package.json" (
  echo No se encuentra el frontend en:
  echo %FRONTEND%
  pause
  exit /b 1
)

echo Iniciando backend local...
start "Decarb Backend" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%BACKEND%'; python -m uvicorn main:app --host 127.0.0.1 --port 8010"

echo Iniciando frontend local...
start "Decarb Frontend" powershell -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%FRONTEND%'; if (-not (Test-Path -LiteralPath 'node_modules')) { npm.cmd install }; npm.cmd run dev"

echo Esperando a que la aplicacion este lista...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$api='%API_URL%'; $app='%LANDING_URL%';" ^
  "$deadline=(Get-Date).AddSeconds(45);" ^
  "do { try { Invoke-WebRequest -UseBasicParsing $api -TimeoutSec 2 | Out-Null; $apiOk=$true } catch { $apiOk=$false }; Start-Sleep -Milliseconds 700 } until ($apiOk -or (Get-Date) -gt $deadline);" ^
  "$deadline=(Get-Date).AddSeconds(45);" ^
  "do { try { Invoke-WebRequest -UseBasicParsing $app -TimeoutSec 2 | Out-Null; $appOk=$true } catch { $appOk=$false }; Start-Sleep -Milliseconds 700 } until ($appOk -or (Get-Date) -gt $deadline);" ^
  "Start-Process $app"

echo.
echo Landing abierta en %LANDING_URL%
echo Herramienta principal en %TOOL_URL%
echo Puedes cerrar esta ventana. Para parar la app, cierra las ventanas "Decarb Backend" y "Decarb Frontend".
pause
