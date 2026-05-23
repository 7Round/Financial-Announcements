@echo off
chcp 65001 >nul
cd /d "E:\GF\eco\基于公告信息的上市公司股价趋势预测挑战赛"

echo ============ Pipeline Status ============

if not exist run_all.log (
    echo Status: NOT STARTED
    pause
    exit /b
)

echo.
echo --- Last 10 lines of log ---
powershell -Command "Get-Content run_all.log -Tail 10"
echo -----------------------------

echo.

echo Steps:
findstr /c:"Step 2/4" run_all.log >nul 2>&1
if %errorlevel% equ 0 (echo   [OK] 1/4 PDF extraction) else (echo   [..] 1/4 PDF extraction)

findstr /c:"Step 3/4" run_all.log >nul 2>&1
if %errorlevel% equ 0 (echo   [OK] 2/4 Sliding FinBERT) else (echo   [..] 2/4 Sliding FinBERT)

findstr /c:"Step 4/4" run_all.log >nul 2>&1
if %errorlevel% equ 0 (echo   [OK] 3/4 TF-IDF+Features) else (echo   [..] 3/4 TF-IDF+Features)

findstr /c:"All Done" run_all.log >nul 2>&1
if %errorlevel% equ 0 (echo   [OK] 4/4 Training) else (
    findstr /c:"Step 4/4" run_all.log >nul 2>&1
    if %errorlevel% equ 0 (echo   [..] 4/4 Training - RUNNING) else (echo   [..] 4/4 Training)
)

echo.
echo Files:
if exist train_text_full.csv (
    powershell -Command "$f=Get-Item 'train_text_full.csv'; Write-Host '  train_text_full.csv' $f.Length 'bytes'"
) else (echo   train_text_full.csv - not yet)
if exist sliding_finbert.npz echo   sliding_finbert.npz - done
if not exist sliding_finbert.npz echo   sliding_finbert.npz - not yet
if exist features_built.npz echo   features_built.npz - done
if not exist features_built.npz echo   features_built.npz - not yet

echo.
echo Check errors: findstr FAILED run_all.log
echo.
pause
