@echo off
chcp 65001 >nul
cd /d "E:\GF\eco\基于公告信息的上市公司股价趋势预测挑战赛"

echo ========================================
echo  Start Full Pipeline
echo  Log: run_all.log
echo ========================================

echo [%time%] Step 1/4: Extracting PDF text... > run_all.log
python 02_pdf_extraction.py --all >> run_all.log 2>&1
if %errorlevel% neq 0 (
    echo Step 1 FAILED >> run_all.log
    echo Step 1 FAILED - check run_all.log
    pause
    exit /b 1
)

echo [%time%] Step 2/4: Sliding FinBERT... >> run_all.log
python 06_sliding_finbert.py --input train_text_full.csv >> run_all.log 2>&1
if %errorlevel% neq 0 (
    echo Step 2 FAILED >> run_all.log
    echo Step 2 FAILED - check run_all.log
    pause
    exit /b 1
)

echo [%time%] Step 3/4: Building features... >> run_all.log
python 06b_build_features.py --input train_text_full.csv >> run_all.log 2>&1
if %errorlevel% neq 0 (
    echo Step 3 FAILED >> run_all.log
    echo Step 3 FAILED - check run_all.log
    pause
    exit /b 1
)

echo [%time%] Step 4/4: Training... >> run_all.log
python 07_hybrid_train.py >> run_all.log 2>&1
if %errorlevel% neq 0 (
    echo Step 4 FAILED >> run_all.log
    echo Step 4 FAILED - check run_all.log
    pause
    exit /b 1
)

echo All Done! >> run_all.log
echo ========================================
echo  All 4 steps completed!
echo  Check run_all.log for results.
echo ========================================
pause
