@echo off
REM Full NCF-to-MetaNCF ablation on Amazon Fashion (4 runs). Uses ..\code as the data folder on this PC.
cd /d "%~dp0"
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" "%USERPROFILE%\anaconda3"
call conda activate tf2
python run_ablation.py --data_dir ..\code
echo Finished. You can close this window.
pause
