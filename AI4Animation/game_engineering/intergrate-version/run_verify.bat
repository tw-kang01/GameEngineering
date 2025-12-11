@echo off
call %USERPROFILE%\anaconda3\Scripts\activate.bat
call conda activate GameEngineering
cd /d c:\Users\KTW\GameEngineering\AI4Animation\game_engineering\categorical\PyTorch\Models\CodebookMatching
python verify_gmd_integration.py
pause
