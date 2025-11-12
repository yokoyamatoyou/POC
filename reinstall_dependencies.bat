@echo off
chcp 65001 >nul
echo ========================================
echo 依存関係の再インストールスクリプト
echo ========================================
echo.

REM 現在のディレクトリを取得
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

echo [1/3] Python環境の確認...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [エラー] Pythonがインストールされていません。
    pause
    exit /b 1
)
python --version
echo.

echo [2/3] 既存パッケージのアンインストール（オプション）...
set /p UNINSTALL="既存のパッケージをアンインストールしますか？ (y/N): "
if /i "%UNINSTALL%"=="y" (
    echo アンインストール中...
    pip freeze > temp_requirements.txt
    pip uninstall -y -r temp_requirements.txt >nul 2>&1
    del temp_requirements.txt
    echo 完了
    echo.
)

echo [3/3] requirements.txtから依存関係をインストール...
echo この処理には数分かかる場合があります...
echo.

REM requirements.txtが存在するか確認
if not exist "requirements.txt" (
    echo [エラー] requirements.txtが見つかりません。
    pause
    exit /b 1
)

REM 依存関係をインストール
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo.
    echo [エラー] 依存関係のインストールに失敗しました。
    echo.
    echo PyTorchのDLLエラーが発生している場合は、fix_pytorch_dll_error.batを実行してください。
    pause
    exit /b 1
)

echo.
echo ========================================
echo インストール完了
echo ========================================
echo.
echo インストールされたパッケージを確認します...
pip list | findstr /i "torch streamlit openai easyocr"
echo.
echo アプリケーションを起動する準備ができました。
pause



