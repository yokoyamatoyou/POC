@echo off
chcp 65001 >nul
echo ========================================
echo PyTorch DLLエラー修正スクリプト
echo ========================================
echo.

REM 管理者権限の確認
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [警告] このスクリプトは管理者権限で実行することを推奨します。
    echo Visual C++ ランタイムライブラリのインストールには管理者権限が必要な場合があります。
    echo.
    pause
)

echo [1/4] Python環境の確認...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [エラー] Pythonがインストールされていません。
    echo Python 3.11以上をインストールしてください。
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] 既存のPyTorch関連パッケージをアンインストール...
pip uninstall -y torch torchvision torchaudio easyocr >nul 2>&1
echo 完了
echo.

echo [3/4] Visual C++ ランタイムライブラリの確認...
echo Visual C++ 2015-2022 Redistributable がインストールされているか確認します。
echo 未インストールの場合は、以下のURLからダウンロードしてインストールしてください:
echo https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
pause

echo [4/4] CPU版PyTorchと依存関係を再インストール...
echo この処理には数分かかる場合があります...
echo.

REM CPU版PyTorchを明示的にインストール（インデックスURLを指定）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if %errorLevel% neq 0 (
    echo [エラー] PyTorchのインストールに失敗しました。
    echo インターネット接続を確認してください。
    pause
    exit /b 1
)

REM easyocrを再インストール
pip install easyocr>=1.7.0,<2.0.0
if %errorLevel% neq 0 (
    echo [警告] easyocrのインストールに失敗しました。
    echo PyTorchはインストールされましたが、easyocrの再インストールが必要です。
)

echo.
echo ========================================
echo インストール完了
echo ========================================
echo.
echo PyTorchのバージョンを確認します...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
if %errorLevel% neq 0 (
    echo [エラー] PyTorchのインポートに失敗しました。
    echo Visual C++ ランタイムライブラリが不足している可能性があります。
    echo.
    echo 対処方法:
    echo 1. Visual C++ 2015-2022 Redistributable (x64) をインストール
    echo 2. システムを再起動
    echo 3. このスクリプトを再実行
    pause
    exit /b 1
)

echo.
echo [成功] PyTorchが正常にインストールされました。
echo.
echo アプリケーションを再起動してください。
pause



