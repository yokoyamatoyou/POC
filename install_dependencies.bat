@echo off
chcp 65001 >nul
echo ========================================
echo RAG SaaS 軽量版POC - 依存関係インストール
echo ========================================
echo.

REM Pythonのバージョンチェック
python --version >nul 2>&1
if errorlevel 1 (
    echo [エラー] Pythonがインストールされていません。
    echo Python 3.8以上をインストールしてください。
    pause
    exit /b 1
)

echo [1/4] Pythonバージョン確認中...
python --version
echo.

echo [2/4] pipのアップグレード中...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [警告] pipのアップグレードに失敗しましたが、続行します。
)
echo.

echo [3/4] 依存関係のインストール中...
echo この処理には数分かかる場合があります...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [エラー] 依存関係のインストールに失敗しました。
    pause
    exit /b 1
)
echo.

echo [4/4] インストール完了確認中...
python -c "import streamlit; import openai; print('✅ 主要ライブラリのインポート成功')" 2>nul
if errorlevel 1 (
    echo [警告] 一部のライブラリのインポートに失敗しました。
    echo 詳細はエラーメッセージを確認してください。
)
echo.

echo ========================================
echo インストール完了
echo ========================================
echo.
echo 次のステップ:
echo 1. Tesseract OCRをインストールしてください（README.mdを参照）
echo 2. .envファイルを作成してOPENAI_API_KEYを設定してください
echo 3. start_app.batを実行してアプリケーションを起動してください
echo.
pause








