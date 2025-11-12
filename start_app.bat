@echo off
chcp 65001 >nul
echo ========================================
echo RAG SaaS 軽量版POC - アプリケーション起動
echo ========================================
echo.

REM カレントディレクトリをスクリプトの場所に変更
cd /d "%~dp0"

REM Pythonのバージョンチェック
python --version >nul 2>&1
if errorlevel 1 (
    echo [エラー] Pythonがインストールされていません。
    echo Python 3.8以上をインストールしてください。
    pause
    exit /b 1
)

REM .envファイルの確認
if not exist .env (
    echo [警告] .envファイルが見つかりません。
    echo.
    echo .envファイルを作成してください:
    echo 1. env.exampleをコピーして.envを作成
    echo 2. .envファイルを編集してOPENAI_API_KEYを設定
    echo.
    if exist env.example (
        echo env.exampleをコピーして.envを作成しますか？ (Y/N)
        set /p answer=
        if /i "%answer%"=="Y" (
            copy env.example .env >nul
            echo .envファイルを作成しました。
            echo .envファイルを編集してOPENAI_API_KEYを設定してください。
            pause
            exit /b 0
        )
    )
    echo.
    echo アプリケーションを起動しますが、OPENAI_API_KEYが設定されていない場合、
    echo 一部の機能が動作しない可能性があります。
    echo.
    timeout /t 3 >nul
)

REM Streamlitのインストール確認
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [エラー] Streamlitがインストールされていません。
    echo install_dependencies.batを実行してください。
    pause
    exit /b 1
)

echo [起動中] Streamlitアプリケーションを起動しています...
echo.
echo ブラウザが自動的に開きます。
echo 開かない場合は、以下のURLにアクセスしてください:
echo http://localhost:8501
echo.
echo アプリケーションを停止するには、Ctrl+Cを押してください。
echo.

REM Streamlitアプリケーションを起動
streamlit run app.py

if errorlevel 1 (
    echo.
    echo [エラー] アプリケーションの起動に失敗しました。
    echo エラーメッセージを確認してください。
    pause
    exit /b 1
)








