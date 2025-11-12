# RAG SaaS 軽量版（POC用）

別PCでのPOC（Proof of Concept）用に作成された軽量版です。機能は完全版と同じですが、作業記録（WORKLOG、AGENTSなど）を除外しています。

## 特徴

- **完全独立**: `poc_lightweight_v2` ディレクトリだけで動作します（親ディレクトリへの依存なし）
- **全機能対応**: すべてのファイル形式に対応（PDF、DOCX、Excel、画像、OCRなど）
- **POCモード**: 認証なしで即座に使用可能
- **別PC移動対応**: このディレクトリをそのまま別PCにコピーするだけで使用可能

## 必要な環境

### 必須

- **Python 3.8以上**
- **OpenAI APIキー**
- **インターネット接続**（依存関係のインストール時、およびアプリケーション実行時）

### OCR機能を使用する場合（画像・PDF・Excel処理）

OCR機能を使用する場合は、以下のソフトウェアを**別途インストール**する必要があります：

#### 1. Tesseract OCR（必須）

画像やPDF内の文字を認識するために必要です。

**Windows:**
1. [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) からインストーラーをダウンロード
2. インストール実行（推奨: `tesseract-ocr-w64-setup-5.3.3.20231005.exe`）
3. インストール先（通常: `C:\Program Files\Tesseract-OCR`）を環境変数 `PATH` に追加
4. **日本語言語パック**をインストール（インストーラーのオプションで選択可能）

**確認方法:**
```cmd
tesseract --version
```

**macOS:**
```bash
brew install tesseract
brew install tesseract-lang  # 日本語言語パック
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-jpn  # 日本語言語パック
```

#### 2. EasyOCR（自動インストール）

- Pythonパッケージとして `pip install` で自動インストールされます
- **初回実行時にモデルファイル（約200MB）をダウンロードします**
- インターネット接続が必要です
- 日本語と英語がデフォルトで有効です

#### 3. OpenCV（自動インストール）

- Pythonパッケージとして `pip install` で自動インストールされます
- 画像処理ライブラリです

### アーカイブ処理を使用する場合（ZIP/RAR/7Z）

アーカイブ処理機能を使用する場合、以下の依存関係が**オプション**です：

#### RAR/7Zサポート（オプション）

- **ZIP/TAR**: 標準ライブラリで対応（追加インストール不要）
- **RAR**: `rarfile` パッケージとシステムの `unrar` または `unar` が必要
- **7Z**: `py7zr` パッケージが必要

**RAR/7Zサポートを有効にする場合:**

```bash
# RARサポート（Windowsの場合、unrarも必要）
pip install rarfile

# 7Zサポート
pip install py7zr
```

**注意**: RAR/7Zサポートがインストールされていない場合、ZIP/TARファイルは処理できますが、RAR/7Zファイルはエラーになります。

## セットアップ手順

### 方法1: BATファイルを使用（推奨）

#### 1. 依存関係のインストール

```cmd
install_dependencies.bat
```

このBATファイルは以下を自動実行します：
- Pythonバージョン確認
- pipのアップグレード
- 依存関係のインストール（requirements.txtから）
- インストール完了確認

#### 2. 環境変数の設定

```cmd
# .envファイルを作成（env.exampleをコピー）
copy env.example .env

# .envファイルを編集してOPENAI_API_KEYを設定
notepad .env
```

`.env`ファイルの内容例：
```
OPENAI_API_KEY=sk-your-api-key-here
POC_MODE=true
```

#### 3. Tesseract OCRのインストール（OCR機能を使用する場合）

上記の「OCR機能を使用する場合」セクションを参照してください。

#### 4. アプリケーションの起動

```cmd
start_app.bat
```

ブラウザが自動的に開き、アプリケーションが起動します。

### 方法2: 手動セットアップ

#### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

**注意:** 初回インストールには時間がかかります（特にEasyOCR）。

#### 2. 環境変数の設定

```bash
# .envファイルを作成
cp env.example .env

# .envファイルを編集してOPENAI_API_KEYを設定
# Windows: notepad .env
# macOS/Linux: nano .env
```

#### 3. Tesseract OCRのインストール（OCR機能を使用する場合）

上記の「OCR機能を使用する場合」セクションを参照してください。

#### 4. アプリケーションの起動

```bash
streamlit run app.py
```

ブラウザが自動的に開き、アプリケーションが起動します。

## 機能

### 実装済み機能

- ✅ ナレッジ検索（ベクトル検索 + BM25ハイブリッド検索）
- ✅ ナレッジ登録（すべてのファイル形式に対応）
  - PDF（文字・画像・OCR対応）
  - DOCX（画像混じり文書対応）
  - Excel（複数シート・セル結合対応）
  - 画像（Vision + GPT-4.1 mini 二段階推論）
  - CSV/TSV
  - JSON/YAML
  - XML
  - コードファイル（Python/Java/VBA）
  - アーカイブ（ZIP/TAR/RAR/7Z）
    - ZIP/TAR: 標準ライブラリで対応（追加インストール不要）
    - RAR/7Z: オプション依存関係が必要（上記「アーカイブ処理を使用する場合」を参照）
- ✅ 基本的なUI（検索、ナレッジ管理、分析、設定タブ）
- ✅ OCR機能（Tesseract、EasyOCR）
- ✅ 画像処理（OpenCV、Pillow）
- ✅ 文書処理（PDF、DOCX、Excel）

## トラブルシューティング

### OpenAI APIキーが設定されていない

`.env`ファイルに`OPENAI_API_KEY`を設定してください。

### 依存関係のインストールエラー

Python 3.8以上を使用していることを確認してください。

### PyTorchのDLLエラー（[WinError 1114] ダイナミック リンク ライブラリ (DLL) 初期化ルーチンの実行に失敗しました）

**症状:**
```
Error loading "C:\Users\...\torch\lib\c10.dll" or one of its dependencies.
```

**原因:**
- Visual C++ ランタイムライブラリの不足
- PyTorchのバージョンとシステムの互換性の問題
- easyocrがtorchに依存しているが、正しくインストールされていない

**解決方法:**

1. **専用の修正スクリプトを実行（推奨）:**
   ```cmd
   fix_pytorch_dll_error.bat
   ```
   このスクリプトは以下を自動実行します：
   - 既存のPyTorch関連パッケージのアンインストール
   - CPU版PyTorchの再インストール
   - easyocrの再インストール

2. **Visual C++ ランタイムライブラリをインストール:**
   - [Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) をダウンロードしてインストール
   - インストール後、システムを再起動

3. **依存関係を再インストール:**
   ```cmd
   reinstall_dependencies.bat
   ```

4. **手動でCPU版PyTorchをインストール:**
   ```cmd
   pip uninstall -y torch torchvision torchaudio easyocr
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install easyocr
   ```

### Streamlitの警告（use_container_width will be removed）

**症状:**
```
Please replace `use_container_width` with `width`.
```

**解決方法:**
- この警告は既に修正済みです（`width='stretch'`に置き換え済み）
- 警告が表示される場合は、最新のコードを使用していることを確認してください

### Tesseract OCRが見つからない

Tesseract OCRをインストールし、環境変数`PATH`に追加してください。詳細は上記の「OCR機能を使用する場合」セクションを参照してください。

### EasyOCRのモデルダウンロードが失敗する

インターネット接続を確認してください。初回実行時にモデルファイル（約200MB）をダウンロードします。

### アプリケーションが起動しない

ログを確認し、エラーメッセージを参照してください。ログは`logs/app.log`に出力されます。

### ポート8501が既に使用されている

別のStreamlitアプリケーションが実行中の可能性があります。以下のコマンドでポートを変更できます：

```bash
streamlit run app.py --server.port 8502
```

## 別PCへの移動方法

1. **`poc_lightweight_v2` ディレクトリ全体を別PCにコピー**
   - ZIPファイルに圧縮して移動することも可能です

2. **別PCで依存関係のインストール:**
   ```cmd
   install_dependencies.bat
   ```

3. **環境変数の設定:**
   ```cmd
   copy env.example .env
   # .envファイルを編集してOPENAI_API_KEYを設定
   ```

4. **Tesseract OCRのインストール（OCR機能を使用する場合）**

5. **アプリケーションの起動:**
   ```cmd
   start_app.bat
   ```

## 注意事項

- この軽量版はPOC用途を想定していますが、すべての機能が含まれています
- 本番環境でも使用可能です
- 元のデータ（docs_store、metadata.db等）は変更されません
- 作業記録（WORKLOG、AGENTSなど）は含まれていません

## ライセンス

本プロジェクトのライセンスに準拠します。
