"""
Phase 10-3: Error Handler
エラーハンドリング強化と型ヒント完全実装
"""
import asyncio
import functools
import logging
import sys
import traceback
import time
from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
from dataclasses import dataclass, field
from enum import Enum
import inspect
import threading
from contextlib import contextmanager


class ErrorSeverity(Enum):
    """エラー重要度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """エラーカテゴリ"""
    VALIDATION = "validation"
    NETWORK = "network"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """エラーコンテキスト"""
    error_type: Type[Exception]
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float
    context_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    function_name: str = ""
    file_name: str = ""
    line_number: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None

    @property
    def message(self) -> str:
        """互換性のため旧属性名 `message` を提供する"""
        return self.error_message


@dataclass
class ErrorMetrics:
    """エラーメトリクス"""
    total_errors: int = 0
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_type: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[ErrorContext] = field(default_factory=list)
    error_rate: float = 0.0
    last_error_time: Optional[float] = None


class ErrorHandler:
    """エラーハンドリング強化クラス"""
    
    def __init__(self, 
                 log_level: int = logging.INFO,
                 enable_recovery: bool = True,
                 max_retries: int = 3,
                 max_error_history: int = 1000):
        self.log_level = log_level
        self.enable_recovery = enable_recovery
        self.max_retries = max_retries
        self.max_error_history = max_error_history
        
        # ログ設定
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # エラー履歴
        self.error_history: List[ErrorContext] = []
        self.metrics = ErrorMetrics()
        
        # リトライ管理
        self.retry_counts: Dict[str, int] = {}
        
        # ロック
        self._lock = threading.Lock()
        
        # エラー回復戦略
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.DATABASE: self._recover_database_error,
            ErrorCategory.FILE_SYSTEM: self._recover_file_system_error,
            ErrorCategory.MEMORY: self._recover_memory_error,
            ErrorCategory.TIMEOUT: self._recover_timeout_error,
        }
    
    def handle_error(self,
                    exception: Exception,
                    context_data: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.UNKNOWN) -> ErrorContext:
        """エラーハンドリング"""
        if context_data is None:
            context_data = {}

        # エラーコンテキスト作成
        error_context = self._create_error_context(
            exception, context_data, severity, category
        )

        # エラー処理実行
        self._process_error(error_context)

        return error_context
    
    def _create_error_context(self,
                             exception: Exception,
                             context_data: Dict[str, Any],
                             severity: ErrorSeverity,
                             category: ErrorCategory) -> ErrorContext:
        """エラーコンテキスト作成"""
        # スタックトレース取得
        stack_trace = traceback.format_exc()
        
        # 呼び出し元情報取得
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        
        function_name = ""
        file_name = ""
        line_number = 0
        
        if caller_frame:
            function_name = caller_frame.f_code.co_name
            file_name = caller_frame.f_code.co_filename
            line_number = caller_frame.f_lineno
        
        return ErrorContext(
            error_type=type(exception),
            error_message=str(exception),
            severity=severity,
            category=category,
            timestamp=time.time(),
            context_data=context_data,
            stack_trace=stack_trace,
            function_name=function_name,
            file_name=file_name,
            line_number=line_number
        )
    
    def _process_error(self, error_context: ErrorContext) -> None:
        """エラー処理実行"""
        with self._lock:
            self._add_error_to_history(error_context)
            self._update_metrics(error_context)
            self._log_error(error_context)

            if self.enable_recovery:
                self._attempt_recovery(error_context)
    
    def _add_error_to_history(self, error_context: ErrorContext) -> None:
        """エラー履歴に追加"""
        self.error_history.append(error_context)
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
    
    def _update_metrics(self, error_context: ErrorContext) -> None:
        """メトリクス更新"""
        self.metrics.total_errors += 1
        
        # 重要度別カウント
        severity = error_context.severity
        self.metrics.errors_by_severity[severity] = (
            self.metrics.errors_by_severity.get(severity, 0) + 1
        )
        
        # カテゴリ別カウント
        category = error_context.category
        self.metrics.errors_by_category[category] = (
            self.metrics.errors_by_category.get(category, 0) + 1
        )
        
        # タイプ別カウント
        error_type = error_context.error_type.__name__
        self.metrics.errors_by_type[error_type] = (
            self.metrics.errors_by_type.get(error_type, 0) + 1
        )
        
        # 最近のエラー
        self.metrics.recent_errors.append(error_context)
        if len(self.metrics.recent_errors) > 100:
            self.metrics.recent_errors.pop(0)
        
        # 最終エラー時間
        self.metrics.last_error_time = error_context.timestamp
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """エラーログ出力"""
        log_level = self._get_log_level(error_context.severity)
        
        self.logger.log(
            log_level,
            f"Error in {error_context.function_name} "
            f"({error_context.file_name}:{error_context.line_number}): "
            f"{error_context.error_message}",
            extra={
                "error_type": error_context.error_type.__name__,
                "severity": error_context.severity.value,
                "category": error_context.category.value,
                "context_data": error_context.context_data
            }
        )
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """重要度からログレベル取得"""
        severity_levels = {
            ErrorSeverity.LOW: logging.DEBUG,
            ErrorSeverity.MEDIUM: logging.INFO,
            ErrorSeverity.HIGH: logging.WARNING,
            ErrorSeverity.CRITICAL: logging.ERROR
        }
        return severity_levels.get(severity, logging.INFO)
    
    def _attempt_recovery(self, error_context: ErrorContext) -> None:
        """エラー回復試行"""
        recovery_func = self.recovery_strategies.get(error_context.category)
        if recovery_func:
            try:
                recovery_func(error_context)
            except Exception as e:
                self.logger.error(f"Recovery failed: {e}")
    
    def _recover_network_error(self, error_context: ErrorContext) -> None:
        """ネットワークエラー回復"""
        self.logger.info("Attempting network error recovery...")
        # ネットワーク接続の再試行
        time.sleep(1.0)
    
    def _recover_database_error(self, error_context: ErrorContext) -> None:
        """データベースエラー回復"""
        self.logger.info("Attempting database error recovery...")
        # データベース接続の再試行
        time.sleep(0.5)
    
    def _recover_file_system_error(self, error_context: ErrorContext) -> None:
        """ファイルシステムエラー回復"""
        self.logger.info("Attempting file system error recovery...")
        # ファイル操作の再試行
        time.sleep(0.1)
    
    def _recover_memory_error(self, error_context: ErrorContext) -> None:
        """メモリエラー回復"""
        self.logger.info("Attempting memory error recovery...")
        # ガベージコレクション実行
        import gc
        gc.collect()
    
    def _recover_timeout_error(self, error_context: ErrorContext) -> None:
        """タイムアウトエラー回復"""
        self.logger.info("Attempting timeout error recovery...")
        # タイムアウト時間の調整
        time.sleep(0.5)
    
    def get_error_metrics(self) -> ErrorMetrics:
        """エラーメトリクス取得"""
        with self._lock:
            # エラー率計算
            current_time = time.time()
            if self.metrics.last_error_time:
                time_diff = current_time - self.metrics.last_error_time
                self.metrics.error_rate = self.metrics.total_errors / max(time_diff, 1.0)
            
            return self.metrics
    
    def get_recent_errors(self, limit: int = 10) -> List[ErrorContext]:
        """最近のエラー取得"""
        with self._lock:
            return self.error_history[-limit:]
    
    def clear_error_history(self) -> None:
        """エラー履歴クリア"""
        with self._lock:
            self.error_history.clear()
            self.metrics = ErrorMetrics()
    
    def retry_with_backoff(self, 
                          func: Callable,
                          max_retries: int = 3,
                          backoff_factor: float = 2.0,
                          base_delay: float = 1.0) -> Any:
        """バックオフ付きリトライ"""
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries:
                    raise e
                
                delay = base_delay * (backoff_factor ** attempt)
                self.logger.warning(f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {e}")
                time.sleep(delay)
    
    @contextmanager
    def error_context(self, 
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.UNKNOWN,
                     context_data: Dict[str, Any] = None):
        """エラーコンテキストマネージャー"""
        if context_data is None:
            context_data = {}
        
        try:
            yield
        except Exception as e:
            self.handle_error(e, context_data, severity, category)
            raise


def error_handler(severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 reraise: bool = True):
    """エラーハンドラーデコレーター"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = ErrorHandler()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context_data = {
                    "function": func.__name__,
                    "args": str(args),
                    "kwargs": str(kwargs)
                }
                handler.handle_error(e, context_data, severity, category)
                if reraise:
                    raise
                return None
        return wrapper
    return decorator


async def async_error_handler(handler: ErrorHandler,
                             func: Callable,
                             *args,
                             **kwargs) -> Any:
    """非同期エラーハンドラー"""
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as e:
        context_data = {
            "function": func.__name__,
            "args": str(args),
            "kwargs": str(kwargs),
            "async": True
        }
        handler.handle_error(e, context_data, ErrorSeverity.MEDIUM, ErrorCategory.UNKNOWN)
        raise


def validate_input(data: Any, 
                  expected_type: Type,
                  allow_none: bool = False) -> bool:
    """入力検証"""
    if data is None and allow_none:
        return True
    
    if not isinstance(data, expected_type):
        raise TypeError(f"Expected {expected_type.__name__}, got {type(data).__name__}")
    
    return True


def safe_execute(func: Callable, 
                *args,
                default_value: Any = None,
                **kwargs) -> Any:
    """安全実行"""
    try:
        return func(*args, **kwargs)
    except Exception:
        return default_value


# --- 互換性レイヤ ---

# ErrorDecorator: 旧API の `error_handler=` キーワードを受け取る互換ラッパー
def ErrorDecorator(*, error_handler: Optional[ErrorHandler] = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM, category: ErrorCategory = ErrorCategory.UNKNOWN, reraise: bool = True):
    def decorator(func: Callable) -> Callable:
        handler = error_handler or ErrorHandler()
        # 互換対応: Enum が別実装のものでも name/value で判定する
        cat_name = getattr(category, 'name', None) or getattr(category, 'value', None)
        is_processing_alias = getattr(ErrorCategory, 'PROCESSING', None) is category
        default_return = None if (str(cat_name).upper() == 'PROCESSING' or str(cat_name).lower() == 'processing' or is_processing_alias) else 0

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context_data = {"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                handler.handle_error(e, context_data, severity, category)
                if reraise:
                    raise
                return default_return

        return wrapper

    return decorator


class AsyncErrorHandler(ErrorHandler):
    """互換用ラッパー: 非同期ハンドリングAPIを期待する既存コード向け"""

    async def handle_async_error(self, exception: Exception, context_data: Dict[str, Any], severity: ErrorSeverity, category: ErrorCategory) -> ErrorContext:
        return self.handle_error(exception, context_data or {}, severity, category)


# 単純なグローバルログラッパとして error_logger を提供
error_logger = ErrorHandler()

# 互換: ErrorCategory.PROCESSING が参照されるケースに対応（既存の UNKNOWN を流用）
ErrorCategory.PROCESSING = ErrorCategory.UNKNOWN

# 旧実装では ErrorDecorator がグローバルに公開されていたため、互換のために builtins へ登録
import builtins
import sys

builtins.ErrorDecorator = ErrorDecorator

# phase10.* 名前空間との互換性: 同一モジュールを再利用
sys.modules.setdefault("phase10.error_handler", sys.modules[__name__])


def main():
    """メイン実行関数"""
    print("🛡️ Phase 10-3: エラーハンドリング強化と型ヒント完全実装")
    print("=" * 60)
    
    # エラーハンドラー初期化
    error_handler = _initialize_error_handler()
    
    # テスト実行
    _run_error_handling_tests(error_handler)
    
    print("\n🎉 エラーハンドリング強化テスト完了")


def _initialize_error_handler() -> ErrorHandler:
    """エラーハンドラー初期化"""
    return ErrorHandler(
        log_level=logging.INFO,
        enable_recovery=True,
        max_retries=3
    )


def _run_error_handling_tests(error_handler: ErrorHandler) -> None:
    """エラーハンドリングテスト実行"""
    # デコレーターテスト
    _run_decorator_tests(error_handler)
    
    # エラーメトリクス表示
    _display_error_metrics(error_handler)
    
    # 最近のエラー表示
    _display_recent_errors(error_handler)
    
    # 非同期テスト実行
    _run_async_tests()


def _run_decorator_tests(error_handler: ErrorHandler) -> None:
    """デコレーターテスト実行"""
    print("\n🧪 デコレーターテスト実行中...")
    
    # 正常ケース
    _test_normal_case(error_handler)
    
    # エラーケース
    _test_error_case(error_handler)


def _test_normal_case(error_handler: ErrorHandler) -> None:
    """正常ケーステスト"""
    @error_handler(ErrorSeverity.LOW, ErrorCategory.VALIDATION)
    def normal_function():
        return "正常実行"
    
    try:
        result = normal_function()
        print(f"  正常ケース: {result}")
    except Exception as e:
        print(f"  正常ケースエラー: {e}")


def _test_error_case(error_handler: ErrorHandler) -> None:
    """エラーケーステスト"""
    @error_handler(ErrorSeverity.HIGH, ErrorCategory.VALIDATION)
    def error_function():
        raise ValueError("テストエラー")
    
    try:
        error_function()
    except Exception as e:
        print(f"  エラーケース: {e}")


def _display_error_metrics(error_handler: ErrorHandler) -> None:
    """エラーメトリクス表示"""
    print("\n📊 エラーメトリクス:")
    metrics = error_handler.get_error_metrics()
    
    print(f"  総エラー数: {metrics.total_errors}")
    print(f"  エラー率: {metrics.error_rate:.2f}/秒")
    print(f"  最終エラー時間: {metrics.last_error_time}")
    
    if metrics.errors_by_severity:
        print("  重要度別エラー:")
        for severity, count in metrics.errors_by_severity.items():
            print(f"    {severity.value}: {count}")
    
    if metrics.errors_by_category:
        print("  カテゴリ別エラー:")
        for category, count in metrics.errors_by_category.items():
            print(f"    {category.value}: {count}")


def _display_recent_errors(error_handler: ErrorHandler) -> None:
    """最近のエラー表示"""
    print("\n🔍 最近のエラー:")
    recent_errors = error_handler.get_recent_errors(5)
    
    for i, error in enumerate(recent_errors, 1):
        print(f"  {i}. {error.error_type.__name__}: {error.error_message}")
        print(f"     重要度: {error.severity.value}, カテゴリ: {error.category.value}")


def _run_async_tests() -> None:
    """非同期テスト実行"""
    print("\n🔄 非同期テスト実行中...")
    
    async def async_test():
        error_handler = ErrorHandler()
        
        # 正常ケース
        await _test_async_normal_case(error_handler)
        
        # エラーケース
        await _test_async_error_case(error_handler)
    
    # 非同期テスト実行
    asyncio.run(async_test())


async def _test_async_normal_case(error_handler: ErrorHandler) -> None:
    """非同期正常ケーステスト"""
    async def async_normal_function():
        return "非同期正常実行"
    
    try:
        result = await async_error_handler(error_handler, async_normal_function)
        print(f"  非同期正常ケース: {result}")
    except Exception as e:
        print(f"  非同期正常ケースエラー: {e}")


async def _test_async_error_case(error_handler: ErrorHandler) -> None:
    """非同期エラーケーステスト"""
    async def async_error_function():
        raise RuntimeError("非同期テストエラー")
    
    try:
        await async_error_handler(error_handler, async_error_function)
    except Exception as e:
        print(f"  非同期エラーケース: {e}")


if __name__ == "__main__":
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()








