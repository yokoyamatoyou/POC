"""
パフォーマンスメトリクス収集
"""
import time
import psutil
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PerformanceMetrics:
    """パフォーマンスメトリクスを格納するデータクラス"""
    start_time: float
    end_time: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    processing_time: float = 0.0
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.end_time is None:
            self.end_time = time.time()
        
        self.processing_time = self.end_time - self.start_time
        self.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.cpu_usage_percent = psutil.cpu_percent()

class MetricsCollector:
    """メトリクス収集クラス"""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
    
    def start_timer(self, operation_name: str) -> None:
        """タイマーを開始"""
        self.metrics[operation_name] = PerformanceMetrics(start_time=time.time())
    
    def end_timer(self, operation_name: str) -> PerformanceMetrics:
        """タイマーを終了してメトリクスを取得"""
        if operation_name not in self.metrics:
            raise ValueError(f"Operation '{operation_name}' not found")
        
        self.metrics[operation_name].end_time = time.time()
        return self.metrics[operation_name]
    
    def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """指定された操作のメトリクスを取得"""
        return self.metrics.get(operation_name)
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetrics]:
        """すべてのメトリクスを取得"""
        return self.metrics.copy()
    
    def clear_metrics(self) -> None:
        """メトリクスをクリア"""
        self.metrics.clear()

# グローバルメトリクス収集器
metrics_collector = MetricsCollector()

def get_system_info() -> Dict[str, Any]:
    """システム情報を取得"""
    return {
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
        "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "timestamp": datetime.now().isoformat()
    }
