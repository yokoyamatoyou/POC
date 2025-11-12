"""
ヘルスチェック機能
"""
import psutil
import time
from typing import Dict, Any, List
from datetime import datetime

# 軽量版ではmetricsモジュールを簡易実装
def get_system_info() -> Dict[str, Any]:
    """システム情報を取得（簡易版）"""
    return {
        "platform": "unknown",
        "python_version": "unknown"
    }

def health_check() -> Dict[str, Any]:
    """システムのヘルスチェックを実行"""
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "status": "healthy",
        "checks": {}
    }
    
    try:
        # CPU使用率チェック
        cpu_usage = psutil.cpu_percent(interval=1)
        health_status["checks"]["cpu_usage"] = {
            "value": cpu_usage,
            "status": "healthy" if cpu_usage < 80 else "warning" if cpu_usage < 95 else "critical",
            "threshold": 80
        }
        
        # メモリ使用率チェック
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        health_status["checks"]["memory_usage"] = {
            "value": memory_usage,
            "status": "healthy" if memory_usage < 80 else "warning" if memory_usage < 95 else "critical",
            "threshold": 80,
            "available_gb": memory.available / 1024 / 1024 / 1024
        }
        
        # ディスク使用率チェック
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        health_status["checks"]["disk_usage"] = {
            "value": disk_usage,
            "status": "healthy" if disk_usage < 80 else "warning" if disk_usage < 95 else "critical",
            "threshold": 80,
            "free_gb": disk.free / 1024 / 1024 / 1024
        }
        
        # プロセス数チェック
        process_count = len(psutil.pids())
        health_status["checks"]["process_count"] = {
            "value": process_count,
            "status": "healthy" if process_count < 1000 else "warning" if process_count < 2000 else "critical",
            "threshold": 1000
        }
    except Exception:
        # psutilが利用できない場合はスキップ
        pass
    
    # 全体のステータスを決定
    critical_count = sum(1 for check in health_status["checks"].values() if check.get("status") == "critical")
    warning_count = sum(1 for check in health_status["checks"].values() if check.get("status") == "warning")
    
    if critical_count > 0:
        health_status["status"] = "critical"
    elif warning_count > 0:
        health_status["status"] = "warning"
    
    # システム情報を追加
    health_status["system_info"] = get_system_info()
    
    return health_status

def get_health_summary() -> str:
    """ヘルスチェックの要約を取得"""
    health = health_check()
    
    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "critical": "❌"
    }
    
    emoji = status_emoji.get(health["status"], "❓")
    return f"{emoji} システム状態: {health['status'].upper()}"








