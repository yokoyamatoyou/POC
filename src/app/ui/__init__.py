"""
UIモジュール（軽量版）
"""
from .main_ui import main_app_ui
from .knowledge_ui import knowledge_registration_ui, display_knowledge_base
from .chat_ui import chat_ui, create_answer_generation_ui, display_generated_answer

# 軽量版では以下の機能は除外
# from .pdf_ui import create_pdf_processing_ui, display_pdf_processing_result
# from .dashboard_ui import role_dashboard_ui, time_aware_search_ui
# from .evaluation_ui import advanced_evaluation_ui

__all__ = [
    'main_app_ui',
    'knowledge_registration_ui',
    'display_knowledge_base',
    'chat_ui',
    'create_answer_generation_ui',
    'display_generated_answer',
    # 'create_pdf_processing_ui',
    # 'display_pdf_processing_result',
    # 'role_dashboard_ui',
    # 'time_aware_search_ui',
    # 'advanced_evaluation_ui'
]








