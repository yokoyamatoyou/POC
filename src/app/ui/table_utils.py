"""Streamlit DataFrame表示のゆれを抑えるユーティリティ."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import pandas as pd
import streamlit as st

DataFrameLike = Union[
    pd.DataFrame,
    Mapping[str, Sequence[Any]],
    Sequence[Mapping[str, Any]],
]


def _ensure_dataframe(data: DataFrameLike) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)


def render_stable_dataframe(
    data: DataFrameLike,
    *,
    column_config: Optional[Mapping[str, Any]] = None,
    hide_index: bool = True,
    min_height: int = 160,
    max_height: int = 480,
    row_height: int = 38,
    header_height: int = 56,
    height: Optional[int] = None,
) -> pd.DataFrame:
    """高さを安定化させてDataFrameを描画する.

    StreamlitのDataFrameは描画ごとに高さが変わるとスクロールバーが揺れてしまうため、
    行数に応じて安定した高さを計算し、`st.dataframe` を呼び出す共通処理。

    Args:
        data: DataFrameあるいは類似のデータ構造。
        column_config: 列設定。`st.column_config.*Column` の辞書を想定。
        hide_index: インデックスを非表示にするかどうか。
        min_height: 最低高さ。
        max_height: 最高高さ。
        row_height: 行ごとの高さ係数。
        header_height: ヘッダーの高さ係数。
        height: 固定高さを指定する場合に使用。

    Returns:
        描画に使用したDataFrame（コピーではなく元オブジェクト）。
    """

    df = _ensure_dataframe(data)

    if height is None:
        rows = len(df)
        computed = header_height + row_height * max(rows, 1)
        target_height = max(min_height, min(int(computed), max_height))
    else:
        target_height = height

    st.dataframe(
        df,
        width='stretch',
        hide_index=hide_index,
        column_config=column_config or {},
        height=int(target_height),
    )

    return df






