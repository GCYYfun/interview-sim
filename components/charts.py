"""
图表组件
包含环形进度图、雷达图、词云图等
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utils.styles import COLORS


def render_progress_ring(value: float, max_value: float = 100, label: str = "", size: int = 150):
    """
    渲染环形进度图
    
    Args:
        value: 当前值
        max_value: 最大值
        label: 标签文字
        size: 图表大小
    """
    percentage = (value / max_value) * 100
    
    fig = go.Figure(data=[go.Pie(
        values=[percentage, 100 - percentage],
        hole=0.75,
        marker_colors=[COLORS["primary"], COLORS["gray_200"]],
        textinfo='none',
        hoverinfo='skip',
    )])
    
    fig.update_layout(
        showlegend=False,
        margin=dict(t=0, b=0, l=0, r=0),
        width=size,
        height=size,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(
                text=f"<b>{int(percentage)}%</b>",
                x=0.5, y=0.55,
                font=dict(size=24, color=COLORS["gray_900"]),
                showarrow=False,
            ),
            dict(
                text=label,
                x=0.5, y=0.35,
                font=dict(size=12, color=COLORS["gray_500"]),
                showarrow=False,
            ),
        ]
    )
    
    st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})


def render_radar_chart(dimensions: list, title: str = "能力图谱"):
    """
    渲染雷达图
    
    Args:
        dimensions: 维度数据列表，每项包含 name, score, benchmark
        title: 图表标题
    """
    categories = [d["name"] for d in dimensions]
    scores = [d["score"] for d in dimensions]
    benchmarks = [d.get("benchmark", 4.0) for d in dimensions]
    
    # 闭合图形
    categories = categories + [categories[0]]
    scores = scores + [scores[0]]
    benchmarks = benchmarks + [benchmarks[0]]
    
    fig = go.Figure()
    
    # 基准线
    fig.add_trace(go.Scatterpolar(
        r=benchmarks,
        theta=categories,
        fill='toself',
        fillcolor='rgba(156, 163, 175, 0.2)',
        line=dict(color=COLORS["gray_400"], width=1, dash='dash'),
        name='岗位基准',
    ))
    
    # 候选人得分
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        fillcolor='rgba(37, 99, 235, 0.3)',
        line=dict(color=COLORS["primary"], width=2),
        name='候选人得分',
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                tickvals=[1, 2, 3, 4, 5],
                tickfont=dict(size=10, color=COLORS["gray_500"]),
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color=COLORS["gray_700"]),
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=60, l=60, r=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_bar_chart(data: list, x_key: str, y_key: str, title: str = ""):
    """
    渲染柱状图
    
    Args:
        data: 数据列表
        x_key: X轴字段名
        y_key: Y轴字段名
        title: 图表标题
    """
    x_values = [d[x_key] for d in data]
    y_values = [d[y_key] for d in data]
    
    fig = go.Figure(data=[
        go.Bar(
            x=x_values,
            y=y_values,
            marker_color=COLORS["primary"],
            text=y_values,
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        margin=dict(t=40 if title else 20, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor=COLORS["gray_200"]),
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def render_keyword_cloud(keywords: list):
    """
    渲染关键词云（使用简化的标签形式）
    
    Args:
        keywords: 关键词列表，每项包含 word, count
    """
    max_count = max(kw["count"] for kw in keywords) if keywords else 1
    
    html_parts = []
    for kw in sorted(keywords, key=lambda x: x["count"], reverse=True):
        # 根据词频计算字体大小
        size = 0.75 + (kw["count"] / max_count) * 0.75
        opacity = 0.5 + (kw["count"] / max_count) * 0.5
        
        html_parts.append(f'<span style="display: inline-block; padding: 0.25rem 0.75rem; margin: 0.25rem; border-radius: 9999px; font-size: {size}rem; background: rgba(37, 99, 235, {opacity * 0.2}); color: {COLORS["primary"]}; cursor: default;">{kw["word"]}</span>')
    
    st.markdown(
        f"""<div style="display: flex; flex-wrap: wrap; gap: 0.25rem;">{"".join(html_parts)}</div>""",
        unsafe_allow_html=True
    )


def render_timeline_chart(events: list, title: str = ""):
    """
    渲染时间轴图表
    
    Args:
        events: 事件列表，每项包含 time, label
        title: 图表标题
    """
    for event in events:
        is_key = event.get("is_key_point", False)
        
        st.markdown(f'<div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; padding-left: 1rem; border-left: 2px solid {"var(--primary)" if is_key else "var(--gray-300)"};"><div style="min-width: 60px; font-size: 0.75rem; color: var(--gray-500); font-family: monospace;">{event.get("time", "")}</div><div><div style="font-size: 0.75rem; font-weight: 600; color: {"var(--primary)" if event.get("speaker") == "候选人" else "var(--gray-700)"}; margin-bottom: 0.25rem;">{event.get("speaker", "")}</div><div style="font-size: 0.875rem; color: var(--gray-600); {"background: rgba(37, 99, 235, 0.05); padding: 0.5rem; border-radius: 0.25rem;" if is_key else ""}">{event.get("content", "")}</div></div></div>', unsafe_allow_html=True)
