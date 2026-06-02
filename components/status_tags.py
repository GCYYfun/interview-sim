"""
状态标签组件
"""

import streamlit as st
from utils.styles import COLORS, STATUS_COLORS


def render_status_tag(status: str, style: str = "default") -> str:
    """
    渲染状态标签 HTML
    
    Args:
        status: 状态值 (pending, analyzing, completed, cancelled)
        style: 样式类型 (default, pill)
    
    Returns:
        HTML 字符串
    """
    status_config = {
        "pending": {
            "text": "待面试",
            "bg": "rgba(37, 99, 235, 0.1)",
            "color": COLORS["primary"],
            "icon": "🔵",
        },
        "analyzing": {
            "text": "AI分析中",
            "bg": "rgba(6, 182, 212, 0.1)",
            "color": COLORS["info"],
            "icon": "🔄",
        },
        "completed": {
            "text": "已完成",
            "bg": "rgba(16, 185, 129, 0.1)",
            "color": COLORS["success"],
            "icon": "✅",
        },
        "cancelled": {
            "text": "已取消",
            "bg": "rgba(156, 163, 175, 0.1)",
            "color": COLORS["gray_400"],
            "icon": "⚪",
        },
        "in_progress": {
            "text": "进行中",
            "bg": "rgba(37, 99, 235, 0.1)",
            "color": COLORS["primary"],
            "icon": "🔵",
        },
        "active": {
            "text": "正常",
            "bg": "rgba(16, 185, 129, 0.1)",
            "color": COLORS["success"],
            "icon": "✅",
        },
        "frozen": {
            "text": "已冻结",
            "bg": "rgba(156, 163, 175, 0.1)",
            "color": COLORS["gray_400"],
            "icon": "🔒",
        },
    }
    
    config = status_config.get(status, {
        "text": status,
        "bg": "rgba(156, 163, 175, 0.1)",
        "color": COLORS["gray_500"],
        "icon": "⚪",
    })
    
    if style == "pill":
        return f"""
        <span style="
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 500;
            background: {config['bg']};
            color: {config['color']};
        ">
            {config['icon']} {config['text']}
        </span>
        """
    else:
        return f"""
        <span style="
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 500;
            background: {config['bg']};
            color: {config['color']};
        ">
            {config['text']}
        </span>
        """


def render_priority_tag(priority: str) -> str:
    """渲染优先级标签"""
    priority_config = {
        "high": {"text": "高", "color": COLORS["danger"]},
        "medium": {"text": "中", "color": COLORS["warning"]},
        "low": {"text": "低", "color": COLORS["gray_500"]},
    }
    
    config = priority_config.get(priority, priority_config["low"])
    
    return f"""
    <span style="
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {config['color']};
        margin-right: 0.5rem;
    "></span>
    """


def display_status_tag(status: str, style: str = "pill"):
    """直接显示状态标签"""
    st.markdown(render_status_tag(status, style), unsafe_allow_html=True)
