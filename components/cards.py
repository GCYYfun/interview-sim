"""
卡片组件
包含面试卡片、统计卡片、风险卡片等
"""

import streamlit as st
from utils.styles import COLORS
from utils.mock_data import format_datetime, get_status_display


def render_stat_card(value: str, label: str, icon: str = "", trend: str = None):
    """
    渲染统计卡片
    
    Args:
        value: 数值
        label: 标签
        icon: 图标
        trend: 趋势 (up/down/None)
    """
    trend_html = ""
    if trend:
        if trend == "up":
            trend_html = f'<span style="color: {COLORS["success"]}; font-size: 0.75rem; margin-left: 0.5rem;">↑ 12%</span>'
        elif trend == "down":
            trend_html = f'<span style="color: {COLORS["danger"]}; font-size: 0.75rem; margin-left: 0.5rem;">↓ 5%</span>'
    
    card_html = f'<div class="stat-card"><div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div><div class="stat-value">{value}{trend_html}</div><div class="stat-label">{label}</div></div>'
    st.markdown(card_html, unsafe_allow_html=True)


def render_interview_card(interview: dict, show_actions: bool = True) -> bool:
    """
    渲染面试卡片
    
    Args:
        interview: 面试数据
        show_actions: 是否显示操作按钮
        
    Returns:
        是否点击了卡片
    """
    status_info = get_status_display(interview.get("status", "pending"))
    scheduled_time = interview.get("scheduled_time")
    time_str = format_datetime(scheduled_time, "full") if scheduled_time else ""
    
    # 状态样式配置
    status_cls = status_info.get('class', 'status-pending')
    if status_cls == 'status-pending':
        bg, color = 'rgba(37, 99, 235, 0.1)', COLORS['primary']
    elif status_cls == 'status-analyzing':
        bg, color = 'rgba(6, 182, 212, 0.1)', COLORS['info']
    elif status_cls == 'status-completed':
        bg, color = 'rgba(16, 185, 129, 0.1)', COLORS['success']
    else:
        bg, color = 'rgba(156, 163, 175, 0.1)', COLORS['gray_400']

    # 匹配度HTML
    match_html = f"<span style='margin-left: 1rem;'>📊 匹配度预估：{interview.get('match_score')}%</span>" if interview.get('match_score') else ""
    
    card_html = f'<div class="interview-card"><div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.75rem;"><div><div class="interview-title">🎤 {interview.get("position", "")} - {interview.get("candidate_name", "")}</div><div class="interview-meta">面试官：{interview.get("interviewer", "")}</div></div><span style="display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 500; background: {bg}; color: {color};">{status_info["icon"]} {status_info["text"]}</span></div><div style="display: flex; justify-content: space-between; align-items: center;"><div class="interview-meta"><span>⏰ {time_str}</span>{match_html}</div></div></div>'
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    if show_actions:
        col1, col2, col3 = st.columns(3)
        with col1:
            if interview.get("status") == "pending":
                if st.button("进入会议室", key=f"enter_{interview['id']}", use_container_width=True):
                    st.session_state.current_page = "meeting_room"
                    st.session_state.current_interview = interview
                    st.rerun()
        with col2:
            if interview.get("status") in ["completed", "analyzing"]:
                if st.button("查看报告", key=f"report_{interview['id']}", use_container_width=True):
                    st.session_state.current_page = "analysis_report"
                    st.session_state.current_interview = interview
                    st.rerun()
        with col3:
            if st.button("分享链接", key=f"share_{interview['id']}", use_container_width=True):
                st.toast("链接已复制到剪贴板！", icon="📋")
    
    return False


def render_todo_card(todo: dict):
    """渲染待办事项卡片"""
    priority_colors = {
        "high": COLORS["danger"],
        "medium": COLORS["warning"],
        "low": COLORS["gray_400"],
    }
    
    type_icons = {
        "assign": "👤",
        "review": "📝",
        "alert": "⚠️",
    }
    
    priority_color = priority_colors.get(todo.get('priority', 'low'), COLORS['gray_400'])
    icon = type_icons.get(todo.get('type', ''), '📋')
    title = todo.get('title', '')
    
    card_html = f'<div style="padding: 0.75rem 1rem; border-radius: 0.5rem; background: white; border-left: 3px solid {priority_color}; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.75rem;"><span style="font-size: 1.25rem;">{icon}</span><div style="flex: 1;"><div style="font-size: 0.875rem; color: var(--gray-900);">{title}</div></div><span style="width: 8px; height: 8px; border-radius: 50%; background: {priority_color};"></span></div>'
    st.markdown(card_html, unsafe_allow_html=True)


def render_risk_card(risk: dict):
    """
    渲染风险提醒卡片
    
    Args:
        risk: 风险数据，包含 type, title, description, confidence, severity
    """
    severity_config = {
        "high": {"icon": "🚩", "bg": "rgba(239, 68, 68, 0.1)", "border": COLORS["danger"], "badge": COLORS["danger"]},
        "medium": {"icon": "⚠️", "bg": "rgba(245, 158, 11, 0.1)", "border": COLORS["warning"], "badge": COLORS["warning"]},
        "low": {"icon": "ℹ️", "bg": "rgba(156, 163, 175, 0.1)", "border": COLORS["gray_400"], "badge": COLORS["gray_400"]},
    }
    
    level = risk.get('severity', 'low')
    config = severity_config.get(level, severity_config['low'])
    confidence = risk.get("confidence", 0)
    
    card_html = f'<div style="padding: 1rem; border-radius: 0.5rem; background: {config["bg"]}; border-left: 3px solid {config["border"]}; margin-bottom: 0.75rem;"><div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;"><span style="font-size: 1.25rem;">{config["icon"]}</span><span style="font-weight: 600; color: var(--gray-900);">{risk.get("title", "")}</span><span style="padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.625rem; background: {config.get("badge", COLORS["gray_200"])}; color: white;">{confidence}%</span></div><div style="font-size: 0.875rem; color: var(--gray-600); margin-bottom: 0.5rem;">{risk.get("description", "")}</div></div>'
    st.markdown(card_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("忽略此风险", key=f"ignore_{risk.get('type', '')}_{confidence}", use_container_width=True):
            st.toast("已忽略该风险提醒", icon="✓")
    with col2:
        if st.button("加入面试笔记", key=f"note_{risk.get('type', '')}_{confidence}", use_container_width=True):
            st.toast("已添加到面试笔记", icon="📝")


def render_candidate_card(candidate: dict, is_selected: bool = False):
    """渲染候选人卡片"""
    status_info = get_status_display(candidate.get("status", ""))
    
    status_cls = candidate.get('status')
    if status_cls == 'in_progress':
        status_bg, status_color = 'rgba(37, 99, 235, 0.1)', COLORS['primary']
    elif status_cls == 'completed':
        status_bg, status_color = 'rgba(16, 185, 129, 0.1)', COLORS['success']
    else:
        status_bg, status_color = 'rgba(6, 182, 212, 0.1)', COLORS['info']
    
    bg_color = 'rgba(37, 99, 235, 0.05)' if is_selected else 'white'
    border_color = 'var(--primary)' if is_selected else 'var(--gray-200)'
    
    try:
        updated = format_datetime(candidate.get('latest_interview'), 'relative') if candidate.get('latest_interview') else '-'
    except:
        updated = "-"
        
    card_html = f'<div style="padding: 1rem; border-radius: 0.5rem; background: {bg_color}; border: 1px solid {border_color}; margin-bottom: 0.5rem; cursor: pointer; transition: all 0.2s ease;"><div style="display: flex; justify-content: space-between; align-items: center;"><div><div style="font-weight: 600; color: var(--gray-900); margin-bottom: 0.25rem;">{candidate.get("name", "")}</div><div style="font-size: 0.875rem; color: var(--gray-500);">{candidate.get("position", "")}</div></div><span style="display: inline-flex; align-items: center; padding: 0.25rem 0.5rem; border-radius: 0.25rem; font-size: 0.75rem; background: {status_bg}; color: {status_color};">{status_info["text"]}</span></div><div style="font-size: 0.75rem; color: var(--gray-400); margin-top: 0.5rem;">面试次数：{candidate.get("interview_count", 0)} | 最近：{updated}</div></div>'
    
    st.markdown(card_html, unsafe_allow_html=True)


def render_template_card(template: dict, is_selected: bool = False):
    """渲染模板卡片"""
    bg_color = 'rgba(37, 99, 235, 0.05)' if is_selected else 'white'
    border_color = 'var(--primary)' if is_selected else 'var(--gray-200)'
    
    toggle_bg = 'var(--success)' if template.get('is_active') else 'var(--gray-300)'
    toggle_pos = 'right: 2px;' if template.get('is_active') else 'left: 2px;'
    
    positions_html = ' '.join([f'<span style="padding: 0.125rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; background: var(--gray-100); color: var(--gray-600);">{pos}</span>' for pos in template.get('positions', [])])
    try:
        updated = format_datetime(template.get('updated_at'), 'relative') if template.get('updated_at') else '-'
    except:
        updated = "-"
        
    card_html = f'<div style="padding: 1rem; border-radius: 0.5rem; background: {bg_color}; border: 1px solid {border_color}; margin-bottom: 0.5rem; cursor: pointer;"><div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;"><div style="font-weight: 600; color: var(--gray-900);">{template.get("name", "")}</div><div style="width: 40px; height: 20px; border-radius: 10px; background: {toggle_bg}; position: relative;"><div style="width: 16px; height: 16px; border-radius: 50%; background: white; position: absolute; top: 2px; {toggle_pos} transition: all 0.2s ease;"></div></div></div><div style="display: flex; flex-wrap: wrap; gap: 0.25rem; margin-bottom: 0.5rem;">{positions_html}</div><div style="font-size: 0.75rem; color: var(--gray-400);">更新于：{updated}</div></div>'
    
    st.markdown(card_html, unsafe_allow_html=True)
