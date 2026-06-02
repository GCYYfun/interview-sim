"""
顶部导航栏组件
"""

import streamlit as st
from utils.styles import COLORS


def render_header(user_info: dict, notifications: list = None):
    """
    渲染顶部导航栏
    
    Args:
        user_info: 用户信息字典，包含 name, role, avatar_initial
        notifications: 通知列表
    """
    if notifications is None:
        notifications = []
    
    unread_count = len([n for n in notifications if not n.get("read", False)])
    
    # 构建通知徽章
    badge_html = ""
    if unread_count > 0:
        badge_html = f'<span style="position: absolute; top: -4px; right: -4px; width: 16px; height: 16px; background: #EF4444; color: white; font-size: 0.625rem; border-radius: 50%; display: flex; align-items: center; justify-content: center;">{unread_count}</span>'
    
    # 使用单行HTML,避免Streamlit解析问题
    header_html = f'<div style="background: white; padding: 0.75rem 1.5rem; border-bottom: 1px solid #E5E7EB; display: flex; align-items: center; justify-content: space-between; margin: -1rem -1rem 1rem -1rem;"><div style="font-size: 1.25rem; font-weight: 700; color: #2563EB; display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 1.5rem;">⚙️</span><span>智能面试分析系统</span></div><div style="display: flex; align-items: center; gap: 1rem;"><div style="position: relative; cursor: pointer;"><span style="font-size: 1.25rem;">🔔</span>{badge_html}</div><div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; border-radius: 0.5rem;"><div style="width: 32px; height: 32px; border-radius: 50%; background: #2563EB; color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.875rem;">{user_info.get("avatar_initial", "U")}</div><div><div style="font-size: 0.875rem; font-weight: 500; color: #374151;">{user_info.get("name", "用户")}</div><div style="font-size: 0.75rem; color: #6B7280;">{get_role_display(user_info.get("role", ""))}</div></div></div></div></div>'
    
    st.markdown(header_html, unsafe_allow_html=True)


def get_role_display(role: str) -> str:
    """获取角色显示名称"""
    role_map = {
        "admin": "系统管理员",
        "interviewer": "面试官·技术部",
        "candidate": "候选人",
    }
    return role_map.get(role, role)


def render_search_box():
    """渲染搜索框（使用 Streamlit 原生组件）"""
    search_query = st.text_input(
        "搜索",
        placeholder="搜索候选人、面试编号、岗位...",
        label_visibility="collapsed",
        key="global_search"
    )
    return search_query


def render_notification_dropdown(notifications: list):
    """渲染通知下拉框"""
    with st.expander("📬 通知", expanded=False):
        if not notifications:
            st.info("暂无新通知")
        else:
            for notif in notifications[:5]:
                icon = "🔔" if notif.get("type") == "reminder" else "📋"
                st.markdown(f"""
                <div style="padding: 0.75rem; border-bottom: 1px solid var(--gray-200); cursor: pointer;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <span>{icon}</span>
                        <div>
                            <div style="font-size: 0.875rem; color: var(--gray-900);">{notif.get('title', '')}</div>
                            <div style="font-size: 0.75rem; color: var(--gray-500);">{notif.get('time', '')}</div>
                        </div></div></div>
                """, unsafe_allow_html=True)
