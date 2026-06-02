"""
左侧导航菜单组件
根据角色动态渲染不同的菜单项
"""

import streamlit as st
from utils.styles import COLORS


# 各角色的菜单配置
MENU_CONFIG = {
    "admin": [
        {"key": "dashboard", "icon": "📊", "label": "Dashboard"},
        {"key": "org_management", "icon": "👥", "label": "组织管理"},
        {"key": "template_config", "icon": "📝", "label": "面试模板"},
        {"key": "data_board", "icon": "📈", "label": "数据看板"},
        {"key": "settings", "icon": "⚙️", "label": "系统设置"},
    ],
    "interviewer": [
        {"key": "interview_list", "icon": "🎤", "label": "我的面试"},
        {"key": "candidate_pool", "icon": "👤", "label": "候选人库"},
        {"key": "analysis_report", "icon": "📊", "label": "分析报告"},
        {"key": "calendar", "icon": "📅", "label": "面试日历"},
        {"key": "upload_analysis", "icon": "📤", "label": "上传分析"},
    ],
    "candidate": [
        {"key": "my_interviews", "icon": "🎤", "label": "我的面试"},
        {"key": "authorization", "icon": "🔐", "label": "授权管理"},
    ],
}


def render_sidebar(role: str, current_page: str) -> str:
    """
    渲染侧边栏导航
    
    Args:
        role: 用户角色
        current_page: 当前页面 key
        
    Returns:
        选中的页面 key
    """
    # Logo 区域
    st.sidebar.markdown('<div style="padding: 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1);"><h1 style="font-size: 1.25rem; color: white; margin: 0; display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 1.5rem;">⚙️</span>智能面试系统</h1></div>', unsafe_allow_html=True)
    
    # 角色选择器
    role_options = {
        "admin": "👑 管理员",
        "interviewer": "🎤 面试官", 
        "candidate": "👤 候选人",
    }
    
    selected_role_label = st.sidebar.selectbox(
        "角色切换",
        options=list(role_options.values()),
        index=list(role_options.keys()).index(role) if role in role_options else 0,
        key="role_selector"
    )
    
    # 获取选中的角色 key
    selected_role = [k for k, v in role_options.items() if v == selected_role_label][0]
    
    # 如果角色改变，更新 session state
    if selected_role != role:
        st.session_state.current_role = selected_role
        st.session_state.current_page = get_default_page(selected_role)
        st.rerun()
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # 菜单项
    menu_items = MENU_CONFIG.get(role, [])
    selected_page = current_page
    
    for item in menu_items:
        is_active = current_page == item["key"]
        
        if st.sidebar.button(
            f"{item['icon']}  {item['label']}",
            key=f"nav_{item['key']}",
            width='stretch',
            type="primary" if is_active else "secondary",
        ):
            selected_page = item["key"]
            st.session_state.current_page = item["key"]
            st.rerun()
    
    return selected_page


def get_default_page(role: str) -> str:
    """获取角色的默认页面"""
    default_pages = {
        "admin": "dashboard",
        "interviewer": "interview_list",
        "candidate": "my_interviews",
    }
    return default_pages.get(role, "dashboard")


def render_sidebar_footer():
    """渲染侧边栏底部"""
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    st.sidebar.markdown('<div style="padding: 1rem 0; border-top: 1px solid rgba(255,255,255,0.1); margin-top: auto;"><div style="display: flex; align-items: center; gap: 0.5rem; color: rgba(255,255,255,0.6); font-size: 0.75rem;"><span>💡</span><span>技术支持: support@interview.ai</span></div></div>', unsafe_allow_html=True)
