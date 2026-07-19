"""
Reusable UI Components for the Football Prediction Dashboard.

Provides consistent, animated metric cards, Plotly charts, gauge widgets,
sparklines, status badges, CSS styling, and **dark/light theme support**
used across all dashboard pages.

Usage
-----
    from dashboard.components import (
        init_theme, sidebar_theme_toggle, get_plotly_layout,
        metric_card, gauge_chart, status_badge,
        section_header, render_custom_css,
        confusion_matrix_heatmap, feature_importance_chart,
        render_hero, render_footer, Colors,
    )

    # In Streamlit page:
    init_theme()                          # must be called early
    render_custom_css()                   # injects theme-aware CSS
    sidebar_theme_toggle()                # in sidebar
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════
#  Theme Management
# ═══════════════════════════════════════════════════════════

THEME_KEY = "_dashboard_theme"


def init_theme(default: Literal["dark", "light"] = "dark") -> str:
    """Initialise the theme in session state.
    
    Call this early in every page before any other component functions.
    
    Parameters
    ----------
    default : str
        Default theme (``"dark"`` or ``"light"``).
    
    Returns
    -------
    str
        Current theme name.
    """
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = default
    return st.session_state[THEME_KEY]


def get_current_theme() -> Literal["dark", "light"]:
    """Return the current theme name from session state."""
    return st.session_state.get(THEME_KEY, "dark")


def sidebar_theme_toggle() -> None:
    """Render a theme toggle button in the sidebar.
    
    Call this inside a ``with st.sidebar:`` block or it will render
    inline.  The button toggles between dark and light mode on click.
    """
    current = get_current_theme()
    is_dark = current == "dark"
    label = "🌙 Dark Mode" if is_dark else "☀️ Light Mode"
    
    if st.sidebar.button(label, use_container_width=True, type="secondary"):
        st.session_state[THEME_KEY] = "light" if is_dark else "dark"
        st.rerun()


def sidebar_theme_radio() -> None:
    """Render a theme selector radio group in the sidebar."""
    current = get_current_theme()
    options = ["🌙 Dark", "☀️ Light"]
    idx = 0 if current == "dark" else 1
    
    choice = st.sidebar.radio("Theme", options, index=idx, label_visibility="collapsed")
    new_theme = "dark" if choice.startswith("🌙") else "light"
    if new_theme != current:
        st.session_state[THEME_KEY] = new_theme
        st.rerun()


# ═══════════════════════════════════════════════════════════
#  Colour Palettes — Dark & Light
# ═══════════════════════════════════════════════════════════

class DarkColors:
    """Dark theme colour palette."""
    PRIMARY = "#4fc3f7"
    SUCCESS = "#4caf50"
    WARNING = "#ffc107"
    DANGER = "#f44336"
    INFO = "#2196f3"
    ACCENT = "#7c3aed"
    
    GRADIENT_GREEN = "#81c784"
    GRADIENT_BLUE = "#4fc3f7"
    GRADIENT_GOLD = "#ffd54f"
    GRADIENT_RED = "#ef9a9a"
    GRADIENT_PURPLE = "#b39ddb"
    
    BG_APP = "#0a0d14"
    BG_CARD = "#141824"
    BG_CARD_HOVER = "#1a1f2e"
    BORDER = "#1e2235"
    BORDER_HOVER = "#4fc3f7"
    TEXT_PRIMARY = "#e0e0e0"
    TEXT_SECONDARY = "#6b7280"
    TEXT_MUTED = "#444444"
    CARD_GRADIENT_FROM = "#141824"
    CARD_GRADIENT_TO = "#1a1f2e"
    HERO_GRADIENT = "135deg, #11141e 0%, #0f1928 50%, #11141e 100%"
    GRID_COLOR = "#1e2235"
    ZERO_LINE = "#2a2d3a"
    SELECT_BG = "#141824"
    SIDEBAR_BG = "#0e1117"


class LightColors:
    """Light theme colour palette."""
    PRIMARY = "#0288d1"
    SUCCESS = "#2e7d32"
    WARNING = "#f9a825"
    DANGER = "#c62828"
    INFO = "#0277bd"
    ACCENT = "#6a1b9a"
    
    GRADIENT_GREEN = "#66bb6a"
    GRADIENT_BLUE = "#42a5f5"
    GRADIENT_GOLD = "#ffca28"
    GRADIENT_RED = "#ef5350"
    GRADIENT_PURPLE = "#ab47bc"
    
    BG_APP = "#f5f5f5"
    BG_CARD = "#ffffff"
    BG_CARD_HOVER = "#fafafa"
    BORDER = "#e0e0e0"
    BORDER_HOVER = "#0288d1"
    TEXT_PRIMARY = "#212121"
    TEXT_SECONDARY = "#757575"
    TEXT_MUTED = "#bdbdbd"
    CARD_GRADIENT_FROM = "#ffffff"
    CARD_GRADIENT_TO = "#fafafa"
    HERO_GRADIENT = "135deg, #e3f2fd 0%, #f3e5f5 50%, #e8f5e9 100%"
    GRID_COLOR = "#e0e0e0"
    ZERO_LINE = "#cccccc"
    SELECT_BG = "#ffffff"
    SIDEBAR_BG = "#ffffff"


# ── Fast colour lookup ─────────────────────────────────

_COLOR_SETS: dict[str, Any] = {
    "dark": DarkColors,
    "light": LightColors,
}


def get_colors() -> Any:
    """Return the colour class for the current theme."""
    theme = get_current_theme()
    return _COLOR_SETS.get(theme, DarkColors)


# ── Plotly layout config ───────────────────────────────

def get_plotly_layout() -> dict:
    """Return a Plotly layout dict that matches the current theme."""
    c = get_colors()
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": c.TEXT_PRIMARY, "size": 11},
        "xaxis": {"gridcolor": c.GRID_COLOR, "zerolinecolor": c.ZERO_LINE},
        "yaxis": {"gridcolor": c.GRID_COLOR, "zerolinecolor": c.ZERO_LINE},
    }


# ── Theme-aware CSS ────────────────────────────────────

def _build_theme_css() -> str:
    """Build a complete CSS string using CSS custom properties for theming."""
    return """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', -apple-system, sans-serif; }

    /* ── CSS Custom Properties (set via JS in head, but we use static) ── */
    .stApp { background: var(--bg-app, #0a0d14); }
    .stApp header { background: var(--bg-header, #11141e); }

    /* ── Metric Card ── */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-card-from, #141824) 0%, var(--bg-card-to, #1a1f2e) 100%);
        border: 1px solid var(--border, #1e2235);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        background: linear-gradient(90deg, var(--primary, #4fc3f7), var(--success, #4caf50));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.15);
        border-color: var(--primary, #4fc3f7);
    }
    .metric-card:hover::before { opacity: 1; }
    .metric-value {
        font-size: 1.8rem; font-weight: 800;
        color: var(--text-primary, #e0e0e0);
        line-height: 1.1; letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-secondary, #6b7280);
        text-transform: uppercase; letter-spacing: 0.08em;
        margin-top: 0.3rem; font-weight: 600;
    }
    .metric-delta { font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem; }
    .metric-delta.up { color: var(--success, #4caf50); }
    .metric-delta.down { color: var(--danger, #f44336); }
    .metric-delta.neutral { color: var(--warning, #ffc107); }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.4rem; font-weight: 700;
        color: var(--text-primary, #e0e0e0);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border, #1e2235);
        display: flex; align-items: center; gap: 0.5rem;
    }
    .section-header-sm {
        font-size: 1.05rem; font-weight: 600;
        color: var(--text-secondary, #c0c0c0);
        margin: 1.5rem 0 0.8rem 0;
    }

    /* ── Status Badge ── */
    .badge {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.2rem 0.7rem; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.02em;
        transition: all 0.2s ease;
    }
    .badge-green { background: rgba(76, 175, 80, 0.15); color: var(--success, #81c784); border: 1px solid rgba(76, 175, 80, 0.25); }
    .badge-yellow { background: rgba(255, 193, 7, 0.15); color: var(--warning, #ffd54f); border: 1px solid rgba(255, 193, 7, 0.25); }
    .badge-red { background: rgba(244, 67, 54, 0.15); color: var(--danger, #ef9a9a); border: 1px solid rgba(244, 67, 54, 0.25); }
    .badge-blue { background: rgba(79, 195, 247, 0.15); color: var(--primary, #4fc3f7); border: 1px solid rgba(79, 195, 247, 0.25); }
    .badge-purple { background: rgba(124, 58, 237, 0.15); color: var(--accent, #b39ddb); border: 1px solid rgba(124, 58, 237, 0.25); }

    /* ── Status Dot ── */
    .status-dot {
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin-right: 6px;
        animation: pulse-dot 2s infinite;
    }
    .status-dot.green { background: var(--success, #4caf50); box-shadow: 0 0 8px rgba(76,175,80,0.5); }
    .status-dot.yellow { background: var(--warning, #ffc107); box-shadow: 0 0 8px rgba(255,193,7,0.5); }
    .status-dot.red { background: var(--danger, #f44336); box-shadow: 0 0 8px rgba(244,67,54,0.5); }
    .status-dot.blue { background: var(--primary, #4fc3f7); box-shadow: 0 0 8px rgba(79,195,247,0.5); }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.85); }
    }

    /* ── Hero ── */
    .hero {
        background: linear-gradient(var(--hero-gradient, 135deg, #11141e 0%, #0f1928 50%, #11141e 100%));
        border: 1px solid var(--border, #1e2235);
        border-radius: 20px; padding: 2.5rem 3rem;
        margin-bottom: 1.5rem; position: relative; overflow: hidden;
    }
    .hero h1 {
        font-size: 2.4rem; font-weight: 800;
        margin: 0 0 0.3rem 0;
        background: linear-gradient(90deg, var(--primary, #4fc3f7), var(--success, #81c784), var(--warning, #ffd54f));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em; line-height: 1.2;
    }
    .hero-sub { color: var(--text-secondary, #6b7280); font-size: 1rem; margin: 0; line-height: 1.5; }
    .hero-badge {
        display: inline-block;
        background: rgba(79, 195, 247, 0.1);
        border: 1px solid rgba(79, 195, 247, 0.2);
        color: var(--primary, #4fc3f7);
        padding: 0.2rem 0.8rem; border-radius: 20px;
        font-size: 0.72rem; font-weight: 600;
        margin-top: 0.5rem; margin-right: 0.5rem;
    }

    /* ── Info row ── */
    .info-row {
        display: flex; align-items: center;
        gap: 0.5rem; padding: 0.3rem 0;
        color: var(--text-secondary, #9ca3af); font-size: 0.82rem;
    }

    /* ── Skeleton loading ── */
    .skeleton {
        background: linear-gradient(90deg, var(--border, #1e2235) 25%, var(--bg-card-to, #252a40) 50%, var(--border, #1e2235) 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 10px; height: 80px; margin-bottom: 1rem;
    }
    @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    /* ── Progress bar ── */
    .progress-container { background: var(--border, #1e2235); border-radius: 10px; height: 10px; overflow: hidden; margin: 0.3rem 0; }
    .progress-fill { height: 100%; border-radius: 10px; transition: width 1s ease; }

    /* ── Data table tweaks ── */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border, #1e2235);
        border-radius: 12px; overflow: hidden;
    }
    div[data-testid="stDataFrame"] th {
        background: var(--bg-card-from, #141824) !important;
        color: var(--text-secondary, #6b7280) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }

    /* ── Selectbox ── */
    div[data-baseweb="select"] > div {
        background: var(--select-bg, #141824) !important;
        border: 1px solid var(--border, #1e2235) !important;
        border-radius: 10px !important;
    }

    /* ── Button ── */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79,195,247,0.3);
    }

    /* ── Expander ── */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border, #1e2235);
        border-radius: 12px;
        background: var(--bg-card-from, #141824);
    }

    /* ── Number input / slider ── */
    div[data-baseweb="input"] > div {
        background: var(--select-bg, #141824) !important;
        border: 1px solid var(--border, #1e2235) !important;
        border-radius: 10px !important;
    }
    div[role="slider"] { background: var(--primary, #4fc3f7) !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg, #0e1117) !important;
        border-right: 1px solid var(--border, #1e2235);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary, #9ca3af); }
    section[data-testid="stSidebar"] hr { border-color: var(--border, #1e2235); }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-app, #0a0d14); }
    ::-webkit-scrollbar-thumb { background: var(--border, #1e2235); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-muted, #555); }

    @media (max-width: 768px) {
        .hero { padding: 1.5rem; }
        .hero h1 { font-size: 1.6rem; }
        .metric-value { font-size: 1.3rem; }
    }
</style>
"""


def _inject_css_variables() -> str:
    """Return a style block that declares CSS custom properties for the current theme."""
    c = get_colors()
    return f"""
<style>
    :root {{
        --bg-app: {c.BG_APP};
        --bg-header: {c.BG_APP};
        --bg-card-from: {c.CARD_GRADIENT_FROM};
        --bg-card-to: {c.CARD_GRADIENT_TO};
        --border: {c.BORDER};
        --primary: {c.PRIMARY};
        --success: {c.SUCCESS};
        --warning: {c.WARNING};
        --danger: {c.DANGER};
        --accent: {c.ACCENT};
        --text-primary: {c.TEXT_PRIMARY};
        --text-secondary: {c.TEXT_SECONDARY};
        --text-muted: {c.TEXT_MUTED};
        --select-bg: {c.SELECT_BG};
        --sidebar-bg: {c.SIDEBAR_BG};
        --hero-gradient: {c.HERO_GRADIENT};
        --grid-color: {c.GRID_COLOR};
    }}
</style>
"""


# ═══════════════════════════════════════════════════════════
#  Render functions
# ═══════════════════════════════════════════════════════════

def render_custom_css() -> None:
    """Inject the dashboard's theme-aware CSS into the page.

    Must be called after ``init_theme()`` so the current theme is known.
    Injects CSS variable definitions followed by the component CSS.
    """
    st.markdown(_inject_css_variables(), unsafe_allow_html=True)
    st.markdown(_build_theme_css(), unsafe_allow_html=True)


def section_header(title: str, emoji: str = "") -> None:
    """Render a styled section header with optional emoji."""
    st.markdown(
        f'<div class="section-header">{emoji} {title}</div>',
        unsafe_allow_html=True,
    )


def section_header_sm(title: str) -> None:
    """Render a smaller section header."""
    st.markdown(
        f'<div class="section-header-sm">{title}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    col: Any,
    value: str,
    label: str,
    delta: str | None = None,
    up: bool | None = None,
    color: str | None = None,
    help_text: str | None = None,
) -> None:
    """Render an animated metric card with optional delta indicator."""
    delta_class = ""
    delta_arrow = ""
    if up is True:
        delta_class = "up"
        delta_arrow = chr(9650)  # ▲
    elif up is False:
        delta_class = "down"
        delta_arrow = chr(9660)  # ▼
    elif delta and up is None:
        if delta.startswith("+"):
            delta_class = "up"
            delta_arrow = chr(9650)
        elif delta.startswith("-"):
            delta_class = "down"
            delta_arrow = chr(9660)
        else:
            delta_class = "neutral"

    style = f"color: {color};" if color else ""
    help_attr = f'title="{help_text}"' if help_text else ""
    delta_html = ""
    if delta:
        delta_html = (
            f'<div class="metric-delta {delta_class}">{delta_arrow} {delta}</div>'
        )

    col.markdown(
        f'<div class="metric-card" {help_attr}>'
        f'<div class="metric-value" style="{style}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(text: str, variant: Literal["green", "yellow", "red", "blue", "purple"] = "blue") -> None:
    """Render a status badge with colour coding."""
    st.markdown(
        f'<span class="badge badge-{variant}">{text}</span>',
        unsafe_allow_html=True,
    )


def status_dot(color: Literal["green", "yellow", "red", "blue"]) -> None:
    """Render a pulsing status dot."""
    st.markdown(
        f'<span class="status-dot {color}"></span>',
        unsafe_allow_html=True,
    )


def info_row(text: str, icon: str = "•") -> None:
    """Render a simple info row."""
    st.markdown(
        f'<div class="info-row">{icon} {text}</div>',
        unsafe_allow_html=True,
    )


def skeleton_loader(height: int = 80, count: int = 1) -> None:
    """Render shimmer skeleton loading placeholders."""
    for _ in range(count):
        st.markdown(
            f'<div class="skeleton" style="height:{height}px"></div>',
            unsafe_allow_html=True,
        )


def render_hero(
    title: str,
    subtitle: str,
    badges: list[tuple[str, str]] | None = None,
) -> None:
    """Render the main hero section."""
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)
        st.markdown(
            f'<p class="hero-sub">{subtitle}</p>',
            unsafe_allow_html=True,
        )
        if badges:
            badge_html = "".join(
                f'<span class="hero-badge">{icon} {label}</span>'
                for label, icon in badges
            )
            st.markdown(f'<div style="margin-top:0.3rem">{badge_html}</div>', unsafe_allow_html=True)
    with c2:
        from datetime import datetime
        st.markdown(
            f'<div style="text-align:right;padding-top:0.5rem">'
            f'<div style="font-size:0.72rem;color:var(--text-secondary,#6b7280)">'
            f'{datetime.now().strftime("%B %d, %Y · %H:%M")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)


def render_footer() -> None:
    """Render the standard page footer."""
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div style="color:var(--text-muted,#555);font-size:0.72rem">'
            'Football Prediction Dashboard</div>',
            unsafe_allow_html=True,
        )
    with c2:
        from datetime import datetime
        st.markdown(
            f'<div style="color:var(--text-muted,#555);font-size:0.72rem;text-align:center">'
            f'Last updated: {datetime.now().strftime("%H:%M:%S")}</div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div style="color:var(--text-muted,#555);font-size:0.72rem;text-align:right">'
            'Built with Streamlit + Plotly</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
#  Convenience exports
# ═══════════════════════════════════════════════════════════

class Colors:
    """Convenience access to the **current theme's** colours.
    
    Usage::
    
        c = Colors
        st.markdown(f'<div style="color:{c.PRIMARY}">...</div>')
    
    .. note::
        This is a dynamic proxy — it reads the theme from session state
        each time an attribute is accessed, so it always returns the
        active theme's colour.
    """
    @classmethod
    def _get_c(cls) -> Any:
        return get_colors()
    
    @classmethod
    @property
    def PRIMARY(cls) -> str: return cls._get_c().PRIMARY
    @classmethod
    @property
    def SUCCESS(cls) -> str: return cls._get_c().SUCCESS
    @classmethod
    @property
    def WARNING(cls) -> str: return cls._get_c().WARNING
    @classmethod
    @property
    def DANGER(cls) -> str: return cls._get_c().DANGER
    @classmethod
    @property
    def INFO(cls) -> str: return cls._get_c().INFO
    @classmethod
    @property
    def ACCENT(cls) -> str: return cls._get_c().ACCENT
    @classmethod
    @property
    def GRADIENT_GREEN(cls) -> str: return cls._get_c().GRADIENT_GREEN
    @classmethod
    @property
    def GRADIENT_BLUE(cls) -> str: return cls._get_c().GRADIENT_BLUE
    @classmethod
    @property
    def GRADIENT_GOLD(cls) -> str: return cls._get_c().GRADIENT_GOLD
    @classmethod
    @property
    def GRADIENT_RED(cls) -> str: return cls._get_c().GRADIENT_RED
    @classmethod
    @property
    def GRADIENT_PURPLE(cls) -> str: return cls._get_c().GRADIENT_PURPLE
    @classmethod
    @property
    def BG_CARD(cls) -> str: return cls._get_c().BG_CARD
    @classmethod
    @property
    def BORDER(cls) -> str: return cls._get_c().BORDER
    @classmethod
    @property
    def TEXT_PRIMARY(cls) -> str: return cls._get_c().TEXT_PRIMARY
    @classmethod
    @property
    def TEXT_SECONDARY(cls) -> str: return cls._get_c().TEXT_SECONDARY
    @classmethod
    @property
    def TEXT_MUTED(cls) -> str: return cls._get_c().TEXT_MUTED


# ═══════════════════════════════════════════════════════════
#  Plotly Chart Builders (theme-aware via get_plotly_layout)
# ═══════════════════════════════════════════════════════════

def _layout(**kw: Any) -> dict:
    """Return a base Plotly layout dict merged with the current theme."""
    layout = get_plotly_layout()
    layout.update(kw)
    return layout


def gauge_chart(
    col: Any,
    label: str,
    value: float,
    target: float = 50.0,
    unit: str = "",
    lower_better: bool = False,
    height: int = 200,
) -> None:
    """Render a Plotly gauge/indicator chart in the given column."""
    c = get_colors()
    if lower_better:
        met = value <= target
        color = c.SUCCESS if met else c.WARNING if value <= target * 1.5 else c.DANGER
    else:
        met = value >= target
        color = c.SUCCESS if met else c.WARNING if value >= target * 0.7 else c.DANGER

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        number={"suffix": unit, "font": {"color": color, "size": 28}, "valueformat": ".1f"},
        delta={
            "reference": target, "suffix": unit, "position": "bottom",
            "font": {"size": 11},
            "increasing": {"color": c.SUCCESS},
            "decreasing": {"color": c.DANGER},
        },
        gauge={
            "axis": {"range": [0, max(target * 2, value * 1.5, 1)],
                     "tickfont": {"color": c.TEXT_SECONDARY, "size": 9}},
            "bar": {"color": color, "thickness": 0.4},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [
                {"range": [0, target * 0.7], "color": f"rgba(76, 175, 80, 0.08)"},
                {"range": [target * 0.7, target * 1.3], "color": f"rgba(255, 193, 7, 0.08)"},
                {"range": [target * 1.3, max(target * 2, value * 1.5, 1)], "color": f"rgba(244, 67, 54, 0.08)"},
            ] if not lower_better else [
                {"range": [target * 1.3, max(target * 2, value * 1.5, 1)], "color": f"rgba(76, 175, 80, 0.08)"},
                {"range": [target * 0.7, target * 1.3], "color": f"rgba(255, 193, 7, 0.08)"},
                {"range": [0, target * 0.7], "color": f"rgba(244, 67, 54, 0.08)"},
            ],
            "threshold": {
                "line": {"color": c.TEXT_MUTED, "width": 2},
                "thickness": 0.6, "value": target,
            },
        },
    ))

    fig.update_layout(
        height=height, margin=dict(l=20, r=20, t=10, b=40),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": c.TEXT_PRIMARY},
    )
    fig.add_annotation(
        text=f"<b>{label}</b> &middot; Target: {target}{unit}",
        xref="paper", yref="paper", x=0.5, y=-0.05,
        showarrow=False, font={"color": c.TEXT_SECONDARY, "size": 10},
    )
    col.plotly_chart(fig, use_container_width=True)


def confusion_matrix_heatmap(
    cm: list[list[int]],
    labels: list[str] | None = None,
    title: str = "Confusion Matrix",
    height: int = 400,
) -> go.Figure:
    """Build a styled confusion matrix heatmap."""
    c = get_colors()
    if labels is None:
        labels = ["Away Win", "Draw", "Home Win"]

    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums > 0) * 100
    hover_text = [
        [f"Actual: {labels[i]}<br>Predicted: {labels[j]}<br>Count: {int(cm_arr[i][j])}<br>Pct: {cm_pct[i][j]:.1f}%"
         for j in range(len(labels))]
        for i in range(len(labels))
    ]

    fig = go.Figure(data=go.Heatmap(
        z=cm_arr, x=labels, y=labels,
        text=[[str(int(v)) for v in row] for row in cm_arr],
        texttemplate="%{text}", textfont={"color": "#fff", "size": 13},
        hovertemplate="%{customdata}<extra></extra>", customdata=hover_text,
        colorscale=[
            [0, "rgba(15, 20, 40, 0.9)"],
            [0.25, "rgba(25, 60, 120, 0.9)"],
            [0.5, "rgba(40, 100, 180, 0.9)"],
            [0.75, f"rgba(79, 195, 247, 0.85)"],
            [1, f"rgba(129, 199, 132, 0.9)"],
        ],
        showscale=True,
        colorbar=dict(title="Count", titleside="right",
                      tickfont={"color": c.TEXT_SECONDARY, "size": 9},
                      titlefont={"color": c.TEXT_SECONDARY, "size": 9}),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=c.TEXT_PRIMARY, size=14)),
        xaxis=dict(title="Predicted", titlefont=dict(color=c.TEXT_SECONDARY),
                   tickfont=dict(color=c.TEXT_PRIMARY)),
        yaxis=dict(title="Actual", titlefont=dict(color=c.TEXT_SECONDARY),
                   tickfont=dict(color=c.TEXT_PRIMARY), autorange="reversed"),
        height=height,
        **_layout(),
    )
    return fig


def feature_importance_chart(
    features: dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 20,
    height: int = 400,
) -> go.Figure:
    """Build a horizontal bar chart of feature importances."""
    c = get_colors()
    sorted_items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    max_val = max(abs(v) for v in values) if values else 1
    bar_colors = [
        f"rgba(79, 195, 247, {0.3 + 0.7 * abs(v) / max_val})"
        if v >= 0 else f"rgba(239, 154, 154, {0.3 + 0.7 * abs(v) / max_val})"
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=c.TEXT_PRIMARY, size=14)),
        xaxis=dict(title="Importance", titlefont=dict(color=c.TEXT_SECONDARY), gridcolor=c.GRID_COLOR),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        height=height, **_layout(),
    )
    return fig


def radar_chart(
    categories: list[str],
    values_dict: dict[str, list[float]],
    title: str = "Model Comparison",
    height: int = 400,
) -> go.Figure:
    """Build a radar/spider chart for multi-model comparison."""
    c = get_colors()
    fig = go.Figure()
    colors_list = [c.PRIMARY, c.SUCCESS, c.GRADIENT_GOLD, c.GRADIENT_PURPLE, c.DANGER]

    for i, (model_name, values) in enumerate(values_dict.items()):
        values_closed = values + [values[0]]
        categories_closed = categories + [categories[0]]
        color = colors_list[i % len(colors_list)]
        fig.add_trace(go.Scatterpolar(
            r=values_closed, theta=categories_closed, fill="toself",
            name=model_name, line=dict(color=color, width=2),
            fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.15)",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=c.TEXT_PRIMARY, size=14)),
        polar=dict(
            radialaxis=dict(visible=True,
                          range=[0, max(max(v) for v in values_dict.values()) * 1.1],
                          gridcolor=c.GRID_COLOR,
                          tickfont={"color": c.TEXT_SECONDARY, "size": 9}),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(font={"color": c.TEXT_PRIMARY}, orientation="h", y=-0.15),
        height=height, **_layout(),
    )
    return fig


def bankroll_growth_chart(
    history: list[float],
    initial: float | None = None,
    height: int = 350,
) -> go.Figure:
    """Build a styled bankroll growth area chart."""
    c = get_colors()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history, mode="lines", name="Bankroll",
        line=dict(color=c.PRIMARY, width=2.5),
        fill="tozeroy", fillcolor=f"rgba(79, 195, 247, 0.1)",
        hovertemplate="Bet %{x}<br>%{y:.2f}<extra></extra>",
    ))
    if initial is not None and len(history) > 0:
        fig.add_hline(y=initial, line_dash="dash", line_color=c.TEXT_MUTED, line_width=1,
                      annotation_text=f"Initial: {initial:,.0f}",
                      annotation_font=dict(color=c.TEXT_SECONDARY, size=10))
        fig.add_hrect(y0=initial, y1=max(history), fillcolor=c.SUCCESS, opacity=0.03, layer="below", line_width=0)
        fig.add_hrect(y0=min(history), y1=initial, fillcolor=c.DANGER, opacity=0.03, layer="below", line_width=0)
    fig.update_layout(
        xaxis=dict(title="Bet Sequence", gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        yaxis=dict(title="Bankroll", gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        height=height, hovermode="x unified", **_layout(),
    )
    return fig


def drawdown_chart(
    history: list[float],
    height: int = 200,
) -> go.Figure:
    """Build a drawdown chart showing peak-to-trough declines."""
    c = get_colors()
    peak = np.maximum.accumulate(history)
    drawdown_vals = (history - peak) / peak * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=drawdown_vals, mode="lines", name="Drawdown",
        line=dict(color=c.DANGER, width=1.5),
        fill="tozeroy", fillcolor="rgba(244, 67, 54, 0.1)",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(title="Drawdown (%)", gridcolor=c.GRID_COLOR, tickfont=dict(size=9), zerolinecolor=c.ZERO_LINE),
        xaxis=dict(gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        height=height, hovermode="x unified", **_layout(),
    )
    return fig


def area_trend_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Trend",
    color: str | None = None,
    height: int = 300,
    y_label: str | None = None,
) -> go.Figure:
    """Build a general-purpose area trend chart."""
    c = get_colors()
    color = color or c.PRIMARY
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[x_col], y=df[y_col],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(color=color, size=4),
        fill="tozeroy",
        fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4))}, 0.08)",
        hovertemplate="%{x}<br>%{y:.4f}<extra></extra>", name=title,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color=c.TEXT_PRIMARY, size=13)),
        xaxis=dict(gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        yaxis=dict(title=y_label or y_col, gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        height=height, hovermode="x unified", **_layout(),
    )
    return fig


def comparison_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "Comparison",
    color_col: str | None = None,
    color_scale: str = "Viridis",
    height: int = 350,
    y_label: str | None = None,
    horizontal: bool = True,
) -> go.Figure:
    """Build a bar chart for comparing entities (models, strategies, etc.)."""
    c = get_colors()
    if color_col is None:
        color_col = y_col

    if horizontal:
        fig = go.Figure()
        sorted_df = df.sort_values(y_col, ascending=True)
        vals = sorted_df[y_col].values
        names = sorted_df[x_col].values
        max_v = max(abs(v) for v in vals) if len(vals) > 0 else 1
        bar_colors = [
            f"rgba(79, 195, 247, {0.3 + 0.7 * abs(v) / max_v})"
            if v >= 0 else f"rgba(239, 154, 154, {0.3 + 0.7 * abs(v) / max_v})"
            for v in vals
        ]
        fig.add_trace(go.Bar(
            x=vals, y=names, orientation="h",
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
    else:
        fig = px.bar(
            df.sort_values(y_col, ascending=False), x=x_col, y=y_col,
            color=color_col if color_col != x_col else None,
            color_continuous_scale=color_scale, title=title,
        )
        fig.update_traces(hovertemplate="%{x}: %{y:.4f}<extra></extra>")

    fig.update_layout(
        title=dict(text=title, font=dict(color=c.TEXT_PRIMARY, size=13)),
        xaxis=dict(gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        yaxis=dict(title=y_label or y_col, gridcolor=c.GRID_COLOR, tickfont=dict(size=9)),
        height=height, **_layout(),
    )
    return fig
