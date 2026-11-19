"""
Dashboard Utilities Module for EduPredict AI.
Handles CSS styling injections, custom layout KPI cards,
pagination slicing logic, and download encoders for CSV/Excel data files.
"""

import streamlit as st
import pandas as pd
import io
import os
from . import config

def inject_custom_css():
    """
    Injects custom styles for a modern, institutional look.
    Loads and renders layout sheets from assets/styles.css.
    """
    css_path = os.path.join(config.BASE_DIR, "assets", "styles.css")
    if os.path.exists(css_path):
        try:
            with open(css_path, "r", encoding="utf-8") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Failed to load custom styling stylesheet: {e}")
    else:
        st.warning("Custom stylesheet not found at: assets/styles.css")

def metric_card(label, value, subtitle="", accent="hsl(221,83%,55%)"):
    """
    Returns HTML code for a premium dashboard metric card.
    """
    return f"""
    <div class="metric-card" style="border-top-color:{accent};">
        <div class="mc-label">{label}</div>
        <div class="mc-value">{value}</div>
        <div class="mc-sub">{subtitle}</div>
    </div>"""

def progress_bar_html(percentage, color="hsl(221,83%,55%)"):
    """
    Returns HTML code for a progress bar.
    """
    pct = max(0.0, min(100.0, float(percentage)))
    return f"""
    <div class="progress-container">
        <div class="progress-fill" style="width:{pct:.1f}%; background:{color};"></div>
    </div>"""

def render_pagination(total_rows, page_key, rows_per_page=100):
    """
    Renders visual pagination controller in Streamlit.
    Returns (start_idx, end_idx, current_page).
    """
    total_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    
    col_sel, col_btn_prev, col_info, col_btn_next = st.columns([1.5, 1, 3, 1])
    
    with col_sel:
        current_page = st.number_input(
            "Go to Page", min_value=1, max_value=total_pages, value=1, step=1, key=page_key
        )
        
    start_idx = (current_page - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)
    
    with col_info:
        st.markdown(
            f"<div style='margin-top: 30px; text-align: center;'><small style='opacity:.75; font-size: 0.88rem;'>"
            f"Showing records <strong>{start_idx+1:,} - {end_idx:,}</strong> of <strong>{total_rows:,}</strong> "
            f"| Page {current_page} of {total_pages}</small></div>",
            unsafe_allow_html=True
        )
        
    return start_idx, end_idx, current_page

def export_to_csv(df):
    """Encodes a DataFrame as CSV bytes."""
    return df.to_csv(index=False).encode('utf-8')

def export_to_excel(df):
    """Encodes a DataFrame as Excel bytes. Falls back to CSV on missing library."""
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name="EduPredict_Predictions")
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    except Exception:
        # Fallback to plain CSV encoding but flag it
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        return csv_bytes, "text/csv", "csv"

def export_to_markdown(df):
    """Returns markdown string of the dataframe."""
    return df.to_markdown(index=False)

def render_topbar(model_bundle, academic_df=None, active_override=None):
    model_name = "Loading..."
    f1_score   = 0.0
    n_students = 0

    try:
        model_name = (
            active_override or
            model_bundle.get("model_type", "N/A")
        )
        f1_score = model_bundle.get(
            "metrics", {}
        ).get("F1-Score", 0.0)
    except Exception:
        pass

    try:
        n_students = len(academic_df) \
            if academic_df is not None else 0
    except Exception:
        pass

    import streamlit as st
    st.markdown(f"""
    <div style="
        position:fixed; top:0; left:0; right:0;
        height:52px;
        background:linear-gradient(
            90deg,#1a237e 0%,#1565c0 100%
        );
        display:flex; align-items:center;
        justify-content:space-between;
        padding:0 24px; padding-left:60px; z-index:9999;
        box-shadow:0 2px 12px rgba(0,0,0,0.18);
    ">
        <div style="display:flex;
                    align-items:center; gap:10px;">
            <span style="font-size:1.3rem;">🎓</span>
            <span style="color:white; font-weight:700;
                         font-size:1rem;">
                EduPredict AI
            </span>
            <span style="color:rgba(255,255,255,0.45);
                         font-size:0.72rem;
                         letter-spacing:0.06em;
                         text-transform:uppercase;">
                Student Performance Predictor
            </span>
        </div>
        <div style="display:flex; gap:16px;
                    color:rgba(255,255,255,0.75);
                    font-size:0.8rem;
                    align-items:center;">
            <span>🤖 {model_name}</span>
            <span style="opacity:0.4;">|</span>
            <span>📊 {n_students:,} Students</span>
            <span style="opacity:0.4;">|</span>
            <span style="
                background:rgba(255,255,255,0.15);
                padding:3px 10px;
                border-radius:12px;
                color:white; font-weight:600;
            ">F1: {f1_score:.3f}</span>
        </div>
    </div>
    <div style="height:60px;"></div>
    """, unsafe_allow_html=True)

def render_footer(model_bundle):
    model_name = "Loading..."
    f1_score   = 0.0

    try:
        model_name = (
            model_bundle.get("model_type", "N/A") 
            if model_bundle else "N/A"
        )
        f1_score = (
            model_bundle.get("metrics", {}).get("F1-Score", 0.0) 
            if model_bundle else 0.0
        )
    except Exception:
        pass

    import streamlit as st
    st.markdown(f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 36px;
        background: linear-gradient(
            90deg, #0d1117 0%, #161b22 100%
        );
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        z-index: 9999;
        border-top: 1px solid rgba(255,255,255,0.06);
        font-size: 0.72rem;
        color: rgba(255,255,255,0.45);
    ">
        <div>
            🎓 EduPredict AI &nbsp;·&nbsp;
            Academic Intelligence Platform
        </div>
        <div style="display:flex; gap:20px;">
            <span>
                Model: 
                <strong style="color:rgba(255,255,255,0.7)">
                    {model_name}
                </strong>
            </span>
            <span>
                F1: 
                <strong style="color:rgba(255,255,255,0.7)">
                    {f1_score:.3f}
                </strong>
            </span>
            <span>© 2025</span>
        </div>
    </div>
    <!-- Push content above fixed footer -->
    <div style="height:44px;"></div>
    """, unsafe_allow_html=True)
