import streamlit as st
import os

_suite_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_suite_dir)  # ensure working dir is always the suite root
_icon_path = os.path.join(_suite_dir, "Icon (Industrial Engineering Assistant Suite).png")

st.set_page_config(
    page_title="Industrial Engineering Assistant Suite",
    page_icon=_icon_path,
    layout="wide",
)

# Load icons as base64 for header and app cards
import base64

def _load_b64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

_icon_b64     = _load_b64(_icon_path)
_inv_icon_b64 = _load_b64(os.path.join(_suite_dir, "Inventory Assistant for Supply Chain",
                                        "Icon (Inventory Assistant for Supply Chain).png"))
_ops_icon_b64 = _load_b64(os.path.join(_suite_dir, "Operations Assistant for Manufacturing",
                                        "Icon (Operations Assistant for Manufacturing).png"))

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    footer                           { display: none !important; }
    #MainMenu                        { display: none !important; }
    [data-testid="stSidebar"]        { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    .main .block-container           { padding-top: 2rem; padding-bottom: 2rem; }

    /* ── Header ── */
    .suite-header {
        background: linear-gradient(135deg, #0e1a3a 0%, #2e1f6e 35%, #1a3a6e 65%, #0e1a3a 100%);
        border-top: 6px solid #d4a017;
        border-radius: 12px;
        padding: 36px 40px 30px 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .suite-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 8px 0;
        letter-spacing: 0.5px;
    }
    .suite-header .tagline {
        color: #d4a017;
        font-size: 0.88rem;
        margin: 0;
        font-weight: 500;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }
    .suite-header .subtitle {
        color: #a8b8e8;
        font-size: 0.85rem;
        margin-top: 12px;
        letter-spacing: 2.5px;
        font-weight: 500;
        text-transform: uppercase;
        opacity: 0.75;
    }

    /* ── Hierarchy bar ── */
    .hierarchy-bar {
        background: #0e1428;
        border: 1px solid #3d4aae;
        border-radius: 8px;
        padding: 14px 24px;
        margin-bottom: 28px;
        text-align: center;
        color: #a8b8e8;
        font-size: 0.88rem;
        letter-spacing: 0.3px;
    }

    /* ── Active app card ── */
    .app-card {
        background: #0e1428;
        border: 1px solid #3d4aae;
        border-top: 4px solid #d4a017;
        border-radius: 10px;
        padding: 18px 14px;
        height: 300px;
        min-height: 300px;
        box-sizing: border-box;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .app-card h3 {
        color: #d4a017;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 10px 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .app-card h3 img { flex-shrink: 0; }
    .app-card .card-icon { display: block; margin-bottom: 4px; }
    .app-card .app-tag { display: none; }
    .app-card p { color: #a8b8e8; font-size: 0.86rem; line-height: 1.55; margin: 0 0 12px 0; }
    .app-card .features {
        color: #a8b8e8;
        font-size: 0.81rem;
        line-height: 1.65;
        display: inline-block;
        text-align: left;
    }

    /* ── Planned app card ── */
    .app-card-planned {
        background: #080e1e;
        border: 1px solid #141e40;
        border-top: 4px solid #3d4aae;
        border-radius: 10px;
        padding: 18px 14px;
        height: 300px;
        min-height: 300px;
        box-sizing: border-box;
        opacity: 0.55;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .app-card-planned h3 { color: #5a6aaa; font-size: 1.1rem; font-weight: 700; margin: 0 0 10px 0; display: flex; align-items: center; gap: 8px; }
    .app-card-planned .app-tag { display: none; }
    .app-card-planned p { color: #2a3a70; font-size: 0.86rem; line-height: 1.55; margin: 0 0 12px 0; }
    .app-card-planned .features { color: #1a2850; font-size: 0.81rem; line-height: 1.65; }
    .coming-soon-badge {
        background: #141e40;
        color: #4a5a9a;
        border: 1px solid #1a2850;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        display: inline-block;
        margin-top: 8px;
    }

    /* ── Buttons ── */
    [data-testid="stButton"] > button {
        background: #2a2e8e !important;
        color: #d4a017 !important;
        border: 1px solid #d4a017 !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    [data-testid="stButton"] > button:hover {
        background: #3a40ae !important;
        color: #e8c840 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Custom sliding sidebar for suite menu ─────────────────────────────────────
_sidebar_suite_img = f'<img src="data:image/png;base64,{_icon_b64}" style="height:48px; opacity:0.9;">' if _icon_b64 else "🏭"
_sidebar_inv_img   = f'<img src="data:image/png;base64,{_inv_icon_b64}" style="height:18px; vertical-align:middle; margin-right:8px;">' if _inv_icon_b64 else "📦 "
_sidebar_ops_img   = f'<img src="data:image/png;base64,{_ops_icon_b64}" style="height:18px; vertical-align:middle; margin-right:8px;">' if _ops_icon_b64 else "⚙️ "

st.markdown(f"""
<style>
    #suite-toggle {{ display: none; }}
    .suite-tab {{
        position: fixed;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        z-index: 9999;
        background: #1a1a3a;
        border: 1px solid #d4a017;
        border-left: none;
        border-radius: 0 8px 8px 0;
        width: 20px;
        height: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        color: #d4a017;
        font-size: 0.85rem;
        user-select: none;
        transition: left 0.25s ease, background 0.2s;
    }}
    .suite-tab:hover {{ background: #2a2e8e; }}
    .suite-panel {{
        position: fixed;
        left: -230px;
        top: 0;
        width: 230px;
        height: 100vh;
        background: #08091a;
        border-right: 2px solid #3d4aae;
        z-index: 9998;
        transition: left 0.25s ease;
        padding: 80px 18px 24px 18px;
        overflow-y: auto;
        box-shadow: 4px 0 16px rgba(0,0,0,0.5);
    }}
    #suite-toggle:checked ~ .suite-panel {{ left: 0; }}
    #suite-toggle:checked ~ .suite-tab   {{ left: 230px; }}
    .suite-panel .icon-home {{
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }}
    .suite-panel a {{
        display: block;
        padding: 8px 12px;
        border-radius: 6px;
        color: #a8b8e8;
        text-decoration: none;
        font-size: 0.87rem;
        margin-bottom: 4px;
        transition: background 0.15s, color 0.15s;
    }}
    .suite-panel a:hover   {{ background: #1a1a3a; color: #ffffff; }}
    .suite-panel hr {{ border: none; border-top: 1px solid #2a3a6a; margin: 12px 0; }}
    .suite-panel .section-label {{
        color: #5a6aaa;
        font-size: 0.7rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin: 8px 0 6px 0;
        padding: 0 4px;
    }}
    .suite-panel .coming {{
        color: #3a4a8a;
        font-size: 0.82rem;
        padding: 6px 12px;
    }}
    .suite-footer {{
        margin-top: 24px;
        color: #2a3a7a;
        font-size: 0.7rem;
        text-align: center;
    }}
</style>
<input type="checkbox" id="suite-toggle">
<label class="suite-tab" for="suite-toggle">&#9776;</label>
<div class="suite-panel">
    <div class="icon-home">
        {_sidebar_suite_img}
    </div>
    <div class="section-label">APP MENU</div>
    <a href="/inventory_assistant">{_sidebar_inv_img}Inventory Assistant</a>
    <a href="/operations_assistant">{_sidebar_ops_img}Operations Assistant</a>
    <hr>
    <div class="section-label">Coming Soon</div>
    <div class="coming">🔒 Production Assistant</div>
    <div class="coming">🔒 Quality Assistant</div>
    <div class="suite-footer">© 2026 Dyson Ye</div>
</div>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
if _icon_b64:
    st.markdown(f"""
<div class="suite-header">
    <div style="display:inline-flex; align-items:center; justify-content:center; gap:16px; margin-bottom:8px;">
        <img src="data:image/png;base64,{_icon_b64}" style="height:60px; vertical-align:middle;">
        <h1 style="margin:0;">Industrial Engineering Assistant Suite</h1>
    </div>
    <p class="tagline">INVENTORY · PRODUCTION · OPERATIONS · QUALITY</p>
    <p class="subtitle">BUILT FOR ENGINEERS</p>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="suite-header">
    <h1>🏭 Industrial Engineering Assistant Suite</h1>
    <p class="tagline">INVENTORY · PRODUCTION · OPERATIONS · QUALITY</p>
    <p class="subtitle">BUILT FOR ENGINEERS</p>
</div>
""", unsafe_allow_html=True)

# ── S&OP hierarchy strip ───────────────────────────────────────────────────────
st.markdown("""
<div class="hierarchy-bar">
    Each app targets a different layer of your operation. Select where you need it most to get started.
</div>
""", unsafe_allow_html=True)

# ── App cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    _inv_img = f'<img src="data:image/png;base64,{_inv_icon_b64}" style="height:28px; margin-right:10px; vertical-align:middle;">' if _inv_icon_b64 else ""
    st.markdown(f"""
<div class="app-card">
    <h3>{_inv_img}Inventory Assistant for Supply Chain</h3>
    <p style="font-size:0.78rem; letter-spacing:1.2px; text-transform:uppercase; color:#a8b8e8; margin-bottom:12px;">FORECAST DEMAND · OPTIMISE STOCK · REDUCE WASTE</p>
    <p class="features">
        ✦ ABC Inventory Analysis<br>
        ✦ Customer Segmentation<br>
        ✦ Demand Forecast<br>
        ✦ Demand Health<br>
        ✦ Stock Control &amp; Optimisation
    </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if st.button("Open Inventory Assistant →", key="btn_app1", use_container_width=True, type="primary"):
        st.switch_page("pages/inventory_assistant.py")

with col2:
    _ops_img = f'<img src="data:image/png;base64,{_ops_icon_b64}" style="height:28px; margin-right:10px; vertical-align:middle;">' if _ops_icon_b64 else ""
    st.markdown(f"""
<div class="app-card">
    <h3>{_ops_img}Operations Assistant for Manufacturing</h3>
    <p style="font-size:0.78rem; letter-spacing:1.2px; text-transform:uppercase; color:#a8b8e8; margin-bottom:12px;">BALANCE LINES · DIAGNOSE MACHINES<br>· ELIMINATE BOTTLENECKS</p>
    <p class="features">
        ✦ Current State Analysis<br>
        ✦ Line Balancing Simulation<br>
        ✦ Machine Health Diagnostics<br>
        ✦ Discrete Event Simulation
    </p>
</div>
""", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
    if st.button("Open Operations Assistant →", key="btn_app2", use_container_width=True, type="primary"):
        st.switch_page("pages/operations_assistant.py")

with col3:
    st.markdown("""
<div class="app-card-planned">
    <h3>🔒 Production Assistant for Manufacturing</h3>
    <span class="coming-soon-badge" style="margin-top:16px;">COMING SOON</span>
</div>
""", unsafe_allow_html=True)

with col4:
    st.markdown("""
<div class="app-card-planned">
    <h3>🔒 Quality Assistant for Manufacturing</h3>
    <span class="coming-soon-badge" style="margin-top:16px;">COMING SOON</span>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top: 40px;
    padding: 16px 24px;
    border-top: 1px solid #2a3a8e;
    text-align: center;
    color: #4a5a9a;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
">
    © 2026 Dyson Ye · Industrial Engineering Assistant Suite · Built with Streamlit · Data is not stored or shared
</div>
""", unsafe_allow_html=True)
