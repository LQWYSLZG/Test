"""
Build script: merges sidebar navigation into the pages/ wrapper files.
Run: py build_pages.py
"""
import os, base64

root = os.path.dirname(os.path.abspath(__file__))

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode() if os.path.exists(path) else ""

suite_icon = b64(os.path.join(root, "Icon (Industrial Engineering Assistant Suite).png"))
inv_icon   = b64(os.path.join(root, "Inventory Assistant for Supply Chain",
                               "Icon (Inventory Assistant for Supply Chain).png"))
ops_icon   = b64(os.path.join(root, "Operations Assistant for Manufacturing",
                               "Icon (Operations Assistant for Manufacturing).png"))

def make_sidebar_prefix(active_app):
    suite_img = f'<img src="data:image/png;base64,{suite_icon}" style="height:48px; opacity:0.9;">' if suite_icon else "🏭"
    inv_img   = f'<img src="data:image/png;base64,{inv_icon}" style="height:18px; vertical-align:middle; margin-right:8px;">' if inv_icon else "📦 "
    ops_img   = f'<img src="data:image/png;base64,{ops_icon}" style="height:18px; vertical-align:middle; margin-right:8px;">' if ops_icon else "⚙️ "

    inv_class = "active-app" if active_app == "inventory" else ""
    ops_class = "active-app" if active_app == "operations" else ""

    app_title = "Inventory Assistant for Supply Chain" if active_app == "inventory" else "Operations Assistant for Manufacturing"
    icon_file = "Icon (Inventory Assistant for Supply Chain).png" if active_app == "inventory" else "Icon (Operations Assistant for Manufacturing).png"
    app_folder = "Inventory Assistant for Supply Chain" if active_app == "inventory" else "Operations Assistant for Manufacturing"

    return f'''# ── Suite sidebar navigation (injected by build_pages.py) ──────────────────
import streamlit as _suite_st
import os as _suite_os
import base64 as _suite_b64

# Page config — must be first Streamlit call
_root_dir = _suite_os.path.dirname(_suite_os.path.dirname(_suite_os.path.abspath(__file__)))
_page_icon_path = _suite_os.path.join(_root_dir, "{app_folder}", "{icon_file}")
_suite_st.set_page_config(
    page_title="{app_title}",
    page_icon=_page_icon_path if _suite_os.path.exists(_page_icon_path) else "🏭",
    layout="wide",
)

# Session state isolation
_page_key = "_{active_app}_page_visited"
if _page_key not in _suite_st.session_state:
    _keys = [k for k in _suite_st.session_state.keys() if k != _page_key]
    for _k in _keys:
        del _suite_st.session_state[_k]
    _suite_st.session_state[_page_key] = True

_suite_st.markdown("""
<style>
    [data-testid="stSidebar"]        {{ display: none !important; visibility: hidden !important; }}
    [data-testid="collapsedControl"] {{ display: none !important; visibility: hidden !important; opacity: 0 !important; }}
    section[data-testid="stSidebar"] {{ display: none !important; }}
    #suite-toggle {{ display: none; }}
    .suite-tab {{
        position: fixed; left: 0; top: 50vh; transform: translateY(-50%);
        z-index: 9999; background: #1a1a3a; border: 1px solid #d4a017;
        border-left: none; border-radius: 0 8px 8px 0;
        width: 20px; height: 52px; display: flex; align-items: center;
        justify-content: center; cursor: pointer; color: #d4a017;
        font-size: 0.85rem; user-select: none;
        transition: left 0.25s ease, background 0.2s;
    }}
    .suite-tab:hover {{ background: #2a2e8e; }}
    .suite-panel {{
        position: fixed; left: -230px; top: 0; width: 230px; height: 100vh;
        background: #08091a; border-right: 2px solid #3d4aae;
        z-index: 9998; transition: left 0.25s ease;
        padding: 80px 18px 24px 18px; overflow-y: auto;
        box-shadow: 4px 0 16px rgba(0,0,0,0.5);
    }}
    #suite-toggle:checked ~ .suite-panel {{ left: 0; }}
    #suite-toggle:checked ~ .suite-tab   {{ left: 230px; }}
    .suite-panel .icon-home {{ display: flex; justify-content: center; margin-bottom: 20px; }}
    .suite-panel a {{
        display: block; padding: 8px 12px; border-radius: 6px;
        color: #a8b8e8; text-decoration: none; font-size: 0.87rem;
        margin-bottom: 4px; transition: background 0.15s, color 0.15s;
    }}
    .suite-panel a:hover   {{ background: #1a1a3a; color: #ffffff; }}
    .suite-panel .active-app {{ background: #1a1a3a; color: #d4a017 !important; font-weight: 700; cursor: default; }}
    .suite-panel hr {{ border: none; border-top: 1px solid #2a3a6a; margin: 12px 0; }}
    .suite-panel .section-label {{
        color: #5a6aaa; font-size: 0.7rem; letter-spacing: 0.8px;
        text-transform: uppercase; margin: 8px 0 6px 0; padding: 0 4px;
    }}
    .suite-panel .coming {{ color: #3a4a8a; font-size: 0.82rem; padding: 6px 12px; }}
    .suite-footer {{ margin-top: 24px; color: #2a3a7a; font-size: 0.7rem; text-align: center; }}
</style>
<input type="checkbox" id="suite-toggle">
<label class="suite-tab" for="suite-toggle">&#9776;</label>
<div class="suite-panel">
    <div class="icon-home"><a href="/" title="Main Menu">{suite_img}</a></div>
    <div class="section-label">APP MENU</div>
    <a class="{inv_class}" href="/inventory_assistant">{inv_img}Inventory Assistant</a>
    <a class="{ops_class}" href="/operations_assistant">{ops_img}Operations Assistant</a>
    <hr>
    <div class="section-label">Coming Soon</div>
    <div class="coming">🔒 Production Assistant</div>
    <div class="coming">🔒 Quality Assistant</div>
    <div class="suite-footer">© 2026 Dyson Ye</div>
</div>
""", unsafe_allow_html=True)
del _suite_st, _suite_os, _suite_b64, _root_dir, _page_icon_path, _page_key

# ── App code follows ──────────────────────────────────────────────────────────
'''

# ── Build inventory_assistant.py ──────────────────────────────────────────────
inv_app = open(os.path.join(root, "Inventory Assistant for Supply Chain", "app.py"),
               encoding="utf-8-sig").read()

sidebar_inv = make_sidebar_prefix("inventory")
full_inv = sidebar_inv + inv_app

output_inv = os.path.join(root, "pages", "inventory_assistant.py")
with open(output_inv, "w", encoding="utf-8") as f:
    f.write(full_inv)
print(f"Written: {output_inv} ({len(full_inv):,} chars)")

# ── Build operations_assistant.py ─────────────────────────────────────────────
ops_app = open(os.path.join(root, "Operations Assistant for Manufacturing", "app.py"),
               encoding="utf-8-sig").read()

sidebar_ops = make_sidebar_prefix("operations")
full_ops = sidebar_ops + ops_app

output_ops = os.path.join(root, "pages", "operations_assistant.py")
with open(output_ops, "w", encoding="utf-8") as f:
    f.write(full_ops)
print(f"Written: {output_ops} ({len(full_ops):,} chars)")

print("Done.")
