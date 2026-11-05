import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import io
import math

# ── ML / Stats imports ────────────────────────────────────────────────────────
try:
    from scipy.stats import weibull_min
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import pulp
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
import os
# Note: set_page_config is handled by the suite wrapper when running inside the suite.
# When running standalone, uncomment the lines below:
# st.set_page_config(
#     page_title="Operations Assistant for Manufacturing",
#     page_icon=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Icon (Operations Assistant for Manufacturing).png"),
#     layout="wide",
# )

# ── Color theme: Steel Blue + Gold ───────────────────────────────────────────
# Primary dark:   #1a2a3a   (deep navy-steel)
# Primary mid:    #2d5a8e   (steel blue)
# Accent gold:    #d4a017   (same gold as App 1, suite unifier)
# Light bg:       #f0f4f8   (cool off-white)
# Border light:   #b0c8e0   (light steel)

st.markdown("""
<style>
    /* ── Global font ── */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }

    /* ── Hide Streamlit chrome ── */
    [data-testid="stToolbar"]          { display: none !important; }
    [data-testid="stDecoration"]       { display: none !important; }
    footer                             { display: none !important; }
    #MainMenu                          { display: none !important; }

    /* ── Prevent layout shift on rerun ── */
    .main .block-container             { transition: none !important; animation: none !important; }
    [data-testid="stAppViewContainer"] { transition: none !important; }
    iframe                             { transition: none !important; }

    /* ── Hide sidebar ── */
    [data-testid="stSidebar"]          { display: none; }

    /* ── Header banner ── */
    .ops-header {
        background: linear-gradient(135deg, #1a2a3a 0%, #2d5a8e 60%, #1a2a3a 100%);
        border-top: 6px solid #d4a017;
        border-radius: 12px;
        padding: 32px 40px 28px 40px;
        margin-bottom: 24px;
        text-align: center;
        min-height: 130px;
    }
    .ops-header .title-row {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 16px;
        margin-bottom: 8px;
    }
    .ops-header h1 {
        color: #ffffff;
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.5px;
        line-height: 1.2;
        display: inline;
    }
    .ops-header p {
        color: #d4a017;
        font-size: 0.9rem;
        margin: 0;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        border-bottom: 2px solid #d4a017;
        gap: 4px;
    }
    [data-testid="stTabs"] [role="tab"] {
        background: #eaf0f8;
        border-radius: 6px 6px 0 0;
        color: #1a2a3a;
        font-weight: 600;
        padding: 6px 16px;
        border: none;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: #1a2a3a;
        color: #d4a017 !important;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #b0c8e0;
        border-top: 3px solid #d4a017;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }

    /* ── Status badges ── */
    .badge-ok         { background:#e6f0fa; color:#1a2a4a; padding:2px 8px; border-radius:4px; font-weight:700; border:1px solid #90b8e0; }
    .badge-warn       { background:#fdf4e0; color:#6b4f00; padding:2px 8px; border-radius:4px; font-weight:700; border:1px solid #ddc070; }
    .badge-critical   { background:#fdf0f0; color:#8c2020; padding:2px 8px; border-radius:4px; font-weight:700; border:1px solid #e0a8a8; }

    /* ── Primary buttons ── */
    [data-testid="stButton"] > button[kind="primary"] {
        background: #1a2a3a;
        color: #d4a017;
        border: 1px solid #d4a017;
        font-weight: 600;
        border-radius: 6px;
    }
    [data-testid="stButton"] > button[kind="primary"]:hover {
        background: #2d5a8e;
        color: #e8b830;
    }

    /* ── Run Analysis button ── */
    [data-testid="stButton"][data-key="run_analysis_btn"] > button {
        background: #d4a017 !important;
        color: #0d1a2a !important;
        border: none !important;
        font-weight: 900 !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        padding: 10px 0 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    [data-testid="stButton"][data-key="run_analysis_btn"] > button:hover {
        background: #e8b830 !important;
        color: #0d1a2a !important;
    }

    /* ── Metric cards equal spacing ── */
    [data-testid="stMetric"] {
        padding: 12px 16px !important;
    }

    /* ── Subheader gold left-border accent (subtitles only, not main tab headers) ── */
    [data-testid="stHeadingWithActionElements"]:has(h3) {
        border-left: 4px solid #d4a017;
        padding-left: 10px;
        margin-top: 1.4rem !important;
        margin-bottom: 0.8rem !important;
    }

    /* ── Dataframe header ── */
    [data-testid="stDataFrame"] thead tr th {
        background-color: #1a2a3a !important;
        color: #d4a017 !important;
    }
</style>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

STEEL_BLUE  = "#2d5a8e"
NAVY        = "#1a2a3a"
GOLD        = "#d4a017"
LIGHT_BLUE  = "#6aaed6"
ORANGE      = "#e07b39"
GREEN       = "#3d8b5e"
RED         = "#c0392b"
GREY        = "#95a5a6"

DOWNTIME_CATEGORIES = [
    "Mechanical Failure", "Electrical Failure", "Setup / Changeover",
    "Material Shortage", "Quality Issue", "Planned Maintenance",
    "Operator Absence", "Other",
]

# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()

def render_chart(fig) -> None:
    """Render a matplotlib figure as a high-res PNG via st.image to avoid Streamlit's lossy compression."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)

def fmt_time(seconds: float) -> str:
    """Format seconds to a human-readable string."""
    if seconds >= 60:
        return f"{seconds/60:.2f} min"
    return f"{seconds:.1f} s"

# ── Takt time ─────────────────────────────────────────────────────────────────
def calc_takt_time(available_seconds: float, demand_units: float) -> float | None:
    """Takt time = available production time / customer demand."""
    if demand_units <= 0:
        return None
    return available_seconds / demand_units

# ── OEE ───────────────────────────────────────────────────────────────────────
def calc_oee(availability: float, performance: float, quality: float) -> float:
    """OEE = Availability × Performance × Quality (all as 0–1 fractions)."""
    return availability * performance * quality

def oee_rating(oee: float) -> tuple[str, str]:
    """Return (label, color) based on OEE value."""
    if oee >= 0.85:
        return "World Class", GREEN
    elif oee >= 0.60:
        return "Acceptable", GOLD
    else:
        return "Needs Improvement", RED

# ── Line balancing ────────────────────────────────────────────────────────────
def compute_line_balance(df: pd.DataFrame, takt: float | None) -> pd.DataFrame:
    """
    Given a time study df with columns [Workstation, Operation, Cycle Time (s)],
    aggregate by workstation and compute:
      - Total Cycle Time
      - % of Takt (if takt provided)
      - Idle Time (if takt provided)
      - Balance Efficiency
    """
    # Average CT per operation first, then sum per workstation
    # This prevents multiple readings of the same operation from inflating totals
    op_avg = (df.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                .mean()
                .reset_index())
    agg = (op_avg.groupby("Workstation")["Cycle Time (s)"]
                 .sum()
                 .reset_index()
                 .rename(columns={"Cycle Time (s)": "Total CT (s)"}))
    agg["Total CT (min)"] = (agg["Total CT (s)"] / 60).round(3)

    if takt and takt > 0:
        agg["Takt Time (s)"]   = round(takt, 2)
        agg["% of Takt"]       = (agg["Total CT (s)"] / takt * 100).round(1)
        agg["Idle Time (s)"]   = (takt - agg["Total CT (s)"]).clip(lower=0).round(2)
        agg["Bottleneck"]      = agg["Total CT (s)"] >= takt
        # Balance efficiency = sum of all CT / (n stations × takt)
        n = len(agg)
        be = agg["Total CT (s)"].sum() / (n * takt) * 100
        agg["Balance Efficiency (%)"] = round(be, 1)
    return agg.sort_values("Total CT (s)", ascending=False).reset_index(drop=True)

def detect_bottleneck(balance_df: pd.DataFrame) -> str | None:
    """Returns the single worst bottleneck (highest CT station)."""
    if balance_df.empty:
        return None
    return balance_df.iloc[0]["Workstation"]

def detect_all_bottlenecks(balance_df: pd.DataFrame, takt: float | None) -> list:
    """Returns all stations at or over takt time, sorted worst first."""
    if balance_df.empty or not takt:
        return []
    return balance_df[balance_df["Total CT (s)"] >= takt]["Workstation"].tolist()

# ── Throughput ────────────────────────────────────────────────────────────────
def calc_throughput(good_units: float, time_hours: float) -> float | None:
    if time_hours <= 0:
        return None
    return good_units / time_hours

# ═════════════════════════════════════════════════════════════════════════════
# APP ICON
# ═════════════════════════════════════════════════════════════════════════════
import base64, os as _os
_icon_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in dir() else _os.getcwd(),
    "Icon (Operations Assistant for Manufacturing).png"
)
_icon_b64 = ""
if _os.path.exists(_icon_path):
    with open(_icon_path, "rb") as _f:
        _icon_b64 = base64.b64encode(_f.read()).decode()

if "icon_b64" not in st.session_state:
    st.session_state.icon_b64 = _icon_b64
_icon_b64 = st.session_state.icon_b64

# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
if _icon_b64:
    st.markdown(f"""
<div class="ops-header">
    <div class="title-row">
        <img src="data:image/png;base64,{_icon_b64}" style="height:64px; vertical-align:middle; margin-right:18px;">
        <h1>Operations Assistant for Manufacturing</h1>
    </div>
    <p>BALANCE LINES · DIAGNOSE MACHINES · ELIMINATE BOTTLENECKS</p>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="ops-header">
    <h1>🏭 Operations Assistant for Manufacturing</h1>
    <p>BALANCE LINES · DIAGNOSE MACHINES · ELIMINATE BOTTLENECKS</p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# UPLOAD FLOW
# ═════════════════════════════════════════════════════════════════════════════
if "app_running" not in st.session_state:
    st.session_state.app_running = False

if not st.session_state.app_running:
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        with st.expander("💡 Required & Optional Information for Excel File(s)", expanded=False):
            st.markdown("""
**Time Study File** (used in Current State Analysis and Line Balancing Simulation):

| Column | Required | Notes |
|---|---|---|
| `Workstation` | ✅ | Name or ID of each workstation on the line |
| `Operation` | ✅ | Name of the individual task or operation |
| `Cycle Time (s)` | ✅ | Time in seconds to complete the operation once |
| `Min CT (s)` | Optional | Fastest observed time for this operation. Together with Max CT, enables Monte Carlo variability simulation in the Line Balancing Simulation |
| `Max CT (s)` | Optional | Slowest observed time for this operation. Together with Min CT, enables Monte Carlo variability simulation in the Line Balancing Simulation |
| `Product` | Optional | Product or part number. Enables the Product filter in Current State Analysis and the Product Mix control in the Required Setup panel |
| `Line` | Optional | Production line name or ID. When present, a line selector appears at the top of the app to filter all analysis and simulation to a single line |
| `Operator` | Optional | Operator ID or name. Enables the Cycle Time by Operator section in Current State Analysis |
| `Queue Time (s)` | Optional | Time a unit waits in the buffer after this station before moving to the next. One value per station. Auto-populates the queue time inputs in Waste Analysis |
| `Date` | Optional | Date the reading was taken. Enables the Cycle Time Trend Over Time section in Current State Analysis |

---

**Downtime Log File** (used in Machine Health Diagnostics):

| Column | Required | Notes |
|---|---|---|
| `Machine` | ✅ | Machine or line name |
| `Duration (min)` | ✅ | Downtime duration in minutes |
| `Category` | Optional | Reason or failure category. Enables Pareto and root cause breakdown |
| `Date` | Optional | Date of the event. Enables downtime trend analysis, MTBF and MTTR calculations, failure pattern detection, and maintenance interval recommendations |
| `Shift` | Optional | Shift name or number. Enables shift-level downtime comparison |

---

**Production Log File** (used in Machine Health Diagnostics):

| Column | Required | Notes |
|---|---|---|
| `Machine` | ✅ | Machine or line name |
| `Planned Hours` | ✅ | Scheduled production hours per record |
| `Run Hours` | ✅ | Actual hours the machine ran |
| `Good Units` | ✅ | Units that passed quality checks |
| `Total Units` | ✅ | Total units produced including defects |
| `Ideal Cycle Time (s)` | Optional | Theoretical fastest cycle time. Used to calculate the Performance component of OEE |
| `Date` | Optional | Production date. Enables OEE trend charts and shift comparison over time |
| `Shift` | Optional | Shift name or number. Enables shift-level OEE and performance comparison |

&nbsp;
""")
        uploaded = st.file_uploader(
            "📂 UPLOAD YOUR DATA EXCEL FILE(S)",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="main_uploader",
        )
        if uploaded:
            col_run, _ = st.columns([1, 2])
            with col_run:
                if st.button("▶ Run Analysis", type="primary", use_container_width=True, key="run_analysis_btn"):
                    st.session_state.app_running = True
                    st.session_state.uploaded_files = uploaded
                    st.rerun()

    st.markdown("""
<div style="
    margin-top: 45vh;
    padding: 16px 24px;
    border-top: 1px solid #2a3a5e;
    text-align: center;
    color: #3a4a6a;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
">
    © 2026 Dyson Ye · Operations Assistant for Manufacturing · Built with Streamlit · Data is not stored or shared
</div>
""", unsafe_allow_html=True)
    st.stop()

uploaded = st.session_state.uploaded_files
# Upload button and Line Selector rendered together after data loads below

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

# Column name synonyms → standard names
TS_COL_MAP = {
    # Workstation
    "station": "Workstation", "work station": "Workstation", "ws": "Workstation",
    "work center": "Workstation", "workcenter": "Workstation", "cell": "Workstation",
    "line": "Workstation", "machine": "Workstation",
    # Operation
    "task": "Operation", "process": "Operation", "step": "Operation",
    "activity": "Operation", "op": "Operation",
    # Cycle time
    "cycle time": "Cycle Time (s)", "ct": "Cycle Time (s)", "cycle time (s)": "Cycle Time (s)",
    "cycle time (sec)": "Cycle Time (s)", "time (s)": "Cycle Time (s)",
    "time per unit": "Cycle Time (s)", "process time": "Cycle Time (s)",
    "std time": "Cycle Time (s)", "standard time": "Cycle Time (s)",
    "avg ct (s)": "Cycle Time (s)", "avg ct": "Cycle Time (s)",
    # Min / Max CT
    "min ct": "Min CT (s)", "min ct (s)": "Min CT (s)", "minimum ct": "Min CT (s)",
    "min cycle time": "Min CT (s)", "min cycle time (s)": "Min CT (s)",
    "max ct": "Max CT (s)", "max ct (s)": "Max CT (s)", "maximum ct": "Max CT (s)",
    "max cycle time": "Max CT (s)", "max cycle time (s)": "Max CT (s)",
    # Queue time
    "queue time": "Queue Time (s)", "queue time (s)": "Queue Time (s)",
    "buffer time": "Queue Time (s)", "buffer time (s)": "Queue Time (s)",
    "wait time": "Queue Time (s)", "wait time (s)": "Queue Time (s)",
    "queue": "Queue Time (s)",
    # Product
    "part": "Product", "part number": "Product", "sku": "Product",
    "item": "Product", "product name": "Product",
    # Line
    "line": "Line", "production line": "Line", "line name": "Line",
    "line id": "Line", "line number": "Line",
}

DT_COL_MAP = {
    "date": "Date", "event date": "Date", "downtime date": "Date",
    "machine": "Machine", "equipment": "Machine", "asset": "Machine",
    "line": "Machine", "station": "Machine",
    "duration": "Duration (min)", "duration (min)": "Duration (min)",
    "downtime (min)": "Duration (min)", "minutes": "Duration (min)",
    "duration (minutes)": "Duration (min)", "time lost": "Duration (min)",
    "category": "Category", "reason": "Category", "cause": "Category",
    "type": "Category", "failure type": "Category",
    "shift": "Shift",
}

PROD_COL_MAP = {
    "date": "Date", "production date": "Date",
    "machine": "Machine", "equipment": "Machine", "line": "Machine",
    "planned hours": "Planned Hours", "scheduled hours": "Planned Hours",
    "available hours": "Planned Hours",
    "run hours": "Run Hours", "actual hours": "Run Hours",
    "operating hours": "Run Hours", "uptime hours": "Run Hours",
    "good units": "Good Units", "good parts": "Good Units",
    "good output": "Good Units", "accepted units": "Good Units",
    "total units": "Total Units", "total output": "Total Units",
    "total parts": "Total Units", "produced": "Total Units",
    "ideal cycle time": "Ideal Cycle Time (s)", "ideal ct": "Ideal Cycle Time (s)",
    "ideal cycle time (s)": "Ideal Cycle Time (s)",
    "shift": "Shift",
}

SKIP_SHEETS = {"dropdowndata", "dropdown", "lookup", "reference", "config", "settings", "readme"}


@st.cache_data
def load_sheets(files) -> dict[str, pd.DataFrame]:
    """
    Load all sheets from all uploaded files.
    Returns dict of sheet_key → DataFrame (raw, after column harmonisation).
    Auto-detects whether a sheet is time study, downtime, or production log.
    """
    if not isinstance(files, list):
        files = [files]

    sheets: dict[str, pd.DataFrame] = {}

    for file in files:
        fname = getattr(file, "name", "Unknown")
        try:
            xl = pd.ExcelFile(file)
        except Exception:
            continue

        for sheet in xl.sheet_names:
            if sheet.strip().lower() in SKIP_SHEETS:
                continue
            try:
                sdf = xl.parse(sheet)
                sdf.columns = sdf.columns.str.strip()
                sdf = sdf.loc[:, ~sdf.columns.str.startswith("Unnamed")]
                sdf = sdf.loc[:, ~sdf.columns.str.contains(r'\.\d+$', regex=True)]
                sdf = sdf.dropna(how="all")
                key = f"{fname} › {sheet}"
                sheets[key] = sdf
            except Exception:
                continue
    return sheets


def harmonise(df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
    """Rename columns according to a synonym map."""
    rename = {}
    for col in df.columns:
        std = col_map.get(col.lower().strip())
        if std and std not in df.columns:
            rename[col] = std
    return df.rename(columns=rename)


def detect_sheet_type(df: pd.DataFrame) -> str:
    """
    Heuristically identify sheet type.
    Returns: 'time_study' | 'downtime' | 'production' | 'unknown'
    """
    cols_lower = {c.lower().strip() for c in df.columns}

    ts_signals   = {"workstation", "station", "ws", "cycle time", "ct", "operation", "task", "process"}
    dt_signals   = {"duration", "downtime", "failure", "breakdown", "category", "reason", "cause"}
    prod_signals = {"good units", "total units", "run hours", "planned hours", "oee", "availability"}

    ts_score   = len(cols_lower & ts_signals)
    dt_score   = len(cols_lower & dt_signals)
    prod_score = len(cols_lower & prod_signals)

    best = max(ts_score, dt_score, prod_score)
    if best == 0:
        return "unknown"
    if ts_score == best:
        return "time_study"
    if dt_score == best:
        return "downtime"
    return "production"


def clean_time_study(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = harmonise(df, TS_COL_MAP)
    report = []
    required = {"Workstation", "Operation", "Cycle Time (s)"}
    missing  = required - set(df.columns)
    if missing:
        report.append(("❌ Missing", f"Required columns not found: {missing}"))
        return pd.DataFrame(), report

    before = len(df)
    df = df.dropna(subset=["Workstation", "Operation", "Cycle Time (s)"])
    df["Cycle Time (s)"] = pd.to_numeric(df["Cycle Time (s)"], errors="coerce")
    df = df[df["Cycle Time (s)"] > 0].copy()
    removed = before - len(df)
    if removed > 0:
        report.append(("✅ Fixed", f"Removed {removed} row(s) with missing or zero Cycle Time"))

    df["Workstation"] = df["Workstation"].astype(str).str.strip()
    df["Operation"]   = df["Operation"].astype(str).str.strip()
    report.append(("Summary", f"{len(df):,} operations across {df['Workstation'].nunique()} workstations"))
    return df.reset_index(drop=True), report


def clean_downtime(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = harmonise(df, DT_COL_MAP)
    report = []
    required = {"Machine", "Duration (min)"}
    missing  = required - set(df.columns)
    if missing:
        report.append(("❌ Missing", f"Required columns not found: {missing}"))
        return pd.DataFrame(), report

    before = len(df)
    df["Duration (min)"] = pd.to_numeric(df["Duration (min)"], errors="coerce")
    df = df.dropna(subset=["Machine", "Duration (min)"])
    df = df[df["Duration (min)"] > 0].copy()
    removed = before - len(df)
    if removed > 0:
        report.append(("✅ Fixed", f"Removed {removed} row(s) with missing or zero Duration"))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Category" not in df.columns:
        df["Category"] = "Unspecified"
    df["Category"] = df["Category"].fillna("Unspecified").astype(str).str.strip()
    df["Machine"]  = df["Machine"].astype(str).str.strip()
    report.append(("Summary", f"{len(df):,} downtime events across {df['Machine'].nunique()} machine(s)"))
    return df.reset_index(drop=True), report


def clean_production(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    df = harmonise(df, PROD_COL_MAP)
    report = []
    required = {"Machine", "Planned Hours", "Run Hours", "Good Units", "Total Units"}
    missing  = required - set(df.columns)
    if missing:
        report.append(("❌ Missing", f"Required columns not found: {missing}"))
        return pd.DataFrame(), report

    for col in ["Planned Hours", "Run Hours", "Good Units", "Total Units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Machine", "Planned Hours", "Run Hours", "Good Units", "Total Units"])
    removed = before - len(df)
    if removed > 0:
        report.append(("✅ Fixed", f"Removed {removed} row(s) with missing values"))

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if "Ideal Cycle Time (s)" in df.columns:
        df["Ideal Cycle Time (s)"] = pd.to_numeric(df["Ideal Cycle Time (s)"], errors="coerce")
    if "Shift" not in df.columns:
        df["Shift"] = "All Shifts"

    df["Machine"] = df["Machine"].astype(str).str.strip()
    report.append(("Summary", f"{len(df):,} production records across {df['Machine'].nunique()} machine(s)"))
    return df.reset_index(drop=True), report


# ── Load all sheets and auto-classify ────────────────────────────────────────
all_sheets = load_sheets(uploaded)

if not all_sheets:
    st.error("Could not read any sheets from the uploaded file(s). Please check the format.")
    st.stop()

# Build categorized DataFrames
ts_frames, dt_frames, prod_frames = [], [], []
ts_reports, dt_reports, prod_reports = [], [], []

for key, raw_df in all_sheets.items():
    stype = detect_sheet_type(raw_df)
    if stype == "time_study":
        cleaned, report = clean_time_study(raw_df.copy())
        if not cleaned.empty:
            ts_frames.append(cleaned)
            ts_reports.extend([r for r in report if "❌" not in r[0]])
        # silently skip sheets that failed column validation (summary/lookup sheets)
    elif stype == "downtime":
        cleaned, report = clean_downtime(raw_df.copy())
        if not cleaned.empty:
            dt_frames.append(cleaned)
            dt_reports.extend([r for r in report if "❌" not in r[0]])
    elif stype == "production":
        cleaned, report = clean_production(raw_df.copy())
        if not cleaned.empty:
            prod_frames.append(cleaned)
            prod_reports.extend([r for r in report if "❌" not in r[0]])

ts_df   = pd.concat(ts_frames,   ignore_index=True) if ts_frames   else pd.DataFrame()
dt_df   = pd.concat(dt_frames,   ignore_index=True) if dt_frames   else pd.DataFrame()
prod_df = pd.concat(prod_frames, ignore_index=True) if prod_frames else pd.DataFrame()

# Deduplicate report messages (same message can appear from multiple sheets)
def dedup_reports(reports):
    seen = set()
    out  = []
    for status, msg in reports:
        if msg not in seen:
            seen.add(msg)
            out.append((status, msg))
    return out

ts_reports   = dedup_reports(ts_reports)
dt_reports   = dedup_reports(dt_reports)
prod_reports = dedup_reports(prod_reports)

# ═════════════════════════════════════════════════════════════════════════════
# LINE SELECTOR — shown next to Upload New File button when Line column exists
# ═════════════════════════════════════════════════════════════════════════════
_has_line_col = (not ts_df.empty and "Line" in ts_df.columns) or \
                (not dt_df.empty and "Line" in dt_df.columns) or \
                (not prod_df.empty and "Line" in prod_df.columns)

if _has_line_col:
    _all_lines = sorted(set(
        list(ts_df["Line"].dropna().unique() if "Line" in ts_df.columns else []) +
        list(dt_df["Line"].dropna().unique() if "Line" in dt_df.columns else []) +
        list(prod_df["Line"].dropna().unique() if "Line" in prod_df.columns else [])
    ))
    _up_col, _line_col, _ = st.columns([1, 3, 2])
    with _up_col:
        if st.button("🔄 Upload New File", use_container_width=True):
            st.session_state.app_running = False
            for key in ["ts_df", "dt_df", "prod_df", "wi_moves", "wi_stations", "wi_operators",
                        "wi_op_assignments", "wi_op_history", "wi_action_type", "ao_constraints",
                        "ao_suggested_moves", "ao_suggested_op_assigns", "ao_no_operators_warning",
                        "ao_target", "ao_solution_method", "ao_min_ops_needed",
                        "ao_empty_stations", "ao_unused_operators",
                        "ao_stations_over_takt", "ao_stations_to_add", "global_line_filter",
                        "_active_line"]:
                st.session_state.pop(key, None)
            st.rerun()
    with _line_col:
        selected_line = st.selectbox(
            "🏭 Filter by Production Line",
            ["All Lines"] + _all_lines,
            key="global_line_filter",
            help="Filter all analysis and simulation to a specific production line. "
                 "When a line is selected, all features will reflect only that line."
        )
else:
    _up_col, _ = st.columns([1, 5])
    with _up_col:
        if st.button("🔄 Upload New File", use_container_width=True):
            st.session_state.app_running = False
            for key in ["ts_df", "dt_df", "prod_df", "wi_moves", "wi_stations", "wi_operators",
                        "wi_op_assignments", "wi_op_history", "wi_action_type", "ao_constraints",
                        "ao_suggested_moves", "ao_suggested_op_assigns", "ao_no_operators_warning",
                        "ao_target", "ao_solution_method", "ao_min_ops_needed",
                        "ao_empty_stations", "ao_unused_operators",
                        "ao_stations_over_takt", "ao_stations_to_add", "global_line_filter",
                        "_active_line"]:
                st.session_state.pop(key, None)
            st.rerun()
    selected_line = "All Lines"

if selected_line != "All Lines":
    if not ts_df.empty and "Line" in ts_df.columns:
        ts_df = ts_df[ts_df["Line"] == selected_line].copy()
    if not dt_df.empty and "Line" in dt_df.columns:
        dt_df = dt_df[dt_df["Line"] == selected_line].copy()
    if not prod_df.empty and "Line" in prod_df.columns:
        prod_df = prod_df[prod_df["Line"] == selected_line].copy()

# Clear simulation state when line selection changes
_prev_line = st.session_state.get("_active_line", None)
if selected_line != _prev_line:
    st.session_state["_active_line"] = selected_line
    for key in ["wi_moves", "wi_stations", "wi_operators", "wi_op_assignments",
                "wi_op_history", "wi_action_type", "ao_constraints",
                "ao_suggested_moves", "ao_suggested_op_assigns", "ao_no_operators_warning",
                "ao_target", "ao_solution_method", "ao_min_ops_needed",
                "ao_empty_stations", "ao_unused_operators",
                "ao_stations_over_takt", "ao_stations_to_add"]:
        st.session_state.pop(key, None)

# ═════════════════════════════════════════════════════════════════════════════
# REQUIRED SETUP — Takt Time & Product Mix (shared across all tabs)
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("⚙️ Required Setup: Takt Time & Product Mix", expanded=False):

    # ── Takt Time Calculator ──────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**TAKT TIME CALCULATOR**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            days_per_week   = st.number_input("Working days per week", min_value=1, max_value=7, value=5, step=1, key="g_days",
                                              help="How many days per week the line operates. Used to calculate your weekly and monthly output targets.")
        with c2:
            shifts_per_day  = st.number_input("Shifts per day", min_value=1, max_value=3, value=1, step=1, key="g_shifts",
                                              help="How many production shifts run per day on this line. Most factories run 1, 2, or 3 shifts.")
        with c3:
            hours_per_shift = st.number_input("Hours per shift", min_value=1.0, max_value=24.0, value=8.0, step=0.5, key="g_hours",
                                              help="The scheduled length of each shift in hours, including breaks.")
        with c4:
            break_minutes   = st.number_input("Breaks per shift (min)", min_value=0, max_value=120, value=30, step=5, key="g_breaks",
                                              help="Total break time per shift in minutes. This is subtracted from shift hours to get the net productive time.")
        with c5:
            demand_per_day  = st.number_input("Total daily demand (units)", min_value=1, value=400, step=10, key="g_demand",
                                              help="Total units your customers need per day across ALL products combined. This is the number that drives your takt time.")

        net_seconds_per_day = shifts_per_day * ((hours_per_shift * 3600) - (break_minutes * 60))
        takt = calc_takt_time(net_seconds_per_day, demand_per_day) if net_seconds_per_day > 0 else None
        weekly_cap  = demand_per_day * days_per_week
        monthly_cap = weekly_cap * 4
        st.session_state["takt_time_s"] = takt

        if net_seconds_per_day <= 0:
            st.warning("Break time exceeds or equals shift hours. Net productive time is zero or negative. Reduce break minutes or increase shift hours to calculate takt time.")
        elif takt and takt > 0:
            hourly_throughput = 3600 / takt
            gm1, gm2, gm3, gm4, gm5 = st.columns(5)
            gm1.metric("🥁 Takt Time", fmt_time(takt),
                       help="The maximum time allowed per unit at each workstation to meet demand. "
                            "Every operation on your line should complete within this time.")
            gm2.metric("🔋 Net Available / Day", f"{net_seconds_per_day/3600:.2f} hrs",
                       help="Total productive hours available per day after removing break time. "
                            "Calculated as: Shifts x (Shift hours minus Breaks).")
            gm3.metric("⚡ Hourly Throughput", f"{hourly_throughput:.1f} units/hr",
                       help="The number of units your line needs to produce every hour to meet daily demand. "
                            "Calculated as: 3600 seconds divided by takt time.")
            gm4.metric("🎯 Weekly Target", f"{weekly_cap:,} units",
                       help=f"Total units your line needs to produce per week. "
                            f"Calculated as: {demand_per_day:,} units/day x {days_per_week} days.")
            gm5.metric("🏆 Monthly Target", f"{monthly_cap:,} units",
                       help="Estimated monthly output based on your weekly target x 4 weeks.")

    # ── Product Mix ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.caption("**PRODUCT MIX CONTROL**")
        if not ts_df.empty and "Product" in ts_df.columns and ts_df["Product"].nunique() > 1:
            all_products = sorted(ts_df["Product"].dropna().unique().tolist())
            pm1, pm2 = st.columns([1, 3])
            with pm1:
                mix_mode = st.radio(
                    "Mix input as",
                    ["Units", "Percentage"],
                    horizontal=True, key="g_mix_mode",
                    help="Units: enter any relative quantity. Only the ratio matters. "
                         "Percentage: enter the share of each product as a percentage."
                )
            with pm2:
                sel_products = st.multiselect(
                    "Select products to analyse",
                    all_products, default=all_products, key="g_product_multiselect",
                    help="Select one or more products. The analysis uses a weighted average of their cycle times."
                )

            if not sel_products:
                sel_products = all_products

            if len(sel_products) > 1:
                mix_cols = st.columns(min(len(sel_products), 6))
                vol_mix = {}
                for i, prod in enumerate(sel_products):
                    if mix_mode == "Units":
                        vol_mix[prod] = mix_cols[i % 6].number_input(
                            prod, min_value=1, value=100, step=10, key=f"g_vol_{prod}",
                            help=f"Relative production quantity for {prod}. Only the ratio matters."
                        )
                    else:
                        vol_mix[prod] = mix_cols[i % 6].number_input(
                            prod, min_value=1, max_value=100,
                            value=int(100 / len(sel_products)), step=5, key=f"g_vol_{prod}",
                            help=f"Percentage share for {prod}. Normalised automatically."
                        )
                total_vol = sum(vol_mix.values())
                weights = {p: v / total_vol for p, v in vol_mix.items()}
            else:
                weights = {sel_products[0]: 1.0}

            filtered = ts_df[ts_df["Product"].isin(sel_products)]
            per_prod_ct = (filtered.groupby(["Product", "Workstation", "Operation"])["Cycle Time (s)"]
                           .mean().reset_index())
            per_prod_ct["Weighted CT (s)"] = per_prod_ct.apply(
                lambda r: r["Cycle Time (s)"] * weights.get(r["Product"], 0), axis=1
            )
            ts_lb = (per_prod_ct.groupby(["Workstation", "Operation"])["Weighted CT (s)"]
                     .sum().reset_index()
                     .rename(columns={"Weighted CT (s)": "Cycle Time (s)"}))
        else:
            ts_lb = ts_df.copy()
            if not ts_df.empty and "Product" in ts_lb.columns:
                ts_lb = (ts_lb.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                         .mean().reset_index())
            if ts_df.empty:
                st.caption("Upload a time study file with a Product column to enable multi-product mix analysis.")
            else:
                st.caption("No Product column detected. All operations treated as one product.")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Current State Analysis",
    "⚖️ Line Balancing Simulation",
    "🩺 Machine Health Diagnostics",
    "📟 Discrete Event Simulation",
])

# Pre-compute balance_df outside tabs so both tab1 and tab2 can use it
balance_df      = compute_line_balance(ts_lb, takt) if not ts_lb.empty else pd.DataFrame()
bottleneck      = detect_bottleneck(balance_df)
all_bottlenecks = detect_all_bottlenecks(balance_df, takt)
n_stations      = len(balance_df)

# Initialise wi_operators early so Current Line Status metric is correct
if "wi_operators" not in st.session_state:
    if not ts_df.empty and "Operator" in ts_df.columns:
        detected_ops = sorted(ts_df["Operator"].dropna().astype(str).str.strip().unique().tolist())
    else:
        detected_ops = ["Operator A", "Operator B"]
    st.session_state["wi_operators"] = detected_ops

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – TIME & CAPACITY
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("📊 Current State Analysis",
              help="Analyse the current state of your production line from the operation level through to the full value stream. "
                   "Work through the sections from top to bottom as a root cause analysis: start with the Current Line Overview to spot problem stations, "
                   "then use Cycle Time Analysis to pinpoint which specific operations are causing them, "
                   "then check the Trend and Operator sections for deeper context, "
                   "then review Waste Analysis to quantify idle and queue time, "
                   "and finish with Value Stream Summary for a complete picture of where time is being lost. "
                   "Set your Takt Time in the Required Setup panel above before starting.")

    # ── Data status ───────────────────────────────────────────────────────────
    with st.expander("🔍 Data Overview", expanded=False):
        if not ts_df.empty:
            st.success(f"✅ Time study data loaded: {len(ts_df):,} operations across {ts_df['Workstation'].nunique()} workstation(s)")
            for status, msg in ts_reports:
                if "❌" in status:   st.error(f"{status}: {msg}")
                elif "✅" in status: st.success(f"{status}: {msg}")
                elif "⚠️" in status: st.warning(f"{status}: {msg}")
                # skip Summary lines, the header already shows the count
            st.dataframe(ts_df.head(10), use_container_width=True)
        else:
            st.warning("No time study data detected in your uploaded file(s). "
                       "Make sure your file has columns: Workstation, Operation, Cycle Time (s).")

    if ts_df.empty:
        st.info("Upload a time study file with Workstation, Operation, and Cycle Time (s) columns to enable Current State Analysis.")
    else:
        st.subheader("Current Line Overview",
                     help="A high-level snapshot of your production line's current state. "
                          "The bar chart shows the total cycle time for each workstation compared to your takt time target. "
                          "Red bars are over takt and cannot keep up with demand. Gold bars are within 15% of takt and at risk. "
                          "Use this to quickly identify which stations need attention, then scroll down to Cycle Time Analysis to drill into the specific operations causing the problem.")

        if balance_df.empty:
            st.info("No balance data available. Check your time study file has Workstation, Operation and Cycle Time (s) columns.")
        else:
            # Station load bar chart
            fig_ov, ax_ov = plt.subplots(figsize=(max(8, n_stations * 1.2), 4))
            bar_colors_ov = []
            for _, row in balance_df.iterrows():
                if takt and row["Total CT (s)"] >= takt:
                    bar_colors_ov.append(RED)
                elif takt and row["Total CT (s)"] >= takt * 0.85:
                    bar_colors_ov.append(GOLD)
                else:
                    bar_colors_ov.append(STEEL_BLUE)
            ax_ov.bar(balance_df["Workstation"], balance_df["Total CT (s)"], color=bar_colors_ov, alpha=0.85, zorder=2)
            if takt:
                ax_ov.axhline(takt, color=GOLD, linewidth=2, linestyle="--",
                              label=f"Takt Time ({fmt_time(takt)})", zorder=3)
            ax_ov.set_ylabel("Total Cycle Time (s)", fontsize=11)
            ax_ov.tick_params(axis="x", rotation=30, labelsize=9)
            ax_ov.set_title("Station Load vs. Takt Time", fontsize=12, fontweight="bold", pad=8)
            ax_ov.grid(axis="y", alpha=0.3, zorder=1)
            legend_handles_ov = [
                Patch(facecolor=RED,        label="Over takt"),
                Patch(facecolor=GOLD,       label="Near takt (within 15%)"),
                Patch(facecolor=STEEL_BLUE, label="Within takt"),
            ]
            if takt:
                legend_handles_ov.append(plt.Line2D([0],[0], color=GOLD, linewidth=2,
                                                    linestyle="--", label=f"Takt ({fmt_time(takt)})"))
            ax_ov.legend(handles=legend_handles_ov, fontsize=8, loc="upper right")
            plt.tight_layout()
            render_chart(fig_ov)

            # Bottleneck callout
            if takt and bottleneck:
                bn_ct = balance_df.iloc[0]["Total CT (s)"]
                if len(all_bottlenecks) > 1:
                    st.error(f"🔴 **{len(all_bottlenecks)} stations exceed takt time.** "
                             f"Start with **{bottleneck}** (highest load). "
                             "Scroll down to Cycle Time Analysis and filter by that workstation to find which operations are causing it.")
                elif bn_ct > takt:
                    st.error(f"🔴 **{bottleneck}** is over takt at **{fmt_time(bn_ct)}** vs takt {fmt_time(takt)}. "
                             "Scroll down to Cycle Time Analysis and filter by this workstation to identify the root cause operations.")
                else:
                    overload_pct = bn_ct / takt * 100
                    st.warning(f"🟡 **{bottleneck}** is your constraint at **{overload_pct:.0f}% of takt**. "
                               "Monitor closely and use the Cycle Time Analysis below to check variability.")
            elif takt:
                st.success("✅ All stations are within takt time. Use the Cycle Time Analysis below to check for variability risks.")

            # ── Workstation Detail expander ───────────────────────────────────────
            if not balance_df.empty:
                # Build ws_total from raw ts_df (not ts_view filtered) for full picture
                ws_op_avg_ov = (ts_df.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                                .mean().reset_index())
                ws_total_ov = (ws_op_avg_ov.groupby("Workstation")["Cycle Time (s)"]
                               .agg(Operations="count", Avg_CT_s="mean", Min_CT_s="min", Max_CT_s="max")
                               .reset_index())
                ws_total_ov.columns = ["Workstation", "# Operations", "Avg CT (s)", "Min CT (s)", "Max CT (s)"]
                ws_total_ov["Avg CT (s)"] = ws_total_ov["Avg CT (s)"].round(1)
                ws_total_ov["Min CT (s)"] = ws_total_ov["Min CT (s)"].round(1)
                ws_total_ov["Max CT (s)"] = ws_total_ov["Max CT (s)"].round(1)
                if takt:
                    ws_total_ov["Takt Time (s)"]       = round(takt, 2)
                    ws_total_ov["Slowest Op % of Takt"] = (ws_total_ov["Max CT (s)"] / takt * 100).round(1)
                    ws_total_ov["Status"]               = ws_total_ov["Slowest Op % of Takt"].apply(
                        lambda x: "🔴 Slowest Op Over Takt" if x > 100 else ("🟡 Slowest Op Near Takt" if x > 85 else "🟢 OK")
                    )

                def style_ws(row):
                    status = row.get("Status", "")
                    if "🔴" in str(status):
                        return ["background-color:#fdf0f0; color:#1a1a1a"] * len(row)
                    elif "🟡" in str(status):
                        return ["background-color:#fdf4e0; color:#1a1a1a"] * len(row)
                    return [""] * len(row)

                with st.expander("🏷️ Workstation Detail", expanded=False):
                    st.caption("Station-level summary showing operation count, cycle time range, and status vs takt time. "
                               "Slowest Op % of Takt compares the slowest single operation at each station against takt time.")
                    if takt:
                        st.dataframe(ws_total_ov.style.apply(style_ws, axis=1),
                                     use_container_width=True, hide_index=True)
                    else:
                        st.dataframe(ws_total_ov, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Workstation Summary",
                        to_excel_bytes(ws_total_ov),
                        "workstation_summary.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_ws_summary",
                    )

        # ── Cycle Time Analysis ───────────────────────────────────────────────────
        st.subheader("Cycle Time Analysis",
                     help="Shows how long each individual operation on your line actually takes. "
                          "Each bar in the chart represents one task, coloured by which workstation it belongs to. "
                          "The gold dashed line is your takt time target. Any bar above that line means the operation is too slow and will cause the line to fall behind. "
                          "Use the filters to focus on a specific product or workstation. "
                          "The small whiskers on each bar show how consistent the time is across readings. Tall whiskers mean the time varies a lot between cycles, which is a risk even if the average looks fine. "
                          "Open the Operation Cycle Time Detail expander to see AI-powered anomaly detection and variability clustering that automatically flag which operations need attention.")

        f1, f2 = st.columns(2)
        with f1:
            products = ["All Products"]
            if "Product" in ts_df.columns:
                products += sorted(ts_df["Product"].dropna().unique().tolist())
            selected_product = st.selectbox("Filter by Product / Part", products, key="ts_product_filter",
                                            help="When your line produces multiple products, filter to a single product to see its specific cycle times. "
                                                 "Averaging across products can hide performance differences between them. "
                                                 "This filter applies to the raw observed cycle times in this view only. "
                                                 "To weight station load calculations by product volume, use the Product Mix Control in the Required Setup panel above.")
        with f2:
            workstation_opts = ["All Workstations"] + sorted(ts_df["Workstation"].dropna().unique().tolist())
            selected_ws_filter = st.selectbox("Filter by Workstation", workstation_opts, key="ts_ws_filter",
                                              help="Narrow the cycle time chart to a single workstation. "
                                                   "This is especially useful after identifying a bottleneck station in the Current Line Overview above. "
                                                   "Filter to that station to see exactly which operations are driving its high cycle time.")

        ts_view = ts_df.copy()
        if selected_product != "All Products":
            ts_view = ts_view[ts_view["Product"] == selected_product]
        if selected_ws_filter != "All Workstations":
            ts_view = ts_view[ts_view["Workstation"] == selected_ws_filter]

        # Chart: cycle time per operation with variability error bars
        op_stats = (ts_view.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                    .agg(mean="mean", std="std")
                    .reset_index()
                    .rename(columns={"mean": "Avg CT (s)", "std": "Std Dev (s)"}))
        op_stats["Std Dev (s)"] = op_stats["Std Dev (s)"].fillna(0)
        # Sort by workstation then CT descending within each workstation
        op_stats = op_stats.sort_values(["Workstation", "Avg CT (s)"], ascending=[True, False]).reset_index(drop=True)

        if op_stats.empty:
            st.info("No operations found for the selected filters.")
        else:
            fig, ax = plt.subplots(figsize=(min(max(10, len(op_stats) * 0.6), 28), 5))
            workstations = op_stats["Workstation"].unique()
            palette_ws   = [STEEL_BLUE, ORANGE, GREEN, RED, LIGHT_BLUE, GOLD, GREY, "#9b59b6"]
            ws_color_map = {ws: palette_ws[i % len(palette_ws)] for i, ws in enumerate(workstations)}
            bar_colors   = list(op_stats["Workstation"].map(ws_color_map))

            ax.bar(op_stats["Operation"], op_stats["Avg CT (s)"], color=bar_colors, alpha=0.85,
                   yerr=op_stats["Std Dev (s)"], capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555555", "alpha": 0.7})

            if takt:
                ax.axhline(takt, color=GOLD, linewidth=2, linestyle="--", label=f"Takt ({fmt_time(takt)})", zorder=3)

            ax.set_ylabel("Cycle Time (s)", fontsize=11)
            ax.tick_params(axis="x", rotation=65, labelsize=8)
            ax.set_title("Cycle Time per Operation (bars = average, error bars = variability)", fontsize=12, fontweight="bold", pad=10)

            legend_handles = [Patch(color=ws_color_map[ws], label=ws) for ws in workstations]
            if takt:
                legend_handles.append(plt.Line2D([0],[0], color=GOLD, linewidth=2, linestyle="--", label=f"Takt ({fmt_time(takt)})"))
            ax.legend(handles=legend_handles, fontsize=8, loc="upper right")
            plt.tight_layout()
            render_chart(fig)

            # Smart summary callout
            if takt:
                over_takt = op_stats[op_stats["Avg CT (s)"] > takt]
                near_takt = op_stats[(op_stats["Avg CT (s)"] <= takt) & (op_stats["Avg CT (s)"] >= takt * 0.85)]
                high_var  = op_stats[op_stats["Std Dev (s)"] > op_stats["Avg CT (s)"] * 0.2]
                msgs = []
                if not over_takt.empty:
                    msgs.append(f"🔴 **{len(over_takt)} operation(s) exceed takt time:** {', '.join(over_takt['Operation'].tolist())}.")
                if not near_takt.empty:
                    msgs.append(f"🟡 **{len(near_takt)} operation(s) are within 15% of takt** and should be monitored.")
                if not high_var.empty:
                    msgs.append(f"⚠️ **{len(high_var)} operation(s) have high variability** (std dev over 20% of avg): {', '.join(high_var['Operation'].tolist())}.")
                if msgs:
                    st.warning("  \n".join(msgs))
                else:
                    st.success("✅ All operations are within takt time with low variability.")
            else:
                high_var = op_stats[op_stats["Std Dev (s)"] > op_stats["Avg CT (s)"] * 0.2]
                if not high_var.empty:
                    st.warning(f"⚠️ **{len(high_var)} operation(s) have high variability:** {', '.join(high_var['Operation'].tolist())}. Set takt time to see which exceed the target.")

            # Data table + download in expander
            op_stats_dl = op_stats.copy()
            op_stats_dl["Avg CT (s)"] = op_stats_dl["Avg CT (s)"].round(2)
            op_stats_dl["Std Dev (s)"] = op_stats_dl["Std Dev (s)"].round(2)
            with st.expander("🏷️ Operation Cycle Time Detail", expanded=False):
                st.caption("Average and variability of each operation's cycle time based on the current filters. "
                           "AI analysis adds anomaly flags and variability clusters when enough data is available.")

                if ML_AVAILABLE and len(op_stats_dl) >= 3:
                    try:
                        # ── Anomaly Detection (Isolation Forest) ─────────────
                        iso_feats = op_stats_dl[["Avg CT (s)", "Std Dev (s)"]].fillna(0)
                        iso = IsolationForest(contamination=0.15, random_state=42)
                        op_stats_dl["Anomaly"] = iso.fit_predict(iso_feats)
                        op_stats_dl["Anomaly"] = op_stats_dl["Anomaly"].map(
                            {1: "Normal", -1: "⚠️ Flag for review"})
                    except Exception:
                        pass

                    try:
                        # ── Variability Clustering (KMeans, 3 clusters) ───────
                        if len(op_stats_dl) >= 6:
                            km_feats = op_stats_dl[["Avg CT (s)", "Std Dev (s)"]].fillna(0)
                            scaler = StandardScaler()
                            km_scaled = scaler.fit_transform(km_feats)
                            km = KMeans(n_clusters=3, random_state=42, n_init=10)
                            labels = km.fit_predict(km_scaled)
                            centers = scaler.inverse_transform(km.cluster_centers_)
                            avg_cts  = centers[:, 0]
                            avg_stds = centers[:, 1]
                            # Assign labels: highest std = High Variability,
                            # then of remaining two: highest CT = Consistently Slow, other = Stable
                            remaining = list(range(3))
                            high_var_cluster = int(np.argmax(avg_stds))
                            remaining.remove(high_var_cluster)
                            slow_cluster  = remaining[0] if avg_cts[remaining[0]] >= avg_cts[remaining[1]] else remaining[1]
                            stable_cluster = remaining[1] if slow_cluster == remaining[0] else remaining[0]
                            cluster_names = {
                                high_var_cluster: "🔴 High Variability",
                                slow_cluster:     "🟡 Consistently Slow",
                                stable_cluster:   "🟢 Stable",
                            }
                            op_stats_dl["Cluster"] = [cluster_names[l] for l in labels]
                            st.caption(
                                "**Variability Clusters** (KMeans AI): "
                                "🟢 Stable = well-controlled operations, no action needed. "
                                "🟡 Consistently Slow = bottleneck candidates, consider rebalancing in the Line Balancing Simulation. "
                                "🔴 High Variability = process stability problems, check operator training or tooling."
                            )
                    except Exception:
                        pass

                st.dataframe(op_stats_dl, use_container_width=True, hide_index=True)

                # ── Anomaly warning shown after table, before download ─────────
                if ML_AVAILABLE and "Anomaly" in op_stats_dl.columns:
                    flagged = op_stats_dl[op_stats_dl["Anomaly"] == "⚠️ Flag for review"]
                    if not flagged.empty:
                        st.warning(
                            f"**Anomaly Detection flagged {len(flagged)} operation(s) for review:** "
                            f"{', '.join(flagged['Operation'].tolist())}. "
                            "These have unusual cycle time or variability patterns compared to the rest of the line. "
                            "Check whether the readings are correct or if the operation has a process issue."
                        )

                st.download_button(
                    "⬇️ Download Operation Cycle Times",
                    to_excel_bytes(op_stats_dl.dropna(axis=1, how="all")),
                    "operation_cycle_times.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_op_stats",
                )

        # ── CT Trend Over Time ────────────────────────────────────────────────
        st.subheader("Cycle Time Trend Over Time",
                     help="Shows how the average cycle time per operation has changed over different study dates. "
                          "A downward trend means your process is improving. "
                          "An upward trend or sudden spike may indicate a process problem, a new operator, or an equipment issue. "
                          "Select a specific operation from the dropdown to focus on one task, or leave it on the overall average to see the whole line's trend. "
                          "Requires a Date column in your time study file.")

        if "Date" not in ts_view.columns or not ts_view["Date"].notna().any():
            st.info("Add a **Date** column to your time study file to enable cycle time trend analysis. "
                    "Each reading should have a date so the app can track how performance changes over time.")
        else:
            ts_dated = ts_view.copy()
            ts_dated["Date"] = pd.to_datetime(ts_dated["Date"], errors="coerce")
            ts_dated = ts_dated.dropna(subset=["Date"])

            trend_ops = ["All Operations (Average)"] + sorted(ts_dated["Operation"].unique().tolist())
            sel_trend_op = st.selectbox("Select operation", trend_ops, key="trend_op_sel")

            if sel_trend_op == "All Operations (Average)":
                trend_data = (ts_dated.groupby("Date")["Cycle Time (s)"]
                              .mean().reset_index()
                              .rename(columns={"Cycle Time (s)": "Avg CT (s)"}))
            else:
                trend_data = (ts_dated[ts_dated["Operation"] == sel_trend_op]
                              .groupby("Date")["Cycle Time (s)"]
                              .mean().reset_index()
                              .rename(columns={"Cycle Time (s)": "Avg CT (s)"}))

            trend_data = trend_data.sort_values("Date")

            if len(trend_data) >= 2:
                fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
                ax_trend.plot(trend_data["Date"], trend_data["Avg CT (s)"],
                              color=STEEL_BLUE, linewidth=2, marker="o", markersize=5)
                ax_trend.fill_between(trend_data["Date"], trend_data["Avg CT (s)"],
                                      alpha=0.12, color=STEEL_BLUE)
                if takt:
                    ax_trend.axhline(takt, color=GOLD, linewidth=1.5, linestyle="--",
                                     label=f"Takt Time ({fmt_time(takt)})", alpha=0.8)
                ax_trend.set_ylabel("Avg Cycle Time (s)", fontsize=11)
                ax_trend.set_title(f"Cycle Time Trend: {sel_trend_op}", fontsize=12, fontweight="bold")
                ax_trend.legend(fontsize=9)
                ax_trend.grid(alpha=0.3)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                render_chart(fig_trend)
            else:
                st.info("Need at least 2 different study dates to show a trend. "
                        "Make sure your Date column has multiple different dates across your readings.")

        # ── Per-Operator Breakdown ────────────────────────────────────────────
        st.subheader("Cycle Time by Operator",
                     help="Compares how different operators perform on the same operations. "
                          "A large gap between operators on the same task may point to a training need, "
                          "an experience difference, or an ergonomic issue affecting some operators more than others. "
                          "This is useful context before planning cross-training or operator assignments in the Line Balancing Simulation. "
                          "Select a specific operation from the dropdown to compare operators on that task directly. "
                          "Requires an Operator column in your time study file.")

        if "Operator" not in ts_view.columns or not ts_view["Operator"].notna().any():
            st.info("Add an **Operator** column to your time study file to enable operator performance comparison. "
                    "This helps identify training gaps and supports cross-training decisions in the Line Balancing Simulation.")
        else:
            # Filter out NaN/empty operator values before analysis
            ts_op = ts_view[ts_view["Operator"].notna() & (ts_view["Operator"].astype(str).str.strip() != "")].copy()

            op_sel_ops = ["All Operations (Average)"] + sorted(ts_op["Operation"].unique().tolist())
            sel_op_breakdown = st.selectbox("Select operation", op_sel_ops, key="op_breakdown_sel")

            if sel_op_breakdown == "All Operations (Average)":
                op_breakdown = (ts_op.groupby("Operator")["Cycle Time (s)"]
                                .agg(Avg_CT="mean", Min_CT="min", Max_CT="max")
                                .reset_index())
            else:
                op_breakdown = (ts_op[ts_op["Operation"] == sel_op_breakdown]
                                .groupby("Operator")["Cycle Time (s)"]
                                .agg(Avg_CT="mean", Min_CT="min", Max_CT="max")
                                .reset_index())

            op_breakdown.columns = ["Operator", "Avg CT (s)", "Min CT (s)", "Max CT (s)"]
            op_breakdown["Avg CT (s)"] = op_breakdown["Avg CT (s)"].round(1)
            op_breakdown["Min CT (s)"] = op_breakdown["Min CT (s)"].round(1)
            op_breakdown["Max CT (s)"] = op_breakdown["Max CT (s)"].round(1)
            op_breakdown = op_breakdown.sort_values("Avg CT (s)")

            if op_breakdown.empty:
                st.info("No data available for the selected operation and operator combination.")
            else:
                overall_avg = ts_op["Cycle Time (s)"].mean() if sel_op_breakdown == "All Operations (Average)" else \
                              ts_op[ts_op["Operation"] == sel_op_breakdown]["Cycle Time (s)"].mean()

                fig_op, ax_op = plt.subplots(figsize=(max(7, len(op_breakdown) * 1.1), 4))
                bar_c = [RED if v > (takt if takt else float("inf")) else STEEL_BLUE
                         for v in op_breakdown["Avg CT (s)"]]
                ax_op.bar(op_breakdown["Operator"], op_breakdown["Avg CT (s)"],
                          color=bar_c, alpha=0.85,
                          yerr=op_breakdown["Max CT (s)"] - op_breakdown["Avg CT (s)"],
                          capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555555", "alpha": 0.6})
                ax_op.axhline(overall_avg, color=ORANGE, linewidth=1.5, linestyle="--",
                              label=f"Line Average ({overall_avg:.1f}s)")
                if takt:
                    ax_op.axhline(takt, color=GOLD, linewidth=1.5, linestyle="--",
                                  label=f"Takt ({fmt_time(takt)})", alpha=0.8)
                ax_op.set_ylabel("Avg Cycle Time (s)", fontsize=11)
                ax_op.set_title(f"Cycle Time by Operator: {sel_op_breakdown}", fontsize=12, fontweight="bold")
                ax_op.legend(fontsize=9)
                ax_op.grid(axis="y", alpha=0.3)
                plt.xticks(rotation=20, ha="right")
                plt.tight_layout()
                render_chart(fig_op)

                with st.expander("🏷️ Operator Cycle Time Detail", expanded=False):
                    st.caption("Average, minimum and maximum cycle time per operator based on the current operation selection.")
                    st.dataframe(op_breakdown, use_container_width=True, hide_index=True)
                    st.download_button(
                        "⬇️ Download Operator Breakdown",
                        to_excel_bytes(op_breakdown),
                        "operator_breakdown.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_op_breakdown",
                    )

        # ── Waste Analysis ────────────────────────────────────────────────────
        st.subheader("Waste Analysis",
                     help="Quantifies the two main sources of Non-Value-Added time on your line. "
                          "Idle time is the gap between a station finishing a unit and takt time, caused by uneven workloads across stations. "
                          "Queue time is how long a unit sits waiting in a buffer between stations, caused by inventory buildup. "
                          "Use the Line Flow Settings expander to set the physical station order and enter queue times if you have them. "
                          "The numbers here feed directly into the Value Stream Summary below.")

        if not takt:
            st.info("Set takt time in the Required Setup panel above to enable Waste Analysis.")
        elif balance_df.empty:
            st.info("Upload a time study file to see Waste Analysis.")
        else:
            # ── Lead Time section ─────────────────────────────────────────────
            st.markdown("**Lead Time**")

            # Default station order from file
            default_ws_order = list(dict.fromkeys(ts_df["Workstation"].dropna().astype(str).str.strip().tolist()))
            default_ws_order = [ws for ws in default_ws_order if ws in balance_df["Workstation"].values]

            # Queue time inputs — auto-populate from file if Queue Time (s) column exists
            file_queue_times = {}
            if not ts_df.empty and "Queue Time (s)" in ts_df.columns:
                qt_from_file = (ts_df.groupby("Workstation")["Queue Time (s)"]
                                .first().dropna().to_dict())
                file_queue_times = {ws: int(v) for ws, v in qt_from_file.items() if v > 0}

            with st.expander("⚙️ Line Flow Settings", expanded=False):
                # Station sequence
                with st.container(border=True):
                    st.caption("**STATION SEQUENCE**")
                    st.caption("Set the physical order stations appear on your production line. "
                               "The default order is taken from your uploaded file. "
                               "Adjust the sequence numbers so they reflect the actual flow from first to last station. "
                               "Use unique numbers for each station. If two stations share the same number, their order between each other follows the original file order.")
                    seq_cols = st.columns(min(len(default_ws_order), 4))
                    station_seq = {}
                    for i, ws in enumerate(default_ws_order):
                        station_seq[ws] = seq_cols[i % 4].number_input(
                            ws, min_value=1, value=i + 1, step=1,
                            key=f"ws_seq_{ws}"
                        )
                    ordered_stations = sorted(default_ws_order, key=lambda ws: station_seq.get(ws, 999))

                # Queue time inputs
                with st.container(border=True):
                    st.caption("**QUEUE TIME BETWEEN STATIONS**")
                    st.caption("Enter how long a unit typically waits between leaving one station and arriving at the next. "
                               "This is the time the product sits in a buffer doing nothing. "
                               "Enter 0 if there is no buffer between stations. "
                               "These inputs feed directly into the Lead Time Breakdown chart and the Value Stream Summary below.")
                    if file_queue_times:
                        st.success("Queue times pre-populated from your uploaded file. You can adjust them below.")
                    queue_times = {}
                    ws_list_wa = ordered_stations
                    q_cols_wa = st.columns(min(len(ws_list_wa) - 1, 4)) if len(ws_list_wa) > 1 else []
                    for i in range(len(ws_list_wa) - 1):
                        default_q = file_queue_times.get(ws_list_wa[i], 0)
                        queue_times[ws_list_wa[i]] = q_cols_wa[i % 4].number_input(
                            f"After {ws_list_wa[i]}", min_value=0, value=default_q, step=30,
                            key=f"q_{ws_list_wa[i]}"
                        )
                    queue_times[ws_list_wa[-1]] = 0

            total_queue_s_wa = sum(queue_times.values())

            # Lead Time Breakdown chart
            fig_lt_wa, ax_lt_wa = plt.subplots(figsize=(max(8, n_stations * 1.2), 5))
            ordered_ws = [ws for ws in ws_list_wa if ws in balance_df["Workstation"].values]
            vsm_wa_ct = [balance_df[balance_df["Workstation"] == ws]["Total CT (s)"].values[0] for ws in ordered_ws]
            vsm_wa_q  = [queue_times.get(ws, 0) for ws in ordered_ws]
            ax_lt_wa.bar(ordered_ws, vsm_wa_ct, color=STEEL_BLUE, alpha=0.85, label="Work Time")
            ax_lt_wa.bar(ordered_ws, vsm_wa_q, bottom=vsm_wa_ct,
                         color=ORANGE, alpha=0.75, label="Queue Time")
            ax_lt_wa.set_ylabel("Time (s)", fontsize=11)
            ax_lt_wa.set_title("Lead Time Breakdown: Work Time vs Queue Time per Station", fontsize=12, fontweight="bold")
            ax_lt_wa.legend(fontsize=9)
            ax_lt_wa.tick_params(axis="x", rotation=30, labelsize=9)
            ax_lt_wa.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            render_chart(fig_lt_wa)

            if total_queue_s_wa > 0:
                biggest_q_ws  = max(queue_times, key=queue_times.get)
                biggest_q_val = queue_times[biggest_q_ws]
                q_ratio = total_queue_s_wa / sum(vsm_wa_ct) * 100
                st.warning(
                    f"Queue time accounts for **{q_ratio:.0f}%** of your total process lead time. "
                    f"The largest queue is after **{biggest_q_ws}** at **{fmt_time(biggest_q_val)}**. "
                    "Reducing this queue would have the biggest impact on your overall lead time."
                )

            # ── Idle Time section ─────────────────────────────────────────────
            st.markdown("**Idle Time**")

            vsm_idle = balance_df[["Workstation", "Total CT (s)"]].copy()
            vsm_idle["Idle Time (s)"] = (takt - vsm_idle["Total CT (s)"]).clip(lower=0).round(2)
            total_idle_s = vsm_idle["Idle Time (s)"].sum()

            fig_idle, ax_idle = plt.subplots(figsize=(max(8, n_stations * 1.2), 4))
            idle_colors = [GOLD if v > 0 else GREEN for v in vsm_idle["Idle Time (s)"]]
            ax_idle.bar(vsm_idle["Workstation"], vsm_idle["Idle Time (s)"], color=idle_colors, alpha=0.85)
            ax_idle.set_ylabel("Idle Time per Unit (s)", fontsize=11)
            ax_idle.set_title("Idle Time per Workstation", fontsize=12, fontweight="bold")
            ax_idle.tick_params(axis="x", rotation=30, labelsize=9)
            ax_idle.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            render_chart(fig_idle)

            if total_idle_s > 0:
                worst_idle_ws  = vsm_idle.loc[vsm_idle["Idle Time (s)"].idxmax(), "Workstation"]
                worst_idle_val = vsm_idle["Idle Time (s)"].max()
                st.warning(f"Total idle time across all stations: **{fmt_time(total_idle_s)} per unit**. "
                           f"The most idle station is **{worst_idle_ws}** at **{fmt_time(worst_idle_val)}**. "
                           "Use the Line Balancing Simulation to redistribute work and reduce this waste.")
            else:
                st.success("✅ No idle time detected. All stations are fully loaded up to takt time.")

        # ── Value Stream Summary ──────────────────────────────────────────────
        st.subheader("Value Stream Summary",
                     help="Brings together all the analysis from the sections above into a single view of how time is spent across your line. "
                          "VA Time is productive work time that adds value. Idle and queue time are waste. "
                          "The Line Health Score combines balance efficiency, bottleneck severity, and VA ratio into a single number so you can track improvement over time. "
                          "The Line Diagnosis lists every detected issue ranked by severity. "
                          "Recommended Actions gives you a prioritised starting point for improvement. "
                          "Queue times entered in the Waste Analysis section above are carried through here automatically.")

        if takt and not balance_df.empty:
            # Use ordered_stations from Waste Analysis if available, else fall back to file order
            if 'ordered_stations' in locals() and ordered_stations:
                station_order = ordered_stations
            else:
                station_order = list(dict.fromkeys(ts_df["Workstation"].dropna().astype(str).str.strip().tolist()))
            balance_df_ordered = balance_df.set_index("Workstation").reindex(
                [ws for ws in station_order if ws in balance_df["Workstation"].values]
            ).reset_index()

            vsm = balance_df_ordered[["Workstation", "Total CT (s)"]].copy()
            vsm["VA Time (s)"]   = vsm["Total CT (s)"]
            vsm["NVA Time (s)"]  = (takt - vsm["Total CT (s)"]).clip(lower=0).round(2)
            vsm["VA Ratio (%)"]  = (vsm["VA Time (s)"] / takt * 100).clip(upper=100).round(1)
            vsm["NVA Ratio (%)"] = (vsm["NVA Time (s)"] / takt * 100).round(1)
            # Use queue_times from Waste Analysis if available
            qt = locals().get("queue_times", {})
            vsm["Queue After (s)"] = vsm["Workstation"].map(qt).fillna(0).astype(int)
            vsm["Cumulative Lead Time (s)"] = (vsm["Total CT (s)"] + vsm["Queue After (s)"]).cumsum().round(1)
            total_queue_s       = sum(qt.values())
            process_lead_time_s = vsm["Total CT (s)"].sum() + total_queue_s
            total_va            = vsm["VA Time (s)"].sum()
            total_nva           = vsm["NVA Time (s)"].sum()
            total_waste         = total_nva + total_queue_s
            overall_va_pct      = total_va / process_lead_time_s * 100 if process_lead_time_s > 0 else 0

            # ── Metrics ───────────────────────────────────────────────────────
            v1, v2, v3, v4, v5 = st.columns(5)
            v1.metric("📐 Overall VA Ratio", f"{overall_va_pct:.1f}%",
                      help="The share of process lead time that is productive work. "
                           "If queue times are zero, this will show 100% because all time the product spends on the line is being worked on. "
                           "Add queue times in the Waste Analysis section to see how buffers reduce this ratio. "
                           "Operator idle time is tracked separately in the Line Health Score and is not part of this ratio.")
            v2.metric("🏃 Total VA Time", fmt_time(total_va),
                      help="Total productive work time across all stations for one unit. "
                           "This is the theoretical minimum lead time if all queue waits were eliminated.")
            v3.metric("💤 Total Idle Time", fmt_time(total_nva),
                      help="Total time operators spend waiting per unit across all stations. "
                           "Caused by uneven workloads between stations. Can be reduced by rebalancing in the Line Balancing Simulation.")
            v4.metric("📦 Total Queue Time", fmt_time(total_queue_s),
                      help="Total time a unit spends sitting in buffers between stations. "
                           "Enter queue times in the Waste Analysis section above to see this figure. "
                           "Reducing queue time is the most direct way to shorten process lead time.")
            v5.metric("⏱️ Process Lead Time", fmt_time(process_lead_time_s),
                      help="Total time the product spends travelling the line from first station to last, "
                           "measured as the sum of all station work times plus any queue waits between stations. "
                           "This follows the standard Lean value stream mapping definition where lead time tracks the product, not the operator. "
                           "Operator idle time is not included here because the product has already moved on when the operator is waiting.")

            st.markdown("<div style='margin-top:0.3rem;'></div>", unsafe_allow_html=True)

            # ── Line Health Score ─────────────────────────────────────────────
            # Composite score: Balance Efficiency (40%) + Bottleneck Severity (40%) + VA Ratio (20%)
            n_st = len(balance_df)
            total_work_vsm = balance_df["Total CT (s)"].sum()
            be_score = min(100, total_work_vsm / (n_st * takt) * 100) if n_st and takt else 0

            # Bottleneck severity: 100 if all stations within takt, drops as worst station exceeds takt
            worst_ct = balance_df["Total CT (s)"].max() if not balance_df.empty else 0
            if worst_ct <= takt:
                bn_score = 100.0
            else:
                # Penalise proportionally to how far over takt the worst station is
                bn_score = max(0, 100 - ((worst_ct - takt) / takt * 100))

            va_score = min(100, overall_va_pct)

            line_health = round(0.4 * be_score + 0.4 * bn_score + 0.2 * va_score, 1)

            if line_health >= 75:    health_color = "#3d8b5e";   health_label = "Good"
            elif line_health >= 50:  health_color = "#d4a017";   health_label = "Needs Improvement"
            else:                    health_color = "#c0392b";   health_label = "Critical"

            st.markdown(
                f"<div style='padding:12px 16px; border-radius:8px; "
                f"border-left:4px solid {health_color}; background:#1e2d3d;'>"
                f"<span style='font-weight:700; font-size:1rem; color:#ffffff;'>Line Health Score: "
                f"<span style='color:{health_color};'>{line_health:.0f}% ({health_label})</span></span>"
                f"<br><span style='font-size:0.82rem; color:#b0c8e0;'>"
                f"Composite score based on three dimensions: "
                f"Balance Efficiency (40% weight, currently {be_score:.0f}%) measures how evenly work is spread across stations. "
                f"Bottleneck Severity (40% weight, currently {bn_score:.0f}%) measures how far your worst station is from takt time. "
                f"VA Ratio (20% weight, currently {va_score:.0f}%) measures how much of your total lead time is productive work. "
                f"A line can score high on VA Ratio but still score poorly if it has a bottleneck over takt."
                f"</span></div>",
                unsafe_allow_html=True
            )

            st.markdown("<div style='margin-top:1.2rem;'></div>", unsafe_allow_html=True)

            # ── Summary Diagnosis ─────────────────────────────────────────────
            with st.container(border=True):
                st.caption("**LINE DIAGNOSIS**")
                issues = []

                # Category 1: Bottleneck stations — sorted by severity (furthest over takt first)
                bottleneck_issues = []
                for _, row in balance_df.iterrows():
                    ct = row["Total CT (s)"]
                    if ct > takt:
                        bottleneck_issues.append((ct - takt, f"🔴 **{row['Workstation']}** is over takt at **{fmt_time(ct)}** vs takt {fmt_time(takt)}. This station cannot keep up with demand and is the primary bottleneck."))
                    elif ct >= takt * 0.85:
                        bottleneck_issues.append((ct / takt, f"🟡 **{row['Workstation']}** is near takt at **{fmt_time(ct)}** ({ct/takt*100:.0f}% of takt). Monitor closely as any variability may cause it to fall behind."))
                for _, msg in sorted(bottleneck_issues, reverse=True):
                    issues.append(msg)

                # Category 2: Idle time — sorted by idle seconds descending
                idle_issues = []
                for _, row in vsm.iterrows():
                    idle = row["NVA Time (s)"]
                    if idle > takt * 0.3:
                        idle_issues.append((idle, f"💤 **{row['Workstation']}** has **{fmt_time(idle)} of idle time per unit** ({idle/takt*100:.0f}% of takt). Significant operator waiting time that could be reduced by rebalancing."))
                for _, msg in sorted(idle_issues, reverse=True):
                    issues.append(msg)

                # Category 3: Queue time — sorted by queue seconds descending
                queue_issues = []
                for _, row in vsm.iterrows():
                    q = row["Queue After (s)"]
                    if q > 0:
                        pct = q / process_lead_time_s * 100 if process_lead_time_s > 0 else 0
                        queue_issues.append((q, f"📦 Queue after **{row['Workstation']}** is **{fmt_time(q)}** ({pct:.0f}% of total lead time). Inventory is building up here and extending delivery time."))
                for _, msg in sorted(queue_issues, reverse=True):
                    issues.append(msg)

                # Category 4: Overall VA ratio
                if overall_va_pct < 50:
                    issues.append(f"⚠️ **Overall VA Ratio is {overall_va_pct:.1f}%**. Less than half of your total process time is productive work. The majority is waste from imbalance and buffers.")
                elif overall_va_pct < 75:
                    issues.append(f"🟡 **Overall VA Ratio is {overall_va_pct:.1f}%**. There is meaningful room to reduce waste across the line.")

                if issues:
                    for issue in issues:
                        st.markdown(f"- {issue}")
                else:
                    st.success("✅ No significant issues detected. Your line is well balanced with low waste.")

            # ── Prioritised Action Recommendations ───────────────────────────
            with st.container(border=True):
                st.caption("**RECOMMENDED ACTIONS**")
                actions = []

                over_takt_stations = balance_df[balance_df["Total CT (s)"] > takt]["Workstation"].tolist()
                if over_takt_stations:
                    actions.append(f"1. **Fix bottleneck(s) first.** Use the Auto-Optimisation Panel or manually move operations away from {', '.join(over_takt_stations)} in the **Line Balancing Simulation**.")

                high_idle = vsm[vsm["NVA Time (s)"] > takt * 0.3]["Workstation"].tolist()
                if high_idle:
                    actions.append(f"{'2' if actions else '1'}. **Reduce idle time** at {', '.join(high_idle)}. Redistribute operations from overloaded stations using the **Rebalancing Simulation**.")

                if total_queue_s > 0:
                    biggest_q_ws = vsm.loc[vsm["Queue After (s)"].idxmax(), "Workstation"]
                    actions.append(f"{'3' if len(actions) >= 2 else len(actions)+1}. **Reduce queue after {biggest_q_ws}.** Consider smaller batch sizes, pull-based scheduling, or increasing that station's capacity.")

                if not actions:
                    actions.append("Your line is well balanced. Continue monitoring cycle times and variability over time.")

                for action in actions:
                    st.markdown(f"- {action}")

            # ── VSM Table (detail layer) ──────────────────────────────────────
            with st.expander("🏷️ Value Stream Detail", expanded=False):
                st.caption("Full breakdown of VA time, NVA time, queue time and cumulative lead time per station.")
                st.dataframe(vsm, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download VSM Summary",
                    to_excel_bytes(vsm),
                    "vsm_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_vsm",
                )
        else:
            st.info("Set takt time in the Required Setup panel above to enable Value Stream analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – LINE PERFORMANCE (LINE BALANCING + BOTTLENECK)
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("⚖️ Line Balancing Simulation",
              help="Simulate changes to your production line to improve balance and reduce waste. "
                   "The Simulated Line Status section shows your current line metrics. "
                   "Use the Auto-Optimisation Panel to let the app suggest a balanced layout automatically. "
                   "Use the Simulation Control Panel to manually move operations, assign operators, and test different scenarios. "
                   "The Product Mix and Takt Time from the Required Setup panel are applied throughout.")

    with st.expander("🔍 Data Overview", expanded=False):
        if not ts_df.empty:
            st.success(f"✅ Time study data loaded: {len(ts_df):,} operations across {ts_df['Workstation'].nunique()} workstation(s)")
            st.dataframe(ts_df.head(10), use_container_width=True)
        else:
            st.warning("No time study data detected in your uploaded file(s). "
                       "Make sure your file has columns: Workstation, Operation, Cycle Time (s).")

    if ts_df.empty:
        st.info("Upload a time study file with Workstation, Operation, and Cycle Time (s) columns to enable Line Balancing Simulation.")
    else:
        if takt is None:
            st.warning("⚠️ Takt time not set. Set it in the Required Setup panel above to enable full analysis.")

        # ── KPIs ──────────────────────────────────────────────────────────────
        st.subheader("Simulated Line Status",
                     help="A snapshot of your line based on the current simulation settings. "
                          "Stations and Operators update as you make changes in the Rebalancing Simulation below. "
                          "Total Work Content is the sum of all operation cycle times per unit. "
                          "Balance Efficiency tells you how evenly work is spread across stations. 100% is perfect. "
                          "Bottleneck identifies the station with the highest load that is limiting your throughput.")

        n_stations   = len(balance_df)
        total_ct     = balance_df["Total CT (s)"].sum()
        n_operators  = len(st.session_state.get("wi_operators", []))

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🛠️ Stations", n_stations,
                  help="Total number of workstations currently in the simulation.")
        k2.metric("👷 Operators", n_operators,
                  help="Number of operators defined in the Simulation Control Panel. "
                       "Add or remove operators in the Rebalancing Simulation section below.")
        k3.metric("⏳ Total Work Content", fmt_time(total_ct),
                  help="The total operator time needed to produce one unit across all workstations.")
        if takt:
            be = total_ct / (n_stations * takt) * 100
            k4.metric("🏅 Balance Efficiency", f"{be:.1f}%",
                      help="How evenly work is spread across all stations. "
                           "100% means every station carries exactly equal load. "
                           "Below 100% means some stations have idle time while others are busier. "
                           "Over 100% means the total work exceeds what the line can handle at the current takt rate.")
            if len(all_bottlenecks) > 1:
                k5.metric("🔴 Bottleneck", f"{len(all_bottlenecks)} stations",
                          help=f"Multiple stations are over takt time: {', '.join(all_bottlenecks)}. "
                               "Focus on the highest CT station first.")
            else:
                k5.metric("🔴 Bottleneck", bottleneck or "—",
                          help="The workstation with the highest total cycle time. "
                               "This station limits how fast the entire line can run.")
        else:
            k4.metric("🔴 Bottleneck", bottleneck or "—",
                      help="The workstation with the highest total cycle time. Set takt time to see how far it is from your target.")
            k5.metric("🏅 Balance Efficiency", "Set takt time",
                      help="Set takt time in Required Setup above to calculate balance efficiency.")

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)

        if takt and bottleneck:
            bn_ct = balance_df.iloc[0]["Total CT (s)"]
            if len(all_bottlenecks) > 1:
                bn_list = ", ".join(
                    f"**{ws}** ({balance_df[balance_df['Workstation']==ws]['Total CT (s)'].values[0]:.1f}s)"
                    for ws in all_bottlenecks
                )
                st.error(
                    f"🔴 **{len(all_bottlenecks)} stations are over takt time ({takt:.1f}s):** {bn_list}. "
                    "Each of these stations cannot keep up with demand on its own. "
                    "Start with the station that has the highest cycle time and redistribute its work first."
                )
            elif bn_ct > takt:
                st.error(
                    f"🔴 **Bottleneck alert:** Workstation **{bottleneck}** has a total cycle time of "
                    f"**{bn_ct:.1f}s** ({fmt_time(bn_ct)}), which exceeds takt time of **{takt:.1f}s** ({fmt_time(takt)}). "
                    "This workstation cannot keep up with demand at current work content. "
                    "Consider splitting work to another station or reducing setup time."
                )
            else:
                overload_pct = bn_ct / takt * 100
                st.warning(
                    f"🟡 **{bottleneck}** is your constraint at **{overload_pct:.0f}% of takt** "
                    f"({bn_ct:.1f}s vs takt {takt:.1f}s). "
                    "Monitor closely as any variability may cause it to fall behind."
                )

        # ── Rebalancing Simulation ─────────────────────────────────────────────
        if takt:
            st.subheader("Rebalancing Simulation",
                         help="Simulate changes to your line layout and operator assignments. "
                              "Use the Auto-Optimisation Panel to get an AI-powered optimal rebalancing plan automatically. "
                              "The optimiser uses Integer Linear Programming to find the mathematically best station assignment given your constraints, "
                              "which is more accurate than traditional greedy heuristics. "
                              "Use the Simulation Control Panel to manually move operations between stations, "
                              "assign or split operations across operators, and add or remove stations and operators. "
                              "The station load chart shows how each station compares to takt before and after your changes. "
                              "The idle time chart shows how much waiting time exists per station, "
                              "helping you spot where waste is being created or eliminated by your changes. "
                              "If your time study file includes Min CT and Max CT columns, the chart also shows Monte Carlo P95 variability bars. "
                              "These represent the worst realistic station load across 1000 simulated production runs using a triangular distribution. "
                              "A station within takt at average load but over takt at P95 means it will occasionally fall behind under real-world variability, even if it looks fine on average. "
                              "Use the P95 bars to stress-test your balanced line before implementing changes on the shop floor.")

            # ── Session state init ────────────────────────────────────────────
            total_work = balance_df["Total CT (s)"].sum()
            if "wi_moves" not in st.session_state:
                st.session_state["wi_moves"] = []
            if "wi_stations" not in st.session_state:
                st.session_state["wi_stations"] = balance_df["Workstation"].tolist()

            # ── Auto-Optimisation Panel ───────────────────────────────────────
            with st.container(border=True):
                st.caption("**AUTO-OPTIMISATION PANEL**")
                st.caption("Set constraints for each operation, choose an optimisation target, then run. "
                           "The optimiser uses Integer Linear Programming (ILP) to find the mathematically optimal station assignment. "
                           "For most lines this completes in under 2 seconds. For very large lines it may take up to 10 seconds, after which it returns the best solution found within that time. "
                           "If the solver finds the proven best result you will see an Optimal badge. If it returns a strong but unverified result you will see a Best found badge. "
                           "If ILP is unavailable it falls back to a Longest Processing Time (LPT) heuristic, a fast greedy method that still produces a good result. "
                           "The suggested plan can be applied directly to the Simulation Control Panel below.")

                # Build base operation list
                base_ops_ao = (ts_lb.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                               .mean().reset_index()
                               .rename(columns={"Cycle Time (s)": "Avg CT (s)"}))
                base_ops_ao = base_ops_ao.sort_values(["Workstation", "Operation"]).reset_index(drop=True)

                # Init constraint state
                if "ao_constraints" not in st.session_state:
                    st.session_state["ao_constraints"] = {
                        row["Operation"]: {
                            "order": idx + 1,
                            "locked": False,
                            "splittable": True,
                        }
                        for idx, row in base_ops_ao.iterrows()
                    }
                # Constraint table
                with st.expander("⚙️ Operation Constraints", expanded=False):
                    st.caption("Set sequence order, lock operations to their current station, or uncheck Splittable to prevent an operation from being shared across multiple operators.")
                    ao_cols = st.columns([3, 2, 1, 1, 1])
                    ao_cols[0].markdown("**Operation**")
                    ao_cols[1].markdown("**Current Station**")
                    ao_cols[2].markdown("**Sequence**")
                    ao_cols[3].markdown("**Locked**")
                    ao_cols[4].markdown("**Splittable**")
                    for idx2, row in base_ops_ao.iterrows():
                        op_name = row["Operation"]
                        if op_name not in st.session_state["ao_constraints"]:
                            st.session_state["ao_constraints"][op_name] = {"order": idx2 + 1, "locked": False, "splittable": True}
                        c = st.session_state["ao_constraints"][op_name]
                        # Ensure order is never below 1
                        if c["order"] < 1:
                            c["order"] = idx2 + 1
                        r0, r1, r2, r3, r4 = st.columns([3, 2, 1, 1, 1])
                        r0.markdown(f"{op_name} ({row['Avg CT (s)']:.1f}s)")
                        r1.markdown(row["Workstation"])
                        new_order = r2.number_input("", min_value=1, value=c["order"], step=1,
                                                    key=f"ao_order_{idx2}_{op_name}", label_visibility="collapsed")
                        new_locked = r3.checkbox("", value=c["locked"],
                                                 key=f"ao_lock_{idx2}_{op_name}", label_visibility="collapsed")
                        new_splittable = r4.checkbox("", value=c["splittable"],
                                                     key=f"ao_split_{idx2}_{op_name}", label_visibility="collapsed")
                        st.session_state["ao_constraints"][op_name]["order"]      = new_order
                        st.session_state["ao_constraints"][op_name]["locked"]     = new_locked
                        st.session_state["ao_constraints"][op_name]["splittable"] = new_splittable

                # Optimisation target
                ao_target = st.radio(
                    "Optimisation target",
                    ["Balanced (Recommended)", "Minimise stations", "Minimise operators"],
                    horizontal=True, key="ao_target",
                    help="Choose what the optimiser should prioritise when suggesting how to redistribute operations. "
                         "Balanced (Recommended) tries to evenly spread work across both stations and operators at the same time. This is the best starting point for most factories. "
                         "Minimise stations packs operations as tightly as possible to reduce the number of workstations needed, which is useful when floor space is limited. "
                         "Minimise operators distributes work to reduce headcount, which is useful when labour cost is the main concern. "
                         "There is no single universally best answer since it depends on your factory priorities, which is why you can choose the target that fits your situation."
                )

                if st.button("🤖 Run Auto-Optimisation", type="primary", key="ao_run_btn"):
                    constraints = st.session_state["ao_constraints"]
                    current_ops_list = st.session_state.get("wi_operators", ["Operator A", "Operator B"])

                    # Separate locked and free operations
                    locked_ops  = {r["Operation"]: r["Workstation"]
                                   for _, r in base_ops_ao.iterrows()
                                   if constraints.get(r["Operation"], {}).get("locked", False)}
                    free_ops    = base_ops_ao[~base_ops_ao["Operation"].isin(locked_ops)].copy()

                    # Sort free ops by sequence order, then by CT descending (LPT)
                    free_ops["seq"] = free_ops["Operation"].map(
                        lambda o: constraints.get(o, {}).get("order", 999))
                    free_ops = free_ops.sort_values(["seq", "Avg CT (s)"], ascending=[True, False])

                    # Build station bins from current state (respecting locks)
                    current_stations_ao = st.session_state["wi_stations"]
                    station_loads_ao = {ws: 0.0 for ws in current_stations_ao}
                    # Add locked op loads to their stations
                    for _, row in base_ops_ao.iterrows():
                        op = row["Operation"]
                        if op in locked_ops:
                            ws = locked_ops[op]
                            if ws in station_loads_ao:
                                station_loads_ao[ws] += row["Avg CT (s)"]

                    # ── ILP Optimisation (PuLP) with LPT fallback ────────────
                    suggested_moves = []
                    op_station_assignments = {row["Operation"]: row["Workstation"]
                                              for _, row in base_ops_ao.iterrows()}
                    solution_method = "Heuristic (LPT)"

                    ilp_solved = False
                    if ML_AVAILABLE and takt and len(free_ops) > 0 and len(current_stations_ao) > 0:
                        try:
                            ops_list  = free_ops["Operation"].tolist()
                            cts_dict  = dict(zip(free_ops["Operation"], free_ops["Avg CT (s)"]))
                            stn_list  = current_stations_ao
                            n_ops     = len(ops_list)
                            n_stn     = len(stn_list)

                            prob = pulp.LpProblem("LineBalance", pulp.LpMinimize)

                            # x[i][j] = 1 if operation i assigned to station j
                            x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
                                  for j in range(n_stn)]
                                 for i in range(n_ops)]

                            # Cycle time of each station
                            station_ct = [
                                pulp.lpSum(cts_dict[ops_list[i]] * x[i][j] for i in range(n_ops))
                                + station_loads_ao.get(stn_list[j], 0)
                                for j in range(n_stn)
                            ]

                            # Auxiliary variable for max station load
                            max_load = pulp.LpVariable("max_load", lowBound=0)

                            # Each operation assigned to exactly one station
                            for i in range(n_ops):
                                prob += pulp.lpSum(x[i][j] for j in range(n_stn)) == 1

                            # max_load >= each station's CT (always required)
                            for j in range(n_stn):
                                prob += max_load >= station_ct[j]

                            # Target-specific objective — set ONCE based on target
                            if ao_target == "Minimise stations":
                                station_used = [pulp.LpVariable(f"used_{j}", cat="Binary")
                                                for j in range(n_stn)]
                                for j in range(n_stn):
                                    for i in range(n_ops):
                                        prob += station_used[j] >= x[i][j]
                                # Minimise number of stations used, with max_load as tiebreaker
                                prob += pulp.lpSum(station_used) + 0.01 * max_load

                            elif ao_target == "Minimise operators":
                                # Minimise total idle time = maximise station utilisation
                                # Fewer idle stations means fewer operators needed
                                total_idle = pulp.lpSum(
                                    takt - station_ct[j] for j in range(n_stn)
                                )
                                prob += total_idle

                            else:
                                # Balanced: minimise max station load
                                prob += max_load

                            # Sequence constraints: respect order numbers
                            seq_map = {row["Operation"]: constraints.get(row["Operation"], {}).get("order", 999)
                                       for _, row in free_ops.iterrows()}
                            sorted_ops = sorted(ops_list, key=lambda o: seq_map.get(o, 999))
                            for k in range(len(sorted_ops) - 1):
                                op_a = sorted_ops[k]
                                op_b = sorted_ops[k + 1]
                                if seq_map.get(op_a, 999) < seq_map.get(op_b, 999):
                                    i_a = ops_list.index(op_a)
                                    i_b = ops_list.index(op_b)
                                    # Station index of op_a must be <= station index of op_b
                                    prob += (pulp.lpSum(j * x[i_a][j] for j in range(n_stn)) <=
                                             pulp.lpSum(j * x[i_b][j] for j in range(n_stn)))

                            solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=10)
                            prob.solve(solver)

                            status_str = pulp.LpStatus[prob.status]
                            if status_str in ("Optimal", "Not Solved"):
                                # "Not Solved" with timeLimit means solver timed out but may have an incumbent
                                assignments_found = {}
                                for i, op in enumerate(ops_list):
                                    for j, ws in enumerate(stn_list):
                                        val = pulp.value(x[i][j])
                                        if val is not None and val > 0.5:
                                            assignments_found[op] = ws
                                # Only use if all operations were assigned
                                if len(assignments_found) == len(ops_list):
                                    for op, ws in assignments_found.items():
                                        op_station_assignments[op] = ws
                                        orig_rows = free_ops[free_ops["Operation"] == op]["Workstation"].values
                                        orig = orig_rows[0] if len(orig_rows) > 0 else ws
                                        if ws != orig:
                                            suggested_moves.append({
                                                "Operation": op,
                                                "From": orig,
                                                "To": ws,
                                                "CT (s)": round(cts_dict[op], 1),
                                            })
                                    if status_str == "Optimal":
                                        solution_method = "Optimal (ILP)"
                                    else:
                                        solution_method = "Best found (ILP)"
                                    ilp_solved = True
                        except Exception:
                            ilp_solved = False

                    if not ilp_solved:
                        # Fallback: LPT greedy heuristic
                        for _, row in free_ops.iterrows():
                            op   = row["Operation"]
                            ct   = row["Avg CT (s)"]
                            orig = row["Workstation"]
                            valid_stations = {ws: load for ws, load in station_loads_ao.items()
                                              if load + ct <= (takt if takt else float("inf"))}
                            if not valid_stations:
                                best_ws = min(station_loads_ao, key=station_loads_ao.get)
                            else:
                                best_ws = max(valid_stations, key=valid_stations.get)
                            station_loads_ao[best_ws] += ct
                            op_station_assignments[op] = best_ws
                            if best_ws != orig:
                                suggested_moves.append({
                                    "Operation": op,
                                    "From": orig,
                                    "To": best_ws,
                                    "CT (s)": round(ct, 1),
                                })
                        solution_method = "Heuristic (LPT)"

                    st.session_state["ao_solution_method"] = solution_method

                    # ── Operator Assignment LP (respects Splittable constraint) ──
                    suggested_op_assignments = {}
                    no_operators_warning     = False
                    min_ops_needed           = None

                    all_ops_sorted = base_ops_ao.copy()
                    all_ops_sorted["seq"] = all_ops_sorted["Operation"].map(
                        lambda o: constraints.get(o, {}).get("order", 999))
                    all_ops_sorted = all_ops_sorted.sort_values(["seq", "Avg CT (s)"],
                                                                 ascending=[True, False])
                    total_work_ao = all_ops_sorted["Avg CT (s)"].sum()

                    if not current_ops_list:
                        no_operators_warning = True
                    else:
                        if ao_target in ("Minimise stations", "Minimise operators") and takt:
                            min_ops_needed  = math.ceil(total_work_ao / takt)
                            n_assign_ops    = min(min_ops_needed, len(current_ops_list))
                            assign_ops_list = current_ops_list[:n_assign_ops]
                        elif ao_target in ("Minimise stations", "Minimise operators") and not takt:
                            assign_ops_list = current_ops_list
                        else:
                            assign_ops_list = current_ops_list

                        ops_for_assign  = all_ops_sorted["Operation"].tolist()
                        cts_for_assign  = dict(zip(all_ops_sorted["Operation"], all_ops_sorted["Avg CT (s)"]))
                        n_aops          = len(ops_for_assign)
                        n_aop_workers   = len(assign_ops_list)
                        splittable_map  = {row["Operation"]: constraints.get(row["Operation"], {}).get("splittable", True)
                                           for _, row in all_ops_sorted.iterrows()}

                        op_assigned = False
                        if ML_AVAILABLE and n_aops > 0 and n_aop_workers > 0:
                            if n_aop_workers == 1:
                                # Single operator: assign everything to them, no LP needed
                                for op in ops_for_assign:
                                    suggested_op_assignments[op] = {assign_ops_list[0]: 1.0}
                                op_assigned = True
                            else:
                                try:
                                    op_prob = pulp.LpProblem("OpAssign", pulp.LpMinimize)

                                    f = {}
                                    for i, op in enumerate(ops_for_assign):
                                        for k, wk in enumerate(assign_ops_list):
                                            is_splittable = splittable_map.get(op, True)
                                            cat = "Continuous" if is_splittable else "Binary"
                                            f[(i, k)] = pulp.LpVariable(f"f_{i}_{k}",
                                                                         lowBound=0, upBound=1, cat=cat)

                                    for i in range(n_aops):
                                        op_prob += pulp.lpSum(f[(i, k)] for k in range(n_aop_workers)) == 1

                                    op_loads = [
                                        pulp.lpSum(cts_for_assign[ops_for_assign[i]] * f[(i, k)]
                                                   for i in range(n_aops))
                                        for k in range(n_aop_workers)
                                    ]

                                    max_op_load = pulp.LpVariable("max_op_load", lowBound=0)
                                    op_prob += max_op_load

                                    for k in range(n_aop_workers):
                                        op_prob += max_op_load >= op_loads[k]

                                    op_solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=5)
                                    op_prob.solve(op_solver)

                                    if pulp.LpStatus[op_prob.status] in ("Optimal", "Not Solved"):
                                        all_assigned = True
                                        for i, op in enumerate(ops_for_assign):
                                            assigns = {}
                                            for k, wk in enumerate(assign_ops_list):
                                                val = pulp.value(f[(i, k)])
                                                if val is not None and val > 0.01:
                                                    assigns[wk] = val
                                            if assigns:
                                                total_f = sum(assigns.values())
                                                if total_f > 0:
                                                    suggested_op_assignments[op] = {
                                                        wk: round(v / total_f, 3)
                                                        for wk, v in assigns.items()
                                                    }
                                                else:
                                                    all_assigned = False
                                            else:
                                                all_assigned = False
                                        op_assigned = all_assigned
                                except Exception:
                                    op_assigned = False

                        if not op_assigned:
                            # Fallback: LPT bin-packing (no splitting)
                            op_bins = {op: 0.0 for op in assign_ops_list}
                            for _, row in all_ops_sorted.iterrows():
                                ct      = row["Avg CT (s)"]
                                best_op = min(op_bins, key=op_bins.get)
                                op_bins[best_op] += ct
                                suggested_op_assignments[row["Operation"]] = {best_op: 1.0}

                    st.session_state["ao_suggested_moves"]      = suggested_moves
                    st.session_state["ao_suggested_op_assigns"] = suggested_op_assignments
                    st.session_state["ao_no_operators_warning"] = no_operators_warning
                    st.session_state["ao_min_ops_needed"]       = min_ops_needed

                    # ── Compute structural suggestions using projected simulation loads ──
                    # Project what wi_loads would look like after applying suggested_moves
                    # Start from the current balance_df loads
                    projected_loads = {row["Workstation"]: row["Total CT (s)"]
                                       for _, row in balance_df.iterrows()}
                    # Add any newly added stations (from existing wi_moves, not suggested moves)
                    for move in st.session_state.get("wi_moves", []):
                        if move["Type"] == "Add Station" and move["To"] not in projected_loads:
                            projected_loads[move["To"]] = 0.0
                        elif move["Type"] == "Remove Station" and move["From"] in projected_loads:
                            del projected_loads[move["From"]]
                    # Apply existing wi_moves to get current simulation state
                    for move in st.session_state.get("wi_moves", []):
                        if move["Type"] == "Move Operation":
                            from_ws = move["From"]
                            to_ws   = move["To"]
                            op_ct   = move["Avg CT (s)"]
                            if from_ws in projected_loads and to_ws in projected_loads:
                                projected_loads[from_ws] = max(0.0, projected_loads[from_ws] - op_ct)
                                projected_loads[to_ws]   += op_ct
                    # Now apply the SUGGESTED moves on top
                    for move in suggested_moves:
                        from_ws = move["From"]
                        to_ws   = move["To"]
                        op_ct   = move["CT (s)"]
                        if from_ws in projected_loads and to_ws in projected_loads:
                            projected_loads[from_ws] = max(0.0, projected_loads[from_ws] - op_ct)
                            projected_loads[to_ws]   += op_ct

                    # Empty stations: projected load is zero (or near-zero due to float rounding)
                    empty_stations = [ws for ws, load in projected_loads.items() if load < 0.01]

                    # Stations over takt
                    stations_over_takt = []
                    if takt:
                        stations_over_takt = [ws for ws, load in projected_loads.items() if load > takt]
                        min_stations_needed = math.ceil(total_work_ao / takt)
                        stations_to_add = max(0, min_stations_needed - len(current_stations_ao))
                    else:
                        min_stations_needed = None
                        stations_to_add = 0

                    # Unused operators: operators with zero assignment after optimisation
                    assigned_operators = set()
                    for assigns in suggested_op_assignments.values():
                        assigned_operators.update(assigns.keys())
                    unused_operators = [op for op in current_ops_list if op not in assigned_operators]

                    st.session_state["ao_empty_stations"]    = empty_stations
                    st.session_state["ao_unused_operators"]  = unused_operators
                    st.session_state["ao_stations_over_takt"] = stations_over_takt
                    st.session_state["ao_stations_to_add"]   = stations_to_add
                    st.rerun()

                # Show no-operators warning if flagged
                if st.session_state.get("ao_no_operators_warning"):
                    st.warning("No operators are defined. Add operators in the Simulation Control Panel below to get operator assignment suggestions.")

                # Warn if minimise mode used without takt
                if (st.session_state.get("ao_solution_method") and
                        not takt and
                        st.session_state.get("ao_target") in ("Minimise stations", "Minimise operators")):
                    st.warning("Takt time is not set. Set it in the Required Setup panel above to enable the Minimise stations and Minimise operators targets. Operator assignments defaulted to balanced distribution.")

                # Show suggestions if available
                if st.session_state.get("ao_suggested_moves") is not None:
                    suggested_moves      = st.session_state["ao_suggested_moves"]
                    suggested_op_assigns = st.session_state.get("ao_suggested_op_assigns", {})

                    if not suggested_moves and not suggested_op_assigns:
                        st.success("✅ Your line is already optimally balanced. No changes needed.")
                    else:
                        method = st.session_state.get("ao_solution_method", "")
                        if method == "Optimal (ILP)":
                            st.success("**Optimal solution found** using Integer Linear Programming (ILP). "
                                       "This is the mathematically best possible station assignment given your constraints. "
                                       "This suggestion does not account for physical station layout or equipment requirements. Review each move before applying.")
                        elif method == "Best found (ILP)":
                            st.info("**Best solution found within 10 seconds** using Integer Linear Programming (ILP). "
                                    "Your line is large so the solver reached its time limit before proving optimality, but this is still a strong result from a thorough search. "
                                    "This suggestion does not account for physical station layout or equipment requirements. Review each move before applying.")
                        else:
                            st.info("**Good estimate found** using Longest Processing Time (LPT) heuristic. "
                                    "This approach assigns the most time-consuming operations first to balance load across stations. "
                                    "It produces a strong result quickly but cannot guarantee the absolute best solution. "
                                    "This suggestion does not account for physical station layout or equipment requirements. Review each move before applying.")
                        if suggested_moves:
                            st.markdown(f"**Suggested station moves ({len(suggested_moves)}):**")
                            st.dataframe(pd.DataFrame(suggested_moves), use_container_width=True, hide_index=True)

                        # ── Structural suggestions: empty stations / stations needed ──
                        empty_stations    = st.session_state.get("ao_empty_stations", [])
                        unused_operators  = st.session_state.get("ao_unused_operators", [])
                        stations_over_takt = st.session_state.get("ao_stations_over_takt", [])
                        stations_to_add   = st.session_state.get("ao_stations_to_add", 0)

                        if stations_to_add > 0:
                            st.warning(
                                f"**{len(stations_over_takt)} station(s) still exceed takt time after this plan.** "
                                f"Your total work content requires at least {stations_to_add} more station(s) to bring everything within takt. "
                                "Use 'Add station' in the Simulation Control Panel below to add them, move operations in, then re-run optimisation."
                            )
                        elif stations_over_takt:
                            st.warning(
                                f"**{len(stations_over_takt)} station(s) still exceed takt time after this plan: {', '.join(stations_over_takt)}.** "
                                "This is the best possible balance given your current station count and constraints. "
                                "One or more individual operations may have a cycle time larger than takt, making it impossible to stay within takt on any station they are assigned to. "
                                "Consider splitting those operations or increasing takt time."
                            )

                        if empty_stations:
                            st.warning(
                                f"**{len(empty_stations)} station(s) will be empty after this plan: "
                                f"{', '.join(empty_stations)}.** "
                                "If you want to reduce your station count, remove these from the Simulation Control Panel below using 'Remove station', then re-run optimisation."
                            )

                        if suggested_op_assigns:
                            op_assign_rows = [
                                {"Operation": op,
                                 "Assigned To": ", ".join(
                                     f"{o} ({round(f*100, 1)}%)" for o, f in assigns.items()
                                 )}
                                for op, assigns in suggested_op_assigns.items()
                            ]
                            st.markdown(f"**Suggested operator assignments ({len(op_assign_rows)}):**")
                            st.dataframe(pd.DataFrame(op_assign_rows), use_container_width=True, hide_index=True)

                        # ── Structural suggestions: unused operators ──────────
                        if unused_operators:
                            st.warning(
                                f"**{len(unused_operators)} operator(s) have no assignments in this plan: "
                                f"{', '.join(unused_operators)}.** "
                                "If you want to reduce headcount, remove these from the Operators Management section above, then re-run optimisation."
                            )

                        # ── Minimum operators needed (Minimise stations + Minimise operators) ──
                        min_ops = st.session_state.get("ao_min_ops_needed")
                        if min_ops is not None:
                            n_current = len(st.session_state.get("wi_operators", []))
                            if min_ops > n_current:
                                st.warning(
                                    f"**Minimum operators needed: {min_ops}**, but you only have {n_current} defined. "
                                    f"Add at least {min_ops - n_current} more operator(s) in the Operators Management section above, then re-run optimisation."
                                )
                            elif min_ops == n_current:
                                st.success(
                                    f"Your current {n_current} operator(s) is exactly the minimum needed to cover all operations within takt time."
                                )

                        ap1, ap2 = st.columns([1, 1])
                        with ap1:
                            if st.button("✅ Apply to Simulation", type="primary", key="ao_apply_btn"):
                                for move in suggested_moves:
                                    st.session_state["wi_moves"].append({
                                        "Type": "Move Operation",
                                        "Operation": move["Operation"],
                                        "From": move["From"],
                                        "To": move["To"],
                                        "Avg CT (s)": move["CT (s)"],
                                    })
                                if suggested_op_assigns:
                                    if "wi_op_assignments" not in st.session_state:
                                        st.session_state["wi_op_assignments"] = {}
                                    st.session_state["wi_op_assignments"].update(suggested_op_assigns)
                                st.session_state["ao_suggested_moves"]      = None
                                st.session_state["ao_suggested_op_assigns"] = {}
                                st.rerun()
                        with ap2:
                            if st.button("❌ Dismiss", key="ao_dismiss_btn"):
                                st.session_state["ao_suggested_moves"]      = None
                                st.session_state["ao_suggested_op_assigns"] = {}
                                st.rerun()

            # ── Simulation Control Panel (includes Operator Setup) ────────────
            with st.container(border=True):
                st.caption("**SIMULATION CONTROL PANEL**")

                # ── Operator Setup ────────────────────────────────────────────
                if "wi_operators" not in st.session_state:
                    if not ts_df.empty and "Operator" in ts_df.columns:
                        detected_ops = sorted(ts_df["Operator"].dropna().astype(str).str.strip().unique().tolist())
                    else:
                        detected_ops = ["Operator A", "Operator B"]
                    st.session_state["wi_operators"] = detected_ops

                ops_col1, ops_col2 = st.columns([3, 1])
                with ops_col1:
                    new_op_name = st.text_input("Operators Management", placeholder="e.g. Operator D", key="wi_new_op_input",
                                                help="Define the operators available on this line. "
                                                     "Type a name and click Add Operator to add. "
                                                     "Click any operator button to remove them. "
                                                     "If your time study file has an Operator column, operators are pre-populated automatically. "
                                                     "Use Undo Last to reverse the most recent add or remove. "
                                                     "Use Reset Operators to restore the original operator list from your uploaded file.")
                with ops_col2:
                    st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
                    if st.button("➕ Add Operator", type="primary", key="wi_add_op_btn"):
                        name = new_op_name.strip()
                        if name and name not in st.session_state["wi_operators"]:
                            if "wi_op_history" not in st.session_state:
                                st.session_state["wi_op_history"] = []
                            st.session_state["wi_op_history"].append({
                                "action": "add",
                                "operator": name,
                                "operators_snapshot": list(st.session_state["wi_operators"]),
                                "assignments_snapshot": {k: dict(v) for k, v in st.session_state.get("wi_op_assignments", {}).items()},
                            })
                            st.session_state["wi_operators"].append(name)
                            st.rerun()

                if st.session_state["wi_operators"]:
                    op_display_cols = st.columns(min(len(st.session_state["wi_operators"]), 6))
                    for i, op in enumerate(st.session_state["wi_operators"]):
                        with op_display_cols[i % 6]:
                            if st.button(f"🗑️ {op}", key=f"wi_rm_op_{i}",
                                         help=f"Click to remove {op} from the simulation"):
                                # Save to history before removing
                                if "wi_op_history" not in st.session_state:
                                    st.session_state["wi_op_history"] = []
                                st.session_state["wi_op_history"].append({
                                    "action": "remove",
                                    "operator": op,
                                    "operators_snapshot": list(st.session_state["wi_operators"]),
                                    "assignments_snapshot": {k: dict(v) for k, v in st.session_state.get("wi_op_assignments", {}).items()},
                                })
                                st.session_state["wi_operators"].remove(op)
                                # Clean up wi_moves
                                st.session_state["wi_moves"] = [
                                    m for m in st.session_state["wi_moves"]
                                    if not (m["Type"] in ("Assign Operator", "Split Operation") and op in m.get("Operators", []))
                                ]
                                # Clean up wi_op_assignments
                                updated_assignments = {}
                                for op_name, assigns in st.session_state.get("wi_op_assignments", {}).items():
                                    if op not in assigns:
                                        updated_assignments[op_name] = assigns
                                    else:
                                        remaining = {o: f for o, f in assigns.items() if o != op}
                                        if not remaining:
                                            pass
                                        else:
                                            total = sum(remaining.values())
                                            updated_assignments[op_name] = {o: f / total for o, f in remaining.items()}
                                st.session_state["wi_op_assignments"] = updated_assignments
                                st.rerun()

                # Undo / Reset for operators
                if st.session_state.get("wi_op_history"):
                    ob1, ob2 = st.columns([1, 1])
                    with ob1:
                        if st.button("↩️ Undo Last", key="wi_undo_op_btn"):
                            last = st.session_state["wi_op_history"].pop()
                            st.session_state["wi_operators"] = last["operators_snapshot"]
                            st.session_state["wi_op_assignments"] = last["assignments_snapshot"]
                            st.rerun()
                    with ob2:
                        if st.button("🔄 Reset Operators", key="wi_reset_op_btn"):
                            if not ts_df.empty and "Operator" in ts_df.columns:
                                reset_ops = sorted(ts_df["Operator"].dropna().astype(str).str.strip().unique().tolist())
                            else:
                                reset_ops = ["Operator A", "Operator B"]
                            st.session_state["wi_operators"] = reset_ops
                            st.session_state["wi_op_history"] = []
                            # Re-seed assignments from file
                            reset_asgn = {}
                            if not ts_df.empty and "Operator" in ts_df.columns:
                                base_ops = (ts_lb.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                                            .mean().reset_index())
                                for _, row in base_ops.iterrows():
                                    op_rows = ts_df[ts_df["Operation"] == row["Operation"]]["Operator"].dropna()
                                    if not op_rows.empty:
                                        assigned = str(op_rows.mode()[0]).strip()
                                        if assigned in reset_ops:
                                            reset_asgn[row["Operation"]] = {assigned: 1.0}
                            st.session_state["wi_op_assignments"] = reset_asgn
                            st.rerun()

                st.divider()

                current_stations  = st.session_state["wi_stations"]
                current_operators = st.session_state.get("wi_operators", ["Operator A"])

                # Persist Action Menu selection across reruns
                if "wi_action_type" not in st.session_state:
                    st.session_state["wi_action_type"] = "Move operation to station"

                # Build operation list reflecting simulated station moves
                op_list = (ts_lb.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                           .mean().reset_index()
                           .rename(columns={"Cycle Time (s)": "Avg CT (s)"}))
                for move in st.session_state["wi_moves"]:
                    if move["Type"] == "Move Operation":
                        op_list.loc[op_list["Operation"] == move["Operation"], "Workstation"] = move["To"]

                # Build operator assignment state: {operation: {operator: fraction 0-1}}
                if "wi_op_assignments" not in st.session_state:
                    # Seed from file's Operator column if available
                    init_assignments = {}
                    if not ts_df.empty and "Operator" in ts_df.columns:
                        for op_name in op_list["Operation"].unique():
                            op_rows = ts_df[ts_df["Operation"] == op_name]["Operator"].dropna()
                            if not op_rows.empty:
                                assigned = str(op_rows.mode()[0]).strip()
                                if assigned in current_operators:
                                    init_assignments[op_name] = {assigned: 1.0}
                    st.session_state["wi_op_assignments"] = init_assignments

                op_list["Label"] = (op_list["Operation"] + " [" +
                                    op_list["Workstation"] + "] (" +
                                    op_list["Avg CT (s)"].round(1).astype(str) + "s)")

                action_type = st.radio(
                    "Action Menu",
                    ["Move operation to station",
                     "Assign / split operation across operators",
                     "Add station", "Remove station"],
                    horizontal=True, key="wi_action_type",
                    help="Choose what change to make. Each change stacks on previous ones. Use Undo or Reset to go back."
                )

                # ── Action: Move operation to station ─────────────────────────
                if action_type == "Move operation to station":
                    # Station filter first for easier navigation
                    station_filter_move = st.selectbox(
                        "Operations filter by station",
                        ["All stations"] + sorted(op_list["Workstation"].unique().tolist()),
                        key="wi_station_filter_move",
                        help="Narrow the operation list to a specific station."
                    )
                    filtered_ops_move = op_list if station_filter_move == "All stations" \
                        else op_list[op_list["Workstation"] == station_filter_move]

                    wi_col1, wi_col2, wi_col3 = st.columns([3, 2, 1])
                    with wi_col1:
                        sel_op_label = st.selectbox(
                            "Operation to move",
                            ["(select)"] + filtered_ops_move["Label"].tolist(),
                            key="wi_op_sel",
                            help="Pick the operation. Its current station is shown in brackets."
                        )
                    with wi_col2:
                        sel_target_ws = st.selectbox(
                            "Move to station",
                            current_stations,
                            key="wi_target_ws",
                            help="Pick the destination station."
                        )
                    with wi_col3:
                        st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
                        if st.button("➕ Apply Change", type="primary", key="wi_add_btn"):
                            if sel_op_label != "(select)":
                                op_row    = op_list[op_list["Label"] == sel_op_label].iloc[0]
                                source_ws = op_row["Workstation"]
                                if sel_target_ws != source_ws:
                                    st.session_state["wi_moves"].append({
                                        "Type": "Move Operation",
                                        "Operation": op_row["Operation"],
                                        "From": source_ws,
                                        "To": sel_target_ws,
                                        "Avg CT (s)": round(op_row["Avg CT (s)"], 1),
                                    })
                                    st.rerun()

                # ── Action: Assign / split operation across operators ──────────
                elif action_type == "Assign / split operation across operators":
                    if not current_operators:
                        st.warning("Add operators in the Operators section above first.")
                    else:
                        st.caption("Assign one operation to one or more operators. "
                                   "Enter the % of that operation each operator handles. "
                                   "Fractions must sum to 100%.")
                        # Station filter first for easier navigation
                        station_filter_assign = st.selectbox(
                            "Operations filter by station",
                            ["All stations"] + sorted(op_list["Workstation"].unique().tolist()),
                            key="wi_station_filter_assign",
                            help="Narrow the operation list to a specific station."
                        )
                        filtered_ops_assign = op_list if station_filter_assign == "All stations" \
                            else op_list[op_list["Workstation"] == station_filter_assign]

                        sp_col1, sp_col2 = st.columns([2, 3])
                        with sp_col1:
                            sel_op_for_assign = st.selectbox(
                                "Operation to assign",
                                ["(select)"] + filtered_ops_assign["Label"].tolist(),
                                key="wi_assign_op_sel",
                                help="Pick the operation you want to assign or split."
                            )
                        with sp_col2:
                            sel_assignees = st.multiselect(
                                "Operator(s) involved",
                                current_operators,
                                key="wi_assign_ops_sel",
                                help="Select all operators who will share this operation."
                            )

                        if sel_op_for_assign != "(select)" and sel_assignees:
                            pct_cols = st.columns(min(len(sel_assignees), 6))
                            pct_inputs = {}
                            default_pct = int(100 / len(sel_assignees))
                            for i, op_name in enumerate(sel_assignees):
                                pct_inputs[op_name] = pct_cols[i % 6].number_input(
                                    f"{op_name} %",
                                    min_value=0, max_value=100,
                                    value=default_pct, step=5,
                                    key=f"wi_pct_{op_name}_{sel_op_for_assign}"
                                )
                            total_pct = sum(pct_inputs.values())
                            if total_pct != 100:
                                st.warning(f"Percentages currently sum to **{total_pct}%**. They must total exactly 100% before applying.")
                            else:
                                st.success(f"✅ Total: 100%")
                            if st.button("➕ Apply Change", type="primary",
                                         key="wi_apply_assign_btn",
                                         disabled=(total_pct != 100)):
                                op_row = op_list[op_list["Label"] == sel_op_for_assign].iloc[0]
                                assignments = {op: pct / 100.0 for op, pct in pct_inputs.items()}
                                st.session_state["wi_op_assignments"][op_row["Operation"]] = assignments
                                st.session_state["wi_moves"].append({
                                    "Type": "Assign Operator" if len(sel_assignees) == 1 else "Split Operation",
                                    "Operation": op_row["Operation"],
                                    "From": op_row["Workstation"],
                                    "To": op_row["Workstation"],
                                    "Avg CT (s)": round(op_row["Avg CT (s)"], 1),
                                    "Assignments": assignments,
                                    "Operators": list(assignments.keys()),
                                })
                                st.rerun()

                # ── Action: Add station ───────────────────────────────────────
                elif action_type == "Add station":
                    na_col1, na_col2 = st.columns([3, 1])
                    with na_col1:
                        new_ws_name = st.text_input(
                            "New station name",
                            placeholder="e.g. WS-07 Quality Check",
                            key="wi_new_ws_name"
                        )
                    with na_col2:
                        st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
                        if st.button("🆕 Add Station", type="primary", key="wi_add_ws_btn"):
                            if new_ws_name.strip() and new_ws_name.strip() not in current_stations:
                                st.session_state["wi_stations"].append(new_ws_name.strip())
                                st.session_state["wi_moves"].append({
                                    "Type": "Add Station", "Operation": "",
                                    "From": "", "To": new_ws_name.strip(), "Avg CT (s)": 0,
                                })
                                st.rerun()
                    st.caption("After adding, switch to 'Move operation to station' to assign operations to it.")

                # ── Action: Remove station ────────────────────────────────────
                elif action_type == "Remove station":
                    rm_col1, rm_col2 = st.columns([3, 1])
                    with rm_col1:
                        ws_to_remove = st.selectbox("Station to remove", current_stations, key="wi_rm_ws_sel")
                    with rm_col2:
                        st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)
                        if st.button("🗑️ Remove Station", type="primary", key="wi_rm_ws_btn"):
                            if ws_to_remove in st.session_state["wi_stations"]:
                                st.session_state["wi_stations"].remove(ws_to_remove)
                                st.session_state["wi_moves"].append({
                                    "Type": "Remove Station", "Operation": "",
                                    "From": ws_to_remove, "To": "", "Avg CT (s)": 0,
                                })
                                st.rerun()
                    ops_at_rm = op_list[op_list["Workstation"] == ws_to_remove]
                    if not ops_at_rm.empty:
                        st.warning(
                            f"**{ws_to_remove}** has {len(ops_at_rm)} operation(s) assigned "
                            f"({ops_at_rm['Avg CT (s)'].sum():.1f}s total). "
                            "Move them first to avoid losing their work content from the simulation."
                        )
                    else:
                        st.caption("No operations assigned. Safe to remove.")

                # ── Undo / Reset ──────────────────────────────────────────────
                if st.session_state["wi_moves"]:
                    rb1, rb2 = st.columns([1, 1])
                    with rb1:
                        if st.button("↩️ Undo Last Change", key="wi_undo_btn"):
                            removed = st.session_state["wi_moves"].pop()
                            if removed["Type"] == "Add Station" and removed["To"] in st.session_state["wi_stations"]:
                                st.session_state["wi_stations"].remove(removed["To"])
                            elif removed["Type"] == "Remove Station":
                                st.session_state["wi_stations"].insert(0, removed["From"])
                            elif removed["Type"] in ("Assign Operator", "Split Operation"):
                                # Revert to previous assignment for that operation
                                op_name = removed["Operation"]
                                remaining = [m for m in st.session_state["wi_moves"]
                                             if m["Type"] in ("Assign Operator", "Split Operation")
                                             and m["Operation"] == op_name]
                                if remaining:
                                    st.session_state["wi_op_assignments"][op_name] = remaining[-1]["Assignments"]
                                else:
                                    st.session_state["wi_op_assignments"].pop(op_name, None)
                            st.rerun()
                    with rb2:
                        if st.button("🔄 Reset All Changes", key="wi_reset_btn"):
                            st.session_state["wi_moves"]    = []
                            st.session_state["wi_stations"] = balance_df["Workstation"].tolist()
                            # Re-seed operator assignments from file defaults
                            reset_assignments = {}
                            if not ts_df.empty and "Operator" in ts_df.columns:
                                for op_name in op_list["Operation"].unique():
                                    op_rows = ts_df[ts_df["Operation"] == op_name]["Operator"].dropna()
                                    if not op_rows.empty:
                                        assigned = str(op_rows.mode()[0]).strip()
                                        if assigned in st.session_state.get("wi_operators", []):
                                            reset_assignments[op_name] = {assigned: 1.0}
                            st.session_state["wi_op_assignments"] = reset_assignments
                            st.rerun()

            # ── Current Operation Assignments (reference expander) ─────────────
            with st.expander("🏷️ Current Operation Assignment Detail", expanded=False):
                st.caption("Shows operations grouped by their current simulated station. Use this as a reference when selecting operations in the Action Menu above.")
                for ws in sorted(op_list["Workstation"].unique()):
                    ws_ops = op_list[op_list["Workstation"] == ws].copy()
                    total_ws_ct = ws_ops["Avg CT (s)"].sum()
                    if takt:
                        pct = total_ws_ct / takt * 100
                        icon = "🔴" if pct >= 100 else ("🟡" if pct >= 85 else "🟢")
                    else:
                        icon = "📌"
                    st.markdown(f"{icon} **{ws}** | Total CT: {fmt_time(total_ws_ct)}")
                    display_ops = ws_ops[["Operation", "Avg CT (s)"]].copy()
                    # Show operator assignment if exists
                    wi_op_assignments_ref = st.session_state.get("wi_op_assignments", {})
                    display_ops["Assigned To"] = display_ops["Operation"].apply(
                        lambda op: ", ".join(
                            f"{o} ({round(f*100, 1)}%)" for o, f in wi_op_assignments_ref[op].items()
                        ) if op in wi_op_assignments_ref else "Unassigned"
                    )
                    st.dataframe(display_ops, use_container_width=True, hide_index=True)

            # ── Apply all accumulated changes to simulate ─────────────────────
            wi_loads = {row["Workstation"]: row["Total CT (s)"]
                        for _, row in balance_df.iterrows()}
            for move in st.session_state["wi_moves"]:
                if move["Type"] == "Add Station" and move["To"] not in wi_loads:
                    wi_loads[move["To"]] = 0.0
            for move in st.session_state["wi_moves"]:
                if move["Type"] == "Remove Station" and move["From"] in wi_loads:
                    del wi_loads[move["From"]]
            for move in st.session_state["wi_moves"]:
                if move["Type"] == "Move Operation":
                    from_ws = move["From"]
                    to_ws   = move["To"]
                    op_ct   = move["Avg CT (s)"]
                    if from_ws in wi_loads and to_ws in wi_loads:
                        wi_loads[from_ws] = max(0.0, wi_loads[from_ws] - op_ct)
                        wi_loads[to_ws]   += op_ct

            # ── Compute operator loads from assignments ────────────────────────
            wi_op_assignments = st.session_state.get("wi_op_assignments", {})
            current_operators = st.session_state.get("wi_operators", [])
            op_loads_after = {op: 0.0 for op in current_operators}

            # Build current operation→station map after all moves
            op_station_map = {}
            base_op_list = (ts_lb.groupby(["Workstation", "Operation"])["Cycle Time (s)"]
                            .mean().reset_index()
                            .rename(columns={"Cycle Time (s)": "Avg CT (s)"}))
            for _, row in base_op_list.iterrows():
                op_station_map[row["Operation"]] = {"station": row["Workstation"],
                                                     "ct": row["Avg CT (s)"]}
            for move in st.session_state["wi_moves"]:
                if move["Type"] == "Move Operation" and move["Operation"] in op_station_map:
                    op_station_map[move["Operation"]]["station"] = move["To"]
                elif move["Type"] == "Move Operation" and move["Operation"] not in op_station_map:
                    # Operation moved to a new station not in original data — add it
                    op_station_map[move["Operation"]] = {"station": move["To"], "ct": move["Avg CT (s)"]}

            # Accumulate operator loads
            for op_name, info in op_station_map.items():
                ct = info["ct"]
                if op_name in wi_op_assignments:
                    for operator, fraction in wi_op_assignments[op_name].items():
                        if operator in op_loads_after:
                            op_loads_after[operator] += ct * fraction
                # Unassigned operations don't contribute to any operator load

            # ── Monte Carlo variability (if Min CT / Max CT available) ───────
            all_ws = list(wi_loads.keys())
            before_vals_chart = [balance_df[balance_df["Workstation"] == ws]["Total CT (s)"].values[0]
                                 if ws in balance_df["Workstation"].values else 0
                                 for ws in all_ws]
            after_vals_chart  = [wi_loads[ws] for ws in all_ws]
            n_ws              = len(all_ws)
            bar_colors_after  = [RED if v > takt else GREEN for v in after_vals_chart]

            has_op_data = any(v > 0 for v in op_loads_after.values())

            has_variability = (not ts_df.empty and
                               "Min CT (s)" in ts_df.columns and
                               "Max CT (s)" in ts_df.columns)
            mc_p50_loads = {}
            mc_p95_loads = {}
            if has_variability:
                N_SIM = 1000
                # Build per-operation triangular distributions
                op_mc = (ts_df.groupby("Operation").agg(
                    avg=("Cycle Time (s)", "mean"),
                    mn=("Min CT (s)", "mean"),
                    mx=("Max CT (s)", "mean"),
                ).reset_index())
                # For each simulation, compute station loads
                station_sim_loads = {ws: [] for ws in all_ws}
                for _ in range(N_SIM):
                    sim_op_ct = {}
                    for _, row in op_mc.iterrows():
                        mn  = min(row["mn"], row["avg"]) if pd.notna(row["mn"]) else row["avg"] * 0.8
                        mx  = max(row["mx"], row["avg"]) if pd.notna(row["mx"]) else row["avg"] * 1.2
                        avg = row["avg"]
                        sim_op_ct[row["Operation"]] = float(np.random.triangular(mn, avg, mx))
                    # Compute station loads for this simulation
                    sim_ws_loads = {ws: 0.0 for ws in all_ws}
                    for op_name, info in op_station_map.items():
                        ws = info["station"]
                        if ws in sim_ws_loads:
                            sim_ws_loads[ws] += sim_op_ct.get(op_name, info["ct"])
                    for ws in all_ws:
                        station_sim_loads[ws].append(sim_ws_loads.get(ws, 0.0))
                mc_p50_loads = {ws: float(np.percentile(vals, 50)) for ws, vals in station_sim_loads.items()}
                mc_p95_loads = {ws: float(np.percentile(vals, 95)) for ws, vals in station_sim_loads.items()}

            # ── Station load chart ─────────────────────────────────────────────
            fig_wi, ax_wi = plt.subplots(figsize=(max(8, n_ws * 1.2), 5))
            x     = np.arange(n_ws)
            width = 0.25 if has_variability else 0.35
            ax_wi.bar(x - width, before_vals_chart, width, color=STEEL_BLUE, alpha=0.4, label="Current (Avg)")
            ax_wi.bar(x, after_vals_chart, width, color=bar_colors_after, alpha=0.85,
                      label="After changes" if st.session_state["wi_moves"] else "After (Avg)")
            if has_variability:
                p95_vals = [mc_p95_loads.get(ws, 0) for ws in all_ws]
                p50_vals = [mc_p50_loads.get(ws, 0) for ws in all_ws]
                ax_wi.bar(x + width, p95_vals, width, color=ORANGE, alpha=0.75, label="P95 (Worst Realistic)")
                # P50 line markers
                for i, v in enumerate(p50_vals):
                    ax_wi.hlines(v, x[i] + width - width/2, x[i] + width + width/2,
                                 colors=NAVY, linewidth=2, label="P50 (Median)" if i == 0 else "")
            ax_wi.axhline(takt, color=GOLD, linewidth=2, linestyle="--",
                          label=f"Takt ({fmt_time(takt)})", zorder=3)
            ax_wi.set_xticks(x)
            ax_wi.set_xticklabels(all_ws, rotation=25, ha="right", fontsize=9)
            ax_wi.set_ylabel("Station Load (s)", fontsize=11)
            title = "Station Load: Before vs After"
            if has_variability:
                title += " (with Monte Carlo P95)"
            ax_wi.set_title(title, fontsize=12, fontweight="bold")
            ax_wi.grid(axis="y", alpha=0.3)
            legend_handles_ws = [
                Patch(facecolor=STEEL_BLUE, alpha=0.4, label="Before"),
                Patch(facecolor=GREEN, alpha=0.85, label="After: within takt"),
                Patch(facecolor=RED,   alpha=0.85, label="After: over takt"),
            ]
            if has_variability:
                legend_handles_ws += [
                    Patch(facecolor=ORANGE, alpha=0.75, label="P95 Worst Realistic"),
                    plt.Line2D([0], [0], color=NAVY, linewidth=2, label="P50 Median"),
                ]
            legend_handles_ws.append(
                plt.Line2D([0], [0], color=GOLD, linewidth=2, linestyle="--",
                           label=f"Takt ({fmt_time(takt)})")
            )
            ax_wi.legend(handles=legend_handles_ws, fontsize=8, loc="upper right")
            plt.tight_layout()
            render_chart(fig_wi)

            # ── Idle Time Comparison Chart ─────────────────────────────────────
            before_idle_vals = [max(0, takt - (balance_df[balance_df["Workstation"] == ws]["Total CT (s)"].values[0]
                                               if ws in balance_df["Workstation"].values else 0))
                                for ws in all_ws]
            after_idle_vals  = [max(0, takt - wi_loads.get(ws, 0)) for ws in all_ws]

            fig_idle_cmp, ax_idle_cmp = plt.subplots(figsize=(max(8, n_ws * 1.2), 4))
            x_idle = np.arange(n_ws)
            w_idle = 0.35
            ax_idle_cmp.bar(x_idle - w_idle/2, before_idle_vals, w_idle,
                            color=STEEL_BLUE, alpha=0.4, label="Idle Before")
            ax_idle_cmp.bar(x_idle + w_idle/2, after_idle_vals, w_idle,
                            color=ORANGE, alpha=0.85, label="Idle After")
            ax_idle_cmp.set_xticks(x_idle)
            ax_idle_cmp.set_xticklabels(all_ws, rotation=25, ha="right", fontsize=9)
            ax_idle_cmp.set_ylabel("Idle Time per Unit (s)", fontsize=11)
            ax_idle_cmp.set_title("Station Idle Time: Before vs After", fontsize=12, fontweight="bold")
            ax_idle_cmp.legend(fontsize=9)
            ax_idle_cmp.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            render_chart(fig_idle_cmp)

            if has_variability:
                at_risk = [ws for ws in all_ws if mc_p95_loads.get(ws, 0) > takt and wi_loads.get(ws, 0) <= takt]

            # ── Operator load chart (only when assignments exist) ──────────────
            if has_op_data:
                op_names_chart = list(op_loads_after.keys())
                op_vals_chart  = [op_loads_after[op] for op in op_names_chart]
                op_colors      = [RED if v > takt else GREEN for v in op_vals_chart]
                n_ops = len(op_names_chart)

                fig_op, ax_op = plt.subplots(figsize=(max(8, n_ops * 1.2), 5))
                ax_op.bar(op_names_chart, op_vals_chart, color=op_colors, alpha=0.85)
                ax_op.axhline(takt, color=GOLD, linewidth=2, linestyle="--",
                              label=f"Takt ({fmt_time(takt)})", zorder=3)
                ax_op.set_ylabel("Operator Load (s)", fontsize=11)
                ax_op.set_title("Operator Load vs Takt", fontsize=12, fontweight="bold")
                ax_op.tick_params(axis="x", rotation=25, labelsize=9)
                ax_op.grid(axis="y", alpha=0.3)
                ax_op.legend(fontsize=8)
                plt.tight_layout()
                render_chart(fig_op)

                # Operator utilization table
                op_util_rows = []
                for op_name in op_names_chart:
                    load = op_loads_after[op_name]
                    util = load / takt * 100 if takt else 0
                    if util > 100:   status = "🔴 Overloaded"
                    elif util < 50:  status = "🟡 Underutilised"
                    else:            status = "🟢 OK"
                    # Which operations assigned to this operator
                    ops_assigned = [op for op, assigns in wi_op_assignments.items()
                                    if op_name in assigns]
                    op_util_rows.append({
                        "Operator": op_name,
                        "Load (s)": round(load, 1),
                        "Utilization %": round(util, 1),
                        "Status": status,
                        "Operations": ", ".join(ops_assigned) if ops_assigned else "None assigned",
                    })
                op_util_df = pd.DataFrame(op_util_rows)
                st.markdown("**Operator Utilization Summary**")
                st.dataframe(op_util_df, use_container_width=True, hide_index=True)

                overloaded_ops = [r["Operator"] for r in op_util_rows if "Overloaded" in r["Status"]]
                underutil_ops  = [r["Operator"] for r in op_util_rows if "Underutilised" in r["Status"]]
                if overloaded_ops:
                    st.error(f"🔴 Overloaded operators (load exceeds takt): {', '.join(overloaded_ops)}. "
                             "Split their operations across more operators or reduce their workload.")
                if underutil_ops:
                    st.info(f"🟡 Underutilised operators (below 50% of takt): {', '.join(underutil_ops)}. "
                            "Consider assigning them additional operations or reducing headcount.")

                st.download_button(
                    "⬇️ Download Operator Utilization",
                    to_excel_bytes(op_util_df),
                    "operator_utilization.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_op_util",
                )

                # end operator section
            else:
                pass

            # ── Change log and metrics ─────────────────────────────────────────
            if st.session_state["wi_moves"]:
                st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("**Simulation Change Log**")
                log_rows = []
                for i, move in enumerate(st.session_state["wi_moves"], 1):
                    if move["Type"] == "Move Operation":
                        log_rows.append({"#": i, "Change": "Move Operation",
                                         "Detail": f"{move['Operation']} ({move['Avg CT (s)']}s): {move['From']} → {move['To']}"})
                    elif move["Type"] == "Add Station":
                        log_rows.append({"#": i, "Change": "Add Station", "Detail": move["To"]})
                    elif move["Type"] == "Remove Station":
                        log_rows.append({"#": i, "Change": "Remove Station", "Detail": move["From"]})
                    elif move["Type"] == "Assign Operator":
                        assigns_str = ", ".join(f"{op} 100%" for op in move.get("Operators", []))
                        log_rows.append({"#": i, "Change": "Assign Operator",
                                         "Detail": f"{move['Operation']}: {assigns_str}"})
                    elif move["Type"] == "Split Operation":
                        assigns_str = ", ".join(
                            f"{op} {round(frac*100, 1)}%"
                            for op, frac in move.get("Assignments", {}).items()
                        )
                        log_rows.append({"#": i, "Change": "Split Operation",
                                         "Detail": f"{move['Operation']}: {assigns_str}"})
                log_df = pd.DataFrame(log_rows)
                st.dataframe(log_df, use_container_width=True, hide_index=True)

                before_eff       = balance_df["Total CT (s)"].sum() / (len(balance_df) * takt) * 100 if takt else 0
                after_eff        = sum(wi_loads.values()) / (len(wi_loads) * takt) * 100 if wi_loads and takt else 0
                before_over      = sum(1 for ct in balance_df["Total CT (s)"] if takt and ct > takt)
                after_over       = sum(1 for ct in wi_loads.values() if takt and ct > takt)
                before_idle_s    = sum(max(0, takt - ct) for ct in balance_df["Total CT (s)"]) if takt else 0
                after_idle_s     = sum(max(0, takt - ct) for ct in wi_loads.values()) if takt else 0

                wm1, wm2, wm3, wm4, wm5 = st.columns(5)
                wm1.metric("Station Balance Before", f"{before_eff:.1f}%")
                wm2.metric("Station Balance After",  f"{after_eff:.1f}%",
                           delta=f"{after_eff - before_eff:+.1f}%")
                wm3.metric("Stations Over Takt After", after_over,
                           delta=f"{after_over - before_over}", delta_color="inverse")
                wm4.metric("Total Idle Before", fmt_time(before_idle_s))
                wm5.metric("Total Idle After", fmt_time(after_idle_s),
                           delta=f"{fmt_time(abs(after_idle_s - before_idle_s))} {'saved' if after_idle_s < before_idle_s else 'added'}",
                           delta_color="normal" if after_idle_s <= before_idle_s else "inverse")

                problem_stations = [ws for ws, ct in wi_loads.items() if takt and ct > takt]
                if problem_stations:
                    st.warning(f"Stations still over takt: {', '.join(problem_stations)}. "
                               "Continue rebalancing or add stations.")
                else:
                    st.success("✅ All stations are within takt time after your planned changes.")

                st.download_button(
                    "⬇️ Download Rebalancing Plan",
                    to_excel_bytes(log_df),
                    "rebalancing_plan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_wi_moves",
                )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – MACHINE HEALTH DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.header("🩺 Machine Health Diagnostics",
              help="Diagnose the health and reliability of your machines from overall equipment effectiveness through to failure pattern analysis and maintenance planning. "
                   "Work through the sections from top to bottom: start with OEE Overview to see your overall equipment effectiveness score, "
                   "then use Downtime Analysis to find what is stopping your machines, "
                   "then check Reliability Metrics to understand failure frequency and recovery time, "
                   "then review Trend Analysis to see if things are improving or getting worse over time, "
                   "then compare Shift Performance to separate machine problems from operator or shift problems, "
                   "and finish with Maintenance Intelligence for a data-driven maintenance schedule recommendation.")

    with st.expander("🔍 Data Overview", expanded=False):
        if not prod_df.empty:
            st.success(f"✅ Production log loaded: {len(prod_df):,} records across {prod_df['Machine'].nunique()} machine(s)")
            for status, msg in prod_reports:
                if "❌" in status:   st.error(f"{status}: {msg}")
                elif "✅" in status: st.success(f"{status}: {msg}")
                elif "⚠️" in status: st.warning(f"{status}: {msg}")
            st.dataframe(prod_df.head(10), use_container_width=True)
        else:
            st.info("No production log detected. Upload a file with Planned Hours, Run Hours, Good Units, and Total Units to enable OEE analysis. You can still use Manual Entry in the OEE Overview section.")
        if not dt_df.empty:
            st.success(f"✅ Downtime log loaded: {len(dt_df):,} events across {dt_df['Machine'].nunique()} machine(s)")
            for status, msg in dt_reports:
                if "❌" in status:   st.error(f"{status}: {msg}")
                elif "✅" in status: st.success(f"{status}: {msg}")
                elif "⚠️" in status: st.warning(f"{status}: {msg}")
            st.dataframe(dt_df.head(10), use_container_width=True)
        else:
            st.info("No downtime log detected. Upload a file with Machine, Duration (min), Date, and Category columns to enable Downtime Analysis, Reliability Metrics, and Maintenance Intelligence.")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — OEE OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("OEE Overview",
                 help="OEE (Overall Equipment Effectiveness) measures how productively your machines are being used. "
                      "It combines three components: Availability (was the machine running when it should be?), "
                      "Performance (was it running at the intended speed?), and Quality (were the parts good?). "
                      "A score of 85% or above is considered world class. "
                      "Use this section to identify which of the three components is dragging your score down, "
                      "then scroll to Downtime Analysis to find the specific causes.")

    oee_source = st.radio("Input method", ["📝 Manual Entry", "📂 From Production Log"],
                          horizontal=True, key="oee_source")

    if oee_source == "📝 Manual Entry":
        c1, c2, c3 = st.columns(3)
        with c1:
            planned_h  = st.number_input("Planned production time (hrs)", min_value=0.1, value=8.0, step=0.5)
            downtime_h = st.number_input("Unplanned downtime (hrs)", min_value=0.0, value=1.0, step=0.1)
        with c2:
            ideal_ct_s   = st.number_input("Ideal cycle time (s/unit)", min_value=0.1, value=30.0, step=0.5)
            total_units  = st.number_input("Total units produced", min_value=0, value=800, step=10)
        with c3:
            good_units   = st.number_input("Good units (no defects)", min_value=0, value=760, step=10)

        run_h        = planned_h - downtime_h
        availability = run_h / planned_h if planned_h > 0 else 0
        ideal_units  = (run_h * 3600) / ideal_ct_s if ideal_ct_s > 0 else 0
        performance  = min(total_units / ideal_units, 1.0) if ideal_units > 0 else 0
        quality      = good_units / total_units if total_units > 0 else 0
        oee_val      = calc_oee(availability, performance, quality)
        oee_label, oee_color = oee_rating(oee_val)

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("📊 OEE", f"{oee_val*100:.1f}%",
                  delta=oee_label, delta_color="off")
        o2.metric("🟢 Availability", f"{availability*100:.1f}%")
        o3.metric("⚡ Performance",  f"{performance*100:.1f}%")
        o4.metric("✅ Quality",      f"{quality*100:.1f}%")

        # OEE waterfall chart
        fig_oee, ax_oee = plt.subplots(figsize=(7, 4))
        labels  = ["Availability", "Performance", "Quality", "OEE"]
        values  = [availability * 100, performance * 100, quality * 100, oee_val * 100]
        colors_ = [STEEL_BLUE, LIGHT_BLUE, GREEN, RED if oee_val < 0.6 else (GOLD if oee_val < 0.85 else GREEN)]
        bars    = ax_oee.bar(labels, values, color=colors_, alpha=0.85, width=0.5)
        ax_oee.axhline(85, color=GREEN, linewidth=1.5, linestyle="--", alpha=0.7, label="World Class (85%)")
        ax_oee.axhline(60, color=GOLD,  linewidth=1.5, linestyle=":",  alpha=0.7, label="Acceptable (60%)")
        for bar, val in zip(bars, values):
            ax_oee.text(bar.get_x() + bar.get_width() / 2, val + 1.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax_oee.set_ylim(0, 115)
        ax_oee.set_ylabel("Score (%)", fontsize=11)
        ax_oee.set_title("OEE Breakdown", fontsize=13, fontweight="bold", pad=10)
        ax_oee.legend(fontsize=9)
        ax_oee.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        render_chart(fig_oee)

        st.markdown(
            f"<div style='padding:12px 16px; border-radius:8px; border-left:4px solid "
            f"{'#c0392b' if oee_val < 0.6 else ('#d4a017' if oee_val < 0.85 else '#3d8b5e')}; background:#1e2d3d; margin-top:0.5rem;'>"
            f"<span style='font-weight:700; font-size:1rem; color:#ffffff;'>OEE: {oee_val*100:.1f}% "
            f"<span style='color:{'#c0392b' if oee_val < 0.6 else ('#d4a017' if oee_val < 0.85 else '#3d8b5e')};'>({oee_label})</span></span>"
            f"<br><span style='font-size:0.82rem; color:#b0c8e0;'>"
            f"{'🔴 Availability is your biggest loss. Unplanned downtime is the primary drag on OEE. Scroll to Downtime Analysis to find the causes.' if availability < min(performance, quality) else ''}"
            f"{'🔴 Performance is your biggest loss. The machine is running but not at its intended speed. Check for micro-stops, slow cycles, or reduced feed rates.' if performance < min(availability, quality) and performance < availability else ''}"
            f"{'🔴 Quality is your biggest loss. The machine is running but producing defective parts. Review your quality rejection log for root causes.' if quality < min(availability, performance) and quality < availability and quality < performance else ''}"
            f"{'✅ All three OEE components are strong. Focus on maintaining current performance.' if availability >= 0.9 and performance >= 0.95 and quality >= 0.99 else ''}"
            f"</span></div>",
            unsafe_allow_html=True
        )

    else:
        # From production log
        if prod_df.empty:
            st.info("No production log detected. Upload a file with Planned Hours, Run Hours, Good Units, Total Units columns.")
        else:
            machines = ["All Machines"] + sorted(prod_df["Machine"].unique().tolist())
            sel_machine = st.selectbox("Filter by Machine / Line", machines, key="oee_machine")

            prod_view = prod_df if sel_machine == "All Machines" else prod_df[prod_df["Machine"] == sel_machine]

            # Aggregate OEE by machine
            oee_rows = []
            for machine, grp in prod_df.groupby("Machine"):
                plan_h  = grp["Planned Hours"].sum()
                run_h   = grp["Run Hours"].sum()
                good_u  = grp["Good Units"].sum()
                total_u = grp["Total Units"].sum()
                avail   = run_h / plan_h if plan_h > 0 else 0

                if "Ideal Cycle Time (s)" in grp.columns and grp["Ideal Cycle Time (s)"].notna().any():
                    ict     = grp["Ideal Cycle Time (s)"].mean()
                    ideal_u = (run_h * 3600) / ict if ict > 0 else 0
                    perf    = min(total_u / ideal_u, 1.0) if ideal_u > 0 else total_u / (plan_h * 3600) if plan_h > 0 else 0
                else:
                    # Fallback: performance estimated from run rate vs planned rate
                    perf = min(total_u / (run_h * 3600 / 60), 1.0) if run_h > 0 else 0
                    if perf == 0 or perf > 1:
                        perf = 1.0  # can't calculate without ideal CT, assume 100%

                qual    = good_u / total_u if total_u > 0 else 0
                oee_v   = calc_oee(avail, perf, qual)
                label, _ = oee_rating(oee_v)

                oee_rows.append({
                    "Machine":         machine,
                    "Planned Hrs":     round(plan_h, 1),
                    "Run Hrs":         round(run_h, 1),
                    "Availability %":  round(avail * 100, 1),
                    "Performance %":   round(perf  * 100, 1),
                    "Quality %":       round(qual  * 100, 1),
                    "OEE %":           round(oee_v * 100, 1),
                    "Rating":          label,
                    "Good Units":      int(good_u),
                    "Total Units":     int(total_u),
                })

            oee_summary = pd.DataFrame(oee_rows).sort_values("OEE %")

            # OEE bar chart per machine
            fig_oeem, ax_oeem = plt.subplots(figsize=(max(8, len(oee_summary) * 1.2), 5))
            bar_c = [RED if v < 60 else (GOLD if v < 85 else GREEN) for v in oee_summary["OEE %"]]
            ax_oeem.barh(oee_summary["Machine"], oee_summary["OEE %"], color=bar_c, alpha=0.85)
            ax_oeem.axvline(85, color=GREEN, linewidth=1.5, linestyle="--", label="World Class 85%")
            ax_oeem.axvline(60, color=GOLD,  linewidth=1.5, linestyle=":",  label="Acceptable 60%")
            for i, (_, row) in enumerate(oee_summary.iterrows()):
                ax_oeem.text(row["OEE %"] + 0.5, i, f'{row["OEE %"]:.1f}%', va="center", fontsize=9)
            ax_oeem.set_xlabel("OEE (%)", fontsize=11)
            ax_oeem.set_title("OEE by Machine / Line", fontsize=13, fontweight="bold", pad=10)
            ax_oeem.legend(fontsize=9)
            ax_oeem.set_xlim(0, 110)
            ax_oeem.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            render_chart(fig_oeem)

            def style_oee(row):
                v = row.get("OEE %", 100)
                if v < 60:
                    return ["background-color:#fdf0f0; color:#1a1a1a"] * len(row)
                elif v < 85:
                    return ["background-color:#fdf4e0; color:#1a1a1a"] * len(row)
                return ["background-color:#e6f0fa; color:#1a1a1a"] * len(row)

            st.dataframe(oee_summary.style.apply(style_oee, axis=1), use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download OEE Summary",
                to_excel_bytes(oee_summary),
                "oee_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_oee",
            )

            # OEE trend over time (if Date available)
            if "Date" in prod_view.columns and prod_view["Date"].notna().any():
                st.subheader("OEE Trend Over Time")
                prod_time = prod_view.copy()
                prod_time["Month"] = prod_time["Date"].dt.to_period("M").dt.to_timestamp()
                monthly_oee = []
                for month, grp in prod_time.groupby("Month"):
                    ph = grp["Planned Hours"].sum()
                    rh = grp["Run Hours"].sum()
                    gu = grp["Good Units"].sum()
                    tu = grp["Total Units"].sum()
                    av = rh / ph if ph > 0 else 0
                    ql = gu / tu if tu > 0 else 0
                    pf = 1.0  # simplified without ideal CT
                    monthly_oee.append({"Month": month, "OEE %": round(calc_oee(av, pf, ql) * 100, 1)})
                monthly_oee_df = pd.DataFrame(monthly_oee)

                fig_trend, ax_trend = plt.subplots(figsize=(10, 4))
                ax_trend.plot(monthly_oee_df["Month"], monthly_oee_df["OEE %"],
                              color=STEEL_BLUE, linewidth=2, marker="o", markersize=5)
                ax_trend.fill_between(monthly_oee_df["Month"], monthly_oee_df["OEE %"],
                                      alpha=0.15, color=STEEL_BLUE)
                ax_trend.axhline(85, color=GREEN, linewidth=1.5, linestyle="--", label="World Class 85%", alpha=0.7)
                ax_trend.axhline(60, color=GOLD,  linewidth=1.5, linestyle=":",  label="Acceptable 60%",  alpha=0.7)
                ax_trend.set_ylabel("OEE (%)", fontsize=11)
                ax_trend.set_title(f"Monthly OEE Trend {sel_machine}", fontsize=12, fontweight="bold")
                ax_trend.legend(fontsize=9)
                ax_trend.grid(alpha=0.3)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                render_chart(fig_trend)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — DOWNTIME ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Downtime Analysis",
                 help="Shows what is causing your machines to stop. "
                      "The Pareto chart ranks failure categories by total time lost so you can focus on the biggest causes first. "
                      "Pareto principle: fixing the top 2 or 3 categories typically resolves 80% of your downtime. "
                      "Use the machine filter to drill into a specific piece of equipment. "
                      "Requires a downtime log file with Machine, Duration (min), and Category columns.")

    if dt_df.empty:
        st.info("Upload a downtime log file with Machine, Duration (min), Date, and Category columns to enable Downtime Analysis.")
    else:
        dt_machines = ["All Machines"] + sorted(dt_df["Machine"].unique().tolist())
        sel_dt_machine = st.selectbox("Filter by Machine", dt_machines, key="dt_machine")
        dt_view = dt_df if sel_dt_machine == "All Machines" else dt_df[dt_df["Machine"] == sel_dt_machine]

        # KPIs
        total_dt_min = dt_view["Duration (min)"].sum()
        num_events   = len(dt_view)
        avg_dt       = dt_view["Duration (min)"].mean()
        top_category = dt_view.groupby("Category")["Duration (min)"].sum().idxmax() if not dt_view.empty else "N/A"

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("⏱️ Total Downtime", f"{total_dt_min:,.0f} min ({total_dt_min/60:.1f} hrs)",
                  help="Total minutes of unplanned downtime recorded in the selected period and machine filter.")
        d2.metric("🔢 Downtime Events", f"{num_events:,}",
                  help="Total number of individual downtime events recorded.")
        d3.metric("📊 Avg Duration", f"{avg_dt:.0f} min per event",
                  help="Average duration of a single downtime event. High average duration points to slow recovery. High event count with low average duration points to frequent nuisance faults.")
        d4.metric("🔴 Top Cause", top_category,
                  help="The failure category responsible for the most total downtime. This is your highest-priority target for maintenance improvement.")

        # Pareto by category (duration)
        cat_sum = (dt_view.groupby("Category")["Duration (min)"]
                   .sum().sort_values(ascending=False).reset_index())
        cat_sum["Cumulative %"] = (cat_sum["Duration (min)"].cumsum() /
                                   cat_sum["Duration (min)"].sum() * 100).round(1)

        fig_dt, ax_dt1 = plt.subplots(figsize=(max(8, len(cat_sum) * 1.4), 5))
        ax_dt1.bar(cat_sum["Category"], cat_sum["Duration (min)"], color=STEEL_BLUE, alpha=0.85)
        ax_dt1.set_ylabel("Total Downtime (min)", fontsize=11)
        ax_dt1.tick_params(axis="x", rotation=30, labelsize=9)
        ax_dt2 = ax_dt1.twinx()
        ax_dt2.plot(cat_sum["Category"], cat_sum["Cumulative %"], color=GOLD,
                    marker="o", linewidth=2, markersize=5, label="Cumulative %")
        ax_dt2.axhline(80, color=ORANGE, linewidth=1.5, linestyle="--", alpha=0.7)
        ax_dt2.set_ylabel("Cumulative %", fontsize=11)
        ax_dt2.set_ylim(0, 115)
        ax_dt2.yaxis.set_major_formatter(mticker.PercentFormatter())
        lines2, labels2 = ax_dt2.get_legend_handles_labels()
        ax_dt2.legend(lines2, labels2, loc="center right", fontsize=9)
        ax_dt1.set_title(f"Downtime Pareto by Category: {sel_dt_machine}", fontsize=13, fontweight="bold", pad=10)
        plt.tight_layout()
        render_chart(fig_dt)

        # Callout: which categories fall in the top 80%
        top80_cats = cat_sum[cat_sum["Cumulative %"] <= 80]["Category"].tolist()
        if not top80_cats:
            top80_cats = [cat_sum.iloc[0]["Category"]]
        st.warning(f"The top cause(s) of downtime are: **{', '.join(top80_cats)}**. "
                   "Fixing these will address 80% of your total downtime. Scroll to Reliability Metrics to see how often and how long each lasts.")

        # Downtime detail expander
        with st.expander("🏷️ Downtime Event Detail", expanded=False):
            st.caption("Full list of downtime events with duration and category.")

            # ── Anomaly Detection (Isolation Forest) ──────────────────────────
            if ML_AVAILABLE and len(dt_view) >= 10:
                try:
                    iso_features = dt_view[["Duration (min)"]].copy()
                    iso_features["hour_of_day"] = pd.to_datetime(
                        dt_view["Date"], errors="coerce"
                    ).dt.hour.fillna(0) if "Date" in dt_view.columns else 0
                    iso = IsolationForest(contamination=0.1, random_state=42)
                    dt_view = dt_view.copy()
                    dt_view["Anomaly"] = iso.fit_predict(iso_features)
                    dt_view["Anomaly"] = dt_view["Anomaly"].map({1: "Normal", -1: "⚠️ Unusual"})
                    n_anomalies = (dt_view["Anomaly"] == "⚠️ Unusual").sum()
                    if n_anomalies > 0:
                        st.warning(
                            f"**Anomaly Detection found {n_anomalies} unusual downtime event(s)** out of {len(dt_view)} total. "
                            "These are flagged as outliers because their duration or timing is significantly different from typical events. "
                            "They are marked as 'Unusual' in the table below and may warrant individual investigation."
                        )
                except Exception:
                    pass

            st.dataframe(dt_view, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Downtime Log", to_excel_bytes(dt_view),
                               "downtime_log.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_dt")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — RELIABILITY METRICS (MTBF / MTTR)
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Reliability Metrics",
                 help="MTBF (Mean Time Between Failures) is the average time a machine runs before breaking down. "
                      "Higher is better. A declining MTBF over time means the machine is failing more frequently. "
                      "MTTR (Mean Time To Repair) is the average time to get the machine running again after a failure. "
                      "Lower is better. High MTTR means your recovery process is slow. "
                      "Together these two numbers tell you whether you have a frequency problem, a recovery problem, or both. "
                      "Requires a downtime log file with Machine, Date, and Duration (min) columns.")

    if dt_df.empty:
        st.info("Upload a downtime log with Machine, Date, and Duration (min) columns to calculate MTBF and MTTR.")
    elif "Date" not in dt_df.columns or not dt_df["Date"].notna().any():
        st.info("Add a Date column to your downtime log to enable MTBF and MTTR calculations.")
    else:
        rel_rows = []
        for machine, grp in dt_df.groupby("Machine"):
            grp = grp.dropna(subset=["Date"]).sort_values("Date")
            n_failures = len(grp)
            mttr = grp["Duration (min)"].mean()
            if n_failures >= 2:
                total_span_min = (grp["Date"].max() - grp["Date"].min()).total_seconds() / 60
                total_dt_m = grp["Duration (min)"].sum()
                uptime_min = max(0, total_span_min - total_dt_m)
                mtbf = uptime_min / (n_failures - 1) if n_failures > 1 else uptime_min
            else:
                mtbf = None
            availability_rel = mtbf / (mtbf + mttr) * 100 if mtbf and mttr else None

            # ── Weibull Failure Pattern Analysis ─────────────────────────────
            pattern = "Insufficient data (need 3+ failures)"
            weibull_beta = None
            if ML_AVAILABLE and n_failures >= 3:
                try:
                    # Compute inter-arrival times between failures in minutes
                    failure_times = grp["Date"].dropna()
                    if len(failure_times) >= 3:
                        inter_arr = failure_times.diff().dropna().dt.total_seconds().div(60).values
                        inter_arr = inter_arr[inter_arr > 0]
                        if len(inter_arr) >= 2:
                            shape, loc, scale = weibull_min.fit(inter_arr, floc=0)
                            weibull_beta = round(shape, 2)
                            if shape < 0.9:
                                pattern = f"Burn-in / Decreasing failures (Weibull beta={weibull_beta})"
                            elif shape <= 1.1:
                                pattern = f"Random / Stable failures (Weibull beta={weibull_beta})"
                            else:
                                pattern = f"Wear-out / Increasing failures (Weibull beta={weibull_beta})"
                except Exception:
                    # Fallback to linear slope if Weibull fitting fails
                    if n_failures > 5:
                        inter_failures = grp["Duration (min)"].tolist()
                        trend_slope = np.polyfit(range(len(inter_failures)), inter_failures, 1)[0]
                        pattern = "Wear-out (increasing)" if trend_slope > 0.5 else ("Improving" if trend_slope < -0.5 else "Random / Stable")
            elif n_failures > 5:
                inter_failures = grp["Duration (min)"].tolist()
                trend_slope = np.polyfit(range(len(inter_failures)), inter_failures, 1)[0]
                pattern = "Wear-out (increasing)" if trend_slope > 0.5 else ("Improving" if trend_slope < -0.5 else "Random / Stable")

            rel_rows.append({
                "Machine": machine,
                "Failures": n_failures,
                "MTBF (min)": round(mtbf, 0) if mtbf else "N/A",
                "MTTR (min)": round(mttr, 1),
                "Inherent Availability %": round(availability_rel, 1) if availability_rel else "N/A",
                "Failure Pattern": pattern,
            })

        rel_df = pd.DataFrame(rel_rows).sort_values("Failures", ascending=False)

        def style_rel(row):
            mtbf_v = row.get("MTBF (min)", "N/A")
            mttr_v = row.get("MTTR (min)", 0)
            pattern_v = str(row.get("Failure Pattern", ""))
            if mtbf_v != "N/A" and float(mtbf_v) < float(mttr_v) * 3:
                return ["background-color:#fdf0f0; color:#1a1a1a"] * len(row)
            if "Wear-out" in pattern_v:
                return ["background-color:#fdf4e0; color:#1a1a1a"] * len(row)
            return [""] * len(row)

        st.dataframe(rel_df.style.apply(style_rel, axis=1), use_container_width=True, hide_index=True)

        # Weibull interpretation guide
        weibull_guide = (
            "**Failure Pattern** uses Weibull distribution analysis (a standard reliability engineering model) to classify each machine's failure behaviour. "
            "A Weibull beta below 1 means failures are decreasing over time (typical of early-life defects or burn-in). "
            "Beta near 1 means failures occur randomly with no trend (typical of a healthy machine in service). "
            "Beta above 1 means failures are increasing over time (typical of wear-out or ageing). "
            "Yellow rows are machines in wear-out phase and should be prioritised for preventive maintenance. "
            "Red rows indicate machines where MTBF is less than 3x MTTR, meaning excessive time is spent on repairs relative to uptime."
        )
        if ML_AVAILABLE:
            st.caption(weibull_guide)
        else:
            st.caption(
                "Inherent Availability is calculated as MTBF / (MTBF + MTTR). "
                "Failure Pattern shows whether failure durations are growing (wear-out), shrinking (improving), or stable (random). "
                "Red rows indicate machines where MTBF is less than 3x MTTR."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — TREND ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Trend Analysis",
                 help="Shows whether your OEE and downtime are improving or getting worse over time. "
                      "A rising OEE trend confirms your improvement actions are working. "
                      "A rising downtime trend is an early warning that a machine is degrading. "
                      "Requires Date columns in both your production log and downtime log.")

    has_prod_dates = not prod_df.empty and "Date" in prod_df.columns and prod_df["Date"].notna().any()
    has_dt_dates   = not dt_df.empty  and "Date" in dt_df.columns  and dt_df["Date"].notna().any()

    if not has_prod_dates and not has_dt_dates:
        st.info("Add a Date column to your production log and downtime log to enable trend analysis.")
    else:
        trend_machine_opts = ["All Machines"]
        if not prod_df.empty:
            trend_machine_opts += sorted(prod_df["Machine"].unique().tolist())
        sel_trend_machine = st.selectbox("Filter by Machine", trend_machine_opts, key="trend_machine_sel")

        if has_prod_dates:
            prod_t = prod_df if sel_trend_machine == "All Machines" else prod_df[prod_df["Machine"] == sel_trend_machine]
            prod_t = prod_t.copy()
            prod_t["Month"] = prod_t["Date"].dt.to_period("M").dt.to_timestamp()
            monthly_oee_rows = []
            for month, grp in prod_t.groupby("Month"):
                ph = grp["Planned Hours"].sum()
                rh = grp["Run Hours"].sum()
                gu = grp["Good Units"].sum()
                tu = grp["Total Units"].sum()
                av = rh / ph if ph > 0 else 0
                ql = gu / tu if tu > 0 else 0
                monthly_oee_rows.append({"Month": month, "OEE %": round(calc_oee(av, 1.0, ql) * 100, 1),
                                         "Availability %": round(av * 100, 1), "Quality %": round(ql * 100, 1)})
            monthly_oee_df = pd.DataFrame(monthly_oee_rows)

            fig_t1, ax_t1 = plt.subplots(figsize=(10, 4))
            ax_t1.plot(monthly_oee_df["Month"], monthly_oee_df["OEE %"],
                       color=STEEL_BLUE, linewidth=2, marker="o", markersize=5, label="OEE %")
            ax_t1.plot(monthly_oee_df["Month"], monthly_oee_df["Availability %"],
                       color=ORANGE, linewidth=1.5, linestyle="--", marker="s", markersize=4, label="Availability %")
            ax_t1.plot(monthly_oee_df["Month"], monthly_oee_df["Quality %"],
                       color=GREEN, linewidth=1.5, linestyle=":", marker="^", markersize=4, label="Quality %")
            ax_t1.axhline(85, color=GREEN, linewidth=1, linestyle="--", alpha=0.5, label="World Class 85%")
            ax_t1.axhline(60, color=GOLD,  linewidth=1, linestyle=":",  alpha=0.5, label="Acceptable 60%")
            ax_t1.set_ylabel("Score (%)", fontsize=11)
            ax_t1.set_title(f"OEE Trend: {sel_trend_machine}", fontsize=12, fontweight="bold")
            ax_t1.legend(fontsize=8)
            ax_t1.grid(alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            render_chart(fig_t1)

        if has_dt_dates:
            dt_t = dt_df if sel_trend_machine == "All Machines" else dt_df[dt_df["Machine"] == sel_trend_machine]
            dt_t = dt_t.copy()
            dt_t["Month"] = dt_t["Date"].dt.to_period("M").dt.to_timestamp()
            monthly_dt_rows = (dt_t.groupby("Month")
                               .agg(Total_DT=("Duration (min)", "sum"), Events=("Duration (min)", "count"))
                               .reset_index())

            fig_t2, ax_t2 = plt.subplots(figsize=(10, 4))
            ax_t2.bar(monthly_dt_rows["Month"], monthly_dt_rows["Total_DT"],
                      color=STEEL_BLUE, alpha=0.7, width=20, label="Total Downtime (min)")
            ax_t2b = ax_t2.twinx()
            ax_t2b.plot(monthly_dt_rows["Month"], monthly_dt_rows["Events"],
                        color=RED, linewidth=2, marker="o", markersize=5, label="Event Count")
            ax_t2.set_ylabel("Total Downtime (min)", fontsize=11)
            ax_t2b.set_ylabel("Downtime Events", fontsize=11)
            ax_t2.set_title(f"Monthly Downtime Trend: {sel_trend_machine}", fontsize=12, fontweight="bold")
            lines_a, labels_a = ax_t2.get_legend_handles_labels()
            lines_b, labels_b = ax_t2b.get_legend_handles_labels()
            ax_t2.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")
            ax_t2.grid(axis="y", alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            render_chart(fig_t2)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — SHIFT COMPARISON
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Shift Comparison",
                 help="Compares OEE and downtime across different shifts. "
                      "If OEE is consistently lower on one shift, it points to an operator, supervision, or handover issue rather than an equipment problem. "
                      "If all shifts show similar OEE, the problem is the machine itself. "
                      "Requires a Shift column in both your production log and downtime log.")

    has_prod_shift = not prod_df.empty and "Shift" in prod_df.columns
    has_dt_shift   = not dt_df.empty  and "Shift" in dt_df.columns

    if not has_prod_shift and not has_dt_shift:
        st.info("Add a Shift column to your production log and downtime log to enable shift comparison.")
    else:
        if has_prod_shift:
            shift_oee_rows = []
            for shift, grp in prod_df.groupby("Shift"):
                ph = grp["Planned Hours"].sum()
                rh = grp["Run Hours"].sum()
                gu = grp["Good Units"].sum()
                tu = grp["Total Units"].sum()
                av = rh / ph if ph > 0 else 0
                ql = gu / tu if tu > 0 else 0
                shift_oee_rows.append({"Shift": str(shift), "Availability %": round(av*100,1),
                                       "Quality %": round(ql*100,1), "OEE %": round(calc_oee(av,1.0,ql)*100,1)})
            shift_oee_df = pd.DataFrame(shift_oee_rows).sort_values("OEE %")

            fig_sh, ax_sh = plt.subplots(figsize=(max(6, len(shift_oee_df)*1.4), 4))
            bar_sh_c = [RED if v < 60 else (GOLD if v < 85 else GREEN) for v in shift_oee_df["OEE %"]]
            ax_sh.bar(shift_oee_df["Shift"], shift_oee_df["OEE %"], color=bar_sh_c, alpha=0.85)
            ax_sh.axhline(85, color=GREEN, linewidth=1.5, linestyle="--", label="World Class 85%", alpha=0.7)
            ax_sh.axhline(60, color=GOLD,  linewidth=1.5, linestyle=":",  label="Acceptable 60%",  alpha=0.7)
            for i, row in shift_oee_df.iterrows():
                ax_sh.text(row["Shift"], row["OEE %"] + 1, f'{row["OEE %"]:.1f}%', ha="center", fontsize=10, fontweight="bold")
            ax_sh.set_ylabel("OEE (%)", fontsize=11)
            ax_sh.set_title("OEE by Shift", fontsize=12, fontweight="bold")
            ax_sh.legend(fontsize=8)
            ax_sh.set_ylim(0, 110)
            ax_sh.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            render_chart(fig_sh)

            if len(shift_oee_df) >= 2:
                best_shift  = shift_oee_df.iloc[-1]["Shift"]
                worst_shift = shift_oee_df.iloc[0]["Shift"]
                gap = shift_oee_df.iloc[-1]["OEE %"] - shift_oee_df.iloc[0]["OEE %"]
                if gap > 10:
                    st.warning(f"There is a **{gap:.1f}% OEE gap** between {best_shift} (best) and {worst_shift} (worst). "
                               "A large shift gap typically points to operator skill, supervision, or handover issues rather than the machine itself. "
                               "Check downtime by shift below to confirm.")
                else:
                    st.success(f"Shifts are performing consistently (gap less than 10%). Any remaining OEE issues are more likely equipment-driven than shift-driven.")

        if has_dt_shift:
            shift_dt_rows = []
            for shift, grp in dt_df.groupby("Shift"):
                shift_dt_rows.append({"Shift": str(shift),
                                       "Events": len(grp),
                                       "Total Downtime (min)": round(grp["Duration (min)"].sum(), 0),
                                       "Avg Duration (min)": round(grp["Duration (min)"].mean(), 1)})
            shift_dt_df = pd.DataFrame(shift_dt_rows).sort_values("Total Downtime (min)", ascending=False)
            st.markdown("**Downtime by Shift**")
            st.dataframe(shift_dt_df, use_container_width=True, hide_index=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — MAINTENANCE INTELLIGENCE
    # ─────────────────────────────────────────────────────────────────────────
    st.subheader("Maintenance Intelligence",
                 help="Uses your MTBF and MTTR data to recommend an optimal preventive maintenance interval. "
                      "The goal is to find the maintenance frequency that minimises total cost, balancing planned maintenance cost against the much higher cost of unplanned breakdowns. "
                      "Requires a downtime log with Date and Machine columns to calculate MTBF. "
                      "You also need to provide your cost inputs below.")

    if dt_df.empty or "Date" not in dt_df.columns or not dt_df["Date"].notna().any():
        st.info("Upload a downtime log with Machine and Date columns to enable Maintenance Intelligence.")
    else:
        st.markdown("Enter your cost parameters to calculate the optimal maintenance interval for each machine.")
        mi_col1, mi_col2 = st.columns(2)
        with mi_col1:
            cost_planned = st.number_input("Cost of one planned maintenance ($)",
                                           min_value=0, value=500, step=50, key="mi_cost_planned",
                                           help="Labour, parts, and lost production cost of a single scheduled maintenance stop. "
                                                "This is typically lower than unplanned breakdown cost because you can schedule it at a convenient time.")
        with mi_col2:
            cost_unplanned = st.number_input("Cost of one unplanned breakdown ($)",
                                             min_value=0, value=5000, step=100, key="mi_cost_unplanned",
                                             help="Total cost when the machine breaks down unexpectedly, including emergency labour, rush parts, lost production, and any quality rejects caused by the failure.")

        mi_rows = []
        for machine, grp in dt_df.groupby("Machine"):
            grp = grp.dropna(subset=["Date"]).sort_values("Date")
            n_failures = len(grp)
            if n_failures < 2:
                mi_rows.append({"Machine": machine, "Failures": n_failures,
                                 "MTBF (days)": "N/A", "MTTR (hrs)": round(grp["Duration (min)"].mean()/60, 1),
                                 "Optimal PM Interval (days)": "Need 2+ failures",
                                 "Expected Cost / Day ($)": "N/A", "Recommendation": "Collect more data"})
                continue
            total_span_days = (grp["Date"].max() - grp["Date"].min()).days
            total_dt_days   = grp["Duration (min)"].sum() / (60 * 24)
            uptime_days     = max(0, total_span_days - total_dt_days)
            mtbf_days       = uptime_days / (n_failures - 1) if n_failures > 1 else uptime_days
            mttr_hrs        = grp["Duration (min)"].mean() / 60

            # Optimal PM interval: minimise (cost_planned / T) + (cost_unplanned / MTBF)
            # where T is the maintenance interval in days
            # Optimal T* = sqrt(2 * cost_planned * MTBF / cost_unplanned)
            if cost_unplanned > 0 and mtbf_days > 0:
                optimal_T = (2 * cost_planned * mtbf_days / cost_unplanned) ** 0.5
                optimal_T = max(1, round(optimal_T, 0))
                expected_cost_per_day = (cost_planned / optimal_T) + (cost_unplanned / mtbf_days)
            else:
                optimal_T = None
                expected_cost_per_day = None

            if optimal_T:
                if optimal_T < 7:
                    rec = "Weekly or more frequent maintenance recommended"
                elif optimal_T < 30:
                    rec = f"Maintain every {int(optimal_T)} days (approx. every {int(optimal_T//7)} week(s))"
                elif optimal_T < 90:
                    rec = f"Maintain every {int(optimal_T)} days (approx. monthly)"
                else:
                    rec = f"Maintain every {int(optimal_T)} days (approx. quarterly or less)"
            else:
                rec = "Set cost inputs above to calculate"

            mi_rows.append({
                "Machine": machine,
                "Failures Recorded": n_failures,
                "MTBF (days)": round(mtbf_days, 1),
                "MTTR (hrs)": round(mttr_hrs, 1),
                "Optimal PM Interval (days)": int(optimal_T) if optimal_T else "N/A",
                "Est. Cost / Day ($)": round(expected_cost_per_day, 0) if expected_cost_per_day else "N/A",
                "Recommendation": rec,
            })

        mi_df = pd.DataFrame(mi_rows)
        st.dataframe(mi_df, use_container_width=True, hide_index=True)
        st.caption("Optimal PM Interval is calculated using the cost-minimisation model: T = sqrt(2 x planned cost x MTBF / breakdown cost). "
                   "This is the classical preventive maintenance optimisation formula from reliability engineering. "
                   "It balances the cost of over-maintaining (too many planned stops) against the cost of under-maintaining (too many breakdowns). "
                   "The estimate improves in accuracy as more failure events are recorded.")

        with st.expander("🏷️ Maintenance Plan Detail", expanded=False):
            st.caption("Full breakdown of reliability metrics and recommended intervals per machine. Download to share with your maintenance team.")
            st.dataframe(mi_df, use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download Maintenance Plan", to_excel_bytes(mi_df),
                               "maintenance_plan.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               key="dl_mi")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – DISCRETE EVENT SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.header("📟 Discrete Event Simulation",
              help="Discrete Event Simulation models your factory as a network of machines and workstations, "
                   "then simulates thousands of production runs to predict real-world throughput, queue build-up, and bottlenecks under variability and machine breakdowns. "
                   "Unlike the Line Balancing Simulation which assumes steady-state averages, this simulation accounts for randomness: "
                   "cycle time variation, machine failures, repair times, and buffer capacities all interact dynamically. "
                   "This tab is planned for a future release and will be seeded from your uploaded time study and downtime data.")

    st.info("This feature is coming in a future release. "
            "It will use your time study cycle times from the Current State Analysis tab and your machine reliability data from the Machine Health Diagnostics tab "
            "to run a full factory simulation and predict throughput, WIP levels, and the impact of different maintenance strategies.")
