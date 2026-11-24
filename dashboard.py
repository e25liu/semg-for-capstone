"""
Pitcher sEMG Dashboard (Streamlit)
==================================
Runs YOUR b.py exactly as it is (no changes to b.py), then shows what it produces:
    semg_events.csv, semg_events.png, semg_fatigue.png

Run from the VS Code terminal, inside your project folder:
    python3 -m streamlit run dashboard.py
"""
import contextlib
import glob
import io
import os

import matplotlib
matplotlib.use("Agg")          # so plt.show() in b.py does not open pop-up windows
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from scipy.stats import linregress

import b                       # your analysis file, unchanged

st.set_page_config(page_title="Pitcher sEMG Dashboard", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Session")
player = st.sidebar.text_input("Player", "Player 1")
target = st.sidebar.number_input("Planned throws", 1, 300, 30)
data_dir = st.sidebar.text_input("Recordings folder", ".")
auto = st.sidebar.toggle("Auto-refresh (watch for new/updated files)", value=False)
every = st.sidebar.select_slider("Check every (s)", [5, 10, 30, 60], value=10)

st.sidebar.header("Fatigue status thresholds")
warn_pct = st.sidebar.number_input("Moderate if last-throw MDF drops more than (%)", 1, 50, 5)
stop_pct = st.sidebar.number_input("High if it drops more than (%)", 1, 80, 12)


@st.cache_data(show_spinner="Running b.py on the recording…")
def run_b(path, mtime):            # mtime -> re-runs only when the Excel file changes
    """Run b.py's main() on the chosen file; return its event table and printout."""
    b.FILE = path
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        b.main()                   # saves semg_events.csv / .png and semg_fatigue.png
    plt.close("all")
    return pd.read_csv("semg_events.csv"), out.getvalue()


@st.fragment(run_every=every if auto else None)
def dashboard():
    files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")), key=os.path.getmtime)
    files = [f for f in files if not os.path.basename(f).startswith("~$")]   # skip Excel lock files
    if not files:
        st.warning(f"No .xlsx files found in `{data_dir}`.")
        return
    chosen = st.selectbox("Recording (newest first)", files[::-1], format_func=os.path.basename)

    try:
        tab, printout = run_b(chosen, os.path.getmtime(chosen))
    except Exception as e:
        st.error(f"b.py could not analyze this file: {type(e).__name__}: {e}")
        return

    th = tab[tab["kind"] == "Throw"].reset_index(drop=True)
    nm = int((tab["kind"] == "MVC").sum())
    amp = "pct_mvc" if "pct_mvc" in tab else "rms"
    done, left = len(th), max(0, target - len(th))

    # ---- Same verdict rule as b.py
    fits = {}
    if done >= 3:
        for c in ("mdf_hz", "mnf_hz", amp):
            fits[c] = linregress(th["number"], th[c])
        down = fits["mdf_hz"].slope < 0 and fits["mnf_hz"].slope < 0
        sig = min(fits["mdf_hz"].pvalue, fits["mnf_hz"].pvalue) < 0.05
        verdict = ("Fatigue: MDF and MNF both fall significantly" if down and sig else
                   "Possible fatigue: both fall, not significantly" if down else
                   "No fatigue signature: frequency did not fall across throws")
        level = 2 if down and sig else 1 if down else 0
    else:
        verdict, level = "Not enough throws yet for a trend (need 3)", -1

    # ---- Live status: last throw vs the player's own first 3 throws
    chg = None
    if done >= 3:
        base = th["mdf_hz"].head(3).mean()
        th["mdf_change_%"] = (100 * (th["mdf_hz"] - base) / base).round(1)
        chg = th["mdf_change_%"].iloc[-1]
        status = ("🔴 High" if chg < -stop_pct else
                  "🟡 Moderate" if chg < -warn_pct else "🟢 OK")
    else:
        status = "Calibrating (need 3 throws)"
    last = th.iloc[-1] if done else None

    # ---------------- Header ----------------
    st.title(f"{player}: sEMG session")
    st.caption(f"{os.path.basename(chosen)} · {nm} MVC trial(s)")
    c = st.columns(6)
    c[0].metric("Throws done", done)
    c[1].metric("Throws left", left)
    c[2].metric("Fatigue status", status)
    if last is not None:
        c[3].metric("Last throw MDF", f"{last['mdf_hz']:.1f} Hz",
                    f"{chg:+.1f}% vs first 3" if chg is not None else None)
        c[4].metric("Last throw MNF", f"{last['mnf_hz']:.1f} Hz")
        c[5].metric("Last throw amplitude",
                    f"{last['pct_mvc']:.0f}% MVC" if amp == "pct_mvc" else f"{last['rms']:.3g} RMS")
    st.progress(min(done / target, 1.0), text=f"{done} / {target} throws")
    {2: st.error, 1: st.warning, 0: st.success, -1: st.info}[level](f"**Session trend:** {verdict}")

    # ---------------- Tabs ----------------
    t1, t2, t3, t4, t5, t6 = st.tabs(["Fatigue trend", "Throw detection", "Throw table",
                                      "MVC", "All figures", "b.py output"])
    with t1:
        if os.path.exists("semg_fatigue.png"):
            st.image("semg_fatigue.png", width="stretch")
        if fits:
            names = {"mdf_hz": "MDF (Hz)", "mnf_hz": "MNF (Hz)",
                     "pct_mvc": "Amplitude (% MVC)", "rms": "Amplitude (RMS)"}
            st.dataframe(pd.DataFrame([
                {"Feature": names[k], "Slope per throw": round(v.slope, 4),
                 "p-value": round(v.pvalue, 4), "R²": round(v.rvalue ** 2, 3)}
                for k, v in fits.items()]), hide_index=True, width="stretch")

    with t2:
        if os.path.exists("semg_events.png"):
            st.image("semg_events.png", width="stretch")
        st.caption("Blue = throw, red = MVC, grey = setup. Check every throw is its own block.")

    with t3:
        st.dataframe(th.round(3), hide_index=True, width="stretch")
        with st.expander("All recording blocks (incl. setup and MVC)"):
            st.dataframe(tab.round(3), hide_index=True, width="stretch")
        st.download_button("Download events (CSV)", tab.to_csv(index=False).encode(),
                           f"{player.replace(' ', '_')}_semg_events.csv")

    with t4:
        mv = tab[tab["kind"] == "MVC"]
        if len(mv):
            st.dataframe(mv.round(3), hide_index=True, width="stretch")
        else:
            st.info("No MVC trials detected in this recording.")

    with t5:
        pngs = sorted(glob.glob("*.png"))
        cols = st.columns(2)
        for i, p in enumerate(pngs):
            cols[i % 2].image(p, caption=p, width="stretch")

    with t6:
        st.code(printout or "(no printout)")


dashboard()