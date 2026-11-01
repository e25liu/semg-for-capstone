import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

# ---------- Settings ----------
FILE = "sEMG&IMU_Oct_21.xlsx"   # change if you renamed the file

# The two throws (24-hour clock, HH:MM:SS). CHANGE THESE to your real throw times.
T1_START, T1_END = "18:17:43", "18:17:48"    # throw 1 (earlier)
T2_START, T2_END = "18:30:44", "18:30:47"    # throw 2 (later)

EQUAL_LENGTH = True              # trim both throws to the shorter one for a fair comparison

# Filters
AA_CUTOFF = 450                  # Hz, anti-aliasing low-pass (must be below Nyquist)
AA_ORDER = 4
LOW_CUT = 20                     # Hz, Butterworth band-pass
HIGH_CUT = 400                   # Hz
BP_ORDER = 4

# Median / mean power frequency inside each throw (sliding window)
WIN_S = 1.0                      # window length in seconds
STEP_S = 0.25                    # step in seconds

# Rough rule of thumb for the verdict (not a standard)
DROP_THRESHOLD_PCT = 5.0


def to_seconds(s):
    """Convert 'HH:MM:SS.mmm' to seconds since midnight."""
    h, m, sec = str(s).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


# ---------- Load data (handles repeated header rows) ----------
all_rows = pd.read_excel(FILE, header=None)
if all_rows.shape[1] == 1:       # data still in one column as text
    all_rows = all_rows[0].astype(str).str.split(expand=True)
all_rows = all_rows.iloc[:, :3]
all_rows.columns = ["time_of_day", "time_s", "raw"]

is_data = all_rows["time_of_day"].astype(str).str.match(r"^\d{1,2}:\d{2}:\d{2}")
block_id = (~is_data).cumsum()
all_rows = all_rows[is_data].copy()
all_rows["block"] = block_id[is_data]
all_rows["time_s"] = pd.to_numeric(all_rows["time_s"], errors="coerce")
all_rows["raw"] = pd.to_numeric(all_rows["raw"], errors="coerce")
all_rows = all_rows.dropna(subset=["time_s", "raw"])

for b, g in all_rows.groupby("block"):
    print(f"Block {b}: {len(g)} rows, {g['time_of_day'].iloc[0]} to {g['time_of_day'].iloc[-1]}")


# ---------- Extract one throw: find its block, filter, crop ----------
def extract_throw(name, start, end):
    s, e = to_seconds(start), to_seconds(end)
    df = None
    for b, g in all_rows.groupby("block"):
        tt = g["time_of_day"].apply(to_seconds)
        if ((tt >= s) & (tt <= e)).any():
            df = g.reset_index(drop=True)
            print(f"{name}: using block {b}")
            break
    if df is None:
        raise ValueError(f"{name}: no block contains {start} to {end}. See the block list above.")

    tod = df["time_of_day"].apply(to_seconds).to_numpy()
    t_all = df["time_s"].to_numpy()
    x_all = df["raw"].to_numpy()

    fs = int(round(1.0 / np.median(np.diff(t_all))))
    nyq = fs / 2
    print(f"{name}: sampling rate {fs} Hz, Nyquist {nyq:.0f} Hz")
    assert AA_CUTOFF < nyq, "Anti-aliasing cutoff must be below Nyquist (FS / 2)"
    assert 0 < LOW_CUT < HIGH_CUT < nyq, "Need 0 < LOW_CUT < HIGH_CUT < Nyquist"

    x_centered = x_all - np.mean(x_all)                                   # remove DC offset
    sos_aa = butter(AA_ORDER, AA_CUTOFF, btype="lowpass", fs=fs, output="sos")
    x_f = sosfiltfilt(sos_aa, x_centered)                                 # anti-aliasing
    sos_bp = butter(BP_ORDER, [LOW_CUT, HIGH_CUT], btype="bandpass", fs=fs, output="sos")
    x_f = sosfiltfilt(sos_bp, x_f)                                        # Butterworth band-pass

    m = (tod >= s) & (tod <= e)
    if m.sum() < 128:
        raise ValueError(f"{name} has only {m.sum()} samples. Use a longer window.")
    return dict(
        name=name,
        t=t_all[m] - t_all[m][0],
        raw=x_centered[m],        # unfiltered (DC removed)
        filt=x_f[m],              # anti-aliasing + Butterworth
        fs=fs,
    )


a = extract_throw("Throw 1", T1_START, T1_END)
b = extract_throw("Throw 2", T2_START, T2_END)

if EQUAL_LENGTH:
    dur = min(a["t"][-1], b["t"][-1])
    for d in (a, b):
        k = d["t"] <= dur
        for key in ("t", "raw", "filt"):
            d[key] = d[key][k]
    print(f"Both throws trimmed to {dur:.2f} s for a fair comparison")


# ---------- Spectrum features ----------
def band_features(seg, fs):
    """Band-limited spectrum plus median frequency, mean power frequency, RMS."""
    f, p = welch(seg, fs=fs, nperseg=min(512, len(seg)))
    band = (f >= LOW_CUT) & (f <= HIGH_CUT)
    f, p = f[band], p[band]
    cum = np.cumsum(p)
    mdf = f[np.searchsorted(cum, cum[-1] / 2)]
    mnf = np.sum(f * p) / np.sum(p)
    rms = np.sqrt(np.mean(seg ** 2))
    return f, p, mdf, mnf, rms


def sliding_features(seg, t, fs):
    win, step = int(WIN_S * fs), int(STEP_S * fs)
    times, mdf, mnf = [], [], []
    for i in range(0, len(seg) - win + 1, step):
        _, _, md, mn, _ = band_features(seg[i:i + win], fs)
        mdf.append(md)
        mnf.append(mn)
        times.append(t[i + win // 2])
    return times, mdf, mnf


for d in (a, b):
    _, _, d["mdf"], d["mnf"], d["rms"] = band_features(d["filt"], d["fs"])
    d["f_raw"], d["p_raw"] = welch(d["raw"], fs=d["fs"], nperseg=min(512, len(d["raw"])))
    d["f_flt"], d["p_flt"] = welch(d["filt"], fs=d["fs"], nperseg=min(512, len(d["filt"])))
    d["sl_t"], d["sl_mdf"], d["sl_mnf"] = sliding_features(d["filt"], d["t"], d["fs"])
    print(f"{d['name']}: {len(d['t'])} samples ({d['t'][-1]:.2f} s) | "
          f"MDF {d['mdf']:.1f} Hz | MNF {d['mnf']:.1f} Hz | RMS {d['rms']:.4f}")

mdf_chg = (b["mdf"] - a["mdf"]) / a["mdf"] * 100
mnf_chg = (b["mnf"] - a["mnf"]) / a["mnf"] * 100
rms_chg = (b["rms"] - a["rms"]) / a["rms"] * 100

table = pd.DataFrame(
    {
        "Throw 1": [a["mdf"], a["mnf"], a["rms"]],
        "Throw 2": [b["mdf"], b["mnf"], b["rms"]],
        "Change (%)": [mdf_chg, mnf_chg, rms_chg],
    },
    index=["MDF (Hz)", "MNF (Hz)", "RMS"],
).round(3)
print("\n" + table.to_string())
table.to_csv("semg_throw_comparison.csv")

# ---------- Verdict ----------
if mdf_chg <= -DROP_THRESHOLD_PCT and mnf_chg <= -DROP_THRESHOLD_PCT:
    verdict = "Fatigue-like shift: both MDF and MNF dropped"
    verdict += " and RMS rose (the classic fatigue pattern)." if rms_chg > 0 else ", but RMS did not rise."
elif mdf_chg < 0 and mnf_chg < 0:
    verdict = "Small decrease in both MDF and MNF: possible fatigue, but within the margin of noise."
elif (mdf_chg < 0) != (mnf_chg < 0):
    verdict = "MDF and MNF disagree: no clear fatigue evidence (check for noise or artifact)."
else:
    verdict = "No frequency drop: no sign of fatigue between these two throws."
print("\nVerdict:", verdict)
print("Note: two throws are not enough to confirm fatigue, and rest between them can reverse it.")

# =====================================================================
# FIGURE 1: each throw in its own column, same scales for fair comparison
#   row 1: unfiltered | row 2: filtered | row 3: frequency spectrum
# =====================================================================
fig1, ax = plt.subplots(3, 2, figsize=(13, 11), sharey="row")
colors = {"Throw 1": "tab:blue", "Throw 2": "tab:red"}

for col, d in enumerate((a, b)):
    c = colors[d["name"]]
    ax[0, col].plot(d["t"], d["raw"], color=c, linewidth=0.7)
    ax[0, col].set_title(f"{d['name']}: unfiltered signal")
    ax[0, col].set_xlabel("Time (s)")

    ax[1, col].plot(d["t"], d["filt"], color=c, linewidth=0.7)
    ax[1, col].set_title(f"{d['name']}: after anti-aliasing + Butterworth")
    ax[1, col].set_xlabel("Time (s)")

    ax[2, col].semilogy(d["f_raw"], d["p_raw"], color="gray", alpha=0.7, label="Unfiltered")
    ax[2, col].semilogy(d["f_flt"], d["p_flt"], color=c, label="Filtered")
    ax[2, col].axvline(d["mdf"], color="black", linestyle="--", label=f"MDF {d['mdf']:.0f} Hz")
    ax[2, col].axvline(d["mnf"], color="green", linestyle=":", label=f"MNF {d['mnf']:.0f} Hz")
    ax[2, col].set_title(f"{d['name']}: frequency spectrum")
    ax[2, col].set_xlabel("Frequency (Hz)")
    ax[2, col].legend()

ax[0, 0].set_ylabel("Amplitude")
ax[1, 0].set_ylabel("Amplitude")
ax[2, 0].set_ylabel("Power")
fig1.suptitle("Throw 1 vs Throw 2 (same scales in each row)", fontsize=13)
fig1.tight_layout()

# =====================================================================
# FIGURE 2: median frequency and mean power frequency, side by side
#   left: Throw 1 over time | middle: Throw 2 over time | right: summary
# =====================================================================
fig2, a2 = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.8]})

all_vals = a["sl_mdf"] + a["sl_mnf"] + b["sl_mdf"] + b["sl_mnf"] + [a["mdf"], a["mnf"], b["mdf"], b["mnf"]]
lo, hi = min(all_vals) * 0.9, max(all_vals) * 1.1

for k, d in enumerate((a, b)):
    if len(d["sl_t"]) >= 1:
        a2[k].plot(d["sl_t"], d["sl_mdf"], marker="o", color="tab:purple", label="MDF")
        a2[k].plot(d["sl_t"], d["sl_mnf"], marker="s", color="tab:orange", label="MNF")
    a2[k].axhline(d["mdf"], color="tab:purple", linestyle="--", alpha=0.5)
    a2[k].axhline(d["mnf"], color="tab:orange", linestyle="--", alpha=0.5)
    a2[k].set_ylim(lo, hi)
    a2[k].set_title(f"{d['name']}: MDF and MNF over time")
    a2[k].set_xlabel("Time within throw (s)")
    a2[k].set_ylabel("Frequency (Hz)")
    a2[k].legend()

labels = ["MDF", "MNF"]
x = np.arange(len(labels))
w = 0.35
bars1 = a2[2].bar(x - w / 2, [a["mdf"], a["mnf"]], w, color="tab:blue", label="Throw 1")
bars2 = a2[2].bar(x + w / 2, [b["mdf"], b["mnf"]], w, color="tab:red", label="Throw 2")
for bar in list(bars1) + list(bars2):
    a2[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               f"{bar.get_height():.1f}", ha="center", va="bottom")
a2[2].set_xticks(x)
a2[2].set_xticklabels(labels)
a2[2].set_ylabel("Frequency (Hz)")
a2[2].set_ti