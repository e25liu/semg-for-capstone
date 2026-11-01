import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch

# ---------- Settings ----------
FILE = "sEMG&IMU_Oct_21.xlsx"

# start time (HH:MM:SS) and length in seconds for each throw
THROWS = {
    "Throw 1": ("18:17:43", 5),
    "Throw 2": ("18:30:44", 3),
}

LOW, HIGH, ORDER = 20, 400, 4     # Butterworth band-pass (Hz)
WIN_S, STEP_S = 1.0, 0.25         # sliding window for MDF/MNF over time (s)
DROP_PCT = 5.0                    # rough rule of thumb for the verdict (not a standard)
COLORS = {"Throw 1": "tab:blue", "Throw 2": "tab:red"}


def to_sec(s):
    h, m, x = str(s).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


# ---------- Load data (skips repeated header rows, tracks blocks) ----------
df = pd.read_excel(FILE, header=None)
if df.shape[1] == 1:
    df = df[0].astype(str).str.split(expand=True)
df = df.iloc[:, :3]
df.columns = ["tod", "t", "raw"]
is_data = df["tod"].astype(str).str.match(r"^\d{1,2}:\d{2}:\d{2}")
df["block"] = (~is_data).cumsum()
df = df[is_data].copy()
df["t"] = pd.to_numeric(df["t"], errors="coerce")
df["raw"] = pd.to_numeric(df["raw"], errors="coerce")
df = df.dropna(subset=["t", "raw"])
df["tod_s"] = df["tod"].apply(to_sec)


# ---------- Cut out one throw: filter its block, keep only start -> start + duration ----------
def get_throw(name, start, dur):
    s = to_sec(start)
    hit = df[df["tod_s"] >= s]
    hit = hit[hit["tod_s"] <= s + dur]
    if hit.empty:
        raise ValueError(f"{name}: no data between {start} and +{dur}s")
    g = df[df["block"] == hit["block"].iloc[0]].reset_index(drop=True)

    fs = round(1 / np.median(np.diff(g["t"])))
    x = g["raw"].to_numpy()
    x = x - x.mean()                                            # remove DC offset
    sos = butter(ORDER, [LOW, HIGH], btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)

    i0 = (g["tod_s"] >= s).idxmax()                             # first sample of the throw
    t = g["t"].to_numpy() - g["t"].iloc[i0]
    keep = (t >= 0) & (t < dur)
    return dict(name=name, fs=fs, t=t[keep], raw=x[keep], filt=y[keep])


def spectrum(seg, fs):
    f, p = welch(seg, fs=fs, nperseg=min(512, len(seg)))
    return f, p


def features(seg, fs):
    f, p = spectrum(seg, fs)
    band = (f >= LOW) & (f <= HIGH)
    f, p = f[band], p[band]
    cum = np.cumsum(p)
    mdf = f[np.searchsorted(cum, cum[-1] / 2)]                  # median frequency
    mnf = np.sum(f * p) / np.sum(p)                             # mean power frequency
    rms = np.sqrt(np.mean(seg ** 2))
    return mdf, mnf, rms


def sliding(seg, t, fs):
    win, step = int(WIN_S * fs), int(STEP_S * fs)
    out = [(t[i + win // 2], *features(seg[i:i + win], fs)[:2])
           for i in range(0, len(seg) - win + 1, step)]
    return list(zip(*out)) if out else ([], [], [])


throws = [get_throw(n, s, d) for n, (s, d) in THROWS.items()]
a, b = throws

for d in throws:
    d["mdf"], d["mnf"], d["rms"] = features(d["filt"], d["fs"])
    d["sl_t"], d["sl_mdf"], d["sl_mnf"] = sliding(d["filt"], d["t"], d["fs"])
    print(f"{d['name']}: {d['fs']} Hz, {len(d['t'])} samples ({d['t'][-1]:.2f} s) | "
          f"MDF {d['mdf']:.1f} Hz | MNF {d['mnf']:.1f} Hz | RMS {d['rms']:.4f}")

# ---------- Comparison table + verdict ----------
chg = {k: (b[k] - a[k]) / a[k] * 100 for k in ("mdf", "mnf", "rms")}
table = pd.DataFrame(
    {"Throw 1": [a["mdf"], a["mnf"], a["rms"]],
     "Throw 2": [b["mdf"], b["mnf"], b["rms"]],
     "Change (%)": [chg["mdf"], chg["mnf"], chg["rms"]]},
    index=["MDF (Hz)", "MNF (Hz)", "RMS"],
).round(3)
print("\n" + table.to_string())
table.to_csv("semg_throw_comparison.csv")

if chg["mdf"] <= -DROP_PCT and chg["mnf"] <= -DROP_PCT:
    verdict = "Fatigue-like shift: MDF and MNF both dropped" + \
              (" and RMS rose." if chg["rms"] > 0 else ", but RMS did not rise.")
elif chg["mdf"] < 0 and chg["mnf"] < 0:
    verdict = "Small drop in MDF and MNF: possible fatigue, within the margin of noise."
elif (chg["mdf"] < 0) != (chg["mnf"] < 0):
    verdict = "MDF and MNF disagree: no clear fatigue evidence."
else:
    verdict = "No frequency drop: no sign of fatigue between these two throws."
print("\nVerdict:", verdict)
print("Note: two throws can't confirm fatigue, and rest between them can reverse it.")

# ---------- Figure 1: signals and spectra (same y-scale per row) ----------
fig1, ax = plt.subplots(3, 2, figsize=(13, 10), sharey="row")
for col, d in enumerate(throws):
    c = COLORS[d["name"]]
    ax[0, col].plot(d["t"], d["raw"], color=c, lw=0.7)
    ax[0, col].set_title(f"{d['name']}: unfiltered")
    ax[1, col].plot(d["t"], d["filt"], color=c, lw=0.7)
    ax[1, col].set_title(f"{d['name']}: band-pass {LOW}-{HIGH} Hz")
    for row in (0, 1):
        ax[row, col].set_xlabel("Time (s)")
    ax[2, col].semilogy(*spectrum(d["raw"], d["fs"]), color="gray", alpha=0.7, label="Unfiltered")
    ax[2, col].semilogy(*spectrum(d["filt"], d["fs"]), color=c, label="Filtered")
    ax[2, col].axvline(d["mdf"], color="black", ls="--", label=f"MDF {d['mdf']:.0f} Hz")
    ax[2, col].axvline(d["mnf"], color="green", ls=":", label=f"MNF {d['mnf']:.0f} Hz")
    ax[2, col].set_title(f"{d['name']}: spectrum")
    ax[2, col].set_xlabel("Frequency (Hz)")
    ax[2, col].legend()
ax[0, 0].set_ylabel("Amplitude")
ax[1, 0].set_ylabel("Amplitude")
ax[2, 0].set_ylabel("Power")
fig1.suptitle("Throw 1 vs Throw 2", fontsize=13)
fig1.tight_layout()

# ---------- Figure 2: MDF / MNF over time, plus summary bars ----------
fig2, a2 = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1, 1, 0.8]})
vals = [v for d in throws for v in (*d["sl_mdf"], *d["sl_mnf"], d["mdf"], d["mnf"])]
lo, hi = min(vals) * 0.9, max(vals) * 1.1

for k, d in enumerate(throws):
    a2[k].plot(d["sl_t"], d["sl_mdf"], marker="o", color="tab:purple", label="MDF")
    a2[k].plot(d["sl_t"], d["sl_mnf"], marker="s", color="tab:orange", label="MNF")
    a2[k].axhline(d["mdf"], color="tab:purple", ls="--", alpha=0.5)
    a2[k].axhline(d["mnf"], color="tab:orange", ls="--", alpha=0.5)
    a2[k].set_ylim(lo, hi)
    a2[k].set_title(f"{d['name']}: MDF and MNF over time")
    a2[k].set_xlabel("Time within throw (s)")
    a2[k].set_ylabel("Frequency (Hz)")
    a2[k].legend()

x, w = np.arange(2), 0.35
for off, d in ((-w / 2, a), (w / 2, b)):
    bars = a2[2].bar(x + off, [d["mdf"], d["mnf"]], w, color=COLORS[d["name"]], label=d["name"])
    a2[2].bar_label(bars, fmt="%.1f")
a2[2].set_xticks(x)
a2[2].set_xticklabels(["MDF", "MNF"])
a2[2].set_ylabel("Frequency (Hz)")
a2[2].set_title("Summary")
a2[2].legend()
fig2.tight_layout()

fig1.savefig("semg_fig1.png", dpi=150)
fig2.savefig("semg_fig2.png", dpi=150)
plt.show()