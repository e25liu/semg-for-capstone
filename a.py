"""
sEMG muscle-fatigue analysis using labelled throw times, plus a detector that is
scored against those labels so it can be reused on other people's recordings.

Pipeline: remove DC offset -> anti-aliasing low-pass -> Butterworth band-pass
          -> (optional 60 Hz notch) -> envelope for detection
Features per throw: MDF, MNF, RMS from the band-passed (not rectified) signal.
Fatigue = MDF and MNF falling across throws, usually with RMS rising.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch, find_peaks
from scipy.stats import linregress

# ================= Settings =================
FILE = "sEMG&IMU_Oct_21.xlsx"
AA_CUTOFF = 500                 # Hz, anti-aliasing low-pass (capped below Nyquist)
LOW, HIGH, ORDER = 20, 450, 4   # Hz, Butterworth band-pass (fine at 2000 Hz sampling)
NOTCH_60 = False                # True if the spectrum shows a 60 Hz spike
ENV_HZ = 10                     # envelope smoothing (Hz)

# Labelled events: (name, start, end) as MM:SS elapsed from the start of the recording.
# "Throw ..." entries are used for the fatigue trend; MVCs are used only as a reference.
LABELS = [
    ("Throw 2",  "16:59", "17:04"), ("Throw 3",  "17:43", "17:48"),
    ("Throw 4",  "18:16", "18:21"), ("Throw 5",  "18:59", "19:05"),
    ("Throw 6",  "20:03", "20:08"), ("Throw 7",  "20:52", "20:56"),
    ("Throw 8",  "21:19", "21:25"), ("Throw 9",  "22:13", "22:19"),
    ("Throw 10", "22:44", "22:50"), ("Throw 11", "23:25", "23:32"),
    ("Throw 12", "23:44", "23:47"), ("Throw 13", "24:25", "24:28"),
    ("Throw 14", "24:50", "24:54"), ("Throw 15", "25:24", "25:29"),
    ("Throw 16", "25:46", "25:50"), ("Throw 17", "26:22", "26:27"),
    ("Throw 18", "26:52", "26:56"), ("Throw 19", "27:24", "27:28"),
    ("Throw 20", "27:46", "27:50"), ("Throw 21", "28:32", "28:38"),
    ("Throw 22", "29:11", "29:15"), ("MVC 2",    "29:35", "29:50"),
    ("Throw 23", "30:44", "30:47"), ("Throw 24", "31:08", "31:11"),
    ("Throw 25", "32:53", "32:55"), ("Throw 26", "33:34", "33:39"),
    ("MVC 3",    "34:13", "34:21"),
]

TOL_S = 2.0                     # a detection counts as a hit within this many seconds of a label


def mmss(s):
    m, sec = str(s).strip().replace(".", ":").split(":")
    return int(m) * 60 + float(sec)


# ================= Load =================
df = pd.read_excel(FILE, header=None)
if df.shape[1] == 1:
    df = df[0].astype(str).str.split(expand=True)
df = df.iloc[:, :3]
df.columns = ["tod", "t", "raw"]
df = df[df["tod"].astype(str).str.match(r"^\d{1,2}:\d{2}:\d{2}")].copy()   # drop header rows
df["raw"] = pd.to_numeric(df["raw"], errors="coerce")
df["t"] = pd.to_numeric(df["t"], errors="coerce")
df = df.dropna(subset=["t", "raw"]).reset_index(drop=True)


def clock_s(s):
    h, m, x = str(s).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


x = df["raw"].to_numpy()
d = np.diff(df["t"].to_numpy())
fs = int(round(1 / np.median(d[d > 0])))

# Labels are elapsed time on the WALL CLOCK. The file has gaps between recordings, so
# locate each label by its clock time rather than by sample number.
tod_s = df["tod"].apply(clock_s).to_numpy()
t = tod_s - tod_s[0]                            # elapsed seconds, matching the labels
print(f"{len(x)} samples at {fs} Hz = {len(x)/fs/60:.1f} min of data "
      f"spread over {t[-1]/60:.1f} min of clock time (the difference is gaps)")
print(f"Nyquist {fs/2:.0f} Hz")

# ================= Filter =================
x = x - x.mean()                                                      # 1. remove DC offset
aa = min(AA_CUTOFF, 0.9 * fs / 2)                                     # keep below Nyquist
x = sosfiltfilt(butter(4, aa, "lowpass", fs=fs, output="sos"), x)     # 2. anti-aliasing
y = sosfiltfilt(butter(ORDER, [LOW, HIGH], "bandpass", fs=fs, output="sos"), x)   # 3. band-pass
if NOTCH_60:                                                          # 4. mains notch
    bn, an = iirnotch(60, Q=30, fs=fs)
    y = filtfilt(bn, an, y)
env = np.clip(sosfiltfilt(butter(4, ENV_HZ, "lowpass", fs=fs, output="sos"), np.abs(y)), 0, None)
print(f"Filters: anti-aliasing {aa:.0f} Hz, band-pass {LOW}-{HIGH} Hz order {ORDER}"
      + (", 60 Hz notch" if NOTCH_60 else ""))


# ================= Features =================
def features(seg):
    f, p = welch(seg, fs=fs, nperseg=min(1024, len(seg)))
    band = (f >= LOW) & (f <= HIGH)
    f, p = f[band], p[band]
    cum = np.cumsum(p)
    mdf = f[np.searchsorted(cum, cum[-1] / 2)]      # median frequency
    mnf = np.sum(f * p) / np.sum(p)                 # mean power frequency
    return mdf, mnf, np.sqrt(np.mean(seg ** 2))     # plus RMS


# ================= Part 1: analysis from the labels =================
rows = []
missing = []
for name, s0, e0 in LABELS:
    idx = np.where((t >= mmss(s0)) & (t <= mmss(e0)))[0]
    if len(idx) < 256:
        missing.append(f"{name} ({s0}-{e0}, {len(idx)} samples)")
        continue
    s, e = idx[0], idx[-1] + 1
    mdf, mnf, rms = features(y[s:e])
    rows.append(dict(event=name, kind="MVC" if "MVC" in name else "Throw",
                     number=int(name.split()[-1]), start_s=round(mmss(s0), 1),
                     duration_s=round(mmss(e0) - mmss(s0), 1),
                     env_peak=env[s:e].max(), mdf_hz=mdf, mnf_hz=mnf, rms=rms))
print(f"\n{len(rows)} of {len(LABELS)} labelled events have data in this file")
if missing:
    print(f"Not in the file ({len(missing)}), so left out of everything below:")
    for m in missing:
        print("  " + m)
if not rows:
    raise ValueError("None of the labelled times matched any data. Check that the label times "
                     "are elapsed MM:SS from the start of the recording.")
tab = pd.DataFrame(rows)
mvc_rms = tab.loc[tab["kind"] == "MVC", "rms"].mean()
tab["pct_mvc"] = tab["rms"] / mvc_rms * 100
tab.to_csv("semg_labeled.csv", index=False)
print("\n=== Features per labelled event ===")
print(tab.round(2).to_string(index=False))

th = tab[tab["kind"] == "Throw"]
fits = {}
print("\n=== Fatigue trend across throws (negative frequency slope = fatigue) ===")
for c in ("mdf_hz", "mnf_hz", "pct_mvc"):
    lr = linregress(th["number"], th[c])
    fits[c] = lr
    pct = lr.slope * (th["number"].max() - th["number"].min()) / th[c].iloc[0] * 100
    print(f"  {c:8s}: {lr.slope:+.4g} per throw, p = {lr.pvalue:.4f}, "
          f"R^2 = {lr.rvalue**2:.2f}, total change {pct:+.1f}%")

down = fits["mdf_hz"].slope < 0 and fits["mnf_hz"].slope < 0
sig = min(fits["mdf_hz"].pvalue, fits["mnf_hz"].pvalue) < 0.05
if down and sig:
    verdict = "Fatigue: MDF and MNF both fall significantly across throws"
    verdict += " and amplitude rises." if fits["pct_mvc"].slope > 0 else ", but amplitude does not rise."
elif down:
    verdict = "Possible fatigue: MDF and MNF both fall, but not significantly (p > 0.05)."
elif fits["mdf_hz"].slope < 0 or fits["mnf_hz"].slope < 0:
    verdict = "Unclear: MDF and MNF disagree."
else:
    verdict = "No fatigue signature: frequency did not fall across throws."
print("\n" + verdict)

# ================= Part 2: detector scored against the labels =================
# Generalizable rule: envelope peaks, spaced apart, above a level set by THIS recording.
found = [(n, a, b) for n, a, b in LABELS
         if ((t >= mmss(a)) & (t <= mmss(b))).sum() >= 256]
label_mid = np.array([(mmss(a) + mmss(b)) / 2 for n, a, b in found])
ref = np.median([env[(t >= mmss(a)) & (t <= mmss(b))].max() for n, a, b in found])
print(f"\n=== Detector tuning (reference peak = median labelled peak = {ref:.4g}) ===")
print("  sep_s  frac   found  hits  missed  false")

best = None
for sep in (5, 10, 15, 20):
    for frac in (0.15, 0.25, 0.35, 0.5):
        pk, _ = find_peaks(env, distance=int(sep * fs), height=frac * ref)
        pt = t[pk]
        hits = sum(np.any(np.abs(pt - m) <= TOL_S) for m in label_mid)
        false = len(pt) - sum(np.any(np.abs(label_mid - p) <= TOL_S) for p in pt)
        print(f"  {sep:<6} {frac:<6} {len(pt):<6} {hits:<5} {len(label_mid)-hits:<7} {false}")
        score = hits - false
        if best is None or score > best[0]:
            best = (score, sep, frac, pk)

_, sep, frac, pk = best
print(f"\nBest: peaks at least {sep} s apart, above {frac:.0%} of the reference peak "
      f"-> {len(pk)} detected, {sum(np.any(np.abs(t[pk]-m) <= TOL_S) for m in label_mid)}"
      f"/{len(label_mid)} labels found")
print("For another person, replace `ref` with their own MVC peak (or the median peak of their "
      "strongest bursts) so the threshold scales to them.")

# ================= Figures =================
fig1, ax = plt.subplots(2, 1, sharex=True, figsize=(18, 7))
step = max(1, fs // 50)
ax[0].plot(t[::step], y[::step], color="gray", lw=0.4)
ax[1].plot(t[::step], env[::step], color="black", lw=0.8)
ax[1].axhline(frac * ref, color="green", ls="--", label=f"Detector level ({frac:.0%} of reference)")
ymax = env.max()
for name, s0, e0 in found:                      # only events that have data in the file
    c = "tab:red" if "MVC" in name else "tab:blue"
    for a in ax:
        a.axvspan(mmss(s0), mmss(e0), color=c, alpha=0.3)
    ax[1].text((mmss(s0) + mmss(e0)) / 2, ymax * 1.02,
               name.replace("Throw ", "T"), ha="center", va="bottom",
               fontsize=7, rotation=90, color=c)
ax[1].set_ylim(0, ymax * 1.25)
ax[1].plot(t[pk], env[pk], "kx", ms=6, label="Auto-detected peak")
ax[0].set_title("Filtered sEMG with labelled events (blue = throw, red = MVC)")
ax[0].set_ylabel("Amplitude")
ax[1].set_title("Envelope, labels, and auto-detected peaks")
ax[1].set_xlabel("Time from start of recording (s)")
ax[1].set_ylabel("Envelope")
ax[1].legend(loc="upper right")
fig1.tight_layout()
fig1.savefig("semg_labeled.png", dpi=150)

fig2, a2 = plt.subplots(1, 3, figsize=(15, 4.5))
for a, c, title, unit in [(a2[0], "mdf_hz", "Median frequency (MDF)", "Hz"),
                          (a2[1], "mnf_hz", "Mean power frequency (MNF)", "Hz"),
                          (a2[2], "pct_mvc", "Amplitude", "% of MVC")]:
    a.plot(th["number"], th[c], "o-", color="tab:blue")
    xs = th["number"].to_numpy()
    a.plot(xs, fits[c].intercept + fits[c].slope * xs, "r--",
           label=f"{fits[c].slope:+.3g}/throw (p={fits[c].pvalue:.3f})")
    a.legend()
    a.set_title(title)
    a.set_xlabel("Throw number")
    a.set_ylabel(unit)
fig2.suptitle(verdict)
fig2.tight_layout()
fig2.savefig("semg_fatigue.png", dpi=150)
plt.show()