"""
sEMG throw/MVC detection and fatigue analysis - pronator teres, baseball throwing.

Each event is one LabScribe recording block (the elapsed-Time column restarts at
every new recording), so detection needs no amplitude threshold and transfers to
other people's files unchanged.

The throw instant is the largest excursion of the band-passed signal (the raw
signal peak). The envelope is used only to measure how long the muscle stays
active, which is what separates a brief throw from a sustained MVC.
"""
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch
from scipy.stats import linregress

# ================= Settings (measured from this file) =================
FILE = "sEMG&IMU_Oct_21.xlsx"

AA_CUTOFF = 500          # Hz anti-aliasing low-pass, auto-capped at 0.9*Nyquist per block
LOW, HIGH, ORDER = 20, 450, 4    # Hz Butterworth band-pass, HIGH auto-capped per block
NOTCH_60 = False         # measured: no mains hum (60 Hz power 0.3-3.0x its neighbours)

MIN_EVENT_S = 5.0        # ignore recording blocks shorter than this
MVC_MIN_ACTIVE_S = 10  # >= this much continuous activity means MVC, not throw
ENV_HZ = 5               # envelope smoothing, used only to measure active duration
ACTIVE_FRAC = 0.25       # "active" = envelope above this fraction of the block's peak
FIRST_EVENT_BLOCK = 3    # blocks 1-2 are setup; event numbering starts here
WIN_S = 0.5              # analysis window centred on the peak, for MDF/MNF/RMS


def main():
    # ---------- Load (TimeOfDay holds real Excel time values, not text) ----------
    ws = openpyxl.load_workbook(FILE, read_only=True, data_only=True)["in"]
    tod, el, raw = [], [], []
    for r in ws.iter_rows(values_only=True):
        a, b, c = r[0], r[1], r[2]
        if not isinstance(a, datetime.time):
            continue                                  # header row between recordings
        try:
            el.append(float(b)); raw.append(float(c))
        except (TypeError, ValueError):
            continue
        tod.append(a.hour * 3600 + a.minute * 60 + a.second + a.microsecond / 1e6)
    tod, el, raw = np.array(tod), np.array(el), np.array(raw)
    t0 = tod[0]
    clock = lambda s: (datetime.datetime(2000, 1, 1)
                       + datetime.timedelta(seconds=float(s))).strftime("%H:%M:%S.%f")[:-3]
    lapse = lambda s: f"{int((s - t0) // 60):02d}:{(s - t0) % 60:05.2f}"
    print(f"{len(raw)} samples, {clock(tod[0])} to {clock(tod[-1])}")

    # ---------- Events = recording blocks ----------
    edges = [0] + list(np.where(np.diff(el) <= 0)[0] + 1) + [len(el)]
    print(f"{len(edges)-1} recording blocks\n")

    rows, traces = [], []
    nt = nm = 0
    for k in range(len(edges) - 1):
        s, e = edges[k], edges[k + 1]
        d = np.diff(el[s:e]); d = d[d > 0]
        if len(d) == 0:
            continue
        fs = int(round(1 / np.median(d)))     # per block: this file mixes 1000 and 2000 Hz
        t = el[s:e] - el[s]
        if t[-1] < MIN_EVENT_S:
            continue
        hi = min(HIGH, 0.45 * fs)

        x = raw[s:e] - raw[s:e].mean()                                    # 1. DC offset
        x = sosfiltfilt(butter(4, min(AA_CUTOFF, 0.9 * fs / 2), "lowpass",
                               fs=fs, output="sos"), x)                   # 2. anti-aliasing
        y = sosfiltfilt(butter(ORDER, [LOW, hi], "bandpass",
                               fs=fs, output="sos"), x)                   # 3. band-pass
        if NOTCH_60:                                                      # 4. mains notch
            bn, an = iirnotch(60, Q=30, fs=fs)
            y = filtfilt(bn, an, y)

        env = np.clip(sosfiltfilt(butter(4, ENV_HZ, "lowpass", fs=fs, output="sos"),
                                  np.abs(y)), 0, None)
        active = (env > ACTIVE_FRAC * env.max()).sum() / fs

        p = int(np.argmax(np.abs(y)))                 # the throw instant: raw signal peak
        h = int(WIN_S * fs / 2)
        if active >= MVC_MIN_ACTIVE_S:
            on = np.where(env > ACTIVE_FRAC * env.max())[0]
            w = y[on[0]:on[-1] + 1]                   # MVC: the whole sustained hold
        else:
            w = y[max(0, p - h):min(len(y), p + h)]   # throw: fixed window on the peak
        f, pw = welch(w, fs=fs, nperseg=min(512, len(w)))
        m = (f >= LOW) & (f <= hi); f, pw = f[m], pw[m]
        cum = np.cumsum(pw)

        if k + 1 < FIRST_EVENT_BLOCK:
            kind, num = "Setup", 0
        elif active >= MVC_MIN_ACTIVE_S:
            nm += 1; kind, num = "MVC", nm
        else:
            nt += 1; kind, num = "Throw", nt

        rows.append(dict(block=k + 1, kind=kind, number=num, fs=fs,
                         peak_elapsed=lapse(tod[s] + t[p]), peak_clock=clock(tod[s] + t[p]),
                         peak_val=round(y[p], 3), active_s=round(active, 2),
                         mdf_hz=f[np.searchsorted(cum, cum[-1] / 2)],
                         mnf_hz=np.sum(f * pw) / np.sum(pw),
                         rms=np.sqrt(np.mean(w ** 2)),
                         mvc_ref=(np.sqrt(np.convolve(w ** 2, np.ones(min(len(w), fs))
                                                      / min(len(w), fs), "valid").max())
                                  if active >= MVC_MIN_ACTIVE_S else np.nan)))
        traces.append((tod[s] - t0, t, y, env, kind, num))

    tab = pd.DataFrame(rows)
    mv = tab[tab["kind"] == "MVC"]
    if len(mv):
        tab["pct_mvc"] = tab["rms"] / mv["mvc_ref"].max() * 100   # strongest MVC, peak 1 s RMS
    tab.to_csv("semg_events.csv", index=False)
    print(tab.round(3).to_string(index=False))
    print(f"\n{nt} throws, {nm} MVC")

    # ---------- Fatigue trend ----------
    th = tab[tab["kind"] == "Throw"].reset_index(drop=True)
    amp = "pct_mvc" if "pct_mvc" in tab else "rms"
    fits = {}
    print("\nTrend across throws (falling MDF and MNF = fatigue):")
    for c in ("mdf_hz", "mnf_hz", amp):
        lr = linregress(th["number"], th[c])
        fits[c] = lr
        print(f"  {c:8s}: {lr.slope:+.3f}/throw, p = {lr.pvalue:.3f}, R2 = {lr.rvalue**2:.2f}")
    down = fits["mdf_hz"].slope < 0 and fits["mnf_hz"].slope < 0
    sig = min(fits["mdf_hz"].pvalue, fits["mnf_hz"].pvalue) < 0.05
    verdict = ("Fatigue: MDF and MNF both fall significantly" if down and sig else
               "Possible fatigue: both fall, not significantly" if down else
               "No fatigue signature: frequency did not fall across throws")
    print("\n" + verdict)

    # ---------- Figures ----------
    fig, ax = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    ymax = max(e.max() for _, _, _, e, _, _ in traces)
    for off, t, y, env, kind, num in traces:
        c = {"MVC": "tab:red", "Throw": "tab:blue", "Setup": "tab:gray"}[kind]
        st = max(1, len(t) // 2000)
        ax[0].plot(off + t[::st], y[::st], color=c, lw=0.4)
        ax[1].plot(off + t[::st], env[::st], color=c, lw=0.8)
        if kind != "Setup":
            ax[1].text(off + t[len(t) // 2], ymax * 1.02,
                       f"{'M' if kind == 'MVC' else ''}{num}",
                       ha="center", fontsize=6, rotation=90, color=c)
    ax[0].set_title(f"Filtered sEMG: {nt} throws (blue), {nm} MVC (red)")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_title("Envelope per recording block")
    ax[1].set_xlabel("Time from start of session (s)")
    ax[1].set_ylim(0, ymax * 1.2)
    fig.tight_layout(); fig.savefig("semg_events.png", dpi=140)

    fig2, a2 = plt.subplots(1, 3, figsize=(15, 4.5))
    lab = {"mdf_hz": ("Median frequency (MDF)", "Hz"),
           "mnf_hz": ("Mean power frequency (MNF)", "Hz"),
           "pct_mvc": ("Amplitude", "% of MVC"), "rms": ("Amplitude (RMS)", "")}
    for a, c in zip(a2, ("mdf_hz", "mnf_hz", amp)):
        a.plot(th["number"], th[c], "o-", color="tab:blue")
        xs = th["number"].to_numpy()
        a.plot(xs, fits[c].intercept + fits[c].slope * xs, "r--",
               label=f"{fits[c].slope:+.3g}/throw (p={fits[c].pvalue:.3f})")
        a.legend(); a.set_title(lab[c][0]); a.set_xlabel("Throw number"); a.set_ylabel(lab[c][1])
    fig2.suptitle(verdict)
    fig2.tight_layout(); fig2.savefig("semg_fatigue.png", dpi=140)
    print("\nSaved semg_events.csv, semg_events.png, semg_fatigue.png")
    plt.show()


if __name__ == "__main__":
    main()