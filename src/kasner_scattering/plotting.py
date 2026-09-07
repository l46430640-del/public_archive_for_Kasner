"""Rebuild the published spatial comparison from every exported profile node."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .propagation.inputs import ROOT


def plot_results(output):
    output = Path(output).resolve()
    build = (ROOT / "build").resolve()
    if build not in output.parents:
        raise ValueError("figure output must be strictly below build/")
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / "data/tables/response_profiles.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .7, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.9), sharey=True)
    fig.subplots_adjust(left=.115, right=.974, bottom=.21, top=.80, wspace=.13)
    for i, (ax, panel) in enumerate(zip(axes, ("a", "b"))):
        data = [r for r in rows if r["panel"] == panel]
        def values(key):
            return np.array([float(r[key]) for r in data])
        x = values("y")/values("width")
        scale = 100/values("p0")
        ax.plot(x, values("coherent")*scale, color="#0072B2", lw=1.5, label="Coherent prediction")
        ax.plot(x, values("omitted_cross")*scale, color="#D55E00", ls="--", lw=1.2, label="Cross term omitted")
        pick = np.unique(np.r_[np.arange(0,len(x),6 if panel=="a" else 3),len(x)-1])
        ax.plot(x[pick], (values("nonlinear")*scale)[pick], ls="none", marker="o", ms=3,
                mfc="white", mec="#222222", mew=.8, label="Nonlinear evolution")
        ax.set_xlim(-.255, .255)
        ax.set_xticks([-.25, 0, .25])
        ax.set_xlabel(r"$y/w$")
        ax.set_ylim(-1.14, .05)
        ax.axhline(0, color=".75", lw=.6, zorder=-1)
        ax.set_title((r"(a) $\delta=10^{-6}$, $\lambda_1$" if i==0 else r"(b) $\delta=10^{-5}$, $\lambda_3$"), fontsize=10, pad=10, loc="left")
    axes[0].set_ylabel(r"$100\,\Delta p_\psi/|p_{\psi,0}|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(.53,1.025), fontsize=8.5, columnspacing=1.4, handlelength=2.2)
    pdf, png = output / "response_profiles.pdf", output / "response_profiles.png"
    fig.savefig(pdf, metadata={"Creator": "kasner-scattering", "CreationDate": None, "ModDate": None})
    fig.savefig(png, dpi=180, metadata={"Software": "kasner-scattering"})
    plt.close(fig)
    return [pdf, png]
