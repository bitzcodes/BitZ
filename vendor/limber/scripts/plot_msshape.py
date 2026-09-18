#!/usr/bin/env python3
"""Plot the MultiSwap-shaped benchmark comparison: IntMod-Spartan vs the
shape-matched plain-Spartan native baseline.

Reads criterion's JSON estimates from `target/criterion/` for the
`msshape_c{10,12,14}` configs of both benches and emits prover-time and
verifier-time figures (PDF + PNG) under `docs/plots/`.

Run the benches first:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan_modp -- msshape
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench spartan_synthetic -- msshape
then:
    python3 scripts/plot_msshape.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRITERION = os.path.join(ROOT, "target", "criterion")
OUTDIR = os.path.join(ROOT, "docs", "plots")

# log2(num_cons) of the sweep points; vars = 2^(lc+1).
SIZES = [10, 12, 14]

SERIES = {
    "IntMod-Spartan": "imod_spartan_modp",
    "Spartan (native baseline)": "spartan_synthetic",
}

STYLE = {
    "IntMod-Spartan": dict(color="#c0392b", marker="o"),
    "Spartan (native baseline)": dict(color="#2c5f8a", marker="s"),
}


def read_estimate(group: str, phase: str, lc: int):
    """Return (median_ms, lo_ms, hi_ms) from criterion's estimates.json,
    or None if the benchmark hasn't been run."""
    # Criterion sanitizes the benchmark id "prove/msshape_c10" into the
    # directory name "prove_msshape_c10".
    d = os.path.join(CRITERION, group, f"{phase}_msshape_c{lc}", "new", "estimates.json")
    if not os.path.exists(d):
        return None
    with open(d) as f:
        e = json.load(f)
    med = e["median"]
    to_ms = 1e-6
    return (
        med["point_estimate"] * to_ms,
        med["confidence_interval"]["lower_bound"] * to_ms,
        med["confidence_interval"]["upper_bound"] * to_ms,
    )


def plot_phase(phase: str, title: str, outname: str) -> bool:
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    data = {}
    for label, group in SERIES.items():
        pts = [(lc, read_estimate(group, phase, lc)) for lc in SIZES]
        pts = [(lc, est) for lc, est in pts if est is not None]
        if not pts:
            print(f"warning: no data for {group}/{phase} — skipping series")
            continue
        xs = [lc for lc, _ in pts]
        ys = [est[0] for _, est in pts]
        yerr_lo = [est[0] - est[1] for _, est in pts]
        yerr_hi = [est[2] - est[0] for _, est in pts]
        ax.errorbar(
            xs,
            ys,
            yerr=[yerr_lo, yerr_hi],
            label=label,
            linewidth=1.6,
            markersize=5,
            capsize=2,
            **STYLE[label],
        )
        data[label] = dict(pts)

    if len(data) < 2:
        plt.close(fig)
        return False

    # Annotate the overhead ratio at each common size.
    imod, base = data["IntMod-Spartan"], data["Spartan (native baseline)"]
    for lc in SIZES:
        if lc in imod and lc in base:
            ratio = imod[lc][0] / base[lc][0]
            ax.annotate(
                f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×",
                (lc, imod[lc][0]),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color="#c0392b",
            )

    ax.set_xlabel("Constraints")
    ax.set_ylabel(f"{title} time (ms)")
    # Log only when the data genuinely spans decades; otherwise a linear
    # axis from zero (a narrow log range yields unreadable ticks like
    # 3x10^1).
    all_y = [est[0] for pts in data.values() for est in pts.values()]
    if max(all_y) / min(all_y) >= 4:
        ax.set_yscale("log")
        lo, hi = min(all_y) / 1.25, max(all_y) * 1.3
        ticks = [c * 10**k for k in range(0, 7) for c in (1, 2, 5) if lo <= c * 10**k <= hi]
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        ax.set_ylim(bottom=0)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"$2^{{{lc}}}$" for lc in SIZES])
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.margins(y=0.15)  # headroom so ratio labels clear the frame
    # Legend above the axes so it can never occlude either series.
    ax.legend(
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        borderaxespad=0.2,
    )
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{outname}.{ext}")
        fig.savefig(path, dpi=200)
        print(f"wrote {os.path.relpath(path, ROOT)}")
    plt.close(fig)
    return True


def emit_latex_table() -> bool:
    """Write docs/plots/msshape_table.tex: booktabs table of the sweep."""
    rows = []
    for lc in SIZES:
        ip = read_estimate("imod_spartan_modp", "prove", lc)
        bp = read_estimate("spartan_synthetic", "prove", lc)
        iv = read_estimate("imod_spartan_modp", "verify", lc)
        bv = read_estimate("spartan_synthetic", "verify", lc)
        if None in (ip, bp, iv, bv):
            continue
        rows.append(
            f"    $2^{{{lc}}}$ & {ip[0]:.0f} & {bp[0]:.1f} & "
            f"{ip[0] / bp[0]:.0f}$\\times$ & {iv[0]:.1f} & {bv[0]:.1f} & "
            f"{iv[0] / bv[0]:.1f}$\\times$ \\\\"
        )
    if not rows:
        return False
    body = "\n".join(rows)
    tex = (
        "% Auto-generated by scripts/plot_msshape.py — do not edit by hand.\n"
        "\\begin{table}[t]\n"
        "  \\centering\n"
        "  \\caption{MultiSwap-shaped workload: \\sysname{} vs.\\ a\n"
        "    shape-matched native Spartan baseline (same constraint and\n"
        "    variable counts, full-width witness values). Gates are random\n"
        "    multiplications modulo the Tom-256 base-field prime, which the\n"
        "    native system cannot express in one constraint; the baseline\n"
        "    proves native gates of the same shape. Criterion medians,\n"
        "    10 samples; prover time includes witness generation and\n"
        "    commitment on both sides.}\n"
        "  \\label{tab:msshape}\n"
        "  \\begin{tabular}{lrrrrrr}\n"
        "    \\toprule\n"
        "    & \\multicolumn{3}{c}{Prover (ms)} & \\multicolumn{3}{c}{Verifier (ms)} \\\\\n"
        "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n"
        "    Constraints & Ours & Spartan & Ratio & Ours & Spartan & Ratio \\\\\n"
        "    \\midrule\n"
        f"{body}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, "msshape_table.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return True


def emit_pgfplots(phase: str, title: str, outname: str) -> bool:
    """Write a self-contained pgfplots tikzpicture (native LaTeX version
    of the matplotlib figure) to docs/plots/{outname}.tex. Requires
    \\usepackage{pgfplots} + \\pgfplotsset{compat=1.18} in the preamble;
    place via \\input inside a figure/subfigure."""
    series = {}
    for label, group in SERIES.items():
        pts = [(lc, read_estimate(group, phase, lc)) for lc in SIZES]
        series[label] = [(lc, est) for lc, est in pts if est is not None]
    if any(not pts for pts in series.values()):
        return False

    def coords(label):
        return " ".join(
            f"({lc},{est[0]:.3f}) +- ({est[0] - est[1]:.3f},{est[2] - est[0]:.3f})"
            for lc, est in series[label]
        )

    # Nice fixed-point y ticks (1/2/5 per decade within the data range) —
    # avoids pgfplots' fractional-exponent ticks (e.g. 10^{1.5}) on
    # narrow log ranges.
    all_y = [est[0] for pts in series.values() for _, est in pts]
    lo, hi = min(all_y) / 1.25, max(all_y) * 1.3
    if max(all_y) / min(all_y) >= 4:
        yticks = [
            c * 10**k
            for k in range(0, 7)
            for c in (1, 2, 5)
            if lo <= c * 10**k <= hi
        ]
        ytick_list = ",".join(str(t) for t in yticks)
        yaxis_opts = (
            "    ymode=log,\n"
            f"    ytick={{{ytick_list}}}, log ticks with fixed point,\n"
        )
    else:
        step = 10
        top = int(hi // step + 1) * step
        lin_ticks = ",".join(str(t) for t in range(0, top + 1, step))
        yaxis_opts = f"    ymin=0,\n    ytick={{{lin_ticks}}},\n"

    imod = dict(series["IntMod-Spartan"])
    base = dict(series["Spartan (native baseline)"])
    ratio_nodes = "\n".join(
        f"    \\node[above=2pt, ourscolor, font=\\scriptsize] at "
        f"(axis cs:{lc},{imod[lc][0]:.3f}) "
        f"{{{imod[lc][0] / base[lc][0]:.0f}$\\times$}};"
        if imod[lc][0] / base[lc][0] >= 10
        else f"    \\node[above=2pt, ourscolor, font=\\scriptsize] at "
        f"(axis cs:{lc},{imod[lc][0]:.3f}) "
        f"{{{imod[lc][0] / base[lc][0]:.1f}$\\times$}};"
        for lc in SIZES
        if lc in imod and lc in base
    )
    ticks = ",".join(str(lc) for lc in SIZES)
    ticklabels = ",".join(f"$2^{{{lc}}}$" for lc in SIZES)

    tex = (
        "% Auto-generated by scripts/plot_msshape.py — do not edit by hand.\n"
        "% Preamble: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}\n"
        "\\colorlet{ourscolor}{red!70!black}\n"
        "\\colorlet{basecolor}{blue!50!black}\n"
        "\\begin{tikzpicture}\n"
        "  \\begin{axis}[\n"
        # scale only axis + fixed-width y tick labels: both panels get an

        # identical axis box and identical left offset, so the stacked

        # prover/verifier figures align regardless of tick label widths.

        "    scale only axis, width=0.84\\linewidth, height=4.6cm,\n"
        "    yticklabel style={font=\\scriptsize, text width=2.4em, align=right},\n"
        "    xticklabel style={font=\\scriptsize},\n"
        + yaxis_opts
        + f"    xtick={{{ticks}}}, xticklabels={{{ticklabels}}},\n"
        "    xlabel={Constraints}, ylabel={" + title + " time (ms)},\n"
        "    grid=both, grid style={gray!20},\n"
        "    % legend above the axes so it never occludes either series\n"
        "    legend style={at={(0.5,1.03)}, anchor=south, legend columns=-1,\n"
        "      font=\\scriptsize, draw=none, /tikz/every even column/.append style={column sep=8pt}},\n"
        "    enlarge y limits={upper, value=0.2},\n"
        "    error bars/y dir=both, error bars/y explicit,\n"
        "  ]\n"
        "    \\addplot[ourscolor, thick, mark=*, mark size=1.8pt]\n"
        f"      coordinates {{{coords('IntMod-Spartan')}}};\n"
        "    \\addlegendentry{IntMod-Spartan}\n"
        "    \\addplot[basecolor, thick, mark=square*, mark size=1.8pt]\n"
        f"      coordinates {{{coords('Spartan (native baseline)')}}};\n"
        "    \\addlegendentry{Spartan (native baseline)}\n"
        f"{ratio_nodes}\n"
        "  \\end{axis}\n"
        "\\end{tikzpicture}\n"
    )
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"{outname}.tex")
    with open(path, "w") as f:
        f.write(tex)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return True


def main():
    ok = True
    ok &= plot_phase("prove", "Prover", "msshape_prove")
    ok &= plot_phase("verify", "Verifier", "msshape_verify")
    ok &= emit_latex_table()
    ok &= emit_pgfplots("prove", "Prover", "msshape_prove_pgf")
    ok &= emit_pgfplots("verify", "Verifier", "msshape_verify_pgf")
    if not ok:
        print(
            "\nmissing data — run:\n"
            '  RUSTFLAGS="-C target-cpu=native" cargo bench --bench imod_spartan_modp -- msshape\n'
            '  RUSTFLAGS="-C target-cpu=native" cargo bench --bench spartan_synthetic -- msshape'
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
