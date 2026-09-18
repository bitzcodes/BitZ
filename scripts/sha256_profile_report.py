#!/usr/bin/env python3
"""Render a full-range SHA-256 timing/constraint overview from zkperf summary JSON."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


COMPONENTS = (
    ("Message schedule", 48, 7_776, 10_224),
    ("Round logic", 128, 45_568, 31_104),
    ("Feed-forward", 8, 776, 264),
    ("Initial lifts + constant one", 0, 0, 769),
)


def decile(metric: dict, percentile: int) -> float:
    return float(metric["decilesMsExact"][f"p{percentile}"])


def build_rows(summary: dict) -> list[dict]:
    rows = []
    series = sorted(
        summary["series"],
        key=lambda item: item["parameters"]["input"]["sha256_compressions"],
    )
    for item in series:
        inputs = item["parameters"]["input"]
        security = item["parameters"]["security"]
        proving = item["metrics"]["proving"]
        verification = item["metrics"]["verification"]
        compressions = inputs["sha256_compressions"]
        exponent = compressions.bit_length() - 1
        prover_ms = float(proving["medianMsExact"])
        rows.append(
            {
                "exponent": exponent,
                "compressions": compressions,
                "prime_bits": security["prime_bits"],
                "outer_rounds": exponent + 8,
                "live_constraints": inputs["constraints"],
                "padded_rows": inputs["num_rows"],
                "source_cells": inputs["padded_source_cells"],
                "assignment_cells": inputs["padded_assignment_cells"],
                "map_nonzeros": inputs["conceptual_map_nonzeros"],
                "c_nonzeros": inputs["conceptual_c_nonzeros"],
                "prover_median_ms": prover_ms,
                "prover_p10_ms": decile(proving, 10),
                "prover_p90_ms": decile(proving, 90),
                "verify_median_ms": float(verification["medianMsExact"]),
                "verify_p10_ms": decile(verification, 10),
                "verify_p90_ms": decile(verification, 90),
                "throughput": compressions / (prover_ms / 1_000),
                "n": proving["n"],
                "warmups": item["warmupN"],
            }
        )
    return rows


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_constraint_csv(output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("component", "rows_per_compression", "c_nonzeros", "map_nonzeros"))
        writer.writerows(COMPONENTS)
        writer.writerow(("Total", 184, 54_120, 42_361))


def render_html(rows: list[dict], summary: dict, output: Path) -> None:
    environments = {
        (series["environment"]["cpu"], series["environment"]["threads"])
        for series in summary["series"]
    }
    if len(environments) != 1:
        raise ValueError("range overview requires one CPU/thread configuration")
    cpu, threads = environments.pop()
    measured = sum(row["n"] for row in rows)
    warmups = sum(row["warmups"] for row in rows)
    sample_counts = {row["n"] for row in rows}
    warmup_counts = {row["warmups"] for row in rows}
    if len(sample_counts) == 1 and len(warmup_counts) == 1:
        samples_per_point = sample_counts.pop()
        warmups_per_point = warmup_counts.pop()
        campaign = (
            f"Every point is {warmups_per_point} warmup plus "
            f"{samples_per_point} verified samples"
        )
    else:
        campaign = f"The campaign contains {warmups} warmups and {measured} verified samples"
    embedded = json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")
    rendered = TEMPLATE.replace("__DATA__", embedded)
    rendered = rendered.replace("__CPU__", escape(str(cpu)))
    rendered = rendered.replace("__THREADS__", str(threads))
    rendered = rendered.replace("__CAMPAIGN__", campaign)
    rendered = rendered.replace("__TOTAL_RUNS__", str(measured + warmups))
    rendered = rendered.replace("__WARMUPS__", str(warmups))
    rendered = rendered.replace("__MEASURED__", str(measured))
    output.write_text(rendered)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SHA-256 Paper128 full-range profile</title>
<style>
:root{color-scheme:dark;--bg:#081018;--panel:#101b27;--line:#26384a;--text:#e7f0f8;--muted:#9fb0c0;--cyan:#50d5ff;--amber:#ffbe55;--green:#62dfa0;--red:#ff768c}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% -15%,#18334a 0,transparent 35%),var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}.wrap{max-width:1280px;margin:auto;padding:34px 24px 60px}h1{font-size:30px;line-height:1.15;margin:0 0 7px}.lede{color:var(--muted);max-width:900px;margin:0 0 22px}.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}.card,.panel{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:13px;box-shadow:0 14px 35px #0005}.card{padding:14px 16px}.card b{font-size:20px;display:block}.card span{color:var(--muted);font-size:12px}.grid{display:grid;grid-template-columns:1.35fr .85fr;gap:14px}.panel{padding:18px;margin-top:14px}.panel h2{font-size:17px;margin:0 0 3px}.panel p{color:var(--muted);margin:0 0 10px;font-size:12px}.chart{width:100%;height:auto;display:block}.axis{stroke:#547087;stroke-width:1}.gridline{stroke:#223343;stroke-width:1}.tick{fill:#9fb0c0;font-size:11px}.legend{display:flex;gap:18px;margin:8px 0 0;color:var(--muted);font-size:12px}.swatch{display:inline-block;width:11px;height:3px;vertical-align:middle;margin-right:6px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:8px 9px;border-bottom:1px solid #203141;text-align:right;white-space:nowrap}th{color:#a9b9c8;font-size:11px;text-transform:uppercase;letter-spacing:.04em}th:first-child,td:first-child{text-align:left}tr:hover td{background:#152333}.notes{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.note{border-left:3px solid var(--cyan);padding:6px 10px;color:var(--muted)}a{color:var(--cyan)}.tip{position:fixed;display:none;pointer-events:none;background:#04090eee;border:1px solid #416078;padding:8px 10px;border-radius:8px;font-variant-numeric:tabular-nums;box-shadow:0 8px 25px #0008;z-index:2}@media(max-width:850px){.cards,.notes{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}}@media(max-width:540px){.cards,.notes{grid-template-columns:1fr}}
</style>
</head>
<body><main class="wrap">
<h1>SHA-256 Paper128 · full supported range</h1>
<p class="lede">Observed on __CPU__ with __THREADS__ Rayon workers. __CAMPAIGN__; ribbons are sample P10–P90 deciles, not confidence intervals. The prime field is transcript-selected at 112/113 bits inside 128-bit storage, while commitments remain over GF(2<sup>128</sup>).</p>
<div class="cards">
 <div class="card"><b id="peakThroughput"></b><span>peak prover throughput</span></div>
 <div class="card"><b>71.875%</b><span>live constraint-row occupancy</span></div>
 <div class="card"><b>62.430%</b><span>assignment-grid occupancy</span></div>
 <div class="card"><b>__TOTAL_RUNS__ runs</b><span>__WARMUPS__ warmups + __MEASURED__ measured</span></div>
</div>
<div class="grid">
 <section class="panel"><h2>Latency across the paper range</h2><p>Logarithmic y-axis; hover points for exact median and sample deciles.</p><svg id="latency" class="chart" viewBox="0 0 760 390"></svg><div class="legend"><span><i class="swatch" style="background:var(--cyan)"></i>Prover</span><span><i class="swatch" style="background:var(--amber)"></i>Verifier</span></div></section>
 <section class="panel"><h2>Prover throughput</h2><p>The two-chunk virtual-BitZ row-weight decomposition begins at 2<sup>14</sup>.</p><svg id="throughput" class="chart" viewBox="0 0 480 390"></svg><div class="legend"><span><i class="swatch" style="background:var(--green)"></i>compressions/s</span></div></section>
</div>
<section class="panel"><h2>Measured range and exact constraint geometry</h2><p>Global counts are conceptual tensor repetitions; the implementation retains one local CSC relation and repeats it implicitly.</p><div class="table-wrap"><table><thead><tr><th>Batch</th><th>Prime</th><th>Outer rounds</th><th>Live constraints</th><th>Padded rows</th><th>Prover median</th><th>P10–P90</th><th>Verifier</th><th>Throughput</th></tr></thead><tbody id="rows"></tbody></table></div></section>
<section class="panel"><h2>Per-compression constraint breakdown</h2><div class="table-wrap"><table><thead><tr><th>Component</th><th>Rows</th><th>C nonzeros</th><th>Virtual-map edges</th><th>Share of rows</th></tr></thead><tbody>
<tr><td>Message schedule</td><td>48</td><td>7,776</td><td>10,224</td><td>26.09%</td></tr><tr><td>Round logic</td><td>128</td><td>45,568</td><td>31,104</td><td>69.57%</td></tr><tr><td>Feed-forward</td><td>8</td><td>776</td><td>264</td><td>4.35%</td></tr><tr><td>Initial lifts + constant one</td><td>0</td><td>0</td><td>769</td><td>0%</td></tr><tr><td><b>Total</b></td><td><b>184</b></td><td><b>54,120</b></td><td><b>42,361</b></td><td><b>100%</b></td></tr>
</tbody></table></div></section>
<section class="panel notes"><div class="note"><b>Observed chronology</b><br><a href="intervals.html">Open the interactive interval timeline</a> for nested phases, math, recurrence coordinates, and per-interval P10–P90.</div><div class="note"><b>CPU hotspots</b><br><a href="../cpu/cpu_hotspots.md">Open the Time Profiler report</a>. It covers a 10-second proving prefix, so its percentages are hotspot evidence rather than end-to-end phase shares.</div><div class="note"><b>Expected discontinuities</b><br>Exact-product projection turns parallel at 2<sup>8</sup>; the row-weight decomposition changes from one chunk to two at 2<sup>14</sup>.</div><div class="note"><b>Root boundary</b><br>The timeline root is a complete verified trial. The prover metric excludes verification, public relation setup, and deterministic input generation.</div></section>
</main><div id="tip" class="tip"></div>
<script>
const data=__DATA__,NS='http://www.w3.org/2000/svg',tip=document.getElementById('tip');
const svg=(name,attrs={})=>{const e=document.createElementNS(NS,name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e};
const fmtMs=v=>v>=1000?(v/1000).toFixed(2)+' s':v.toFixed(v<10?2:1)+' ms';
function axes(root,W,H,m,yTicks,yMap){for(const yv of yTicks){const y=yMap(yv);root.append(svg('line',{x1:m.l,y1:y,x2:W-m.r,y2:y,class:'gridline'}));const t=svg('text',{x:m.l-8,y:y+4,'text-anchor':'end',class:'tick'});t.textContent=fmtMs(yv);root.append(t)}data.forEach((d,i)=>{const x=m.l+i*(W-m.l-m.r)/(data.length-1);const t=svg('text',{x,y:H-m.b+20,'text-anchor':'middle',class:'tick'});t.textContent='2^'+d.exponent;root.append(t)});root.append(svg('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));root.append(svg('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}))}
function latency(){const root=document.getElementById('latency'),W=760,H=390,m={l:67,r:18,t:18,b:42},lo=10,hi=20000,y=v=>m.t+(Math.log10(hi)-Math.log10(v))/(Math.log10(hi)-Math.log10(lo))*(H-m.t-m.b),x=i=>m.l+i*(W-m.l-m.r)/(data.length-1);axes(root,W,H,m,[10,30,100,300,1000,3000,10000],y);for(const [prefix,color] of [['prover','#50d5ff'],['verify','#ffbe55']]){const band=[...data.map((d,i)=>[x(i),y(d[prefix+'_p90_ms'])]),...data.slice().reverse().map((d,j)=>[x(data.length-1-j),y(d[prefix+'_p10_ms'])])];root.append(svg('polygon',{points:band.map(p=>p.join(',')).join(' '),fill:color,opacity:.12}));root.append(svg('polyline',{points:data.map((d,i)=>x(i)+','+y(d[prefix+'_median_ms'])).join(' '),fill:'none',stroke:color,'stroke-width':2.4}));data.forEach((d,i)=>{const c=svg('circle',{cx:x(i),cy:y(d[prefix+'_median_ms']),r:4.5,fill:color,stroke:'#081018','stroke-width':2});c.onmousemove=e=>show(e,`2^${d.exponent} ${prefix}<br><b>${fmtMs(d[prefix+'_median_ms'])}</b><br>P10–P90 ${fmtMs(d[prefix+'_p10_ms'])}–${fmtMs(d[prefix+'_p90_ms'])}<br>n=${d.n}`);c.onmouseleave=hide;root.append(c)})}}
function throughput(){const root=document.getElementById('throughput'),W=480,H=390,m={l:63,r:16,t:18,b:42},hi=Math.ceil(Math.max(...data.map(d=>d.throughput))/1000)*1000,y=v=>m.t+(hi-v)/hi*(H-m.t-m.b),x=i=>m.l+i*(W-m.l-m.r)/(data.length-1);for(let v=0;v<=hi;v+=2000){const yy=y(v);root.append(svg('line',{x1:m.l,y1:yy,x2:W-m.r,y2:yy,class:'gridline'}));const t=svg('text',{x:m.l-8,y:yy+4,'text-anchor':'end',class:'tick'});t.textContent=(v/1000).toFixed(0)+'k';root.append(t)}data.forEach((d,i)=>{const t=svg('text',{x:x(i),y:H-m.b+20,'text-anchor':'middle',class:'tick'});t.textContent='2^'+d.exponent;root.append(t)});const boundary=(x(6)+x(7))/2;root.append(svg('line',{x1:boundary,y1:m.t,x2:boundary,y2:H-m.b,stroke:'#ff768c','stroke-dasharray':'5 5'}));const bt=svg('text',{x:boundary+5,y:m.t+12,class:'tick',fill:'#ff768c'});bt.textContent='2 chunks';root.append(bt);root.append(svg('polyline',{points:data.map((d,i)=>x(i)+','+y(d.throughput)).join(' '),fill:'none',stroke:'#62dfa0','stroke-width':2.6}));data.forEach((d,i)=>{const c=svg('circle',{cx:x(i),cy:y(d.throughput),r:4.8,fill:'#62dfa0',stroke:'#081018','stroke-width':2});c.onmousemove=e=>show(e,`2^${d.exponent}<br><b>${Math.round(d.throughput).toLocaleString()} compressions/s</b><br>${fmtMs(d.prover_median_ms)} prover median`);c.onmouseleave=hide;root.append(c)});root.append(svg('line',{x1:m.l,y1:H-m.b,x2:W-m.r,y2:H-m.b,class:'axis'}));root.append(svg('line',{x1:m.l,y1:m.t,x2:m.l,y2:H-m.b,class:'axis'}))}
function show(e,text){tip.innerHTML=text;tip.style.display='block';tip.style.left=Math.min(innerWidth-220,e.clientX+13)+'px';tip.style.top=Math.max(8,e.clientY-40)+'px'}function hide(){tip.style.display='none'}
document.getElementById('peakThroughput').textContent=Math.round(Math.max(...data.map(d=>d.throughput))).toLocaleString()+' /s';document.getElementById('rows').innerHTML=data.map(d=>`<tr><td>2<sup>${d.exponent}</sup> (${d.compressions.toLocaleString()})</td><td>${d.prime_bits} bit</td><td>${d.outer_rounds}</td><td>${d.live_constraints.toLocaleString()}</td><td>${d.padded_rows.toLocaleString()}</td><td>${fmtMs(d.prover_median_ms)}</td><td>${fmtMs(d.prover_p10_ms)}–${fmtMs(d.prover_p90_ms)}</td><td>${fmtMs(d.verify_median_ms)}</td><td>${Math.round(d.throughput).toLocaleString()}/s</td></tr>`).join('');latency();throughput();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    rows = build_rows(summary)
    if [row["exponent"] for row in rows] != list(range(7, 17)):
        raise SystemExit("expected exactly the full SHA exponent range 7..16")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "range-metrics.csv")
    write_constraint_csv(args.out_dir / "constraint-profile.csv")
    render_html(rows, summary, args.out_dir / "range-overview.html")


if __name__ == "__main__":
    main()
