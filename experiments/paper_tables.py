#!/usr/bin/env python3
"""
Paper-ready LaTeX table bodies for the TCPS revision (results/tables/paper/*.tex).

Rules (decided with the author): "EXACT" is the journal model (caex). The conference
model (ex) appears in exactly one row of the component-ablation table and nowhere else.
Standard rows: mean +- std over seeds 42-46, best checkpoint (lowest meta-test loss).
LOOCV rows: seed 42, mean +- std over held-out subject folds.
Dice / IoU: macro average over classes; ROC-AUC: binary (2 classes) or one-vs-rest (3).
"""
import glob, json, os, re, statistics as st, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/tables/paper"); os.makedirs(OUT, exist_ok=True)
MODELS = ["segmenter", "cnn", "transformer", "unet", "caex"]
NAME = {"segmenter": "Segmenter", "cnn": "CNN", "transformer": "Transformer", "unet": "U-Net", "caex": "EXACT"}
DS = {"physiq": "PhysiQ", "spar": "SPAR", "mmfit": "MM-Fit"}
M = ["test/Dice Score", "test/IoU", "test/ROC-AUC"]

runs = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results/runs/*.json"))}
std, loocv, tagged = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
for n, d in runs.items():
    if "-subject-" in n:
        loocv[(d["model"], d["dataset"])].append(d); continue
    (tagged[(d["model"], d["dataset"], d["tag"])] if d.get("tag") else std[(d["model"], d["dataset"])]).append(d)

def ms(v, key, w="best"):
    xs = [r[w][key] for r in v]; return st.mean(xs), (st.pstdev(xs) if len(xs) > 1 else 0.0), len(xs)
def cell(m, s): return f"\\revise{{{m:.3f} $\\pm$ {s:.3f}}}"  # regenerated numbers are this-iteration changes
def decorate(rows):
    """rows: list of (label, [(mean,std),...]) -> bold best / underline second per column"""
    out = []
    ncol = len(rows[0][1])
    best, second = [], []
    for c in range(ncol):
        vals = sorted({r[1][c][0] for r in rows}, reverse=True)
        best.append(vals[0]); second.append(vals[1] if len(vals) > 1 else None)
    for lab, cells in rows:
        tex = []
        for c, (m, s) in enumerate(cells):
            t = cell(m, s)
            if m == best[c]: t = f"\\textbf{{{t}}}"
            elif m == second[c]: t = f"\\underline{{{t}}}"
            tex.append(t)
        out.append((lab, tex))
    return out
MACRO = {"table_standard.tex": "TableStandardRows", "table_loocv.tex": "TableLoocvRows",
         "table_rest_intervals.tex": "TableRestIntervalsRows", "table_ood.tex": "TableOodRows",
         "table_ablation.tex": "TableAblationRows", "table_sensitivity.tex": "TableSensitivityRows",
         "table_realtime.tex": "TableRealtimeRows", "table_edge.tex": "TableEdgeRows"}
def write(name, text):
    # Each file defines a macro holding the rows. \input inside a tabular breaks on current LaTeX
    # (end-of-file hooks run between the last row and \bottomrule), so the manuscript \inputs the
    # file outside the table and uses the macro inside it.
    body = "\n".join(l for l in text.rstrip("\n").splitlines() if not l.startswith("%"))
    text = f"\\newcommand{{\\{MACRO[name]}}}{{%\n{body}\n}}\n"
    open(os.path.join(OUT, name), "w").write(text); print(f"--- {name}\n{text}")

# 1. standard and LOOCV, three datasets
for src, fname in ((std, "table_standard.tex"), (loocv, "table_loocv.tex")):
    blocks = []
    for ds in DS:
        rows = [(NAME[m], [ms(src[(m, ds)], k)[:2] for k in M]) for m in MODELS if (m, ds) in src]
        n = ms(src[(MODELS[0], ds)], M[0])[2]
        dec = decorate(rows)
        lines = [f"\\multirow{{{len(dec)}}}{{*}}{{{DS[ds]}}}"]
        for i, (lab, tex) in enumerate(dec):
            lab = f"\\revise{{{lab}}}" if lab == "EXACT" else lab
            lines.append(f"                 & {lab:12s} & " + " & ".join(tex) + " \\\\" + (" \\midrule" if i == len(dec) - 1 and ds != "mmfit" else " \\bottomrule" if i == len(dec) - 1 else ""))
        blocks.append("\n".join(lines) + f"  % n={n} per row")
    write(fname, "\n".join(blocks) + "\n")

# 2. three-class rest-interval table (PhysiQ)
rows = [(NAME[m], [ms(tagged[(m, "physiq", "c3")], k)[:2] for k in M]) for m in MODELS]
write("table_rest_intervals.tex", "\n".join(f"{lab:12s} & " + " & ".join(tex) + " \\\\" for lab, tex in decorate(rows)) + "\n")

# 3. OOD: few-shot first (deployment protocol), zero-shot second
ood = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results/ood/*.json"))}
def o(n): d = ood[n]; return [(d["dice_mean"], d["dice_std"]), (d["iou_mean"], d["iou_std"]), (d["rocauc_mean"], d["rocauc_std"])]
rows = [(NAME[m], o("caex_both" if m == "caex" else f"{m}_inner") + o(f"{m}_zeroshot")) for m in MODELS]
dec = decorate(rows)
lines = [f"{lab:12s} & " + " & ".join(tex) + " \\\\" for lab, tex in dec]
write("table_ood.tex", "% Method & few-shot Dice & IoU & ROC-AUC & zero-shot Dice & IoU & ROC-AUC\n" + "\n".join(lines) + "\n")

# 4. component ablation (PhysiQ in-distribution + PhyTMo few-shot); one conference row
ABL = [("full", "EXACT (all components)"), ("no_rotaug", "$-$ orientation aug."), ("no_norm", "$-$ IMU norm."),
       ("no_adabn", "$-$ AdaBN"), ("no_gate", "$-$ SE-gate"), ("no_groupgate", "$-$ group gate"),
       ("norm_only", "backbone + IMU norm."), ("norm_adabn", "\\quad + AdaBN"), ("norm_adabn_se", "\\quad + AdaBN + SE-gate")]
def abl_row(tag):
    ind = std[("caex", "physiq")] if tag == "full" else tagged[("caex", "physiq", tag)]
    few = o("caex_both" if tag == "full" else f"caex-{tag}_both")
    return [ms(ind, M[0])[:2], ms(ind, M[1])[:2]] + few
def arow(lab, cells): return f"{lab} & " + " & ".join(cell(*c) for c in cells) + " \\\\"
lines = [arow(lab, abl_row(tag)) for tag, lab in ABL[:6]] + ["\\midrule"] + \
        [arow(lab, abl_row(tag)) for tag, lab in ABL[6:]] + ["\\midrule"] + \
        [arow("backbone only~\\cite{wang2025exact}", [ms(std[("ex", "physiq")], M[0])[:2], ms(std[("ex", "physiq")], M[1])[:2]] + o("ex_inner"))]
write("table_ablation.tex", "\n".join(lines) + "\n")

# 5. sensitivity: the default run once, then only the varied settings
def srow(lab, v): return f"{lab} & " + " & ".join(cell(*ms(v, k)[:2]) for k in M) + " \\\\"
base = std[("caex", "physiq")]; T = tagged
lines = [srow("Default (5-shot, 1 inner step, 10\\,s window)", base), "\\midrule",
         srow("Shots: 1", T[("caex","physiq","shot1")]), srow("\\phantom{Shots: }3", T[("caex","physiq","shot3")]), srow("\\phantom{Shots: }10", T[("caex","physiq","shot10")]), "\\midrule",
         srow("Inner steps: 3", T[("caex","physiq","inner3")]), srow("\\phantom{Inner steps: }5", T[("caex","physiq","inner5")]), "\\midrule",
         srow("Window: 5\\,s (250 samples)", T[("caex","physiq","win250")]), srow("\\phantom{Window: }15\\,s (750 samples)", T[("caex","physiq","win750")])]
write("table_sensitivity.tex", "\n".join(lines) + "\n")

# 6. real-time (from aggregate's CSV-free path: read JSONs directly)
rt = collections.defaultdict(list)
for f in glob.glob(os.path.join(ROOT, "results/realtime_eval/*_seed*.json")):
    j = json.load(open(f)); rt[j["model_name"]].append(j)
lines = []
for m in (MODELS if rt else []):   # results/realtime_eval/*.json are not in the repo; without them keep the committed table
    v = rt[m]
    size = st.mean(x["memory"]["model_size_mb"] for x in v); params = v[0]["compute"]["params"]; macs = v[0]["compute"]["macs"]
    lat = [x["latency"]["batch_1"]["total_ms"]["mean"] for x in v]; fwd = [x["latency"]["batch_1"]["inference_ms"]["mean"] for x in v]
    adapt = [x["adaptation"]["maml_first_order"]["total_latency_ms"]["mean"] for x in v]
    lines.append(f"{NAME[m]:12s} & \\revise{{{size:.2f}}} & \\revise{{{params/1e3:.1f}}} & \\revise{{{macs/1e6:.1f}}} & \\revise{{{st.mean(lat):.2f} $\\pm$ {st.pstdev(lat):.2f}}} & \\revise{{{st.mean(fwd):.2f} $\\pm$ {st.pstdev(fwd):.2f}}} & \\revise{{{st.mean(adapt):.1f}}} \\\\")
if not rt: print("--- table_realtime.tex: no results/realtime_eval/*.json here, keeping the committed file")
if rt: write("table_realtime.tex", "% Model & Size (MB) & Params (K) & MACs (M) & Latency ms end-to-end & forward only & 5-step adaptation ms\n" + "\n".join(lines) + "\n")

# 7. on-device latency (Core ML on iPhone / Apple Watch), from results/edge/*.txt written by the EXACTBench app.
#    Only blocks of the multi-model app ("on-device benchmark  model: ...") are used; each file may hold several
#    stamped runs; cells are mean +- std over runs of the per-run MEDIAN latency (300 timed windows per run):
#    watchOS pauses the foreground app when the display dims, which stalls 1-3 windows per run for seconds and
#    makes per-run means meaningless on the watch (p50 is stable; p95 is also reported in the text).
EDGE_DEV = [("iPhone18,2", "iPhone 17 Pro Max"), ("Watch7,5", "Apple Watch Ultra 2")]
EDGE_UNITS = [("all (Neural Engine allowed)", "NE"), ("cpuOnly", "CPU")]
EDGE_MODEL = {"segmenter_physiq": "segmenter", "cnn_physiq": "cnn", "transformer_physiq": "transformer",
              "unet_physiq": "unet", "EXACT_physiq": "caex"}
EDGE_VARIANT = [("EXACT_physiq_win250", "EXACT, 5\\,s window (250)"), ("EXACT_physiq_win750", "EXACT, 15\\,s window (750)")]
edge = collections.defaultdict(list)       # (device, model, units) -> [per-run median ms]
edge_p95 = collections.defaultdict(list)   # (device, model, units) -> [per-run p95 ms]
sustained = []                             # raw sustained blocks for the text
for f in sorted(glob.glob(os.path.join(ROOT, "results/edge/*.txt"))):
    txt = open(f).read()
    for m in re.finditer(r"on-device benchmark  model: (\S+)\n\s*device: (\S+) .*?power: ([^\n]*)\n\s*compute units: (.+?)  window.*?\n\s*latency ms\s+mean ([\d.]+)\s+p50 ([\d.]+)\s+p95 ([\d.]+)", txt):
        model, dev, power, units, mean, p50, p95 = m.groups()
        assert power.startswith("unplugged"), f"{f}: run on charger ({power}), exclude it"
        edge[(dev, model, units.strip())].append(float(p50)); edge_p95[(dev, model, units.strip())].append(float(p95))
    sustained += [b for b in re.findall(r"sustained streaming benchmark.*?overall ms[^\n]*", txt, re.S)]
def edge_cell(dev, model, units):
    v = edge.get((dev, model, units))
    if not v: return "--"
    return f"\\revise{{{st.mean(v):.2f} $\\pm$ {st.pstdev(v):.2f}}}" if len(v) > 1 else f"\\revise{{{v[0]:.2f}}}"   # single run: no std
def pkg_mb(name):
    p = os.path.join(ROOT, "experiments/edge/models", name + ".mlpackage")
    return sum(os.path.getsize(os.path.join(d, x)) for d, _, fs in os.walk(p) for x in fs) / 1e6
if edge:
    lines = []
    def edge_row(label, model):
        cells = [edge_cell(dev, model, u) for dev, _ in EDGE_DEV for u, _ in EDGE_UNITS]
        return f"{label:34s} & \\revise{{{pkg_mb(model):.2f}}} & " + " & ".join(cells) + " \\\\"
    for m in MODELS:
        key = next(k for k, v in EDGE_MODEL.items() if v == m)
        lines.append(edge_row(NAME[m], key))
    lines.append("\\midrule")
    for key, label in EDGE_VARIANT:
        lines.append(edge_row(label, key))
    nruns = sorted({len(v) for v in edge.values()})
    write("table_edge.tex", "% Model & Core ML size (MB) & iPhone NE & iPhone CPU & Watch NE & Watch CPU   (ms per window, per-run median, mean +- std over "
          + "/".join(map(str, nruns)) + " runs)\n" + "\n".join(lines) + "\n")
    print("--- edge p95 (mean over runs of per-run p95, ms)")
    for (dev, model, units), v in sorted(edge_p95.items()): print(f"  {dev:11s} {model:22s} {units:28s} p95 {st.mean(v):.2f}   p50 {st.mean(edge[(dev, model, units)]):.2f}")
    for b in sustained: print("--- sustained\n" + b)
