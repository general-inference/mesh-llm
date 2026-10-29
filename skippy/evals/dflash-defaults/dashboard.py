"""Renders one commit's DFlash results as a self-contained HTML dashboard.

The page is static SVG and HTML with no scripts or network fetches, because MLflow shows
HTML artifacts in a sandboxed frame. Hovering a mark shows its exact value through SVG
<title>; the tables at the bottom carry every number without hovering.

Forms follow the data's job: a dumbbell per model and machine for off -> shipped
default (before -> after per item); diverging heatmaps for gain, which has a natural
midpoint at 1.0x; small multiples for the trend across commits, one line per machine
in a fixed categorical order.
"""

import html
import math

# Reference palette (dataviz skill), light and dark steps of the same roles.
CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--ring:#fcfcfb;
--n3:#c43a3a;--n2:#ec8483;--n1:#f6c3c2;--z:#f0efec;--p1:#b7d3f6;--p2:#6da7ec;--p3:#2a78d6;--p4:#184f95;--p5:#0d366b;
--t-n3:#fff;--t-n2:#0b0b0b;--t-n1:#0b0b0b;--t-z:#0b0b0b;--t-p1:#0b0b0b;--t-p2:#0b0b0b;--t-p3:#fff;--t-p4:#fff;--t-p5:#fff}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--ring:#1a1a19;
--n3:#e66767;--n2:#a33836;--n1:#5c2726;--z:#383835;--p1:#1c3d66;--p2:#1c5cab;--p3:#2a78d6;--p4:#5598e7;--p5:#9ec5f4;
--t-n3:#0b0b0b;--t-n2:#fff;--t-n1:#fff;--t-z:#f0efec;--t-p1:#fff;--t-p2:#fff;--t-p3:#fff;--t-p4:#0b0b0b;--t-p5:#0b0b0b}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
.viz-root{max-width:1080px;margin:0 auto;padding:24px}
h1{font-size:20px;font-weight:600;margin:0 0 4px}
h2{font-size:15px;font-weight:600;margin:0 0 2px}
.sub{color:var(--ink2);margin:0 0 16px}
.note{color:var(--ink2);font-size:12px;margin:2px 0 12px}
.tiles{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 20px}
.tile{background:var(--surface);border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:12px 16px;min-width:200px;flex:1}
.tile .label{color:var(--ink2);font-size:12px}
.tile .value{font-size:28px;font-weight:600;margin:2px 0}
.tile .delta{color:var(--ink2);font-size:12px}
.card{background:var(--surface);border:1px solid rgba(11,11,11,.10);border-radius:8px;padding:16px 16px 12px;margin:0 0 16px}
.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin:4px 0 8px;align-items:center}
.key{display:inline-block;vertical-align:middle;margin-right:6px}
svg{display:block;width:100%;height:auto}
svg text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.multiples{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--ink2);font-weight:600}
th.num{text-align:right}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.all td{font-weight:600}
"""

GROUPS = [("prose", "Prose"), ("code", "Code"), ("math", "Math"), ("all", "All")]


def esc(value):
    return html.escape(str(value))


def num(value, digits=1):
    return "—" if value is None else f"{value:.{digits}f}"


def ratio(after, before):
    return after / before if after and before else None


def heat_class(gain):
    """Diverging bucket on log2(gain): gray around 1.0x, blue faster, red slower."""
    if gain is None:
        return None
    x = math.log2(gain)
    if abs(x) < 0.05:
        return "z"
    if x < 0:
        return "n1" if x > -0.2 else "n2" if x > -0.5 else "n3"
    return "p1" if x < 0.5 else "p2" if x < 1 else "p3" if x < 1.5 else "p4" if x < 2 else "p5"


def heat_text(bucket):
    # Each step carries its own label ink per mode: the ramp runs dark-to-light on the
    # dark surface, so the cell that needs white text in one mode needs black in the other.
    return f"var(--t-{bucket})"


def dumbbell(rows):
    rows = [r for r in rows if r["off"].get("all") and r["default"].get("all")]
    if not rows:
        return ""
    left, right, row_h, top = 230, 150, 34, 8
    width = 1000
    plot = width - left - right
    top_value = max(r["default"]["all"] for r in rows) * 1.05
    scale = lambda v: left + plot * v / top_value
    height = top + row_h * len(rows) + 28
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Decode speed, speculation off and shipped default">']
    for tick in nice_ticks(top_value):
        x = scale(tick)
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - 24}" stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 8}" font-size="11" fill="var(--muted)" text-anchor="middle">{tick:g}</text>')
    for i, r in enumerate(rows):
        y = top + row_h * i + row_h / 2
        off, on = r["off"]["all"], r["default"]["all"]
        gain = ratio(on, off)
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" font-size="13" fill="var(--ink)" text-anchor="end">'
                     f'{esc(r["model"])} <tspan fill="var(--ink2)">· {esc(r["machine"])}</tspan></text>')
        parts.append(f'<line x1="{scale(off):.1f}" x2="{scale(on):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--axis)" stroke-width="2" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{scale(off):.1f}" cy="{y:.1f}" r="5" fill="var(--muted)" stroke="var(--ring)" stroke-width="2">'
                     f'<title>{esc(r["model"])} · {esc(r["machine"])}: speculation off {off:.1f} tok/s</title></circle>')
        parts.append(f'<circle cx="{scale(on):.1f}" cy="{y:.1f}" r="5" fill="var(--s1)" stroke="var(--ring)" stroke-width="2">'
                     f'<title>{esc(r["model"])} · {esc(r["machine"])}: shipped default {on:.1f} tok/s ({gain:.2f}x)</title></circle>')
        tail = "off by default" if not r["default_on"] else f"{gain:.2f}×"
        parts.append(f'<text x="{max(scale(on), scale(off)) + 12:.1f}" y="{y + 4:.1f}" font-size="13" fill="var(--ink)">'
                     f'<tspan font-weight="600">{on:.1f}</tspan><tspan fill="var(--ink2)"> tok/s · {tail}</tspan></text>')
    parts.append("</svg>")
    legend = ('<div class="legend"><span><svg class="key" width="12" height="12" viewBox="0 0 12 12" style="display:inline;width:12px">'
              '<circle cx="6" cy="6" r="5" fill="var(--muted)"/></svg>Speculation off</span>'
              '<span><svg class="key" width="12" height="12" viewBox="0 0 12 12" style="display:inline;width:12px">'
              '<circle cx="6" cy="6" r="5" fill="var(--s1)"/></svg>Shipped default</span></div>')
    return legend + "".join(parts)


def heatmap(rows, columns, cell):
    """rows: list of row dicts; columns: [(key, label)]; cell(row, key) -> (gain, detail) or None."""
    if not rows:
        return ""
    left, cw, ch, gap, top = 230, 96, 30, 2, 22
    width = left + len(columns) * (cw + gap)
    height = top + len(rows) * (ch + gap) + 4
    parts = [f'<svg viewBox="0 0 {width} {height}" style="max-width:{width}px" role="img" aria-label="Gain heatmap">']
    for j, (_, label) in enumerate(columns):
        parts.append(f'<text x="{left + j * (cw + gap) + cw / 2:.1f}" y="14" font-size="12" fill="var(--ink2)" text-anchor="middle">{esc(label)}</text>')
    for i, r in enumerate(rows):
        y = top + i * (ch + gap)
        parts.append(f'<text x="{left - 12}" y="{y + ch / 2 + 4:.1f}" font-size="13" fill="var(--ink)" text-anchor="end">'
                     f'{esc(r["model"])} <tspan fill="var(--ink2)">· {esc(r["machine"])}</tspan></text>')
        for j, (key, label) in enumerate(columns):
            x = left + j * (cw + gap)
            found = cell(r, key)
            if not found or found[0] is None:
                parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" fill="none" stroke="var(--grid)"/>'
                             f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" font-size="12" fill="var(--muted)" text-anchor="middle">—</text>')
                continue
            gain, detail = found
            bucket = heat_class(gain)
            parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" style="fill:var(--{bucket})">'
                         f'<title>{esc(r["model"])} · {esc(r["machine"])} · {esc(label)}: {gain:.2f}x ({esc(detail)})</title></rect>')
            parts.append(f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" font-size="12" font-weight="600" '
                         f'fill="{heat_text(bucket)}" text-anchor="middle" pointer-events="none">{gain:.2f}×</text>')
    parts.append("</svg>")
    scale = "".join(f'<span><svg class="key" width="14" height="12" viewBox="0 0 14 12" style="display:inline;width:14px">'
                    f'<rect width="14" height="12" rx="3" style="fill:var(--{b})"/></svg>{esc(t)}</span>'
                    for b, t in [("n3", "< 0.71×"), ("n1", "0.87–0.97×"), ("z", "≈ 1×"), ("p1", "1.04–1.41×"),
                                 ("p2", "1.41–2×"), ("p3", "2–2.83×"), ("p4", "2.83–4×"), ("p5", "≥ 4×")])
    return f'<div class="legend">{scale}</div>' + "".join(parts)


def nice_ticks(top):
    step = 10 ** math.floor(math.log10(top / 4)) if top > 0 else 1
    for m in (1, 2, 5, 10):
        if top / (step * m) <= 6:
            step *= m
            break
    return [step * k for k in range(0, int(top / step) + 1)]


def trend(history, models, machines):
    """One small chart per model: shipped-default decode tok/s per commit, one line per machine."""
    colors = {machine: f"var(--s{i + 1})" for i, machine in enumerate(machines[:3])}
    charts = []
    for model in models:
        series = {m: [(i, h["values"].get(f"{model}|{m}")) for i, h in enumerate(history)] for m in machines}
        series = {m: [(i, v) for i, v in pts if v] for m, pts in series.items()}
        series = {m: pts for m, pts in series.items() if pts}
        if not series:
            continue
        width, height, left, right, top, bottom = 340, 190, 40, 96, 12, 36
        plot_w, plot_h = width - left - right, height - top - bottom
        top_value = max(v for pts in series.values() for _, v in pts) * 1.15
        n = max(len(history) - 1, 1)
        x = lambda i: left + (plot_w * i / n if len(history) > 1 else plot_w / 2)
        y = lambda v: top + plot_h * (1 - v / top_value)
        parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(model)} trend">']
        for tick in nice_ticks(top_value):
            parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" stroke="var(--grid)"/>'
                         f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" font-size="11" fill="var(--muted)" text-anchor="end">{tick:g}</text>')
        for i, h in enumerate(history):
            parts.append(f'<text x="{x(i):.1f}" y="{height - bottom + 16}" font-size="11" fill="var(--muted)" '
                         f'text-anchor="middle">{esc(h["short"][:7])}</text>')
        for machine, pts in series.items():
            color = colors.get(machine, "var(--muted)")
            if len(pts) > 1:
                d = " ".join(f"{'M' if k == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for k, (i, v) in enumerate(pts))
                parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
            for i, v in pts:
                parts.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="4" fill="{color}" stroke="var(--ring)" stroke-width="2">'
                             f'<title>{esc(machine)} @ {esc(history[i]["short"])}: {v:.1f} tok/s</title></circle>')
            i, v = pts[-1]
            parts.append(f'<text x="{x(i) + 9:.1f}" y="{y(v) + 4:.1f}" font-size="11" fill="var(--ink2)">{esc(machine)} '
                         f'<tspan fill="var(--ink)" font-weight="600">{v:.0f}</tspan></text>')
        parts.append("</svg>")
        charts.append(f'<div><h2 style="font-size:13px">{esc(model)}</h2>{"".join(parts)}</div>')
    legend = "".join(f'<span><svg class="key" width="16" height="12" viewBox="0 0 16 12" style="display:inline;width:16px">'
                     f'<line x1="1" x2="15" y1="6" y2="6" stroke="{c}" stroke-width="2" stroke-linecap="round"/></svg>{esc(m)}</span>'
                     for m, c in colors.items())
    return f'<div class="legend">{legend}</div><div class="multiples">{"".join(charts)}</div>'


def results_table(rows):
    head = ('<tr><th>Machine</th><th>Model</th><th>Workload</th><th class="num">Before</th><th class="num">+DFlash</th>'
            '<th class="num">All together</th><th class="num">Gain</th></tr>')
    body = []
    for r in rows:
        for key, label in GROUPS:
            before, after = r["off"].get(key), r["default"].get(key)
            fixed = (r["fixed"] or {}).get(key)
            note = "" if r["default_on"] or key != "all" else " (off by default)"
            gain = ratio(after, before)
            body.append(f'<tr class="{"all" if key == "all" else ""}"><td>{esc(r["machine"])}</td><td>{esc(r["model"])}</td>'
                        f'<td>{label}</td><td class="num">{num(before)}</td><td class="num">{num(fixed)}</td>'
                        f'<td class="num">{num(after)}{note}</td><td class="num">{num(gain, 2) + "×" if gain else "—"}</td></tr>')
        before, after = r["off_t07"].get("all"), r["default_t07"].get("all")
        gain = ratio(after, before)
        body.append(f'<tr class="all"><td>{esc(r["machine"])}, sampled</td><td>{esc(r["model"])}</td><td>All</td>'
                    f'<td class="num">{num(before)}</td><td class="num">—</td><td class="num">{num(after)}</td>'
                    f'<td class="num">{num(gain, 2) + "×" if gain else "—"}</td></tr>')
        if r["agent_off"]:
            before, after = r["agent_off"].get("all"), r["agent_default"].get("all")
            gain = ratio(after, before)
            body.append(f'<tr><td>{esc(r["machine"])}</td><td>{esc(r["model"])}</td><td>Agentic coding</td>'
                        f'<td class="num">{num(before)}</td><td class="num">—</td><td class="num">{num(after)}</td>'
                        f'<td class="num">{num(gain, 2) + "×" if gain else "—"}</td></tr>')
    return f"<table>{head}{''.join(body)}</table>"


def agent_table(rows):
    body = []
    for r in rows:
        bands = [b for b in r["agent_off"] if b != "all"] + (["all"] if "all" in r["agent_off"] else [])
        for band in bands:
            before, after = r["agent_off"].get(band), r["agent_default"].get(band)
            gain = ratio(after, before)
            body.append(f'<tr class="{"all" if band == "all" else ""}"><td>{esc(r["machine"])}</td><td>{esc(r["model"])}</td>'
                        f'<td>{"All" if band == "all" else esc(band)}</td><td class="num">{num(before)}</td>'
                        f'<td class="num">{num(after)}</td><td class="num">{num(gain, 2) + "×" if gain else "—"}</td></tr>')
    if not body:
        return ""
    return ('<table><tr><th>Machine</th><th>Model</th><th>History</th><th class="num">Before</th>'
            f'<th class="num">All together</th><th class="num">Gain</th></tr>{"".join(body)}</table>')


def tiles(rows, models):
    out = []
    for model in models:
        mine = [r for r in rows if r["model"] == model and r["default_on"]]
        off_by_default = [r["machine"] for r in rows if r["model"] == model and not r["default_on"]]
        gains = [(r["machine"], ratio(r["default"].get("all"), r["off"].get("all"))) for r in mine]
        gains = [(m, g) for m, g in gains if g]
        if not gains:
            continue
        best = max(gains, key=lambda mg: mg[1])
        others = [f"{esc(m)} {g:.2f}×" for m, g in gains if m != best[0]]
        others += [f"{esc(m)}: off by default" for m in off_by_default]
        others = " · ".join(others)
        out.append(f'<div class="tile"><div class="label">{esc(model)} · {esc(best[0])}</div>'
                   f'<div class="value">{best[1]:.2f}×</div><div class="delta">{others or "&nbsp;"}</div></div>')
    return f'<div class="tiles">{"".join(out)}</div>'


def render(data):
    rows, models, machines = data["rows"], data["models"], data["machines"]
    workload_cols = [("prose", "Prose"), ("code", "Code"), ("math", "Math"), ("all", "All"), ("sampled", "Sampled"),
                     ("agent", "Agentic coding")]

    def workload_cell(r, key):
        if key == "sampled":
            before, after = r["off_t07"].get("all"), r["default_t07"].get("all")
        elif key == "agent":
            before, after = r["agent_off"].get("all"), r["agent_default"].get("all")
        else:
            before, after = r["off"].get(key), r["default"].get(key)
        gain = ratio(after, before)
        return (gain, f"{num(before)} -> {num(after)} tok/s") if gain else None

    bands = []
    for r in rows:
        bands += [b for b in r["agent_off"] if b != "all" and b not in bands]
    agent_rows = [r for r in rows if r["agent_off"]]

    def agent_cell(r, key):
        before, after = r["agent_off"].get(key), r["agent_default"].get(key)
        gain = ratio(after, before)
        return (gain, f"{num(before)} -> {num(after)} tok/s") if gain else None

    sections = [
        f'<h1>DFlash speculative decoding · mesh-llm {esc(data["short"])}</h1>',
        f'<p class="sub">{esc(data["label"])} · {esc(data["subject"])} · committed {esc(data["time"][:10])}. '
        "Decode tok/s on held-out prompts, one request at a time; gain = shipped default ÷ speculation off, "
        "both measured in the same job on the same machine.</p>",
        tiles(rows, models),
        '<div class="card"><h2>Speculation off → shipped default</h2>'
        '<p class="note">All prompts, greedy. The gap is the speedup the default ships.</p>' + dumbbell(rows) + "</div>",
        '<div class="card"><h2>Gain by workload</h2>'
        '<p class="note">Blue is faster than speculation off, red slower, gray about the same. Sampled is T=0.7. Agentic coding is real OpenHands histories (4–48K tokens) at 64K context, separate from All.</p>'
        + heatmap(rows, workload_cols, workload_cell) + "</div>",
    ]
    if agent_rows:
        sections.append('<div class="card"><h2>Agentic coding, by history length</h2>'
                        '<p class="note">OpenHands histories from SWE-rebench, 64K context, greedy.</p>'
                        + heatmap(agent_rows, [(b, b) for b in bands] + [("all", "All")], agent_cell) + "</div>")
    if data.get("history"):
        sections.append('<div class="card"><h2>Shipped default across recorded commits</h2>'
                        '<p class="note">All prompts, greedy, decode tok/s. One point per recorded commit, oldest left.</p>'
                        + trend(data["history"], models, machines) + "</div>")
    sections.append('<div class="card"><h2>Results table</h2><p class="note">Before = speculation off. +DFlash = best '
                    'fixed draft length. All together = shipped default with no speculation config.</p>'
                    + results_table(rows) + "</div>")
    if agent_rows:
        sections.append('<div class="card"><h2>Agent transcripts table</h2>' + agent_table(agent_rows) + "</div>")
    return ("<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\">"
            f"<title>DFlash · mesh-llm {esc(data['short'])}</title><style>{CSS}</style></head>"
            f'<body><main class="viz-root">{"".join(sections)}</main></body></html>')
