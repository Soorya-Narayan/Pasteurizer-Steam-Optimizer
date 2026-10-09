"""Self-contained HTML report for a training run (figures referenced relatively)."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#0b0d13;--panel:#141925;--line:#262e40;--text:#e6e9f0;--muted:#8a93a8;--accent:#4cc9f0;--good:#06d6a0;--bad:#ef476f}
*{box-sizing:border-box}body{margin:0;font-family:Inter,system-ui,sans-serif;background:radial-gradient(1200px 600px at 10% -10%,#1b2340 0,var(--bg) 60%);color:var(--text)}
main{max-width:1280px;margin:0 auto;padding:48px 28px 80px}
h1{font-size:34px;margin:0 0 6px;letter-spacing:-.02em;background:linear-gradient(90deg,#4cc9f0,#f72585);-webkit-background-clip:text;color:transparent}
h2{font-size:20px;margin:44px 0 14px}.sub{color:var(--muted);margin:0 0 28px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;transition:transform .2s,border-color .2s}
.card:hover{transform:translateY(-3px);border-color:var(--accent)}.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.card .v{font-size:26px;font-weight:700;margin-top:6px}.good{color:var(--good)}.bad{color:var(--bad)}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;font-size:13px}
th,td{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600;background:#10141e}
td:first-child,th:first-child{text-align:left;font-weight:600}tr.best td{background:rgba(6,214,160,.08)}
figure{margin:0 0 22px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px}
figure img{width:100%;border-radius:8px;display:block}figcaption{color:var(--muted);font-size:13px;margin-top:10px}
code{background:#10141e;padding:2px 6px;border-radius:6px}
"""


def _fmt(v):
    if isinstance(v, float):
        return "–" if pd.isna(v) else f"{v:,.3f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, highlight: str | None = None) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    rows = "".join(
        f"<tr class='{'best' if idx == highlight else ''}'><td>{html.escape(str(idx))}</td>"
        + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>"
        for idx, row in zip(df.index, df.itertuples(index=False))
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"


def write_report(run_dir: Path, summary: dict, metrics: pd.DataFrame, divert: pd.DataFrame,
                 regime: pd.DataFrame, figures: list[tuple[str, str]], best: str) -> Path:
    b, p = metrics.loc[best], metrics.loc["Persistence"]
    skill = b["Skill_vs_Persistence_pct"]
    cards = [
        ("Best model", best, ""),
        (f"Test MAE ({summary['horizon_s']}s ahead)", f"{b['MAE_C']:.3f} °C", ""),
        ("Persistence MAE", f"{p['MAE_C']:.3f} °C", ""),
        ("Skill vs persistence", f"{skill:+.1f} %", "good" if skill > 0 else "bad"),
        ("R² on temperature change", f"{b['R2_delta']:.3f}", ""),
        ("Training samples", f"{summary['n_train']:,}", ""),
    ]
    cards_html = "".join(f"<div class='card'><div class='k'>{k}</div><div class='v {c}'>{html.escape(v)}</div></div>" for k, v, c in cards)
    figs_html = "".join(f"<figure><img src='{src}' alt='{html.escape(cap)}' loading='lazy'><figcaption>{html.escape(cap)}</figcaption></figure>"
                        for src, cap in figures)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HTST Thermal Model · Training Report {summary['run_id']}</title>
<meta name="description" content="Training report for the HTST pasteurizer holding-tube temperature forecasting model.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>
<header><h1>HTST Pasteurizer · Thermal Predictive Model</h1>
<p class="sub">Run <code>{summary['run_id']}</code> · forecasting <code>{summary['target']}</code> {summary['horizon_s']} s ahead ·
generated {datetime.now():%Y-%m-%d %H:%M}</p></header>
<section class="cards">{cards_html}</section>
<section><h2>Test-set accuracy</h2>{_table(metrics, best)}</section>
<section><h2>Divert early warning (forecast &lt; {summary['threshold']} °C + {summary['margin']} °C margin)</h2>{_table(divert, best)}</section>
<section><h2>MAE by time since production start (°C)</h2>{_table(regime)}</section>
<section><h2>Figures</h2>{figs_html}</section>
<section><h2>Data</h2>{_table(pd.DataFrame.from_dict(summary['data'], orient='index', columns=['value']).rename_axis('item'))}</section>
</main></body></html>"""
    out = run_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
