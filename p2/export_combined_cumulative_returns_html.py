from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_REPORTS_DIR = "/Users/raj/ws/asset-pricing/p2/backtests/2026-05-16_23-33-18/reports"
DEFAULT_OUTPUT = "combined_cumulative_returns.html"
TITLE = "Combined Cumulative Returns"
SUBTITLE = "Daily normalized strategy equity across models with one shared benchmark"
COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports_dir", nargs="?", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_monthly_returns(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_result_packet(packet_path: Path) -> dict:
  with packet_path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def find_result_packet(model_dir: Path) -> Path:
  packet_paths = sorted(
    path for path in model_dir.iterdir() if path.is_file() and path.suffix == ".json" and path.stem.isdigit()
  )
  if not packet_paths:
    raise ValueError(f"No result packet JSON found under {model_dir}")
  return max(packet_paths, key=lambda path: path.stat().st_size)


def normalize_dates(timestamps: Iterable[int]) -> list[str]:
  return [
    datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
    for timestamp in timestamps
  ]


def load_daily_series(model_dir: Path) -> tuple[list[str], list[float], list[float]]:
  packet = load_result_packet(find_result_packet(model_dir))

  strategy_equity_values = packet["charts"]["Strategy Equity"]["series"]["Equity"]["values"]
  benchmark_values = packet["charts"]["Benchmark"]["series"]["Benchmark"]["values"]
  if not strategy_equity_values or not benchmark_values:
    raise ValueError(f"Missing daily chart series in {model_dir}")

  initial_equity = float(strategy_equity_values[0][1])
  if initial_equity == 0:
    raise ValueError(f"Initial equity is zero in {model_dir}")

  strategy_dates = normalize_dates(int(point[0]) for point in strategy_equity_values)
  benchmark_dates = normalize_dates(int(point[0]) for point in benchmark_values)
  if strategy_dates != benchmark_dates:
    raise ValueError(f"Daily date index mismatch between equity and benchmark in {model_dir}")

  strategy_nav = [float(point[1]) / initial_equity for point in strategy_equity_values]
  benchmark_nav = [float(point[1]) for point in benchmark_values]
  return strategy_dates, strategy_nav, benchmark_nav


def collect_series(reports_dir: Path) -> tuple[list[str], dict[str, list[float]], list[float]]:
    report_model_dirs = sorted(
        path for path in reports_dir.iterdir() if path.is_dir() and (path / "monthly_returns.csv").exists()
    )
    if not report_model_dirs:
        raise ValueError(f"No model report directories found under {reports_dir}")

    backtest_dir = reports_dir.parent

    dates: list[str] = []
    strategies: dict[str, list[float]] = {}
    benchmark: list[float] = []

    for idx, report_model_dir in enumerate(report_model_dirs):
        model_name = report_model_dir.name
        backtest_model_dir = backtest_dir / model_name
        if not backtest_model_dir.is_dir():
            raise ValueError(f"Backtest directory not found for {model_name}: {backtest_model_dir}")

        model_dates, model_nav, benchmark_nav = load_daily_series(backtest_model_dir)

        if idx == 0:
            dates = model_dates
            benchmark = benchmark_nav
        else:
            if model_dates != dates:
                raise ValueError(f"Date index mismatch for {model_name}")
            if len(benchmark_nav) != len(benchmark):
                raise ValueError(f"Benchmark length mismatch for {model_name}")
        strategies[model_name] = model_nav

    return dates, strategies, benchmark


def build_html(dates: list[str], strategies: dict[str, list[float]], benchmark: list[float]) -> str:
    series_payload = []
    for idx, (model, values) in enumerate(strategies.items()):
        series_payload.append(
            {
                "name": model,
                "values": values,
                "color": COLORS[idx % len(COLORS)],
                "width": 2.5,
                "dash": "",
            }
        )
    series_payload.append(
        {
            "name": "Benchmark",
            "values": benchmark,
            "color": "#111111",
            "width": 2.5,
            "dash": "8 6",
        }
    )

    payload = {
        "dates": dates,
        "series": series_payload,
    }

    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{TITLE}</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #5b6777;
      --grid: #d8dee8;
      --accent: #1f4e78;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 32px;
      background: linear-gradient(180deg, #f8fbff 0%, #edf2f7 100%);
      color: var(--ink);
      font-family: Inter, Segoe UI, Arial, sans-serif;
    }}
    .wrap {{
      max-width: 1320px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid #dbe4ee;
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(16, 24, 40, 0.08);
      padding: 28px 28px 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
    }}
    .sub {{
      margin: 8px 0 20px;
      color: var(--muted);
      font-size: 15px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 16px;
      margin-bottom: 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--ink);
      background: #f8fafc;
      border: 1px solid #e7edf3;
      border-radius: 999px;
      padding: 6px 10px;
    }}
    .swatch {{
      width: 26px;
      height: 0;
      border-top-width: 3px;
      border-top-style: solid;
      display: inline-block;
    }}
    .chart-box {{
      border: 1px solid #e4ebf3;
      border-radius: 16px;
      overflow: hidden;
      background: white;
    }}
    svg {{ width: 100%; height: auto; display: block; }}
    .axis-label {{ font-size: 12px; fill: #52606d; }}
    .tick {{ font-size: 12px; fill: #52606d; }}
    .grid {{ stroke: var(--grid); stroke-width: 1; }}
    .axis {{ stroke: #7b8794; stroke-width: 1.2; }}
    .footer {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 14px;
      color: var(--muted);
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>{TITLE}</h1>
    <p class=\"sub\">{SUBTITLE}</p>
    <div class=\"legend\" id=\"legend\"></div>
    <div class=\"chart-box\">
      <svg id=\"chart\" viewBox=\"0 0 1280 720\" aria-label=\"Combined cumulative return chart\"></svg>
    </div>
    <div class=\"footer\">
      <span>Source: Lean result packet daily equity and benchmark series</span>
      <span id=\"dateRange\"></span>
    </div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const svg = document.getElementById('chart');
    const legend = document.getElementById('legend');
    const width = 1280;
    const height = 720;
    const margin = {{ top: 30, right: 24, bottom: 72, left: 76 }};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const allValues = payload.series.flatMap(s => s.values);
    let minY = Math.min(...allValues);
    let maxY = Math.max(...allValues);
    const pad = (maxY - minY) * 0.08 || 0.05;
    minY -= pad;
    maxY += pad;

    const x = idx => margin.left + (idx / (payload.dates.length - 1)) * innerWidth;
    const y = val => margin.top + (1 - ((val - minY) / (maxY - minY))) * innerHeight;

    function add(tag, attrs, parent = svg) {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs || {{}}).forEach(([key, value]) => el.setAttribute(key, value));
      parent.appendChild(el);
      return el;
    }}

    add('rect', {{ x: 0, y: 0, width, height, fill: '#ffffff' }});

    const gridCount = 6;
    for (let i = 0; i <= gridCount; i += 1) {{
      const value = minY + ((maxY - minY) * i / gridCount);
      const py = y(value);
      add('line', {{ x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: 'grid' }});
      const label = add('text', {{ x: margin.left - 12, y: py + 4, 'text-anchor': 'end', class: 'tick' }});
      label.textContent = value.toFixed(2) + 'x';
    }}

    const xTickIndexes = [];
    const xTickCount = Math.min(8, payload.dates.length);
    for (let i = 0; i < xTickCount; i += 1) {{
      xTickIndexes.push(Math.round(i * (payload.dates.length - 1) / Math.max(1, xTickCount - 1)));
    }}
    [...new Set(xTickIndexes)].forEach(idx => {{
      const px = x(idx);
      add('line', {{ x1: px, y1: margin.top, x2: px, y2: height - margin.bottom, class: 'grid' }});
      const label = add('text', {{ x: px, y: height - margin.bottom + 24, 'text-anchor': 'middle', class: 'tick' }});
      label.textContent = payload.dates[idx];
    }});

    add('line', {{ x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: 'axis' }});
    add('line', {{ x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: 'axis' }});

    const xAxisLabel = add('text', {{ x: margin.left + innerWidth / 2, y: height - 16, 'text-anchor': 'middle', class: 'axis-label' }});
    xAxisLabel.textContent = 'Date';

    const yAxisLabel = add('text', {{ x: 20, y: margin.top + innerHeight / 2, transform: `rotate(-90 20 ${{margin.top + innerHeight / 2}})`, 'text-anchor': 'middle', class: 'axis-label' }});
    yAxisLabel.textContent = 'Cumulative NAV';

    payload.series.forEach(series => {{
      const path = series.values.map((value, idx) => `${{idx === 0 ? 'M' : 'L'}} ${{x(idx).toFixed(2)}} ${{y(value).toFixed(2)}}`).join(' ');
      add('path', {{
        d: path,
        fill: 'none',
        stroke: series.color,
        'stroke-width': series.width,
        'stroke-dasharray': series.dash,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round'
      }});

      const item = document.createElement('div');
      item.className = 'legend-item';
      const swatch = document.createElement('span');
      swatch.className = 'swatch';
      swatch.style.borderTopColor = series.color;
      if (series.dash) swatch.style.borderTopStyle = 'dashed';
      const label = document.createElement('span');
      label.textContent = series.name;
      item.appendChild(swatch);
      item.appendChild(label);
      legend.appendChild(item);

      const lastIdx = series.values.length - 1;
      const endLabel = add('text', {{
        x: x(lastIdx) + 8,
        y: y(series.values[lastIdx]) + 4,
        class: 'tick',
        fill: series.color,
      }});
      endLabel.textContent = series.name;
    }});

    document.getElementById('dateRange').textContent = `${{payload.dates[0]}} to ${{payload.dates[payload.dates.length - 1]}}`;
  </script>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    output_path = reports_dir / args.output
    dates, strategies, benchmark = collect_series(reports_dir)
    output_path.write_text(build_html(dates, strategies, benchmark), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())