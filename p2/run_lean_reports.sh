#!/usr/bin/env bash
set -euo pipefail

# Run Lean backtests per model and generate Lean report per run.
# Usage:
#   ./run_lean_reports.sh [MAX_TICKERS] [MODEL_ID]
# Example:
#   ./run_lean_reports.sh 10
#   ./run_lean_reports.sh 5 OLS-3_L2

MAX_TICKERS="${1:-10}"
MODEL_ID_FILTER="${2:-}"
# Explicitly run Docker without CPU/memory caps for backtests.
LEAN_DOCKER_NO_LIMITS='{"nano_cpus": 0, "mem_limit": 0, "memswap_limit": 0}'

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
P2_DIR="$ROOT_DIR/p2"

TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_DIR="$P2_DIR/backtests/$TS"
REPORTS_REL="backtests/$TS/reports"

mkdir -p "$P2_DIR/ml_inputs/synthetic_data/csv_outputs"
mkdir -p "$RUN_DIR"

# Stage shared ML code and data for Lean Docker runtime.
rm -rf "$P2_DIR/ml_inputs/ml_pipeline"
cp -R "$ROOT_DIR/ml_pipeline" "$P2_DIR/ml_inputs/"
cp "$ROOT_DIR/ml_results_cache/best_params.json" "$P2_DIR/ml_inputs/"
cp "$ROOT_DIR/synthetic_data/csv_outputs/synthetic_predictor_features_all_months.csv" \
   "$P2_DIR/ml_inputs/synthetic_data/csv_outputs/"
cp "$ROOT_DIR/synthetic_data/csv_outputs/ff6_factors_10y.csv" \
   "$P2_DIR/ml_inputs/synthetic_data/csv_outputs/"
cp "$ROOT_DIR/synthetic_data/synthetic_equity_monthly_10y.csv" \
   "$P2_DIR/ml_inputs/synthetic_data/"

# Read model ids from cached parameters.
MODELS=$($PYTHON_BIN - <<'PY'
import json
from pathlib import Path
root = Path('/Users/raj/ws/asset-pricing')
with open(root / 'ml_results_cache' / 'best_params.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(' '.join(data.keys()))
PY
)

if [[ -n "$MODEL_ID_FILTER" ]]; then
  MODELS="$MODEL_ID_FILTER"
fi

cd "$P2_DIR"

echo "Run folder: $RUN_DIR"
echo "Models: $MODELS"

declare -a MODEL_COLS=()
declare -a MODEL_BT_JSON=()

for model in $MODELS; do
  echo "\n=== Backtesting $model ==="
  OUT_DIR="$RUN_DIR/$model"

  lean backtest . \
    --output "$OUT_DIR" \
    --parameter model-id "$model" \
    --parameter max-tickers "$MAX_TICKERS" \
    --parameter report-dir "$REPORTS_REL" \
    --extra-docker-config "$LEAN_DOCKER_NO_LIMITS"

  # Pick the main backtest result packet, not order events or monitor artifacts.
  BT_JSON=$(find "$OUT_DIR" -maxdepth 1 -type f -name '*.json' \
    | grep -E '/[0-9]+\.json$' \
    | grep -v 'order-events' \
    | grep -v 'summary.json' \
    | grep -v 'data-monitor-report' \
    | sort \
    | head -1)

  if [[ -z "$BT_JSON" ]]; then
    echo "No backtest JSON found for $model"
    exit 1
  fi

  mkdir -p "$RUN_DIR/reports/$model"
  lean report \
    --backtest-results "$BT_JSON" \
    --report-destination "$RUN_DIR/reports/$model/lean_report.html" \
    --overwrite

  MODEL_COLS+=("$model")
  MODEL_BT_JSON+=("$model::$BT_JSON")
done

# Build consolidated panel CSV/Excel from Lean backtest JSON statistics.
$PYTHON_BIN - <<PY
import json
from pathlib import Path
import pandas as pd

run_dir = Path(r"$RUN_DIR")
reports = run_dir / 'reports'

panel_rows = [
    'alpha_vs_benchmark',
    'alpha_ff6',
    'beta_mkt_rf',
    'beta_smb',
    'beta_hml',
    'beta_rmw',
    'beta_cma',
    'beta_umd',
    'drawdown',
    'turnover',
    'sharpe_ratio',
    'information_ratio',
    'avg_rebalance_return',
    'mean_return',
    'cumulative_return',
]

model_json_pairs = []
for raw in """${MODEL_BT_JSON[*]}""".split():
    if '::' not in raw:
        continue
    model, bt_json = raw.split('::', 1)
    model_json_pairs.append((model, Path(bt_json)))

models = sorted([m for m, _ in model_json_pairs])
panel = pd.DataFrame(index=panel_rows, columns=models, dtype=float)

for model, bt_path in model_json_pairs:
    with open(bt_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    m = payload.get('statistics', {})
    for row in panel_rows:
        value = m.get(row)
        try:
            panel.loc[row, model] = float(value)
        except (TypeError, ValueError):
            panel.loc[row, model] = float('nan')

panel_csv = reports / 'model_comparison_panel.csv'
panel_xlsx = reports / 'model_comparison_panel.xlsx'
panel.to_csv(panel_csv)
try:
  with pd.ExcelWriter(panel_xlsx, engine='openpyxl') as writer:
    panel.to_excel(writer, sheet_name='Model Panel')
except ImportError:
  panel_xlsx = None

print(f'Panel CSV: {panel_csv}')
if panel_xlsx is None:
  print('Panel Excel: skipped (openpyxl not installed)')
else:
  print(f'Panel Excel: {panel_xlsx}')
PY

echo "\nDone. Outputs in: $RUN_DIR/reports"
