#!/usr/bin/env bash
set -euo pipefail

# Grid search launcher for WGR_UniY.py.
# Splits work across GPUs by noise_dim (nz=1 -> GPU_FOR_NZ1, nz=5 -> GPU_FOR_NZ5),
# runs all combos in parallel per noise_dim, keeps per-run logs, and writes
# markdown tables with mean(std) across reps. Raw logs/metrics live in RUN_DIR.

RESULT_FILE="${RESULT_FILE:-grid_results.md}"
RUN_DIR="${RUN_DIR:-grid_runs}"
REPS="${REPS:-5}"

GPU_FOR_NZ1="${GPU_FOR_NZ1:-0}"
GPU_FOR_NZ5="${GPU_FOR_NZ5:-1}"
declare -A GPU_MAP=([1]="${GPU_FOR_NZ1}" [5]="${GPU_FOR_NZ5}")

mkdir -p "${RUN_DIR}"
: > "${RUN_DIR}/metrics.tsv"
: > "${RUN_DIR}/failures.log"

TRAIN_BATCHES=(5000 4096 2048 1024 512 256 128 64)
# TRAIN_BATCHES=(4096 2048)
LAMBDA_WS=(0 0.2 0.5 0.9 1)
# LAMBDA_WS=(0)
NOISE_DIMS=(1 5)

# Launch per-noise-dim worker in parallel
pids=()
for nz in "${NOISE_DIMS[@]}"; do
  metrics_sub="${RUN_DIR}/metrics_nz${nz}.tsv"
  failures_sub="${RUN_DIR}/failures_nz${nz}.log"
  : > "${metrics_sub}"
  : > "${failures_sub}"
  (
    set -euo pipefail
    gpu="${GPU_MAP[$nz]}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    for bs in "${TRAIN_BATCHES[@]}"; do
      for lw in "${LAMBDA_WS[@]}"; do
        tag="bs${bs}_lw${lw}_nz${nz}"
        run_log="${RUN_DIR}/run_${tag}.log"
        echo "[nz=${nz}] train_batch=${bs} lambda_w=${lw} noise_dim=${nz} reps=${REPS} gpu=${gpu} (log: ${run_log})"
        python Simulation/WGR_UniY.py \
          --train_batch "${bs}" \
          --lambda_w "${lw}" \
          --noise_dim "${nz}" \
          --reps "${REPS}" 2>&1 | tee "${run_log}"

        if ! parsed_line=$(python - "${run_log}" <<'PY'
import sys, re
log_path = sys.argv[1]
lines = open(log_path, "r").read().splitlines()

def extract_block(start_prefix, stop_prefixes):
    vals = []
    collecting = False
    for line in lines:
        if line.startswith(start_prefix):
            collecting = True
        elif collecting and any(line.startswith(p) for p in stop_prefixes):
            break
        if collecting:
            vals.extend([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", line)])
    return vals

ms = extract_block("Mean over reps [L1, L2, MSE(mean), MSE(sd)]:", ["Std over reps  [L1, L2, MSE(mean), MSE(sd)]:"])
ss = extract_block("Std over reps  [L1, L2, MSE(mean), MSE(sd)]:", ["Mean over reps quantiles [0.05, 0.25, 0.50, 0.75, 0.95]:"])
mq = extract_block("Mean over reps quantiles [0.05, 0.25, 0.50, 0.75, 0.95]:", ["Std over reps  quantiles [0.05, 0.25, 0.50, 0.75, 0.95]:"])
sq = extract_block("Std over reps  quantiles [0.05, 0.25, 0.50, 0.75, 0.95]:", [])
if len(mq) >= 10:
    mq = mq[-5:]
if len(sq) >= 10:
    sq = sq[-5:]
if not (len(ms) >= 4 and len(ss) >= 4 and len(mq) >= 5 and len(sq) >= 5):
    sys.exit(1)
ms = ms[-4:]; ss = ss[-4:]; mq = mq[-5:]; sq = sq[-5:]
print("\t".join(map(str, ms + ss + mq + sq)))
PY
        ); then
          echo "Parse failed for ${tag}; see ${run_log}" >> "${failures_sub}"
          continue
        fi
        echo -e "${bs}\t${lw}\t${nz}\t${parsed_line}" >> "${metrics_sub}"
      done
    done
  ) &
  pids+=($!)
done

# Wait for all workers
for pid in "${pids[@]}"; do
  wait "${pid}"
done

# Consolidate metrics and failures
cat "${RUN_DIR}"/metrics_nz*.tsv > "${RUN_DIR}/metrics.tsv"
cat "${RUN_DIR}"/failures_nz*.log > "${RUN_DIR}/failures.log"

# Build markdown tables
python - <<'PY' "${RUN_DIR}/metrics.tsv" "${RESULT_FILE}"
import sys
input_path, out_path = sys.argv[1], sys.argv[2]
rows = []
with open(input_path) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 3 + 4 + 4 + 5 + 5:
            continue
        bs, lw, nz = parts[0:3]
        vals = list(map(float, parts[3:]))
        l1_mean, l2_mean, mse_mean_mean, mse_sd_mean = vals[0:4]
        l1_std, l2_std, mse_mean_std, mse_sd_std = vals[4:8]
        q_means = vals[8:13]
        q_stds = vals[13:18]
        rows.append({
            "bs": bs, "lw": lw, "nz": nz,
            "l1": (l1_mean, l1_std),
            "l2": (l2_mean, l2_std),
            "mmean": (mse_mean_mean, mse_mean_std),
            "msd": (mse_sd_mean, mse_sd_std),
            "q": list(zip(q_means, q_stds))
        })

def fmt(pair):
    m, s = pair
    return f"{m:.3f}({s:.3f})"

lines = []
lines.append("| train_batch | lambda_w | noise_dim | L1 | L2 | MSE_mean | MSE_sd |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(f"| {r['bs']} | {r['lw']} | {r['nz']} | {fmt(r['l1'])} | {fmt(r['l2'])} | {fmt(r['mmean'])} | {fmt(r['msd'])} |")

lines.append("")
lines.append("| train_batch | lambda_w | noise_dim | Q05 | Q25 | Q50 | Q75 | Q95 |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    qcells = " | ".join(fmt(pair) for pair in r["q"])
    lines.append(f"| {r['bs']} | {r['lw']} | {r['nz']} | {qcells} |")

with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"Wrote results to {out_path}")
PY
