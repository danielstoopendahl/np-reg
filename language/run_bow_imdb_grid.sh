#!/usr/bin/env bash
set -u

# Runs a fixed 9-run grid (batch size x learning rate) and stores best val metrics.
# Optional: pass one custom flag/value among
# --dropout, --weight-decay, --np-reg-lambda, --o-reg-lambda.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_SCRIPT="bow_imdb.py"

OUT_DIR="${OUT_DIR:-grid_results}"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_CSV="$OUT_DIR/bow_imdb_grid_summary.csv"

mkdir -p "$LOG_DIR" models

CUSTOM_FLAG=""
CUSTOM_VALUE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dropout|--weight-decay|--np-reg-lambda|--o-reg-lambda)
      if [[ -n "$CUSTOM_FLAG" ]]; then
        echo "Provide only one custom flag."
        exit 1
      fi
      if [[ $# -lt 2 ]]; then
        echo "Missing value for $1"
        exit 1
      fi
      CUSTOM_FLAG="$1"
      CUSTOM_VALUE="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash run_bow_imdb_grid_drop.sh [--dropout v | --weight-decay v | --np-reg-lambda v | --o-reg-lambda v]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Fixed grid from the user specification.
BATCH_SIZES=(32 64 128)
LEARNING_RATES=(3e-6 1e-5 3e-5)

run_id=0
param_name="none"
param_value="none"
if [[ -n "$CUSTOM_FLAG" ]]; then
  param_name="${CUSTOM_FLAG#--}"
  param_value="$CUSTOM_VALUE"
fi

printf "param_name,param_value,batch_size,learning_rate,best_val_loss,best_val_acc_percent,status,log_file\n" > "$SUMMARY_CSV"

echo "Starting fixed grid..."
if [[ -n "$CUSTOM_FLAG" ]]; then
  echo "Using custom parameter: $CUSTOM_FLAG $CUSTOM_VALUE"
fi

for batch_size in "${BATCH_SIZES[@]}"; do
  for learning_rate in "${LEARNING_RATES[@]}"; do
    run_id=$((run_id + 1))

    lr_tag=$(printf "%s" "$learning_rate" | sed 's/+//g; s/\./p/g; s/-/m/g')
    value_tag=$(printf "%s" "$param_value" | sed 's/+//g; s/\./p/g; s/-/m/g')
    log_file="$LOG_DIR/run_${run_id}_${param_name}_${value_tag}_bs_${batch_size}_lr_${lr_tag}.log"

    echo "[run=$run_id] batch_size=$batch_size learning_rate=$learning_rate"

    cmd=("$PYTHON_BIN" "$TARGET_SCRIPT" "--batch-size" "$batch_size" "--lr" "$learning_rate")
    if [[ -n "$CUSTOM_FLAG" ]]; then
      cmd+=("$CUSTOM_FLAG" "$CUSTOM_VALUE")
    fi

    "${cmd[@]}" 2>&1 | tee "$log_file"
    cmd_exit=${PIPESTATUS[0]}

    if [[ $cmd_exit -ne 0 ]]; then
      echo "Run failed with exit code $cmd_exit"
      printf "%s,%s,%s,%s,,,%s,%s\n" \
        "$param_name" "$param_value" "$batch_size" "$learning_rate" "failed" "$log_file" >> "$SUMMARY_CSV"
      continue
    fi

    best_metrics=$(awk '
      BEGIN {
        best_acc = -1;
        best_loss = 1e300;
      }
      {
        if (match($0, /val_loss=[0-9]+\.?[0-9]*/)) {
          loss_str = substr($0, RSTART + 9, RLENGTH - 9);
          loss = loss_str + 0;
          if (loss < best_loss) best_loss = loss;
        }
        if (match($0, /val_acc=[0-9]+\.?[0-9]*%/)) {
          acc_str = substr($0, RSTART + 8, RLENGTH - 9);
          acc = acc_str + 0;
          if (acc > best_acc) best_acc = acc;
        }
      }
      END {
        if (best_acc < 0 || best_loss >= 1e299) {
          print "NA,NA";
        } else {
          printf "%.6f,%.6f", best_loss, best_acc;
        }
      }
    ' "$log_file")

    best_loss=$(printf "%s" "$best_metrics" | cut -d',' -f1)
    best_acc=$(printf "%s" "$best_metrics" | cut -d',' -f2)

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$param_name" "$param_value" "$batch_size" "$learning_rate" "$best_loss" "$best_acc" "ok" "$log_file" >> "$SUMMARY_CSV"

    echo "Saved results: best_val_loss=$best_loss best_val_acc_percent=$best_acc"
  done
done

echo "All done. Summary: $SUMMARY_CSV"
