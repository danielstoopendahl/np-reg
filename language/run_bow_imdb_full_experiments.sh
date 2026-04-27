#!/usr/bin/env bash
set -euo pipefail

# Runs the full IMDb BoW experiment table from:
# "Exjobb models - Final experiment IMDb.csv"
#
# Experiment groups per (vocab_size, hidden_dim):
# - vanilla
# - weight_decay in {1e-5, 1e-4, 1e-3}
# - dropout in {0.1, 0.3, 0.5}
# - layer_norm
# - batch_norm
# - np_reg_lambda in {0.01, 0.1, 1}
#
# Each group is evaluated on the fixed grid:
# batch_size in {32, 64, 128}
# learning_rate in {3e-6, 1e-5, 3e-5}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON_BIN="${PYTHON_BIN:-python}"
TARGET_SCRIPT="bow_imdb.py"
OUT_DIR="${OUT_DIR:-grid_results/full_imdb_experiments}"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_CSV="$OUT_DIR/bow_imdb_full_experiments_summary.csv"
RESUME="${RESUME:-1}"
SEED="${SEED:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$LOG_DIR" models

BATCH_SIZES=(32 64 128)
LEARNING_RATES=(3e-6 1e-5 3e-5)

# (vocab_size hidden_dim) pairs from the CSV.
MODEL_SPECS=(
  "1562 256"
  "3125 256"
  "6250 256"
  "12500 256"
  "25000 256"
  "50000 256"
)

WEIGHT_DECAYS=(1e-5 1e-4 1e-3)
DROPOUTS=(0.1 0.3 0.5)
NP_REGS=(0.001 0.01 0.1)

print_help() {
  cat <<'EOF'
Usage: bash run_bow_imdb_full_experiments.sh [options]

Options:
  --no-resume        Always re-run experiments even if a completed log exists.
  --seed N           Pass a fixed --seed N to bow_imdb.py.
  --extra-args "..." Extra raw arguments forwarded to bow_imdb.py.
  -h, --help         Show this help.

Environment variables:
  PYTHON_BIN         Python executable (default: python)
  OUT_DIR            Output directory (default: grid_results/full_imdb_experiments)
  RESUME             1 to reuse completed logs, 0 to force re-run (default: 1)
  SEED               Same as --seed
  EXTRA_ARGS         Same as --extra-args

Examples:
  bash run_bow_imdb_full_experiments.sh
  bash run_bow_imdb_full_experiments.sh --seed 42
  OUT_DIR=grid_results/final_a100 bash run_bow_imdb_full_experiments.sh
  bash run_bow_imdb_full_experiments.sh --extra-args "--dropout 0 --vocab-size 50000"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-resume)
      RESUME=0
      shift
      ;;
    --seed)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --seed"
        exit 1
      fi
      SEED="$2"
      shift 2
      ;;
    --extra-args)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --extra-args"
        exit 1
      fi
      EXTRA_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_help
      exit 1
      ;;
  esac
done

sanitize_tag() {
  printf "%s" "$1" | sed 's/+//g; s/\./p/g; s/-/m/g; s/\//_/g; s/ /_/g'
}

parse_metrics() {
  local log_file="$1"
  LC_ALL=C awk '
    BEGIN {
      best_acc = -1;
      best_loss = 1e300;
      test_acc = -1;
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
      if (match($0, /Test accuracy=[0-9]+\.?[0-9]*%/)) {
        tacc_str = substr($0, RSTART + 14, RLENGTH - 15);
        test_acc = tacc_str + 0;
      }
    }
    END {
      if (best_acc < 0 || best_loss >= 1e299) {
        best_loss_out = "NA";
        best_acc_out = "NA";
      } else {
        best_loss_out = sprintf("%.6f", best_loss);
        best_acc_out = sprintf("%.6f", best_acc);
      }

      if (test_acc < 0) {
        test_acc_out = "NA";
      } else {
        test_acc_out = sprintf("%.6f", test_acc);
      }

      printf "%s,%s,%s", best_loss_out, best_acc_out, test_acc_out;
    }
  ' "$log_file"
}

append_summary_row() {
  local model_name="$1"
  local vocab_size="$2"
  local hidden_dim="$3"
  local group_name="$4"
  local param_name="$5"
  local param_value="$6"
  local batch_size="$7"
  local learning_rate="$8"
  local best_loss="$9"
  local best_acc="${10}"
  local test_acc="${11}"
  local status="${12}"
  local log_file="${13}"

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$model_name" "$vocab_size" "$hidden_dim" "$group_name" "$param_name" "$param_value" \
    "$batch_size" "$learning_rate" "$best_loss" "$best_acc" "$test_acc" "$status" "$log_file" >> "$SUMMARY_CSV"
}

TOTAL_RUNS_PER_MODEL=$((9 + 27 + 27 + 9 + 9 + 27))
TOTAL_RUNS=$((TOTAL_RUNS_PER_MODEL * ${#MODEL_SPECS[@]}))
RUN_INDEX=0

printf "model_name,vocab_size,hidden_dim,group_name,param_name,param_value,batch_size,learning_rate,best_val_loss,best_val_acc_percent,test_acc_percent,status,log_file\n" > "$SUMMARY_CSV"

echo "Starting full IMDb BoW experiment suite"
echo "Total runs in matrix: $TOTAL_RUNS"
echo "Summary CSV: $SUMMARY_CSV"
if [[ "$RESUME" == "1" ]]; then
  echo "Resume mode: enabled (completed logs are reused)"
else
  echo "Resume mode: disabled (all runs are re-executed)"
fi

run_one() {
  local model_name="$1"
  local vocab_size="$2"
  local hidden_dim="$3"
  local group_name="$4"
  local param_name="$5"
  local param_value="$6"
  local batch_size="$7"
  local learning_rate="$8"
  shift 8
  local extra_flags=("$@")

  RUN_INDEX=$((RUN_INDEX + 1))

  local lr_tag
  lr_tag="$(sanitize_tag "$learning_rate")"
  local pv_tag
  pv_tag="$(sanitize_tag "$param_value")"
  local run_tag
  run_tag="${model_name}_${group_name}_${param_name}_${pv_tag}_bs_${batch_size}_lr_${lr_tag}"
  local log_file
  log_file="$LOG_DIR/${run_tag}.log"

  echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model_name group=$group_name $param_name=$param_value bs=$batch_size lr=$learning_rate"

  if [[ "$RESUME" == "1" && -f "$log_file" ]]; then
    if grep -q "Test loss=" "$log_file"; then
      metrics="$(parse_metrics "$log_file")"
      best_loss="$(printf "%s" "$metrics" | cut -d',' -f1)"
      best_acc="$(printf "%s" "$metrics" | cut -d',' -f2)"
      test_acc="$(printf "%s" "$metrics" | cut -d',' -f3)"

      append_summary_row \
        "$model_name" "$vocab_size" "$hidden_dim" "$group_name" "$param_name" "$param_value" \
        "$batch_size" "$learning_rate" "$best_loss" "$best_acc" "$test_acc" "skipped_existing" "$log_file"
      echo "  reused existing completed run"
      return
    fi
  fi

  cmd=(
    "$PYTHON_BIN" "$TARGET_SCRIPT"
    "--batch-size" "$batch_size"
    "--lr" "$learning_rate"
    "--vocab-size" "$vocab_size"
    "--hidden-dim" "$hidden_dim"
  )

  if [[ -n "$SEED" ]]; then
    cmd+=("--seed" "$SEED")
  fi

  for f in "${extra_flags[@]}"; do
    cmd+=("$f")
  done

  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    extra_split=($EXTRA_ARGS)
    for f in "${extra_split[@]}"; do
      cmd+=("$f")
    done
  fi

  "${cmd[@]}" 2>&1 | tee "$log_file"
  cmd_exit=${PIPESTATUS[0]}

  if [[ $cmd_exit -ne 0 ]]; then
    append_summary_row \
      "$model_name" "$vocab_size" "$hidden_dim" "$group_name" "$param_name" "$param_value" \
      "$batch_size" "$learning_rate" "" "" "" "failed" "$log_file"
    echo "  run failed with exit code $cmd_exit"
    return
  fi

  metrics="$(parse_metrics "$log_file")"
  best_loss="$(printf "%s" "$metrics" | cut -d',' -f1)"
  best_acc="$(printf "%s" "$metrics" | cut -d',' -f2)"
  test_acc="$(printf "%s" "$metrics" | cut -d',' -f3)"

  append_summary_row \
    "$model_name" "$vocab_size" "$hidden_dim" "$group_name" "$param_name" "$param_value" \
    "$batch_size" "$learning_rate" "$best_loss" "$best_acc" "$test_acc" "ok" "$log_file"

  echo "  saved: best_val_loss=$best_loss best_val_acc_percent=$best_acc test_acc_percent=$test_acc"
}

for spec in "${MODEL_SPECS[@]}"; do
  vocab_size="$(printf "%s" "$spec" | awk '{print $1}')"
  hidden_dim="$(printf "%s" "$spec" | awk '{print $2}')"
  model_name="v${vocab_size}_h${hidden_dim}"

  echo "============================================================"
  echo "Model setting: vocab_size=$vocab_size hidden_dim=$hidden_dim"

  # Vanilla
  for batch_size in "${BATCH_SIZES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      run_one "$model_name" "$vocab_size" "$hidden_dim" "vanilla" "none" "none" "$batch_size" "$learning_rate"
    done
  done

  # Weight decay
  for wd in "${WEIGHT_DECAYS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      for learning_rate in "${LEARNING_RATES[@]}"; do
        run_one "$model_name" "$vocab_size" "$hidden_dim" "weight_decay" "weight_decay" "$wd" "$batch_size" "$learning_rate" "--weight-decay" "$wd"
      done
    done
  done

  # Dropout
  for p in "${DROPOUTS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      for learning_rate in "${LEARNING_RATES[@]}"; do
        run_one "$model_name" "$vocab_size" "$hidden_dim" "dropout" "dropout" "$p" "$batch_size" "$learning_rate" "--dropout" "$p"
      done
    done
  done

  # Layer norm
  for batch_size in "${BATCH_SIZES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      run_one "$model_name" "$vocab_size" "$hidden_dim" "layer_norm" "layer_norm" "true" "$batch_size" "$learning_rate" "--layer-norm"
    done
  done

  # Batch norm
  for batch_size in "${BATCH_SIZES[@]}"; do
    for learning_rate in "${LEARNING_RATES[@]}"; do
      run_one "$model_name" "$vocab_size" "$hidden_dim" "batch_norm" "batch_norm" "true" "$batch_size" "$learning_rate" "--batch-norm"
    done
  done

  # NP regularization
  for np_lambda in "${NP_REGS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
      for learning_rate in "${LEARNING_RATES[@]}"; do
        run_one "$model_name" "$vocab_size" "$hidden_dim" "np_reg" "np_reg_lambda" "$np_lambda" "$batch_size" "$learning_rate" "--np-reg-lambda" "$np_lambda"
      done
    done
  done
done

echo "============================================================"
echo "All experiments processed."
echo "Summary saved to: $SUMMARY_CSV"
