#!/bin/bash
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
INPUT_S3_PATH="${INPUT_S3_PATH}"
S3_ENDPOINT_URL_RESULTS="${S3_ENDPOINT_URL_RESULTS}"
S3_URL="https://s3-west.nrp-nautilus.io"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/data/raw"

LOG_FILE="${OUTPUT_DIR}/pipeline_report.txt"
echo "Starting HIPPIE Integration Pipeline..." > "$LOG_FILE"

# ---------------------------------------------------------
# 0. SETUP: GitHub Repo & Dependencies
# ---------------------------------------------------------
echo "Cloning GitHub repository..." | tee -a "$LOG_FILE"
git clone https://github.com/dolmo/folic_acid_hippie.git /workspace/repo >> "$LOG_FILE" 2>&1
cd /workspace/repo

echo "Verifying s5cmd and Python dependencies..." | tee -a "$LOG_FILE"
# If the Docker image already has these, it will pass in seconds. If not, it installs them.
if ! command -v s5cmd &> /dev/null; then
    python3 -c "import urllib.request; urllib.request.urlretrieve('https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz', 's5cmd.tar.gz')"
    tar -xzf s5cmd.tar.gz -C /usr/local/bin/ s5cmd && rm s5cmd.tar.gz
    chmod +x /usr/local/bin/s5cmd
fi

apt-get update && apt-get install -y -q awscli >> "$LOG_FILE" 2>&1
pip install git+https://github.com/braingeneers/HIPPIE.git >> "$LOG_FILE" 2>&1
pip install -r requirements.txt >> "$LOG_FILE" 2>&1

# Dynamically patch config.py to use Kubernetes container paths instead of your local machine paths
echo "Patching config.py paths for Kubernetes environment..." | tee -a "$LOG_FILE"
cat <<EOF > config.py
from pathlib import Path
DATA_RAW_T3 = Path("${PROJECT_ROOT}/data/raw")
HIPPIE_INPUT_T3 = Path("${OUTPUT_DIR}/processed/hippie_input")
DATA_PROCESSED_T3 = Path("${OUTPUT_DIR}/processed/wells")
METADATA_CSV_T3 = HIPPIE_INPUT_T3 / "metadata.csv"
RESULTS_EDA = Path("${OUTPUT_DIR}/eda")

# Fallbacks for T4
DATA_RAW = DATA_RAW_T3
HIPPIE_INPUT = HIPPIE_INPUT_T3
DATA_PROCESSED = DATA_PROCESSED_T3
METADATA_CSV = METADATA_CSV_T3
LABELS_CSV = HIPPIE_INPUT_T3 / "labels.csv"
EOF

# ---------------------------------------------------------
# 1. DOWNLOAD RAW DATA
# ---------------------------------------------------------
echo "Downloading raw input data from S3..." | tee -a "$LOG_FILE"
s5cmd --endpoint-url "${S3_URL}" cp "${INPUT_S3_PATH}*" "${PROJECT_ROOT}/data/raw/" >> "$LOG_FILE" 2>&1
DL_STATUS=$?

if [ $DL_STATUS -ne 0 ]; then
    echo "ERROR: S3 Download failed! Uploading logs and exiting." | tee -a "$LOG_FILE"
    s5cmd --endpoint-url "${S3_URL}" cp "${OUTPUT_DIR}/*" "${S3_ENDPOINT_URL_RESULTS}"
    exit $DL_STATUS
fi

# ---------------------------------------------------------
# 2. RUN PIPELINE SCRIPTS
# ---------------------------------------------------------
echo "Running Step 00: Preprocess (T3)..." | tee -a "$LOG_FILE"
python3 scripts/00b_preprocess_t3.py >> "$LOG_FILE" 2>&1

echo "Running Step 02: Raw Metric EDA..." | tee -a "$LOG_FILE"
python3 scripts/02_raw_metric_eda.py >> "$LOG_FILE" 2>&1

echo "Running Step 03: Train HIPPIE..." | tee -a "$LOG_FILE"
bash scripts/03_train_hippie.sh 10 >> "$LOG_FILE" 2>&1

# Add remaining scripts (04-09) here as needed.
PIPE_STATUS=$?

if [ $PIPE_STATUS -ne 0 ]; then
    echo "ERROR: Pipeline encountered an issue! (Exit code: $PIPE_STATUS)" | tee -a "$LOG_FILE"
else
    echo "SUCCESS: Pipeline finished!" | tee -a "$LOG_FILE"
fi

# ---------------------------------------------------------
# 3. UPLOAD RESULTS
# ---------------------------------------------------------
echo "Uploading all results, plots, and logs back to S3..."
cp -r results/* ${OUTPUT_DIR}/ 2>/dev/null 
s5cmd --endpoint-url "${S3_URL}" cp "${OUTPUT_DIR}/*" "${S3_ENDPOINT_URL_RESULTS}"

exit $PIPE_STATUS