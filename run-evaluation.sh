#!/bin/bash
# NKI-MoE Evaluation Runner
# Runs model evaluation with team_id and member_id tracking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
MODE="evaluate_single"
PLATFORM="trn2"
PROMPT="I believe the meaning of life is"
SEQ_LEN=640
QWEN_MODULE="qwen"
MODEL_PATH="$HOME/Qwen3-30B-A3B/hf_model"
COMPILED_MODEL_PATH="$HOME/Qwen3-30B-A3B/traced_model"
SKIP_COMPILE=false
TARGET_ACCOUNT_ID="195034363981"
SUBMISSION_ID="submission_$(date +%Y%m%d_%H%M%S)"
UPLOAD_TO_S3=false

# Functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }
log_result() { echo -e "${CYAN}[RESULT]${NC} $1"; }

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required:
  -t, --team-id TEAM_ID          Team identifier (required)
  -m, --member-id MEMBER_ID      Team member identifier (required)

Optional:
  -M, --mode MODE                Evaluation mode (default: evaluate_single)
                                 Options: evaluate_single, evaluate_all, validate, generate
  -p, --platform PLATFORM        Platform target (default: trn2)
                                 Options: trn2, trn3
  -P, --prompt PROMPT            Prompt text (default: "I believe the meaning of life is")
  -s, --seq-len LENGTH           Sequence length (default: 640)
  -q, --qwen-module MODULE       Qwen model processing module name (default: qwen)
                                 Examples: qwen, qwen_optimized, qwen_with_nki
  -a, --target-account-id ID     AWS account ID for S3 bucket 
  -S, --submission-id ID         Submission identifier (default: auto-generated timestamp)
  -u, --upload                   Upload results to S3 bucket
  -h, --help                     Show this help message

Examples:
  # Single prompt evaluation on trn2 platform with default prompt
  $0 --team-id TEAM-1234 --member-id john_doe

  # Single prompt evaluation with custom qwen module
  $0 -t TEAM-1234 -m john_doe -q qwen_optimized

  # Single prompt evaluation with S3 bucket upload to custom account
  $0 -t TEAM-1234 -m john_doe -a 123456789012 --upload

  # Evaluate all prompts with custom qwen module
  $0 -t TEAM-1234 -m jane_smith -M evaluate_all -q qwen_with_nki --upload

  # Evaluate single prompt on trn3 platform with custom account
  $0 -t TEAM-1234 -m bob_jones -p trn3 -a 987654321098 --upload

  # All three overrides
  $0 -t TEAM-1234 -m john@company.com \
  --model-path ~/qwen3-30B-A3B/hf_model \
  --compiled-model-path ~/qwen3-30B-A3B/traced_model \
  --skip-compile

  # Custom paths without skipping compile
  $0 -t TEAM-1234 -m john@company.com \
  --model-path /path/qwen3-30B-A3B/hf_model \
  --compiled-model-path /path/qwen3-30B-A3B/traced_model

  # Combined with other flags (NKI module, S3 upload, trn3 platform)
  $0 -t TEAM-1234 -m john@company.com \
  -q qwen_with_nki -p trn3 \
  --model-path ~/qwen3-30B-A3B/hf_model \
  --compiled-model-path ~/qwen3-30B-A3B/traced_model \
  --skip-compile --upload

EOF
}

# set the Python3 environment and wait
 source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
 echo ""
 log_step  "Python 3 environment activated"
 echo ""

# Parse command line arguments
TEAM_ID=""
MEMBER_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--team-id)
            TEAM_ID="$2"
            shift 2
            ;;
        -m|--member-id)
            MEMBER_ID="$2"
            shift 2
            ;;
        -M|--mode)
            MODE="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -P|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -s|--seq-len)
            SEQ_LEN="$2"
            shift 2
            ;;
        -q|--qwen-module)
            QWEN_MODULE="$2"
            shift 2
            ;;
        -a|--target-account-id)
            TARGET_ACCOUNT_ID="$2"
            shift 2
            ;;
        -S|--submission-id)
            SUBMISSION_ID="$2"
            shift 2
            ;;
        -u|--upload)
            UPLOAD_TO_S3=true
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --compiled-model-path)
            COMPILED_MODEL_PATH="$2"
            shift 2
            ;;
        --skip-compile)
            SKIP_COMPILE=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$TEAM_ID" ]; then
    log_error "Team ID in a format of 'TEAM-1234' is required!"
    print_usage
    exit 1
fi

if [ -z "$MEMBER_ID" ]; then
    log_error "Member ID (email) is required!"
    print_usage
    exit 1
fi

# Display configuration
echo ""
log_step "NKI-MoE Inference performance Evaluation Configuration"
echo "=================================="
echo "Team ID:       $TEAM_ID"
echo "Member ID:     $MEMBER_ID"
echo "Mode:          $MODE"
echo "Trainium Platform:      $PLATFORM"
echo "Prompt:        $PROMPT"
echo "Sequence Len:  $SEQ_LEN"
echo "Qwen Module File: $QWEN_MODULE.py"
echo "Model Path:    $MODEL_PATH"
echo "Compiled Path: $COMPILED_MODEL_PATH"
echo "Skip Compile:  $SKIP_COMPILE"
echo "Submission ID: $SUBMISSION_ID"
echo "Upload to S3 bucket:  $UPLOAD_TO_S3"

# CHECK: clean up previous compile results to avoid potential assertion errors
 rm -rf /var/tmp/neuron-compile-cache/*
 rm -rf "$COMPILED_MODEL_PATH"
 echo "CHECK: Neuron compile cache and traced model data are cleaned up.."
 echo "..."
 
# compute the S3 bucket name for uploading artifacts
if [ "$UPLOAD_TO_S3" = true ]; then
    #S3_BUCKET="nki-moe-leaderboard-dev-submissions-${TARGET_ACCOUNT_ID}"
    S3_BUCKET="nki-moe-leaderboard-dev-profiling-data-${TARGET_ACCOUNT_ID}"
    echo "Target Account ID:    $TARGET_ACCOUNT_ID"
    echo "Target S3 Bucket:     s3://$S3_BUCKET"
    echo "S3 Path:     benchmarks/$TEAM_ID/$MEMBER_ID/$SUBMISSION_ID/"
fi
echo "=================================="
echo ""

# Check if main.py exists
if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    log_error "main.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log_error "python3 is not installed or not in PATH"
    exit 1
fi

# Determine benchmarking CSV filename (matching logic in main.py calculate_score function)
if [ -n "$TEAM_ID" ]; then
    BASE_FILENAME="${TEAM_ID}_qwen3-30b-a3b"
    if [ -n "$QWEN_MODULE" ] && [ "$QWEN_MODULE" != "qwen" ]; then
        CSV_FILENAME="${BASE_FILENAME}_${QWEN_MODULE}_score_records.csv"
    else
        CSV_FILENAME="${BASE_FILENAME}_score_records.csv"
    fi
else
    if [ -n "$QWEN_MODULE" ] && [ "$QWEN_MODULE" != "qwen" ]; then
        CSV_FILENAME="qwen3-30b-a3b_${QWEN_MODULE}_score_records.csv"
    else
        CSV_FILENAME="qwen3-30b-a3b_score_records.csv"
    fi
fi

log_step "ATTENTION:  Benchmarking File Name: $CSV_FILENAME"

# Validate qwen module file
QWEN_FILE="${QWEN_MODULE}.py"
log_step "ATTENTION: Validating Submitted QWEN Module Script: $QWEN_FILE"

# Check if qwen module python file exists
if [ ! -f "$SCRIPT_DIR/$QWEN_FILE" ]; then
    log_error "Qwen module source file not found: $QWEN_FILE"
    log_error "Expected script location: $SCRIPT_DIR/$QWEN_FILE"
    exit 1
fi

log_info "✓ Submitted QWEN Python Script file path: $QWEN_FILE"

# Check if file is readable
if [ ! -r "$SCRIPT_DIR/$QWEN_FILE" ]; then
    log_error "QWEN module source file is not readable: $QWEN_FILE"
    exit 1
fi

log_info "✓ QWEN Script File is readable"

# Validate Python3 syntax
log_info "Validating QWEN Script File  Python3 syntax..."
if ! python3 -m py_compile "$SCRIPT_DIR/$QWEN_FILE" 2>/dev/null; then
    log_error "Python syntax validation failed for submitted Script file: $QWEN_FILE"
    log_error "Please fix syntax errors in your QWEN module script"
    python3 -m py_compile "$SCRIPT_DIR/$QWEN_FILE"
    exit 1
fi

log_info "✓ QWEN Script File Python3 syntax is valid - proceeding to security checks..."

# Security checks - scan for dangerous patterns
log_info "Performing QWEN Script File Python3 library security checks..."
SECURITY_ISSUES=0

# Check for dangerous imports
DANGEROUS_IMPORTS="subprocess|os\.system|eval|exec|compile|__import__|pickle|shelve|marshal"
if grep -E "^\s*(import|from).*($DANGEROUS_IMPORTS)" "$SCRIPT_DIR/$QWEN_FILE" > /dev/null 2>&1; then
    log_warn "⚠ Warning: Potentially dangerous imports detected (subprocess, os.system, eval, exec, etc.)"
    log_warn "These may be flagged for security review"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for file operations
if grep -E "(open\s*\(|file\s*\(|\.write\(|\.read\()" "$SCRIPT_DIR/$QWEN_FILE" > /dev/null 2>&1; then
    log_warn "⚠ Warning: File I/O operations detected"
    log_warn "Ensure file operations are necessary and safe"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for network operations
if grep -E "(socket|urllib|requests|http\.client|ftplib)" "$SCRIPT_DIR/$QWEN_FILE" > /dev/null 2>&1; then
    log_warn "⚠ Warning: Network operations detected"
    log_warn "Network access may be restricted during evaluation"
    SECURITY_ISSUES=$((SECURITY_ISSUES + 1))
fi

# Check for system commands
if grep -E "(os\.system|subprocess\.call|subprocess\.run|subprocess\.Popen)" "$SCRIPT_DIR/$QWEN_FILE" > /dev/null 2>&1; then
    log_error "✗ SECURITY RISK: System command execution detected"
    log_error "System commands are not allowed in qwen modules"
    exit 1
fi

# Check for code execution
if grep -E "\b(eval|exec)\s*\(" "$SCRIPT_DIR/$QWEN_FILE" > /dev/null 2>&1; then
    log_error "✗ SECURITY RISK: Dynamic code execution detected (eval/exec)"
    log_error "Dynamic code execution is not allowed"
    exit 1
fi

# Check overall QWEN library file size (max 10MB)
FILE_SIZE=$(stat -f%z "$SCRIPT_DIR/$QWEN_FILE" 2>/dev/null || stat -c%s "$SCRIPT_DIR/$QWEN_FILE" 2>/dev/null)
MAX_SIZE=$((10 * 1024 * 1024))
if [ "$FILE_SIZE" -gt "$MAX_SIZE" ]; then
    log_error "Submitted QWEN Script File size exceeds 10MB limit: $(($FILE_SIZE / 1024 / 1024))MB"
    exit 1
fi

log_info "✓ Submitted QWEN Script File size: $(($FILE_SIZE / 1024))KB"

# Summary
if [ $SECURITY_ISSUES -eq 0 ]; then
    log_info "✓ Security checks passed - no issues detected, tests can continue!"
else
    log_warn "Security checks completed with $SECURITY_ISSUES warning(s)"
    log_warn "Review and resolve warnings above before proceeding to tests!"
fi

echo "END QWEN Script File library security check"
echo ""

# Check AWS CLI if upload is enabled
if [ "$UPLOAD_TO_S3" = true ]; then
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed or not in PATH"
        log_error "Install AWS CLI to enable S3 upload: https://aws.amazon.com/cli/"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        log_error "Run 'aws configure' to set up your credentials"
        exit 1
    fi
    
    log_info "AWS Account access configured successfully"
fi

# Record start time
START_TIME=$(date +%s)
log_info "Starting Inference Benchmarking at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Build Python command as an array for safe argument handling
#  --seq-len "$SEQ_LEN"
#  Needed for proper CSV file generation --qwen "$QWEN_MODULE"
PYTHON_CMD=(python3 main.py
    --mode "$MODE"
    --team-id "$TEAM_ID"
    --member-id "$MEMBER_ID"
    --platform-target "$PLATFORM"
    --model-path "$MODEL_PATH"
    --compiled-model-path "$COMPILED_MODEL_PATH"
    --qwen "$QWEN_MODULE"
)

# Add skip-compile flag ONLY if enabled
if [ "$SKIP_COMPILE" = true ]; then
    PYTHON_CMD+=(--skip-compile)
fi

# Add custom prompt for single evaluation mode
if [ "$MODE" = "evaluate_single" ] || [ "$MODE" = "validate" ] || [ "$MODE" = "generate" ]; then
    PYTHON_CMD+=(--prompt "$PROMPT")
fi

log_step "Initiating QWEN Model Inference performance evaluation..."
log_info "Command: ${PYTHON_CMD[*]}"
echo "=================================================="

# Run the Python script
cd "$SCRIPT_DIR"
if "${PYTHON_CMD[@]}"; then
    EXIT_CODE=0
    log_info "QWEN Model Inference performance evaluation completed successfully!"
else
    EXIT_CODE=$?
    log_error "QWEN Model Inference performance evaluation failed with exit code $EXIT_CODE!"
fi

echo ""

# Calculate execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

log_info "CHECK: Benchmarking TOTAL Execution time: ${MINUTES}m ${SECONDS}s"
echo "=================================================="
echo ""

# Check if CSV file was created/updated
if [ -f "$SCRIPT_DIR/$CSV_FILENAME" ]; then
    log_step "Generated NKI-MOE Benchmark Records File"
    echo "=================================="
    log_result "File: $CSV_FILENAME"
    log_result "Location: $SCRIPT_DIR/$CSV_FILENAME"
    
    # Get file info
    FILE_SIZE=$(ls -lh "$SCRIPT_DIR/$CSV_FILENAME" | awk '{print $5}')
    FILE_LINES=$(wc -l < "$SCRIPT_DIR/$CSV_FILENAME")
    LAST_MODIFIED=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$SCRIPT_DIR/$CSV_FILENAME" 2>/dev/null || stat -c "%y" "$SCRIPT_DIR/$CSV_FILENAME" 2>/dev/null | cut -d'.' -f1)
    
    log_result "Size: $FILE_SIZE"
    log_result "Lines: $FILE_LINES"
    log_result "Last Modified: $LAST_MODIFIED"
    echo ""
    
    
    # Display records for this team member
    log_step "NKI-MOE Inference Metric Records for Team: $TEAM_ID, Member: $MEMBER_ID"
    echo "=================================="
    MEMBER_RECORDS=$(grep "^$TEAM_ID,$MEMBER_ID," "$SCRIPT_DIR/$CSV_FILENAME" 2>/dev/null || echo "")
    if [ -n "$MEMBER_RECORDS" ]; then
        echo "$MEMBER_RECORDS" | column -t -s ','
        RECORD_COUNT=$(echo "$MEMBER_RECORDS" | wc -l)
        echo ""
        log_result "Found $RECORD_COUNT record(s) for this team member"
    else
        log_warn "No records found for team '$TEAM_ID' and member '$MEMBER_ID'"
    fi
    echo ""
    
    # Display summary statistics for this member
    if [ -n "$MEMBER_RECORDS" ]; then
        log_step "Performance Summary"
        echo "=================================="
        
        # Extract scores (assuming final_score is column 11)
        SCORES=$(echo "$MEMBER_RECORDS" | awk -F',' '{print $11}')
        
        if [ -n "$SCORES" ]; then
            # Calculate statistics using awk
            STATS=$(echo "$SCORES" | awk '
                BEGIN { min=999999; max=0; sum=0; count=0 }
                {
                    if ($1 < min) min = $1
                    if ($1 > max) max = $1
                    sum += $1
                    count++
                }
                END {
                    avg = sum / count
                    printf "Min Score: %.4f\nMax Score: %.4f\nAvg Score: %.4f\nTotal Runs: %d\n", min, max, avg, count
                }
            ')
            echo "$STATS"
        fi
        echo ""
    fi
    
    
    # Upload to S3 if enabled
    if [ "$UPLOAD_TO_S3" = true ]; then
        log_step "Uploading Benchmarking Results to the target S3 bucket.."
        echo "=================================="
        
        S3_PATH="s3://$S3_BUCKET/benchmarks/$TEAM_ID/$MEMBER_ID/$SUBMISSION_ID/"
        
        log_info "Uploading benchmark metrics CSV file to S3 Bucket..."
        log_info "Destination: ${S3_PATH}${CSV_FILENAME}"
        
        if aws s3 cp "$SCRIPT_DIR/$CSV_FILENAME" "${S3_PATH}${CSV_FILENAME}"; then
            log_info "✓ CSV file uploaded successfully"
        else
            log_error "Failed to upload CSV file to S3"
            EXIT_CODE=1
        fi
        
        # Upload benchmark report if it exists
        if [ -f "$SCRIPT_DIR/benchmark_report.json" ]; then
            log_info "Uploading benchmark report JSON file to S3..."
            if aws s3 cp "$SCRIPT_DIR/benchmark_report.json" "${S3_PATH}benchmark_report.json"; then
                log_info "✓ Benchmark report uploaded successfully"
            else
                log_warn "Failed to upload benchmark report to S3"
            fi
        fi
        
        # Upload any logit files if they exist
        LOGIT_FILES=$(find "$SCRIPT_DIR" -maxdepth 1 -name "expected_logits_*.pt" 2>/dev/null)
        if [ -n "$LOGIT_FILES" ]; then
            log_info "Uploading logit files to S3..."
            LOGIT_COUNT=0
            echo "$LOGIT_FILES" | while read -r logit_file; do
                if [ -f "$logit_file" ]; then
                    filename=$(basename "$logit_file")
                    if aws s3 cp "$logit_file" "${S3_PATH}${filename}"; then
                        LOGIT_COUNT=$((LOGIT_COUNT + 1))
                    fi
                fi
            done
            log_info "✓ Logit files uploaded"
        fi
        
        echo ""
        log_result "QWEN Model Inference performance benchmark file S3 Upload Summary:"
        echo "=================================="
        log_result "Target Bucket: $S3_BUCKET"
        log_result "Path: benchmarks/$TEAM_ID/$MEMBER_ID/$SUBMISSION_ID/"
        log_result "Files uploaded:"
        log_result "  - $CSV_FILENAME"
        [ -f "$SCRIPT_DIR/benchmark_report.json" ] && log_result "  - benchmark_report.json"
        [ -n "$LOGIT_FILES" ] && log_result "  - expected_logits_*.pt files"
        echo ""
        
        log_info "To View uploaded files:"
        echo "  aws s3 ls ${S3_PATH}"
        echo ""
        log_info "To Download files:"
        echo "  aws s3 cp ${S3_PATH} . --recursive"
        echo ""

        # Now remove uploaded CSV file
	    rm "$SCRIPT_DIR/$CSV_FILENAME"
	    log_info "Removed generated metrics file: $SCRIPT_DIR/$CSV_FILENAME"
	    echo "---------------------------------------------------------"
	
    fi
    
else
    log_warn "CSV metrics file not found: $CSV_FILENAME"
    log_info "The CSV inference metrics file may not have been created if evaluation failed"
fi


# Exit with the same code as the Python script
exit $EXIT_CODE
