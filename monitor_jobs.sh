#!/bin/bash
# Monitor both Option A and Option B jobs in real-time

echo "================================================================"
echo "Monitoring Experiment 3: Option A vs Option B"
echo "================================================================"
echo ""

# Get job IDs (latest two jobs)
JOBS=$(squeue -u $USER -h --format="%i %j" | grep -E "Exp3_Opt[AB]" | awk '{print $1}')
JOB_A=$(squeue -u $USER -h --format="%i %j" | grep "Exp3_OptA" | awk '{print $1}')
JOB_B=$(squeue -u $USER -h --format="%i %j" | grep "Exp3_OptB" | awk '{print $1}')

if [ -z "$JOB_A" ] && [ -z "$JOB_B" ]; then
    echo "✓ Both jobs completed!"
    echo ""
    
    # Check if logs exist
    if [ -f "artifacts_optA/audit_log.jsonl" ] && [ -f "artifacts_optB/audit_log.jsonl" ]; then
        echo "Running comparison analysis..."
        python3 compare_results.py
    else
        echo "⚠️  Audit logs not found yet. Jobs may have failed."
        echo ""
        echo "Check error logs:"
        echo "  cat logs/optA_*.err"
        echo "  cat logs/optB_*.err"
    fi
    exit 0
fi

echo "Job Status:"
squeue -u $USER --format="%.18i %.12j %.8T %.10M %.6D %.20R" | grep -E "JOBID|Exp3_Opt"
echo ""

# Show live progress from logs
if [ -n "$JOB_A" ]; then
    echo "─────────────────────────────────────────────────────────────────"
    echo "Option A Progress (Job $JOB_A):"
    echo "─────────────────────────────────────────────────────────────────"
    if [ -f "logs/optA_${JOB_A}.log" ]; then
        tail -n 20 logs/optA_${JOB_A}.log | grep -E "Epoch|accuracy|loss|ECE|Started|Finished" || echo "  (starting up...)"
    else
        echo "  Log not created yet..."
    fi
    echo ""
fi

if [ -n "$JOB_B" ]; then
    echo "─────────────────────────────────────────────────────────────────"
    echo "Option B Progress (Job $JOB_B):"
    echo "─────────────────────────────────────────────────────────────────"
    if [ -f "logs/optB_${JOB_B}.log" ]; then
        tail -n 20 logs/optB_${JOB_B}.log | grep -E "Epoch|accuracy|loss|ECE|Started|Finished" || echo "  (starting up...)"
    else
        echo "  Log not created yet..."
    fi
    echo ""
fi

echo "================================================================"
echo "Monitoring commands:"
echo "  Watch Option A: tail -f logs/optA_${JOB_A}.log"
echo "  Watch Option B: tail -f logs/optB_${JOB_B}.log"
echo "  Check status:   squeue -u \$USER"
echo "  Re-run monitor: ./monitor_jobs.sh"
echo "================================================================"
