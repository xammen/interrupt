#!/bin/bash
# Run a single experiment: model, condition, run_index
# Usage: bash run_single.sh <model_id> <model_short> <condition> <run_index> <timeout_seconds>

MODEL_ID="$1"
MODEL_SHORT="$2"
CONDITION="$3"
RUN_INDEX="$4"
INTERRUPT_TIMEOUT="${5:-20}"

STUDY_DIR="C:/Users/xammen/interrupt-study"
RESULTS_DIR="${STUDY_DIR}/results/${MODEL_SHORT}_${CONDITION}"
WORKDIR="${STUDY_DIR}/workdirs/${MODEL_SHORT}_${CONDITION}/run_${RUN_INDEX}"

PROMPT="Write a Python module called task_scheduler.py that implements an async task scheduler with the following features: 1. A Task dataclass with fields: id str, name str, priority int 1-10, dependencies list of task IDs, status enum PENDING RUNNING COMPLETED FAILED, retry_count int, max_retries int default 3, created_at datetime, result Optional Any. 2. A TaskScheduler class that: Maintains a priority queue of tasks, Resolves dependency graphs before execution topological sort, Detects circular dependencies and raises CircularDependencyError, Runs independent tasks concurrently using asyncio, Implements exponential backoff retry logic for failed tasks, Has a configurable concurrency limit max parallel tasks, Emits events via an observer pattern on_task_start on_task_complete on_task_fail, Provides a get_execution_plan method that returns the ordered execution groups, Tracks execution metrics total time per-task time retry count. 3. Include proper error handling type hints and docstrings. 4. Write at least 5 unit tests using pytest that cover: Basic task execution, Dependency resolution, Circular dependency detection, Retry logic with exponential backoff, Concurrent execution respecting concurrency limits. Write the complete code in a single response. Do not ask questions just write the code."

mkdir -p "$RESULTS_DIR" "$WORKDIR"

echo "[$(date +%H:%M:%S)] ${MODEL_SHORT}/${CONDITION}/run_${RUN_INDEX} - Starting..."

case "$CONDITION" in
  complete)
    START=$(date +%s)
    cd "$WORKDIR" && opencode run --model "$MODEL_ID" "$PROMPT" > "${RESULTS_DIR}/run_${RUN_INDEX}.txt" 2>&1
    END=$(date +%s)
    DURATION=$((END - START))
    
    # Copy generated files
    cp "$WORKDIR"/task_scheduler.py "${RESULTS_DIR}/run_${RUN_INDEX}_task_scheduler.py" 2>/dev/null
    cp "$WORKDIR"/test_task_scheduler.py "${RESULTS_DIR}/run_${RUN_INDEX}_test_task_scheduler.py" 2>/dev/null
    
    echo "{\"model\":\"${MODEL_ID}\",\"condition\":\"complete\",\"run_index\":${RUN_INDEX},\"duration_seconds\":${DURATION}}" > "${RESULTS_DIR}/run_${RUN_INDEX}_meta.json"
    echo "[$(date +%H:%M:%S)] DONE: ${DURATION}s"
    ;;
    
  interrupted)
    START=$(date +%s)
    
    # Phase 1: Start generation, kill after timeout
    cd "$WORKDIR" && timeout ${INTERRUPT_TIMEOUT} opencode run --model "$MODEL_ID" "$PROMPT" > "${RESULTS_DIR}/run_${RUN_INDEX}_partial.txt" 2>&1
    echo "[$(date +%H:%M:%S)]   Phase 1: Killed after ${INTERRUPT_TIMEOUT}s"
    
    # Clean workdir for fresh generation
    rm -f "$WORKDIR"/task_scheduler.py "$WORKDIR"/test_task_scheduler.py 2>/dev/null
    
    # Phase 2: Fresh re-generation (same prompt, no fragment)
    cd "$WORKDIR" && opencode run --model "$MODEL_ID" "$PROMPT" > "${RESULTS_DIR}/run_${RUN_INDEX}.txt" 2>&1
    END=$(date +%s)
    DURATION=$((END - START))
    
    cp "$WORKDIR"/task_scheduler.py "${RESULTS_DIR}/run_${RUN_INDEX}_task_scheduler.py" 2>/dev/null
    cp "$WORKDIR"/test_task_scheduler.py "${RESULTS_DIR}/run_${RUN_INDEX}_test_task_scheduler.py" 2>/dev/null
    
    PARTIAL_SIZE=$(wc -c < "${RESULTS_DIR}/run_${RUN_INDEX}_partial.txt" 2>/dev/null || echo 0)
    echo "{\"model\":\"${MODEL_ID}\",\"condition\":\"interrupted\",\"run_index\":${RUN_INDEX},\"duration_seconds\":${DURATION},\"interrupt_timeout\":${INTERRUPT_TIMEOUT},\"partial_output_bytes\":${PARTIAL_SIZE}}" > "${RESULTS_DIR}/run_${RUN_INDEX}_meta.json"
    echo "[$(date +%H:%M:%S)] DONE: ${DURATION}s"
    ;;
    
  fragment_continue)
    START=$(date +%s)
    
    # Phase 1: Start generation, kill after timeout to get a fragment
    cd "$WORKDIR" && timeout ${INTERRUPT_TIMEOUT} opencode run --model "$MODEL_ID" "$PROMPT" > "${RESULTS_DIR}/run_${RUN_INDEX}_phase1.txt" 2>&1
    echo "[$(date +%H:%M:%S)]   Phase 1: Got fragment"
    
    # Grab whatever code files were partially written
    FRAGMENT=""
    if [ -f "$WORKDIR/task_scheduler.py" ]; then
      FRAGMENT=$(cat "$WORKDIR/task_scheduler.py")
    fi
    
    # Save fragment
    echo "$FRAGMENT" > "${RESULTS_DIR}/run_${RUN_INDEX}_fragment.txt"
    
    # Phase 2: Feed fragment back and ask to continue
    CONTINUE_PROMPT="I was generating code but got interrupted. Here is what I had so far. Please continue EXACTLY from where this stops and complete the task_scheduler.py module AND write the test file. Do not repeat existing code. Keep the same style. Here is the partial code: ${FRAGMENT} --- END OF PARTIAL CODE --- Now complete the module and write the tests."
    
    # Clean and re-run
    rm -f "$WORKDIR"/task_scheduler.py "$WORKDIR"/test_task_scheduler.py 2>/dev/null
    cd "$WORKDIR" && opencode run --model "$MODEL_ID" "$CONTINUE_PROMPT" > "${RESULTS_DIR}/run_${RUN_INDEX}_phase2.txt" 2>&1
    END=$(date +%s)
    DURATION=$((END - START))
    
    # The combined result is whatever files are now in workdir
    cp "$WORKDIR"/task_scheduler.py "${RESULTS_DIR}/run_${RUN_INDEX}_task_scheduler.py" 2>/dev/null
    cp "$WORKDIR"/test_task_scheduler.py "${RESULTS_DIR}/run_${RUN_INDEX}_test_task_scheduler.py" 2>/dev/null
    
    # Also combine text outputs
    cat "${RESULTS_DIR}/run_${RUN_INDEX}_phase1.txt" "${RESULTS_DIR}/run_${RUN_INDEX}_phase2.txt" > "${RESULTS_DIR}/run_${RUN_INDEX}.txt"
    
    FRAG_SIZE=${#FRAGMENT}
    echo "{\"model\":\"${MODEL_ID}\",\"condition\":\"fragment_continue\",\"run_index\":${RUN_INDEX},\"duration_seconds\":${DURATION},\"interrupt_timeout\":${INTERRUPT_TIMEOUT},\"fragment_size\":${FRAG_SIZE}}" > "${RESULTS_DIR}/run_${RUN_INDEX}_meta.json"
    echo "[$(date +%H:%M:%S)] DONE: ${DURATION}s (fragment: ${FRAG_SIZE} chars)"
    ;;
esac
