#!/bin/bash
ulimit -c unlimited

# Configuration
export LD_PRELOAD=/home/david/anaconda3/envs/myenv/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home/david/anaconda3/envs/myenv/lib:$LD_LIBRARY_PATH
export LEGATE_AUTO_CONFIG=0
export LEGATE_CONFIG="--cpus 4"
export JULIA_NUM_THREADS=4
export LEGATE_TEST=1
export JULIA_DEBUG=Legate

ITER=0
ITERATIONS=1000
while [ $ITER -lt $ITERATIONS ]; do
    ITER=$((ITER+1))
    echo "Iteration $ITER starting..."
    
    # 1. Multi-threaded control (Minimal)
    echo "[DEBUG] Running with JULIA_NUM_THREADS=$JULIA_NUM_THREADS"
    timeout 120s /pool/david/.julia/juliaup/julia-1.11.5+0.x64.linux.gnu/bin/julia --project=. examples/tasking.jl
    RET=$?
    if [ $RET -ne 0 ]; then
        echo "FAILED: JULIA_NUM_THREADS=$JULIA_NUM_THREADS crashed with code $RET on iteration $ITER"
        exit $RET
    fi

    # 2. Package Tests
    echo "[DEBUG] Running Package Tests..."
    export GPUTESTS=0
    timeout 120s /pool/david/.julia/juliaup/julia-1.11.5+0.x64.linux.gnu/bin/julia --project=. test/runtests.jl
    RET=$?
    if [ $RET -ne 0 ]; then
        echo "FAILED: Package tests crashed with code $RET on iteration $ITER"
        exit $RET
    fi
    
    echo "Iteration $ITER successful."
    echo "----------------------------------------"
done
