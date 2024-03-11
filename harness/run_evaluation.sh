#!/bin/bash
python harness/run_evaluation.py \
    --predictions_path "outputs/gpt-3.5-turbo-1106__SWE-bench_oracle__test.jsonl" \
    --swe_bench_tasks "swe_bench_tasks.json" \
    --log_dir "outputs" \
    --testbed "testbed" \
    --skip_existing \
    --timeout 900 \
    --verbose
