#!/bin/bash
python ASYNC.py \
    --ps_hosts=localhost:2222,localhost:2223 \
    --worker_hosts=localhost:2224,localhost:2225 \
    --job_name=worker --task_index=0 --gpu_index=3
