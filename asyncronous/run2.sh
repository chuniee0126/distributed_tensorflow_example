#!/bin/bash
python ASYNC.py \
    --ps_hosts=localhost:2222,localhost:2223 \
    --worker_hosts=localhost:2224,localhost:2225 \
    --job_name=ps --task_index=1 --gpu_index=2
