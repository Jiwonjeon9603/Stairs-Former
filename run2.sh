for i in {1..5}; do
    echo "Run $i"
    CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,32-47 \
    python src/main.py --baseline_run --config=updet-m --env-config=sc2_offline --task-config=toy4 --seed=$i  \
    && CUDA_VISIBLE_DEVICES=2 taskset -c 0-15,32-47 \
    python src/main.py --baseline_run --config=updet-m --env-config=sc2_offline --task-config=toy5 --seed=$i 
done

