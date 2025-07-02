
# for i in {1..5}; do
#     echo "Run $i"
#     CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,32-47 \
#     python src/main.py --baseline_run --config=updet-m --env-config=sc2_offline --task-config=toy0 --seed=$i \
#     && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,32-47 \
#     python src/main.py --baseline_run --config=updet-m --env-config=sc2_offline --task-config=toy1 --seed=$i 
# done

# CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,32-47 \
#     python src/main.py --baseline_run --config=updet-m --env-config=sc2_offline \
#     --task-config=toy0 --seed=0 \

for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=0 python src/main.py --baseline_run --config=odis_bc --env-config=sc2_offline \
    --task-config=marine-hard-medium --seed=$i
done