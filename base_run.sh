

for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=0 python src/main.py --baseline_run --config=stairs --env-config=sc2_offline \
    --task-config=marine-hard-medium --seed=$i
done