CUDA_VISIBLE_DEVICES=0 python src/main.py --baseline_run --config=updet-m --env-config=sc2_offline --task-config=toy0 --seed=0 

# for i in {1..5}; do
#   echo "Run $i"
#   for map in 3m 2s3z 5m_vs_6m; do
#     for suffix in medium expert; do
#       CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,32-47 \
#       python3 src/main.py --config=qmix_bc --env-config=sc2 \
#       with env_args.map_name=${map} h5file_suffix=${suffix}
#     done
#   done
# done
