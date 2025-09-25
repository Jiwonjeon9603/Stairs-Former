# STAIRS: Spatio-Temporal Attention with Interleaved Recursive Structure

This repository provides the official implementation of **STAIRS**, our offline multi-agent reinforcement learning algorithm.  
The code is adapted from the publicly available [ODIS implementation](https://github.com/LAMDA-RL/ODIS).

---

## Installation

1. **StarCraft II and SMAC**

```bash
bash install_sc2.sh
export SC2PATH=/abs/path/to/3rdparty/StarCraftII
```

2. **Python environment**

```bash
conda create -n stairs python=3.10 -y
conda activate stairs
pip install -r requirements.txt
```

3. **SMAC patch**

```bash
git clone https://github.com/oxwhirl/smac.git
pip install -e smac/
bash install_smac_patch.sh
```

---

## Running STAIRS on SMAC

```bash
python src/main.py --baseline_run --config=stairs --env-config=sc2_offline --task-config=marine-hard-expert --seed=1
```

---

## Running STAIRS on Cooperative Navigation (CN)

```bash
python src/main.py --baseline_run --config=stairs --env-config=cn_offline --task-config=cn-expert --seed=1
```

---

## Data Collection (optional)

```bash
python src/main.py --data_collect --config=qmix --env-config=sc2_collect \
  --offline_data_quality=expert --num_episodes_collected=2000 \
  --map_name=5m_vs_6m --save_replay_buffer=False
```

---

## License
This code is released under the **Apache License 2.0**.
