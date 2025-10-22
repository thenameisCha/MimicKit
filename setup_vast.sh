# !/usr/bin/env bash
set -euo pipefail

# --- Find conda and load its shell hook (works even if .bashrc wasn't modified) ---
source /opt/miniforge3/etc/profile.d/conda.sh
# ---- 1) Install Python 3.8 via deadsnakes PPA (system Python) ----
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -y
sudo apt-get install -y python3.8 python3.8-distutils python3.8-venv

# ---- 2) Create conda env with Python 3.8 ----
if conda env list | awk '{print $1}' | grep -qx "isaac_env"; then
  echo "Conda env 'isaac_env' already exists. Skipping creation."
else
  conda create -n isaac_env python=3.8 -y
fi

# ---- 3) Create ~/.bash_aliases with requested alias ----
# (exact string you provided)
mkdir -p "$HOME"
echo "alias iv='conda activate isaac_env && cd workspace/IGRIS_GYM'" > "$HOME/.bash_aliases"
echo "alias train='conda activate isaac_env && cd workspace/IGRIS_GYM/legged_gym/scripts && python train.py'" >> "$HOME/.bash_aliases"
conda activate isaac_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /workspace && tar -xzf /workspace/IsaacGym_Preview_4_Package.tar.gz
cd /workspace/isaacgym/python && pip install -e .
cd /workspace/IGRIS_GYM/rsl_rl && pip install -e .
cd ../ && pip install -e .

cd /workspace/MimicKit && pip install -r requirements.txt