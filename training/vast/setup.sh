# One-time setup on the vast.ai box (RTX 4090 48 GB, torch 2.11 + CUDA 12.8 preinstalled in
# /venv/main). Builds nnue-pytorch's C++ data loader. uv needed --system-certs behind the host's
# TLS proxy.
set -exo pipefail
cd /workspace
[ -d nnue-pytorch ] || git clone https://github.com/official-stockfish/nnue-pytorch
cd nnue-pytorch
git log -1 --format='%h %cd'   # trained at 9f72946 (July 2026)
source /venv/main/bin/activate
uv pip install --system-certs -r requirements.txt cupy-cuda12x
bash compile_data_loader.sh
python train.py --help > /dev/null && echo SETUP OK
