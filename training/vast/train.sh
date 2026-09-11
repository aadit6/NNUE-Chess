# First run: 60 epochs of 100M positions from scratch. No --resume-from-model and no engine
# test net, so nothing starts from an existing network. Run inside tmux.
cd /workspace/nnue-pytorch
source /venv/main/bin/activate
python train.py /dev/shm/data/500M_d9_p*.binpack \
  --features "HalfKAv2_hm^" --l1 1024 \
  --lambda 1.0 --early-fen-skipping 12 --lr 4.375e-4 --gamma 0.995 \
  --gpus 0 --max-epochs 60 --epoch-size 100000000 --validation-size 1000000 \
  --batch-size 16384 --num-workers 12 --threads 4 \
  --network-save-period 5 --default-root-dir /workspace/runs/l1024
