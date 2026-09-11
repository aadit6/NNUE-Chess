# Second run: resume the same run from its last checkpoint (optimizer, scheduler and epoch
# restored, so the learning rate carries on from 3.25e-4) up to the epoch counter's 120. Lightning
# restarts the counter at the checkpoint's final epoch, so the net ends at 121 epochs.
cd /workspace/nnue-pytorch
source /venv/main/bin/activate
python train.py /dev/shm/data/500M_d9_p*.binpack \
  --features "HalfKAv2_hm^" --l1 1024 \
  --resume-from-checkpoint /workspace/runs/l1024/lightning_logs/version_0/checkpoints/last.ckpt \
  --lambda 1.0 --early-fen-skipping 12 --lr 4.375e-4 --gamma 0.995 \
  --gpus 0 --max-epochs 120 --epoch-size 100000000 --validation-size 1000000 \
  --batch-size 16384 --num-workers 12 --threads 4 \
  --network-save-period 10 --default-root-dir /workspace/runs/l1024

# Then export the float weights the agent loads:
#   python training/export_halfka.py \
#     /workspace/runs/l1024/lightning_logs/version_1/checkpoints/last.ckpt nnue_final.pt \
#     --l1 1024 --repo /workspace/nnue-pytorch
