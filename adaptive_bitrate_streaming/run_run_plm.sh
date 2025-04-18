# python run_plm.py \
#     --train \
#     --seed 666 \
#     --plm-type llama \
#     --plm-size tiny \
#     --device cuda:0 \
#     --device-out cuda:0 \
#     --state-feature-dim 256 \
#     --K 20 \
#     --gamma 1.0 \
#     --lr 0.0001 \
#     --num-iters 2 \
#     --num-steps-per-iter 101 \

export CUDA_VISIBLE_DEVICES=1,2,3
python run_plm.py \
    --adapt \
    --grad-accum-steps 32 \
    --plm-type llama \
    --plm-size base \
    --rank 128 \
    --device cuda:0 \
    --lr 0.0001 \
    --warmup-steps 2000 \
    --num-epochs 1 \
    --eval-per-epoch 2