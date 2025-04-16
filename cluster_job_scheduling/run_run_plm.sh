# no need changed to rank and defaults to 128
    # --peft-rank 128 \
# no longer available through args 
    # --freeze-encoder
# lets run with no test first
    # --test \


python run_plm.py \
    --train \
    --seed 666 \
    --plm-type llama \
    --plm-size tiny \
    --device cuda:0 \
    --device-out cuda:0 \
    --state-feature-dim 256 \
    --K 20 \
    --gamma 1.0 \
    --lr 0.0001 \
    --num-iters 2 \
    --num-steps-per-iter 101 \