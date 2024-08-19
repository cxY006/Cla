cd ../

python train.py --gpu-id 0 \
                --arch pvt_v2_b0 \
                --lr 0.001 \
                --batch_size 16 \
                --num-labeled 434 \
                --dataset ucm \
                --seed 5 \

