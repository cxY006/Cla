cd ../

python train.py --gpu-id 0 \
                --arch pvt_v2_b3 \
                --lr 0.001 \
                --batch_size 4 \
                --num-labeled 168 \
                --dataset ucm \
                --seed 5 \

