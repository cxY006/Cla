cd ../

python train.py --gpu-id 0 \
                --arch pvt_tiny \
                --lr 0.001 \
                --batch_size 2 \
                --num-labeled 434 \
                --dataset ucm \
                --seed 5 \
