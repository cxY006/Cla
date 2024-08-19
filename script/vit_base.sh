cd ../

python train.py --gpu-id 0 \
                --arch vit_base_patch16_224 \
                --lr 0.001 \
                --batch-size 2 \
                --num-labeled 434 \
                --dataset ucm \
                --seed 5 \
