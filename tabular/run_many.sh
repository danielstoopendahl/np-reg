#!/bin/bash

echo "vanilla" 

for seed in {10..49}; do
  python har_test.py --hidden-dim 2048 --batch-size 32 --lr 1e-4  --seed $seed | grep "test_acc=" | tail -1 | awk -F'test_acc=' '{print $2}'      
done

echo "layer norm"

for seed in {10..49}; do
  python har_test.py --hidden-dim 2048 --batch-size 32 --lr 1e-4 --layer-norm  --seed $seed | grep "test_acc=" | tail -1 | awk -F'test_acc=' '{print $2}'      
done

echo "batch norm"

for seed in {10..49}; do
  python har_test.py --hidden-dim 2048 --batch-size 32 --lr 1e-4 --batch-norm  --seed $seed | grep "test_acc=" | tail -1 | awk -F'test_acc=' '{print $2}'      
done

echo "np"

for seed in {10..49}; do
  python har_test.py --hidden-dim 2048 --batch-size 32 --lr 1e-4 --np-reg-lambda 0.01 --seed $seed | grep "test_acc=" | tail -1 | awk -F'test_acc=' '{print $2}'      
done

echo "wd"

for seed in {10..49}; do
  python har_test.py --hidden-dim 2048 --batch-size 32 --lr 1e-4 --weight-decay 1e-4 --seed $seed | grep "test_acc=" | tail -1 | awk -F'test_acc=' '{print $2}'      
done

echo "dropout"

for seed in {10..49}; do
  python har_test.py --hidden-dim 2048 --batch-size 64 --lr 3e-4 --dropout 0.1 --seed $seed | grep "test_acc=" | tail -1 | awk -F'test_acc=' '{print $2}'      
done