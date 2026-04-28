#!/bin/bash

for seed in {10..19}; do
	python bow_imdb.py --seed "$seed" --hidden-dim 256 --vocab-size 1562 --batch-size 64 --lr 3e-6 | grep -oP 'Test accuracy=\K[0-9.]+(?=%)'
done


for seed in {10..19}; do
	python bow_imdb.py --seed "$seed" --hidden-dim 256 --vocab-size 1562 --batch-size 32 --lr 3e-6 --layer-norm | grep -oP 'Test accuracy=\K[0-9.]+(?=%)'
done

for seed in {10..19}; do
	python bow_imdb.py --seed "$seed" --hidden-dim 256 --vocab-size 1562 --batch-size 64 --lr 1e-5 --batch-norm | grep -oP 'Test accuracy=\K[0-9.]+(?=%)'
done

for seed in {10..19}; do
	python bow_imdb.py --seed "$seed" --hidden-dim 256 --vocab-size 1562 --batch-size 64 --lr 3e-6 --weight-decay 1e-4 | grep -oP 'Test accuracy=\K[0-9.]+(?=%)'
done

for seed in {10..19}; do
	python bow_imdb.py --seed "$seed" --hidden-dim 256 --vocab-size 1562 --batch-size 32 --lr 3e-6 --dropout 0.1 | grep -oP 'Test accuracy=\K[0-9.]+(?=%)'
done

for seed in {10..19}; do
	python bow_imdb.py --seed "$seed" --hidden-dim 256 --vocab-size 1562 --batch-size 64 --lr 3e-5 --np-reg-lambda 0.01 | grep -oP 'Test accuracy=\K[0-9.]+(?=%)'
done