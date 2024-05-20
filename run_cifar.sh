python3 run_cifar.py --epochs=1500 --batch_size=32 --d_model=128 \
 --lr=0.004 --lr_min=0.001 --weight_decay=0.01 --wd=0.01 \
 --lr_dt=0.001 --min_dt=0.001 --max_dt=10 \
 --epochs_scheduler=200 --warmup=10