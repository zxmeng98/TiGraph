srun --preempt -p 4test --cpus-per-task=8 --gres=gpu:1 --nodes=1 nsys profile --force-overwrite=true -o /mnt/petrelfs/yezhisheng/nsys/od_execution python train.py --iterations 200
