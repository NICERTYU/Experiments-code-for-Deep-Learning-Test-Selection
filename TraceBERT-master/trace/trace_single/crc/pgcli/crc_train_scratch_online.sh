root="/root/Experiments/TraceBERT-master/trace/trace_single"
cd $root

python train_trace_single.py \
--data_dir ../../GitPro/dbcli/pgcli \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online
