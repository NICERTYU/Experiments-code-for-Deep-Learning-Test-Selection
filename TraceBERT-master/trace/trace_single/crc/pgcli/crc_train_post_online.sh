
root="/root/Experiments/TraceBERT-master/trace/trace_single"
cd $root


python train_trace_single.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/single_online_34000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 2000 \
--neg_sampling online
