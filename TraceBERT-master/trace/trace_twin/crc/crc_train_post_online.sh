path=".."

cd $path


python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output/twin1/pgcli/ \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  1



python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output/twin1/pgcli/ \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  2


python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  3


python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  4


python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  5


python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  6


python train_trace_twin.py \
--data_dir ../data/git_data/dbcli/pgcli \
--model_path ../pretrained_model/twin_online_28000 \
--output_dir ./output \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 50 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 500 \
--neg_sampling online \
--twin_type  7
