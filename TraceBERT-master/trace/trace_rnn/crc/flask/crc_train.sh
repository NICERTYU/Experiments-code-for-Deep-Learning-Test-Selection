
ex_path="/root/Experiments/TraceBERT-master/trace/trace_rnn"

cd $ex_path

python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/1-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 1

python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/2-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 2


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/3-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 3


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/4-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 4


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/5-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 5


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/6-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 6


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/7-bigru/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type bi_gru \
--rnn_arch 7



python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/1-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 1

python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/2-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 2


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/3-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 3


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/4-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 4


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/5-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 5


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/6-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 6


python train_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir ./output/7-lstm/flask \
--embd_file_path ./we/proj_embedding.txt \
--exp_name flask \
--valid_step 100 \
--logging_steps 10 \
--gradient_accumulation_steps 8 \
--per_gpu_eval_batch_size 8 \
--num_train_epoch 100 \
--is_embd_trainable \
--hidden_dim 128 \
--rnn_type lstm \
--rnn_arch 7


