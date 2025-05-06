

ex_path="/root/Experiments/TraceBERT-master/trace/trace_rnn"

cd $ex_path



python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/1-bigru/pgcli/pgcli_01-17-10-28-41/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/1-bigru/pgcli/pgcli_eval_test/ \
--rnn_type bi_gru   \
--rnn_arch 1


python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/1-lstm/pgcli/pgcli_01-17-11-16-12/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/1-lstm/pgcli/pgcli_eval_test/   \
--rnn_type lstm \
--rnn_arch 1

python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/2-bigru/pgcli/pgcli_01-17-10-35-23/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/2-bigru/pgcli/pgcli_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 2




python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/2-lstm/pgcli/pgcli_01-17-11-23-09/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/2-lstm/pgcli/pgcli_eval_test/   \
--rnn_type lstm \
--rnn_arch 2

python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/3-bigru/pgcli/pgcli_01-17-10-42-05/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/3-bigru/pgcli/pgcli_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 3




python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/3-lstm/pgcli/pgcli_01-17-11-30-07/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/3-lstm/pgcli/pgcli_eval_test/   \
--rnn_type lstm \
--rnn_arch 3


python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/4-bigru/pgcli/pgcli_01-17-10-48-42/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/4-bigru/pgcli/pgcli_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 4




python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/4-lstm/pgcli/pgcli_01-17-11-36-56/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/4-lstm/pgcli/pgcli_eval_test/  \
--rnn_type lstm \
--rnn_arch 4


python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/5-bigru/pgcli/pgcli_01-17-10-55-31/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/5-bigru/pgcli/pgcli_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 5




python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/5-lstm/pgcli/pgcli_01-17-11-44-06/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/5-lstm/pgcli/pgcli_eval_test/  \
--rnn_type lstm \
--rnn_arch 5


python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/6-bigru/pgcli/pgcli_01-17-11-02-21/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/6-bigru/pgcli/pgcli_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 6




python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/6-lstm/pgcli/pgcli_01-17-11-51-11/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/6-lstm/pgcli/pgcli_eval_test/   \
--rnn_type lstm \
--rnn_arch 6


python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/7-bigru/pgcli/pgcli_01-17-11-09-22/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/7-bigru/pgcli/pgcli_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 7




python eval_trace_rnn.py \
--data_dir ../../GitPro/dbcli/pgcli  \
--model_path ./output/7-lstm/pgcli/pgcli_01-17-11-58-24/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name pgcli  \
--output_dir  ./output/7-lstm/pgcli/pgcli_eval_test/   \
--rnn_type lstm \
--rnn_arch 7



