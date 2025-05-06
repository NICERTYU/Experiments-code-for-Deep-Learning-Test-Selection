ex_path="/root/Experiments/TraceBERT-master/trace/trace_rnn"

cd $ex_path



python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/1-bigru/flask/flask_01-17-07-59-24/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/1-bigru/flask/flask_eval_test/ \
--rnn_type bi_gru \
--rnn_arch 1


python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/1-lstm/flask/flask_01-17-09-09-38/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/1-lstm/flask/flask_eval_test/  \
--rnn_type lstm \
--rnn_arch 1

python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/2-bigru/flask/flask_01-17-08-08-32/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/2-bigru/flask/flask_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 2




python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/2-lstm/flask/flask_01-17-09-18-34/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/2-lstm/flask/flask_eval_test/  \
--rnn_type lstm \
--rnn_arch 2

python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/3-bigru/flask/flask_01-17-08-17-31/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/3-bigru/flask/flask_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 3




python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/3-lstm/flask/flask_01-17-09-27-34/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/3-lstm/flask/flask_eval_test/   \
--rnn_type lstm \
--rnn_arch 3


python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/4-bigru/flask/flask_01-17-08-26-34/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/4-bigru/flask/flask_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 4




python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/4-lstm/flask/flask_01-17-09-36-25/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/4-lstm/flask/flask_eval_test/   \
--rnn_type lstm \
--rnn_arch 4


python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/5-bigru/flask/flask_01-17-08-36-10/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/5-bigru/flask/flask_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 5




python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/5-lstm/flask/flask_01-17-09-58-23/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/5-lstm/flask/flask_eval_test/   \
--rnn_type lstm \
--rnn_arch 5


python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/6-bigru/flask/flask_01-17-08-47-15/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/6-bigru/flask/flask_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 6




python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/6-lstm/flask/flask_01-17-10-07-35/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/6-lstm/flask/flask_eval_test/   \
--rnn_type lstm \
--rnn_arch 6


python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/7-bigru/flask/flask_01-17-08-59-45/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/7-bigru/flask/flask_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 7




python eval_trace_rnn.py \
--data_dir ../../GitPro/pallets/flask \
--model_path ./output/7-lstm/flask/flask_01-17-10-17-57/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name flask  \
--output_dir  ./output/7-lstm/flask/flask_eval_test/   \
--rnn_type lstm \
--rnn_arch 7



