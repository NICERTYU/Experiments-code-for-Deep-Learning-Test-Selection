ex_path="/root/Experiments/TraceBERT-master/trace/trace_rnn"

cd $ex_path



python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/1-bigru/keras/keras_01-17-10-28-41/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/1-bigru/keras/keras_eval_test/ \
--rnn_type bi_gru   \
--rnn_arch 1


python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/1-lstm/keras/keras_01-17-11-18-46/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/1-lstm/keras/keras_eval_test/   \
--rnn_type lstm \
--rnn_arch 1

python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/2-bigru/keras/keras_01-17-10-35-42/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/2-bigru/keras/keras_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 2




python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/2-lstm/keras/keras_01-17-11-26-07/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/2-lstm/keras/keras_eval_test/   \
--rnn_type lstm \
--rnn_arch 2

python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/3-bigru/keras/keras_01-17-10-42-44/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/3-bigru/keras/keras_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 3




python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/3-lstm/keras/keras_01-17-11-33-29/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/3-lstm/keras/keras_eval_test/   \
--rnn_type lstm \
--rnn_arch 3


python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/4-bigru/keras/keras_01-17-10-49-47/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/4-bigru/keras/keras_eval_test/  \
--rnn_type bi_gru \
--rnn_arch 4




python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/4-lstm/keras/keras_01-17-11-40-51/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/4-lstm/keras/keras_eval_test/  \
--rnn_type lstm \
--rnn_arch 4


python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/5-bigru/keras/keras_01-17-10-56-54/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/5-bigru/keras/keras_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 5




python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/5-lstm/keras/keras_01-17-11-48-24/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/5-lstm/keras/keras_eval_test/  \
--rnn_type lstm \
--rnn_arch 5


python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/6-bigru/keras/keras_01-17-11-04-11/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/6-bigru/keras/keras_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 6




python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/6-lstm/keras/keras_01-17-11-55-55/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/6-lstm/keras/keras_eval_test/   \
--rnn_type lstm \
--rnn_arch 6


python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/7-bigru/keras/keras_01-17-11-11-31/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/7-bigru/keras/keras_eval_test/   \
--rnn_type bi_gru \
--rnn_arch 7




python eval_trace_rnn.py \
--data_dir ../../GitPro/keras-team/keras  \
--model_path ./output/7-lstm/keras/keras_01-17-12-03-32/final_model \
--per_gpu_eval_batch_size 4 \
--exp_name keras  \
--output_dir  ./output/7-lstm/keras/keras_eval_test/   \
--rnn_type lstm \
--rnn_arch 7



