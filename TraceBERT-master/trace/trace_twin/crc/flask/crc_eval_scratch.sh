path="../../"

cd $path


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin1/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin1/flask/test \
--twin_type   1


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin2/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin2/flask/test \
--twin_type   2


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin3/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin3/flask/test \
--twin_type   3


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin4/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin4/flask/test \
--twin_type   4


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin5/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin5/flask/test \
--twin_type   5


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin6/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin6/flask/test \
--twin_type   6


python eval_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--model_path /hy-tmp/twin7/flask/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin7/flask/test \
--twin_type   7
