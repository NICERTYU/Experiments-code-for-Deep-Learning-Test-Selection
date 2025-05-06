path="../../"

cd $path


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin1/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin1/keras/test \
--twin_type   1


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin2/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin2/keras/test \
--twin_type   2


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin3/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin3/keras/test \
--twin_type   3


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin4/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin4/keras/test \
--twin_type   4


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin5/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin5/keras/test \
--twin_type   5


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin6/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin6/keras/test \
--twin_type   6


python eval_trace_twin.py \
--data_dir ../../GitPro/keras-team/keras \
--model_path /hy-tmp/twin7/keras/train/final_model   \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin7/keras/test \
--twin_type   7
