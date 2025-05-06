cd ../

python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin1/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin1/pgcli/test \
--twin_type   1


python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin2/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin2/pgcli/test  \
--twin_type   2



python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin3/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin3/pgcli/test \
--twin_type   3


python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin4/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin4/pgcli/test \
--twin_type   4


python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin5/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin5/pgcli/test \
--twin_type   5


python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin6/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin6/pgcli/test \
--twin_type   6


python eval_trace_twin.py \
--data_dir ../../GitPro/dbcli/pgcli \
--model_path /hy-tmp/twin7/pgcli/train/final_model \
--per_gpu_eval_batch_size 4 \
--output_dir ./robert/twin7/pgcli/test \
--twin_type   7

