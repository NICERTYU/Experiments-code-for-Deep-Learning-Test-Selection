# path="../../"

# cd $path


# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin1/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 1


# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin2/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 2


# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin3/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 3


# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin4/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 4



# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin5/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 5


# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin6/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 6


# python train_trace_twin.py \
# --data_dir ../../GitPro/pallets/flask \
# --output_dir /hy-tmp/twin7/flask/train \
# --per_gpu_train_batch_size 4 \
# --per_gpu_eval_batch_size 4 \
# --logging_steps 10 \
# --save_steps 2000 \
# --gradient_accumulation_steps 16 \
# --num_train_epochs 400 \
# --learning_rate 4e-5 \
# --valid_step 1000 \
# --neg_sampling online \
# --twin_type 7



path="../../"

cd $path


python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin1/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 1   \
--code_bert   ./bert/robert


python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin2/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 2 \
--code_bert   ./bert/robert


python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin3/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 3  \
--code_bert   ./bert/robert


python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin4/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 4 \
--code_bert   ./bert/robert



python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin5/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 5  \
--code_bert   ./bert/robert


python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin6/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 6   \
--code_bert   ./bert/robert


python train_trace_twin.py \
--data_dir ../../GitPro/pallets/flask \
--output_dir /hy-tmp/twin7/flask/train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 4 \
--logging_steps 10 \
--save_steps 2000 \
--gradient_accumulation_steps 16 \
--num_train_epochs 400 \
--learning_rate 4e-5 \
--valid_step 1000 \
--neg_sampling online \
--twin_type 7   \
--code_bert   ./bert/robert
