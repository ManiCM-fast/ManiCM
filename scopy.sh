alg_name=${1}
task_name=${2}
config_name=${alg_name}
teacher_addition_info=${3}
addition_info=${4}
seed=${5}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"
teacher_exp_name=${task_name}-dp3-${teacher_addition_info}

cd ManiCM/data/outputs/
mkdir ${exp_name}_seed${seed}
cd ${exp_name}_seed${seed}
mkdir checkpoints
cd ..
mv ${teacher_exp_name}_seed${seed}/checkpoints/latest.ckpt ${exp_name}_seed${seed}/checkpoints