#!/bin/bash
# Table 8 sub-experiment: ZO-SGD with 20% (global) gradient sparsity, OPT-1.3B full fine-tuning on SST2.
# Arguments from zo-bench/sweeps/SST2_opt-1.3b/zo_sgd_sparse_grad/zo_sgd_sparse_grad_cls_ft.yml with the swept
# values lr=1e-6, eps=1e-3, wd=0, sparsity=0.2 (resample steps: code default 1).
# Usage (from the repo root): bash scripts/run_sst2_sparse.sh smoke|full
MODE=${1:-full}
CAP=""; [ "$MODE" = smoke ] && CAP="--max_steps=500"
export WANDB_MODE=offline
cd "$(dirname "$0")/../zo-bench"
python run.py --model_name=facebook/opt-1.3b --task_name=SST2 --output_dir=result/SST2-ft-sparse \
  --num_train_epochs=5 --no_reparam --per_device_train_batch_size=16 --load_best_model_at_end \
  --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 \
  --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification \
  --perturbation_mode=two_side --trainer=zo_sgd --train_set_seed=0 --lr_scheduler_type=constant \
  --eval_steps=500 --save_steps=500 --load_float16 --save_model --sparse_gradient_group=global \
  --learning_rate=1e-6 --zo_eps=1e-3 --weight_decay=0 --gradient_sparsity=0.2 --sparse_gradient_resample_steps=1 $CAP
