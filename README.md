## TPU migration: result curves

This repository records an agent-driven migration of the original code from NVIDIA GPUs to Google Cloud TPUs, part of the study *From GPU to TPU with Coding Agents: What Works, What Fails, and What It Costs*, submitted to the ICSE 2027 Software Engineering in Practice (SEIP) track. The original project's README follows below the figures.

**Full runs: the original code on an NVIDIA L4 GPU and the migrated code on a TPU v5e-1, with the paper's values as references.** ([PDF](migration_figures/curves.pdf))

![curves](migration_figures/curves.png)

**Smoke test before the full run: the original code on the GPU against the migrated code on the TPU.** ([PDF](migration_figures/smoke_compare.pdf))

![smoke_compare](migration_figures/smoke_compare.png)

---

🌄 Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark
====================================================

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[[Paper]](https://arxiv.org/pdf/2402.11592.pdf) [[Code]](https://github.com/ZO-Bench/ZO-LLM) [[Website]](https://sites.google.com/view/zo-tutorial-aaai-2024/)

Official code for the paper **"Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark
"**.

Authors (* Equal Contribution): _[Yihua Zhang](https://www.yihua-zhang.com/)\*, [Pingzhi Li](https://pingzhili.github.io/)\*,
[Junyuan Hong](https://jyhong.gitlab.io/)\*, [Jiaxiang Li](https://jasonjiaxiangli.github.io/)\*,
[Yimeng Zhang](https://damon-demon.github.io/), [Wenqing Zheng](https://wenqing-zheng.github.io/),
[Pin-Yu Chen](https://sites.google.com/site/pinyuchenpage/home), [Jason D. Lee](https://jasondlee88.github.io/),
[Wotao Yin](https://wotaoyin.mathopt.com/), [Mingyi Hong](https://people.ece.umn.edu/~mhong/mingyi.html),
[Zhangyang Wang](https://vita-group.github.io/group.html), [Sijia Liu](https://lsjxjtu.github.io/),
and [Tianlong Chen](https://tianlong-chen.github.io/)_

### Table of Contents

1. **[Overview](#1-overview)**
2. **[Project Structure](#2-project-structure)**
3. **[Getting Started](#3-getting-started)**
4. **[Reproducing Results](#4-reproducing-results)**
5. **[Citation](#5-citation)**

## 1) Overview

This repo contains the source code and reproducing guide of ZO-LLM.
This research endeavor is designed to help researchers better understand the capabilities, limitations and principles
associated with the BP-free, zeroth-order (ZO) optimization as a solution for reducing memory costs during Large
Language Model (LLM) fine-tuning. Our study unveils previously overlooked optimization principles,
highlighting the importance of task alignment, the role of the forward gradient method,
and the balance between algorithm complexity and fine-tuning performance.

This project is organized around the following scopes, including:

1. **Five** LLM families: Roberta, OPT, LLaMA, Vicuna, and Mistral.
2. **Three** task complexities: binary classification, question-answering, and commonsense reasoning.
3. **Four** fine-tuning schemes: full fine-tuning, LoRA, prefix tuning, and prompt tuning.
4. **Six** BP-free optimization methods: ZO-SGD, ZO-SGD-Sign, ZO-SGD-MMT, ZO-SGD-Cons, ZO-Adam, and forward gradient.
5. **Three** novel enhancements to ZO optimization: block-wise descent, hybrid training, and gradient sparsity.

## 2) Project Structure

This project is structured around the hyperparameter sweeping for various tasks & models & tuning schemes & optimization
methods. All optimization methods are implemented in `zo-bench/trainer.py`. Task configurations are defined in
`zo-bench/tasks.py` and `zo-bench/templates.py`. The main entry point is `zo-bench/run.py`.

```
.
├── zo-bench
│   ├── modeling_mistral
│   │   ├─── __init__.py
│   │   ├── configuration_mistral.py
│   │   ├── modleing_mistral.py
│   ├── modeling_llama.py
│   ├── modeling_opt.py
│   ├── modeling_roberta.py
│   ├── prefix_tuning.py
│   ├── prompt_tuning.py
│   ├── run.py
│   ├── tasks.py
│   ├── templates.py
│   ├── test_fake_text_memory.py
│   ├── trainer.py
│   ├── utils.py
│   ├── sweep
│   │   ├── Copa_llama-7b
│   │   │   ├── adam
│   │   │   │   ├── adam_copa_ft.yml
│   │   │   │   ├── adam_copa_lora.yml
│   │   │   │   ├── adam_copa_prefix.yml
│   │   │   │   ├── adam_copa_prompt.yml
│   │   │   ├── forward_grad
│   │   │   │   ├── forward_grad_copa_ft.yml
│   │   │   │   ├── forward_grad_copa_lora.yml
│   │   │   │   ├── forward_grad_copa_prefix.yml
│   │   │   │   ├── forward_grad_copa_prompt.yml
│   │   │   ├── sgd
│   │   │   │   ├── sgd_copa_ft.yml
│   │   │   │   ├── sgd_copa_lora.yml
│   │   │   │   ├── sgd_copa_prefix.yml
│   │   │   │   ├── sgd_copa_prompt.yml
│   │   │   ├── sign_sgd
│   │   │   │   ├── sign_sgd_copa_ft.yml
│   │   │   │   ├── sign_sgd_copa_lora.yml
│   │   │   │   ├── sign_sgd_copa_prefix.yml
│   │   │   │   ├── sign_sgd_copa_prompt.yml
│   │   │   ├── zo_adam
│   │   │   │   ├── zo_adam_copa_ft.yml
│   │   │   │   ├── zo_adam_copa_lora.yml
│   │   │   │   ├── zo_adam_copa_prefix.yml
│   │   │   │   ├── zo_adam_copa_prompt.yml
│   │   │   ├── zo_sgd
│   │   │   │   ├── zo_sgd_copa_ft.yml
│   │   │   │   ├── zo_sgd_copa_lora.yml
│   │   │   │   ├── zo_sgd_copa_prefix.yml
│   │   │   │   ├── zo_sgd_copa_prompt.yml
│   │   │   ├── zo_sgd_conserv
│   │   │   │   ├── zo_sgd_conserv_copa_ft.yml
│   │   │   │   ├── zo_sgd_conserv_copa_lora.yml
│   │   │   │   ├── zo_sgd_conserv_copa_prefix.yml
│   │   │   │   ├── zo_sgd_conserv_copa_prompt.yml
│   │   │   ├── zo_sgd_momen
│   │   │   │   ├── zo_sgd_momen_copa_ft.yml
│   │   │   │   ├── zo_sgd_momen_copa_lora.yml
│   │   │   │   ├── zo_sgd_momen_copa_prefix.yml
│   │   │   │   ├── zo_sgd_momen_copa_prompt.yml
│   │   ├── Copa_llama-13b
│   │   │   ├── ...
│   │   ├── Copa_mistral
│   │   │   ├── ...
│   │   ├── Copa_opt-13b
│   │   │   ├── ...
│   │   ├── Copa_vicuna
│   │   │   ├── ...
│   │   ├── SST2_opt-1.3b
│   │   │   ├── ...
│   │   ├── WinoGrande_llama-7b
│   │   │   ├── ...
│   │   ├── WinoGrande_llama-13b
│   │   │   ├── ...
│   │   ├── WinoGrande_mistral
│   │   │   ├── ...
│   │   ├── WinoGrande_opt-13b
│   │   │   ├── ...
│   │   ├── WinoGrande_vicuna
│   │   │   ├── ...
├── environment.yml
```



## 3) Getting Started

All you need is:

```bash
conda create -n zollm python=3.10
conda activate zollm
pip install -r requirements.txt
```

## 4) Reproducing Results

We provide detailed hyperparameter settings in [sweeps](zo-bench/sweeps), 
where the sweep configuration for tuning a MODEL on TASK under SCHEME with OPTIMIZER is organized as `zo-bench/sweeps/TASK_MODEL/OPTIMIZER/SCHEME.yml`.

An example use of sweep for full fine-tuning LLaMA-7B with ZO-SGD on the COPA task is as follows:

```
~> wandb sweep zo-bench/sweeps/Copa_llama-7b/zo_sgd/zo_sgd_copa_ft.yml
wandb: Creating sweep from: zo-bench/sweeps/Copa_llama-7b/zo_sgd/zo_sgd_copa_ft.yml
wandb: Created sweep with ID: <ID>
wandb: View sweep at: https://wandb.ai/<unique ID>
wandb: Run sweep agent with: wandb agent <unique ID>
~> wandb agent <unique ID>
```
For the Extended Study, please check the following code:

* **Block-wise ZO**: Add the argument `--module_wise_perturbation=True` to the command line. Note that temporarily we only support OPT family models. For example:
```
python run.py --model_name=facebook/opt-1.3b --task_name=SST2 --output_dir=result/SST2-ft-$TAG --num_train_epochs=5 \
--per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps \
--save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 \
--num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=zo_sgd --train_set_seed=0 \
--lr_scheduler_type=constant --save_steps=1000 --load_float16 --learning_rate=1e-8 --zo_eps=0.001 --momentum=0.9 
--weight_decay=0 --module_wise_perturbation=True
```

* **Gradient Pruning**: The corresponding arguments are `sparse_gradient_group`, `gradient_sparsity` and `sparse_gradient_resample_steps`. The options of them can be found in the correpsonding comments in `run.py`. An example sweep configuration is in `zo-bench/sweeps/SST2_opt-1.3b/zo_sgd_sparse_grad/zo_sgd_sparse_grad_cls_ft.yml`.


## 5) Citation

```
@misc{zhang2024revisiting,
      title={Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark}, 
      author={Yihua Zhang and Pingzhi Li and Junyuan Hong and Jiaxiang Li and Yimeng Zhang and Wenqing Zheng and Pin-Yu Chen and Jason D. Lee and Wotao Yin and Mingyi Hong and Zhangyang Wang and Sijia Liu and Tianlong Chen},
      year={2024},
      eprint={2402.11592},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
