# 🏎️ Orbis: Overcoming Challenges of Long-Horizon Prediction in Driving World Models 🏎️
---

## 🧹 Tokenizer


### 🏋️ How to Train
<!-- Fill this section with training instructions for the tokenizer -->
```bash
python main.py  --base configs/stage1.yaml --name stage1 -t --n_gpus 1 --n_nodes 1
```
cont_ratio_trainig
---
### 📦 Features
#### In the config file:

- `cont_ratio_training`: Sets the ratio at which vector quantization is bypassed, allowing the decoder to receive continuous (non-discrete) tokens.
- `only_decoder`: If set to true, only the decoder is trained while other components remain frozen.


### 🔄 To resume training
```bash
python main.py  --base configs/stage1.yaml --resume PATHTOCKPT -t --n_gpus 1 --n_nodes 1 
```

## 🔮 Next Frame Predictor

### 🏋️ How to Train
<!-- Fill this section with training instructions for the predictor -->
```bash
python main.py  --base configs/stage2.yaml --name stage2 -t --n_gpus 1 --n_nodes 1
```
### 📦 Features
#### `--use_fsdp` enables Fully Sharded Data Parallel (FSDP). The image logger is not supported with this option and will be automatically removed.


### 🔁 How to Do Rollout
<!-- Fill this section with rollout/generation instructions -->
```bash
python evaluate/rollout.py --exp_dir EXPDIR --num_gen_frames 120 --num_videos 200 --num_steps 30
```
### Additional points to consider
- Two variables $TK_WORK_DIR and $WM_WORK_DIR are defined that refer to tokenizer and World Model directory. By setting them logs will be automatically save in the specified directory
- In case you need to modify an old checkpoint you can find a template in `ckpt_to_newckpt.ipynb` and modify the checkpoints to a new one that works with current setup.
- callbacks are added in a config file, placed in `callbacks/configs/base.yaml`.
- Tokenizer:
- - `scale_equivariance` is defined for scale equivarinace loss. If the list is empty it will not apply the loss. Additionally `se_weight` is defiend as SE loss temprerature. 
- - `only_decoder` is set to only train decoder.
- World Model
- - patch size in the second stage is the second stage patchifing, or folding, and does not refer to the first stage patch size.
- - You can use enc_scale_calculator.ipynb to calculate enc_scale variable.
---

## ✅ LOGS

- [ ] The baseline model configuration is available in `stage2_baseline.yaml`.
The corresponding trained model can be found at:
`/data/nxtaimraid02/mousakha/repos/orbis/logs_wm/2025-07-31T19-29-11_rope2_lmbgpu20`
