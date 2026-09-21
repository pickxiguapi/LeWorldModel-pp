# LeWM JAX with OGBench IMPALA-small

This directory contains one implementation only: LeWM with OGBench's native
`encoder_modules['impala_small']()` visual encoder. There is no alternate encoder factory.

Only the visual encoder is replaced. The remaining model and objective retain
the LeWM design:

- a 192-dimensional latent representation;
- a two-layer 192 → 2048 → 192 projection MLP with BatchNorm and GELU;
- the LeWM action embedder;
- a six-layer action-conditioned autoregressive predictor with 16 heads;
- prediction MSE plus SIGReg (weight 0.09, 17 knots, 1024 projections);
- three history frames, one future prediction, and frameskip/action chunk 5;
- AdamW at 5e-5, weight decay 1e-3, global gradient clipping at 1.0;
- bf16 compute and 10 training epochs by default.

The Lance loader returns raw `uint8` RGB NHWC images. OGBench's IMPALA encoder
performs `/255` internally, exactly as it does for the existing visual agents.

Training:

```bash
cd impls
python train_lewm_control.py \
  --dataset_path=/path/to/task.lance \
  --save_dir=/path/to/run \
  --exp_name=LeWMJAX_impala_task_bs128_e10
```

Evaluation reconstructs the same model only when the checkpoint architecture
is `lewm_impala_small`; checkpoints from other architectures fail explicitly.
