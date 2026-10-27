# RestoreBench

A PyTorch dense image-reconstruction experiment: learn 2x super-resolution from synthetically degraded 4x4 grayscale digit images back to their original 8x8 resolution. A small residual CNN is compared with a matched bicubic interpolation baseline.

## Run

Python 3.12, CPU sufficient:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
MPLCONFIGDIR=.mplconfig .venv/bin/python -m unittest -v
MPLCONFIGDIR=.mplconfig .venv/bin/python restorebench.py --output artifacts
```

Data comes from scikit-learn's bundled [load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html): 1,797 real 8x8 grayscale handwritten-digit images. The repository does not redistribute raw images. Generated checkpoints are ignored; published results contain aggregate metrics and metric plots only.

## Design

- Seeded, stratified image split: 1,077 train, 270 validation, 450 test. Validation MSE selects the checkpoint; test images are not used for training or checkpoint selection.
- Fixed degradation: 8x8 targets are area-downsampled to 4x4. Bicubic interpolation with `align_corners=False`, clipped to [0,1], is the common reconstruction baseline.
- A 9,857-parameter, three-convolution residual DNN predicts pixel corrections to the bicubic image. Its zero-initialized final convolution starts exactly at the baseline. Training uses MSE, AdamW, weight decay and paired random horizontal flips for 40 epochs.
- Evaluation clips predictions to [0,1] and reports mean per-image PSNR and global mean MSE. The PSNR computation uses data range 1 and caps exact-zero MSE at 1e-12. It is not PSNR computed from the global average MSE.
- Visualizations compare per-image baseline/CNN PSNR and the gain distribution. Twelve tests cover shapes, constant-image behavior, numerical validation, metrics, split separation, gradients and checkpoint restoration.

## Observed local result

On macOS ARM64 CPU with PyTorch 2.10.0, selected epoch 40:

| Test metric | Bicubic | Residual CNN |
| --- | ---: | ---: |
| Mean per-image PSNR | 13.54 dB | 17.05 dB |
| Mean MSE | 0.04623 | 0.02163 |

Mean per-image gain: 3.51 dB; 446/450 images improved. [Raw metrics](results/metrics.json), [evaluation plot](results/evaluation.png), [research report](docs/REPORT.md).

## Scope

This small digit restoration experiment is not state-of-the-art or natural-photo super-resolution. The split is not writer-disjoint; one seed and no augmentation/architecture ablations limit conclusions. There is no GAN, diffusion model, ViT, inpainting, SVBRDF estimation, neural rendering, real camera degradation, mobile deployment or Huawei product integration. Pixel fidelity on this synthetic task does not establish perceptual quality.

The inactive CI template runs the same offline checks and experiment after dependency installation. Code was developed with AI assistance; inspect it and represent your contribution accurately.
