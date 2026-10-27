"""Small 4x4-to-8x8 learned image reconstruction benchmark, not natural-photo SR."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

SEED = 31


def data():
    digits = load_digits()
    images = torch.tensor(digits.images / 16, dtype=torch.float32).unsqueeze(1)
    train, test = train_test_split(np.arange(len(images)), test_size=0.25, stratify=digits.target, random_state=SEED)
    train, validation = train_test_split(train, test_size=0.2, stratify=digits.target[train], random_state=SEED)
    return images, train, validation, test


def degrade(high):
    if high.ndim != 4 or tuple(high.shape[1:]) != (1, 8, 8) or not torch.isfinite(high).all():
        raise ValueError('expected finite N x 1 x 8 x 8 images')
    return F.interpolate(high, size=(4, 4), mode='area')


def bicubic(low):
    return F.interpolate(low, size=(8, 8), mode='bicubic', align_corners=False).clamp(0, 1)


class RestoreCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 1, 3, padding=1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, low):
        if low.ndim != 4 or tuple(low.shape[1:]) != (1, 4, 4) or not torch.isfinite(low).all():
            raise ValueError('expected finite N x 1 x 4 x 4 images')
        base = bicubic(low)
        return base + self.residual(base)


def metrics(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 4 or not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError('expected matching finite image batches')
    mse = (prediction - target).square().mean((1, 2, 3))
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    return {'mean_mse': mse.mean().item(), 'mean_image_psnr_db': psnr.mean().item()}, psnr


def fit(images, train, validation, epochs=40):
    torch.manual_seed(SEED)
    model = RestoreCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    best, selected, state, history = float('inf'), 0, None, []
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(train)).split(64):
            high = images[train[indices.numpy()]].clone()
            # Paired augmentation: flip target before deriving its low-resolution input.
            flip = torch.rand(len(high)) < 0.5
            high[flip] = high[flip].flip(-1)
            loss = F.mse_loss(model(degrade(high)), high)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted = model(degrade(images[validation])).clamp(0, 1)
            score = metrics(predicted, images[validation])[0]['mean_mse']
        history.append({'epoch': epoch + 1, 'validation_mse': score})
        if score < best:
            best, selected = score, epoch + 1
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), selected, history


def experiment(output):
    torch.set_num_threads(1)
    images, train, validation, test = data()
    model, selected, history = fit(images, train, validation)
    truth = images[test]
    low = degrade(truth)
    with torch.inference_mode():
        baseline = bicubic(low)
        learned = model(low).clamp(0, 1)
    base_stats, base_psnr = metrics(baseline, truth)
    learned_stats, learned_psnr = metrics(learned, truth)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / 'restoration.pt')
    result = {'task': 'synthetic 2x degradation/reconstruction of real digit images, 4x4 -> 8x8',
              'seed': SEED, 'train_images': len(train), 'validation_images': len(validation), 'test_images': len(test),
              'epochs': 40, 'selected_epoch': selected, 'parameters': sum(p.numel() for p in model.parameters()),
              'bicubic': base_stats, 'cnn': learned_stats,
              'mean_psnr_gain_db': float((learned_psnr - base_psnr).mean()),
              'images_improved': int((learned_psnr > base_psnr).sum()), 'history': history,
              'limitations': ['Small grayscale digit data with synthetic area downsampling; no natural-photo or mobile validation.',
                              'Random image split, not writer-disjoint; one seed and no architecture/augmentation ablations.',
                              'No GAN, diffusion, ViT, SVBRDF, neural rendering, inpainting or product deployment.']}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(base_psnr.numpy(), learned_psnr.numpy(), s=8, alpha=0.5)
    limits = [float(min(base_psnr.min(), learned_psnr.min())) - 1, float(max(base_psnr.max(), learned_psnr.max())) + 1]
    axes[0].plot(limits, limits, '--', color='gray')
    axes[0].set(xlabel='Bicubic PSNR (dB)', ylabel='CNN PSNR (dB)', title='450 test images')
    axes[1].hist((learned_psnr - base_psnr).numpy(), bins=24, color='#28748d')
    axes[1].set(xlabel='Per-image PSNR gain (dB)', ylabel='Count', title='Restoration error analysis')
    fig.tight_layout()
    fig.savefig(output / 'evaluation.png', dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in result.items() if key != 'history'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    experiment(parser.parse_args().output)
