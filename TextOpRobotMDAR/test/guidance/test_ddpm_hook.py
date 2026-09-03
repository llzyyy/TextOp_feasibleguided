import numpy as np
import torch
from torch import nn

from robotmdar.diffusion.gaussian_diffusion import (
    GaussianDiffusion,
    LossType,
    ModelMeanType,
    ModelVarType,
)


class ZeroCleanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, timesteps, y):
        del timesteps, y
        return torch.zeros_like(x) + self.anchor * 0


def make_diffusion():
    return GaussianDiffusion(
        betas=np.array([0.1, 0.2], dtype=np.float64),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.MSE,
    )


def test_clean_guidance_rewrites_xstart_before_ddpm_posterior():
    diffusion = make_diffusion()
    model = ZeroCleanModel()
    noisy = torch.randn(1, 1, 4)
    called = []

    def clean_guidance(x_t, timestep, pred_xstart, model_kwargs=None):
        called.append((x_t.clone(), timestep.clone(), model_kwargs))
        return pred_xstart + 1.0

    result = diffusion.p_sample(
        model,
        noisy,
        torch.tensor([0]),
        clip_denoised=False,
        clean_guidance_fn=clean_guidance,
        model_kwargs={"y": {}},
    )
    assert len(called) == 1
    torch.testing.assert_close(result["pred_xstart"], torch.ones_like(noisy))
    torch.testing.assert_close(result["sample"], torch.ones_like(noisy), atol=1e-5, rtol=1e-5)


def test_absent_hook_preserves_baseline_result():
    diffusion = make_diffusion()
    model = ZeroCleanModel()
    noise = torch.randn(1, 1, 4)
    torch.manual_seed(11)
    first = diffusion.p_sample_loop(model, noise.shape, noise=noise, clip_denoised=False, model_kwargs={"y": {}})
    torch.manual_seed(11)
    second = diffusion.p_sample_loop(
        model,
        noise.shape,
        noise=noise,
        clip_denoised=False,
        model_kwargs={"y": {}},
        clean_guidance_fn=None,
    )
    torch.testing.assert_close(first, second, atol=0, rtol=0)
