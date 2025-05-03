# Based on Karras et al. (2022) diffusion models for PyTorch
import torch
import k_diffusion as K
from k_diffusion.sampling import BrownianTreeNoiseSampler
from tqdm.auto import trange, tqdm
from stable_audio_tools.inference.sampling import make_cond_model_fn, t_to_alpha_sigma
import typing as tp

@torch.no_grad()
def sample_dpmpp_2m_sde_truncated(model,
                                  x,
                                  sigmas,
                                  truncation_t=None,
                                  extra_args=None,
                                  callback=None,
                                  disable=None,
                                  eta=1.,
                                  s_noise=1.,
                                  noise_sampler=None,
                                  solver_type='midpoint'):
    """DPM-Solver++(2M) SDE."""
    print(f"USING TRUNCATED SAMPLING WITH {truncation_t}")
    if truncation_t is None:
        truncation_t = len(sigmas) - 1
    elif truncation_t >= len(sigmas) - 1:
        truncation_t = len(sigmas) - 1

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None

    for i in trange(truncation_t, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h
    return x

@torch.no_grad()
def sample_dpmpp_2m_sde_truncated_seq(model,
                                  x,
                                  sigmas,
                                  truncation_ts=None,
                                  extra_args=None,
                                  callback=None,
                                  disable=None,
                                  eta=1.,
                                  s_noise=1.,
                                  noise_sampler=None,
                                  solver_type='midpoint'
    ) -> tp.List[torch.Tensor]:
    """DPM-Solver++(2M) SDE."""
    if truncation_ts is None:
        return [sample_dpmpp_2m_sde_truncated(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, eta=eta, s_noise=s_noise, noise_sampler=noise_sampler, solver_type=solver_type)]

    if solver_type not in {'heun', 'midpoint'}:
        raise ValueError('solver_type must be \'heun\' or \'midpoint\'')

    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max) if noise_sampler is None else noise_sampler
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h_last = None

    outs = []

    for i in trange(len(sigmas) - 1, disable=disable):

        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            # DPM-Solver++(2M) SDE
            t, s = -sigmas[i].log(), -sigmas[i + 1].log()
            h = s - t
            eta_h = eta * h

            x = sigmas[i + 1] / sigmas[i] * (-eta_h).exp() * x + (-h - eta_h).expm1().neg() * denoised

            if old_denoised is not None:
                r = h_last / h
                if solver_type == 'heun':
                    x = x + ((-h - eta_h).expm1().neg() / (-h - eta_h) + 1) * (1 / r) * (denoised - old_denoised)
                elif solver_type == 'midpoint':
                    x = x + 0.5 * (-h - eta_h).expm1().neg() * (1 / r) * (denoised - old_denoised)

            if eta:
                x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        old_denoised = denoised
        h_last = h

        if truncation_ts is not None and i in truncation_ts:
            outs.append(denoised)

    return outs
 
def sample_k_truncated(
        model_fn, 
        noise, 
        truncation_t=None,
        init_data=None,
        steps=50, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.01, 
        sigma_max=100, 
        rho=1.0, 
        device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):

    print(f"sample_k_truncated w/ {truncation_t}")
    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma 
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise 
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise

        return sample_dpmpp_2m_sde_truncated(denoiser, x, sigmas, truncation_t=truncation_t, disable=False, callback=callback, extra_args=extra_args)
        
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")
    
def sample_k_truncated_seq(
        model_fn, 
        noise, 
        truncation_ts=None,
        init_data=None,
        steps=50, 
        sampler_type="dpmpp-2m-sde", 
        sigma_min=0.01, 
        sigma_max=100, 
        rho=1.0, 
        device="cuda", 
        callback=None, 
        cond_fn=None,
        **extra_args
    ):
    print(f"sampling truncated seq w/ {truncation_ts}")
    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        # Make the list of sigmas. Sigma values are scalars related to the amount of noise each denoising step has
        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        # Scale the initial noise by sigma 
        noise = noise * sigmas[0]

        if init_data is not None:
            # set the initial latent to the init_data, and noise it with initial sigma
            x = init_data + noise 
        else:
            # SAMPLING
            # set the initial latent to noise
            x = noise

        return sample_dpmpp_2m_sde_truncated_seq(denoiser, x, sigmas, truncation_ts=truncation_ts, disable=False, callback=callback, extra_args=extra_args)
        
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")
