"""Verbatim copy of STEER's `compute_token_weights` for equivalence testing.

Source: https://github.com/zz-haooo/STEER
        verl/trainer/ppo/core_algos.py, lines 580-713 (commit 08add1cc).
Licensed Apache-2.0 by the STEER / verl authors.

DO NOT EDIT.  The whole point of this file is that it is untouched upstream
code; `tests/test_lambda_zero_equiv.py` asserts that STEER-F reproduces it
exactly at lambda=0.  If upstream changes, re-extract rather than hand-patch:

    sed -n '580,713p' <steer>/verl/trainer/ppo/core_algos.py
"""

import torch

def compute_token_weights(
    advantages: torch.Tensor,
    entropys: torch.Tensor,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    token_weight_min: float = 0.8,
    token_weight_max: float = 1.2,
    linear: bool = True,
    mode: str = "symmetric",
) -> torch.Tensor:
    """
    
    Args:
        advantages (torch.Tensor): advantage values, shape (batch_size, response_length)
        entropys (torch.Tensor): entropy values, shape (batch_size, response_length)
        old_log_prob (torch.Tensor): old policy log probabilities, shape (batch_size, response_length)
        log_prob (torch.Tensor): new policy log probabilities, shape (batch_size, response_length)
        response_mask (torch.Tensor): response mask, shape (batch_size, response_length)
        token_weight_min (float): minimum value for token weights, default 0.8
        token_weight_max (float): maximum value for token weights, default 1.2
        linear (bool): whether to use linear mapping strategy, default True
            - True: use linear mapping to [token_weight_min, token_weight_max]
            - False: use exponential mapping token_weights = exp(-k * metric)
        mode (str): entropy control mode, default "symmetric"
            - "symmetric": paper's default; attenuate by |Ω| (both entropy-
              increasing and entropy-decreasing tokens are down-weighted).
            - "asymmetric": respect the sign of Ω, so that only entropy-
              decreasing tokens are attenuated toward token_weight_min while
              entropy-increasing tokens are pulled toward token_weight_max.
              For directional entropy control (e.g. anti-collapse), use
              token_weight_max=1.0 to avoid amplifying entropy-increasing
              tokens.
        
    Returns:
        torch.Tensor: computed token weights, shape (batch_size, response_length)
                     all valid tokens have weights in [token_weight_min, token_weight_max] range
    """

    with torch.no_grad():

        
        # Calculate \delta
        x = torch.exp(log_prob)  # Convert log_prob to probability values
        x = torch.clamp(x, min=1e-8, max=1.0 - 1e-8)

        x_one_minus_x_squared = x  * (1 - x)
        ln_x_plus_h = torch.log(x) + entropys
        
        f_x = x_one_minus_x_squared * ln_x_plus_h
        
        # Calculate A / old_log_prob
        old_prob = torch.exp(old_log_prob)
        old_prob = torch.clamp(old_prob, min=1e-8, max=1.0)

        advantage_over_old_prob = advantages / old_prob
        
        # Calculate \Omega
        if mode == "asymmetric":
            # Signed Ω: leading negative sign matches the sign convention of
            # paper Eq. 8, so Ω > 0 marks entropy-increasing tokens and
            # Ω < 0 marks entropy-decreasing tokens.
            metric = -advantage_over_old_prob * f_x
        else:
            metric = advantage_over_old_prob * f_x
        
        if not torch.isfinite(metric).all():
            print(f"[Token Weighting] Warning: Found non-finite values in metric")
            metric = torch.where(torch.isfinite(metric), metric, torch.zeros_like(metric))
        
        if mode == "symmetric":
            # Calculate abs(\Omega)
            metric = torch.abs(metric)
        # else: mode == "asymmetric" -> keep the sign of Ω
        
        valid_metric = metric[response_mask.bool()]
        if valid_metric.numel() == 0:
            return torch.zeros_like(response_mask, dtype=torch.float)
            
        metric_min = valid_metric.min()
        metric_max = valid_metric.max()
        
        # Initialize token weights
        token_weights = torch.zeros_like(metric, dtype=torch.float)
        
        # Calculate weights only for valid tokens
        valid_mask = response_mask.bool()
        if valid_mask.any():
            valid_metric = metric[valid_mask]
            
            if linear:
                # Linear mapping strategy: map to [token_weight_min, token_weight_max]
                if metric_max > metric_min:
                    scale_factor = (token_weight_max - token_weight_min) / (metric_max - metric_min)
                    if mode == "asymmetric":
                        # Ascending: strongest entropy-decreasing token (most-
                        # negative Ω) -> token_weight_min; strongest entropy-
                        # increasing token (most-positive Ω) -> token_weight_max.
                        valid_weights = token_weight_min + (valid_metric - metric_min) * scale_factor
                    else:
                        # Descending: largest |Ω| -> token_weight_min (paper).
                        valid_weights = token_weight_max - (valid_metric - metric_min) * scale_factor
                else:
                    valid_weights = torch.full_like(valid_metric, (token_weight_min + token_weight_max) / 2)
                valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
            else:
                if mode == "asymmetric":
                    # Exponential mapping over signed Ω. Normalize by |Ω|.max()
                    # so the exponent range is bounded regardless of sign, and
                    # use +k * Ω so Ω > 0 pushes above 1 and Ω < 0 pushes below.
                    abs_max = torch.maximum(
                        valid_metric.abs().max(),
                        torch.tensor(0.02, dtype=metric.dtype, device=metric.device),
                    )
                    k = -torch.log(torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)) / abs_max
                    valid_weights = torch.exp(k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=token_weight_max)
                else:
                    # Exponential mapping strategy: token_weights = exp(-k * metric)
                    k = -torch.log(torch.tensor(token_weight_min, dtype=metric.dtype, device=metric.device)) / \
                        torch.maximum(
                            metric_max,
                            torch.tensor(0.02, dtype=metric.dtype, device=metric.device)
                        )
                    valid_weights = torch.exp(-k * valid_metric)
                    valid_weights = torch.clamp(valid_weights, min=token_weight_min, max=1.0)
            
            # Apply computed weights to valid token positions
            token_weights[valid_mask] = valid_weights
        
        token_weights = token_weights * response_mask.float()
        
        
        return token_weights
