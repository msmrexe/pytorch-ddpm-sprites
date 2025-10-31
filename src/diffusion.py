"""
Diffusion Scheduler for forward (noising) and reverse (denoising) processes.
"""

import torch
from src import config

class DiffusionScheduler:
    """
    Manages the diffusion process scheduler, holding beta, alpha, and
    alpha_bar values, and defining the forward and reverse processes.
    """
    def __init__(self, timesteps, beta1, beta2, device):
        self.timesteps = timesteps
        self.device = device
        
        # Define linear beta schedule
        self.beta = torch.linspace(beta1, beta2, timesteps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(device)
        
        # Pre-calculate terms for forward process
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)

    def add_noise(self, x0, t):
        """
        Forward process (q): Perturbs an image x0 to a specified noise level t.
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        # Get pre-calculated values for batch
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        noise = torch.randn_like(x0).to(self.device)
        
        # Perturb input
        noised_image = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return noised_image, noise

    def p_xt_ddpm(self, xt, t, predicted_noise):
        """
        Reverse process (p_theta) for DDPM.
        Generates x_(t-1) from x_t.
        """
        # Get coefficients for this timestep
        alpha_t = self.alpha[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        
        # Compute the mean for x_(t-1)
        mean = (1 / torch.sqrt(alpha_t)) * (
            xt - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )
        
        # Add noise only if t > 0
        if t.min() > 0:
            noise = torch.randn_like(xt).to(self.device)
            variance = torch.sqrt(beta_t) * noise
        else:
            variance = 0

        x_t_minus_1 = mean + variance
        return x_t_minus_1

    def p_xt_ddim(self, xt, t, t_minus_1, predicted_noise, eta=0.0):
        """
        Reverse process (p_theta) for DDIM.
        Generates x_(t-1) from x_t.
        """
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        
        # Get alpha_bar_t_minus_1
        # If t_minus_1 is -1 (i.e., t=0), set alpha_bar_t_minus_1 to 1.0
        alpha_bar_t_minus_1 = torch.ones_like(alpha_bar_t)
        if t_minus_1.min() >= 0:
             alpha_bar_t_minus_1 = self.alpha_bar[t_minus_1].view(-1, 1, 1, 1)

        # Calculate x0_t (predicted x0)
        x0_t = (xt - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)
        
        # Calculate sigma_t
        sigma_t = eta * torch.sqrt(
            (1 - alpha_bar_t_minus_1) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_minus_1)
        )
        
        # Calculate x_(t-1)
        term1 = torch.sqrt(alpha_bar_t_minus_1) * x0_t
        term2 = torch.sqrt(1 - alpha_bar_t_minus_1 - sigma_t**2) * predicted_noise
        term3 = sigma_t * torch.randn_like(xt)
        
        x_t_minus_1 = term1 + term2 + term3
        return x_t_minus_1
