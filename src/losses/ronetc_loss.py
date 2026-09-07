"""
ronetc_loss.py

Purpose: Implements the Evidential Deep Learning (EDL) loss for RoNeTC.
Implements §3.4 / Section III-D of Wang et al. 2025.

The loss consists of two parts:
1. Bayes-risk Cross-Entropy (L_CE): Measures the fit of the predicted
   Dirichlet distribution to the ground-truth one-hot labels.
   L_CE = sum_k( y_k * (digamma(S) - digamma(alpha_k)) )
   (The paper uses \\psi for the digamma function)
2. KL Divergence (L_KL): Regularises the Dirichlet distribution towards a
   uniform prior for misleading evidence.
   L_KL = KL( D(p | alpha_tilde) || D(p | 1) )
   where alpha_tilde = y + (1 - y) * alpha  (keeps alpha_k for non-target classes).

Total Loss: L = L_CE + lambda_t * L_KL
where lambda_t is an annealing factor that linearly increases over training epochs.

RoNeTC Phase 3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoNeTCLoss(nn.Module):
    """Evidential Deep Learning loss for Subjective Logic opinions.

    Combines Bayes-risk Cross-Entropy with a KL divergence regularizer.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, alpha: torch.Tensor, y: torch.Tensor, lambda_t: float) -> torch.Tensor:
        """
        Computes the RoNeTC loss for a single view or the fused output.

        Parameters
        ----------
        alpha : torch.Tensor
            Shape (B, K). The Dirichlet parameters for each sample.
        y : torch.Tensor
            Shape (B,). The ground-truth class indices.
        lambda_t : float
            The annealing factor for the KL divergence term.

        Returns
        -------
        torch.Tensor
            Shape (B,). The loss per sample (scalar if reduction is used later,
            but here we return the per-sample loss for flexibility).
        """
        # Convert y to one-hot (B, K)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # 1. Bayes-risk Cross Entropy
        # L_CE = sum_k( y_k * (digamma(S) - digamma(alpha_k)) )
        S = torch.sum(alpha, dim=1, keepdim=True)
        # digamma(S) is (B, 1), digamma(alpha) is (B, K)
        term1 = torch.digamma(S) - torch.digamma(alpha)
        loss_ce = torch.sum(y_one_hot * term1, dim=1)  # (B,)

        # 2. KL Divergence Regularization
        # L_KL = KL( D(p | alpha_tilde) || D(p | 1) )
        # alpha_tilde removes the evidence from the ground-truth class
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        loss_kl = self.kl_divergence(alpha_tilde)  # (B,)

        # 3. Total Loss
        return loss_ce + lambda_t * loss_kl

    def kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Computes the KL divergence between Dirichlet(alpha) and Dirichlet(1).

        Parameters
        ----------
        alpha : torch.Tensor
            Shape (B, K).

        Returns
        -------
        torch.Tensor
            Shape (B,). KL divergence for each sample.
        """
        K = self.num_classes
        S = torch.sum(alpha, dim=1, keepdim=True)

        # Log Gamma term: log(Gamma(S)) - sum(log(Gamma(alpha)))
        log_gamma_term = torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)

        # Prior term: sum(log(Gamma(1))) - log(Gamma(K)) = 0 - log(Gamma(K)) = -log(Gamma(K))
        # Since Gamma(K) = (K-1)!
        prior_term = -torch.lgamma(torch.tensor(float(K), device=alpha.device))

        # Digamma term: sum( (alpha - 1) * (digamma(alpha) - digamma(S)) )
        digamma_term = torch.sum(
            (alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True
        )

        kl = log_gamma_term + prior_term + digamma_term
        return kl.squeeze(-1)  # (B,)

def calculate_annealing_factor(epoch: int, annealing_epochs: int, annealing_ceiling: float = 1.0) -> float:
    """Calculates the lambda_t annealing factor.

    lambda_t = min(1.0, epoch / annealing_epochs) * annealing_ceiling
    Assumes epoch is 1-indexed (e.g. epoch 1 to epochs).
    """
    return min(1.0, epoch / annealing_epochs) * annealing_ceiling
