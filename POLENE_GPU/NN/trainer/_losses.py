"""
Pure loss functions — stateless, no side effects.
All functions are static and can be imported independently.
"""

import torch
import torch.nn as nn
from torch.func import jacrev, vmap
import math


def mahalanobis_loss(preds: torch.Tensor, targets: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    Pure Mahalanobis loss:  mean_i [ ||J_i (pred_i - y_i)||^2 ]

    Args:
        preds   : (B, d)
        targets : (B, d)
        J       : (B, d, d)  — the metric matrix (e.g. system Jacobian)

    Returns:
        scalar loss
    """
    diff = (preds - targets).unsqueeze(-1)   # (B, d, 1)
    J_diff = torch.bmm(J, diff)              # (B, d, 1)
    return (J_diff ** 2).sum(dim=(1, 2)).mean()


def blended_mahalanobis_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    J: torch.Tensor,
    lamb: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Blended Mahalanobis loss with M = (1 - λ)I + λJ.

    loss = mean_i [ ||M_i (pred_i - y_i)||^2 ]

    Args:
        preds   : (B, d)
        targets : (B, d)
        J       : (B, d, d)
        lamb    : blending coefficient λ  (already on linear scale, not log)
        device, dtype : used to build the identity matrix on the right device

    Returns:
        scalar loss
    """
    diff = (preds - targets).unsqueeze(-1)            # (B, d, 1)
    d = diff.shape[1]
    I = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)  # (1, d, d)
    M = (1.0 - lamb) * I + lamb * J                  # (B, d, d)
    J_diff = torch.bmm(M, diff)                       # (B, d, 1)
    return (J_diff ** 2).mean(dim=(1, 2)).mean()


def jvp_physics_loss(
    model,
    X: torch.Tensor,
    Y: torch.Tensor,
    J_true: torch.Tensor,
) -> torch.Tensor:
    """
    JVP-based directional Jacobian loss:
        mean_i || J_network(X_i) · (Y_i - X_i)  −  J_true_i · (Y_i - X_i) ||^2

    Args:
        model   : nn.Module with a `tensorInput` keyword (called as model(x, tensorInput=True))
        X       : (B, d) — input points
        Y       : (B, d) — target points
        J_true  : (B, d_out, d_in) — reference Jacobians at X

    Returns:
        scalar loss
    """
    delta = (Y.data - X.data)                         # (B, d)

    def model_fn(x):
        return model(x, tensorInput=True)

    _, Jv_network = torch.func.jvp(model_fn, (X.data,), (delta,))   # (B, d_out)
    Jv_true = torch.bmm(J_true.data, delta.unsqueeze(-1)).squeeze(-1)  # (B, d_out)
    return nn.functional.mse_loss(Jv_network, Jv_true)


def manifoldJacobianLoss(
    model,
    batch_X,
    batch_Y,
    target_jacobians,
):
    """
    Directional Sobolev losses for a batch, using v = Y - X as direction.

    Returns:
        loss_values  : MSE on function values
        loss_jacob   : MSE on directional derivatives  J·v
    """
    v = batch_Y.data - batch_X.data                  # (B, d_in)
    #v_norm = v.norm(dim=1, keepdim=True)
    #v = v / v_norm

    def model_fn(x):
        return model(x, tensorInput=True)

    _, Jv_pred = torch.func.jvp(model_fn, (batch_X.data,), (v,))     # (B, d_out)

    Jv_true = torch.bmm(target_jacobians.data, v.unsqueeze(2)).squeeze(2)  # (B, d_out)
    loss_jacob = nn.functional.mse_loss(Jv_pred, Jv_true)  #TODO: look at jacobian normalisation to prevent overfitting the reactive zones

    return loss_jacob

def manifoldJacobianLoss_cone(
    model,
    batch_X,
    batch_Y,
    target_jacobians,
    alpha=math.pi / 12  # Angle d'ouverture max du cône en radians (ex: 15 degrés)

):
    """
    Directional Sobolev losses for a batch, using v = Y - X as direction.

    Returns:
        loss_values  : MSE on function values
        loss_jacob   : MSE on directional derivatives  J·v
    """
    v = batch_Y.data - batch_X.data  # (B, d_in)

    # -----------------------------------------------------------------
    # 1. Calcul de la norme et de l'axe central (z) du cône
    # -----------------------------------------------------------------
    norm_v = torch.norm(v, p=2, dim=1, keepdim=True)
    # Protection contre la division par zéro (si Y = X)
    norm_v_safe = torch.clamp(norm_v, min=1e-8)
    z = v / norm_v_safe

    # -----------------------------------------------------------------
    # 2. Génération d'une direction aléatoire orthogonale (x)
    # -----------------------------------------------------------------
    r = torch.randn_like(v)
    # Projection de r sur l'hyperplan orthogonal à z : r_perp = r - (r.z)z
    dot_product = torch.sum(r * z, dim=1, keepdim=True)
    r_perp = r - dot_product * z

    norm_r_perp_safe = torch.clamp(torch.norm(r_perp, p=2, dim=1, keepdim=True), min=1e-8)
    x = r_perp / norm_r_perp_safe

    # -----------------------------------------------------------------
    # 3. Échantillonnage de l'angle et construction du vecteur u
    # -----------------------------------------------------------------
    # Tirage d'un angle aléatoire theta uniforme entre [0, alpha] pour chaque élément
    theta = torch.rand(v.shape[0], 1, device=v.device) * alpha

    # Construction du vecteur unitaire dans le cône : cos(theta)*z + sin(theta)*x
    u_dir = z * torch.cos(theta) + x * torch.sin(theta)

    # Remise à l'échelle pour respecter : norme(u) == norme(Y - X)
    # Si norm_v était 0, u sera bien ramené à un vecteur nul grâce à norm_v non-clampé
    u = u_dir * norm_v

    # -----------------------------------------------------------------

    def model_fn(x):
        return model(x, tensorInput=True)

    _, Jv_pred = torch.func.jvp(model_fn, (batch_X.data,), (u,))     # (B, d_out)

    Jv_true = torch.bmm(target_jacobians.data, u.unsqueeze(2)).squeeze(2)  # (B, d_out)
    loss_jacob = nn.functional.mse_loss(Jv_pred, Jv_true)  #TODO: look at jacobian normalisation to prevent overfitting the reactive zones

    return loss_jacob

def directional_sobolev_losses(
    model,
    batch_X,
    batch_Y,
    target_jacobians,
    jacobian_mask=None
):  #TODO: peut être apprendre la jacobienne dans la direction omega et pas Y-X, c'est toujours la direction de la dynamique mais d'un côté on approxime alors que de l'autre on est exacte
    """
    Directional Sobolev losses for a batch, using v = Y - X as direction.

    Returns:
        loss_values  : MSE on function values
        loss_jacob   : MSE on directional derivatives  J·v
    """
    v = batch_Y.data - batch_X.data                  # (B, d_in)

    preds = model(batch_X)  #TODO: look at the way gradient computation works because it seems that it could be some unwanted behaviour in some way here, whene all computation are done on tensors, it makes strange things (new network parameters appear)

    def model_fn(x):
        return model(x, tensorInput=True)

    _, Jv_pred = torch.func.jvp(model_fn, (batch_X.data,), (v,))    # (B, d_out)

    loss_values = nn.functional.mse_loss(preds.data, batch_Y.data)

    if jacobian_mask is None:
        # Comportement identique à l'original
        Jv_true = torch.bmm(target_jacobians.data, v.unsqueeze(2)).squeeze(2)  # (B, d_out)
        loss_jacob = nn.functional.mse_loss(Jv_pred, Jv_true)
    else:
        # On filtre avant le bmm — plus efficace que de masquer après
        v_m    = v[jacobian_mask]                          # (M, d_in)
        J_m    = target_jacobians.data[jacobian_mask]      # (M, d_in, d_out)  ou (M, d_out, d_in) selon convention
        Jv_p_m = Jv_pred[jacobian_mask]                   # (M, d_out)

        Jv_true_m = torch.bmm(J_m, v_m.unsqueeze(2)).squeeze(2)   # (M, d_out)
        loss_jacob = nn.functional.mse_loss(Jv_p_m, Jv_true_m)
    #TODO: look at jacobian normalisation to prevent overfitting the reactive zones

    return loss_values, loss_jacob

def directional_sobolev_losses_cone(
        model,
        batch_X,
        batch_Y,
        target_jacobians,
        alpha=math.pi / 12  # Angle d'ouverture max du cône en radians (ex: 15 degrés)
):
    """
    Directional Sobolev losses for a batch, using a random vector u as direction.
    u is sampled uniformly within a cone of angle alpha around axis v = Y - X,
    and scaled to have the exact same norm as v.
    """
    v = batch_Y.data - batch_X.data  # (B, d_in)

    # -----------------------------------------------------------------
    # 1. Calcul de la norme et de l'axe central (z) du cône
    # -----------------------------------------------------------------
    norm_v = torch.norm(v, p=2, dim=1, keepdim=True)
    # Protection contre la division par zéro (si Y = X)
    norm_v_safe = torch.clamp(norm_v, min=1e-8)
    z = v / norm_v_safe

    # -----------------------------------------------------------------
    # 2. Génération d'une direction aléatoire orthogonale (x)
    # -----------------------------------------------------------------
    r = torch.randn_like(v)
    # Projection de r sur l'hyperplan orthogonal à z : r_perp = r - (r.z)z
    dot_product = torch.sum(r * z, dim=1, keepdim=True)
    r_perp = r - dot_product * z

    norm_r_perp_safe = torch.clamp(torch.norm(r_perp, p=2, dim=1, keepdim=True), min=1e-8)
    x = r_perp / norm_r_perp_safe

    # -----------------------------------------------------------------
    # 3. Échantillonnage de l'angle et construction du vecteur u
    # -----------------------------------------------------------------
    # Tirage d'un angle aléatoire theta uniforme entre [0, alpha] pour chaque élément
    theta = torch.rand(v.shape[0], 1, device=v.device) * alpha

    # Construction du vecteur unitaire dans le cône : cos(theta)*z + sin(theta)*x
    u_dir = z * torch.cos(theta) + x * torch.sin(theta)

    # Remise à l'échelle pour respecter : norme(u) == norme(Y - X)
    # Si norm_v était 0, u sera bien ramené à un vecteur nul grâce à norm_v non-clampé
    u = u_dir * norm_v
    # -----------------------------------------------------------------

    preds = model(batch_X)  #TODO: look at the way gradient computation works because it seems that it could be some unwanted behaviour in some way here, whene all computation are done on tensors, it makes strange things (new network parameters appear)

    def model_fn(x_inp):
        return model(x_inp, tensorInput=True)

    # On évalue la dérivée directionnelle du modèle selon la NOUVELLE direction u
    _, Jv_pred = torch.func.jvp(model_fn, (batch_X.data,), (u,))  # (B, d_out)

    loss_values = nn.functional.mse_loss(preds.data, batch_Y.data)

    # On doit aussi calculer la vraie dérivée directionnelle selon u
    Jv_true = torch.bmm(target_jacobians.data, u.unsqueeze(2)).squeeze(2)  # (B, d_out)

    loss_jacob = nn.functional.mse_loss(Jv_pred, Jv_true)  #TODO: look at jacobian normalisation to prevent overfitting the reactive zones

    return loss_values, loss_jacob


def jac_inv_loss(model, X, J_1: torch.Tensor) -> torch.Tensor:
    """
    Inverse-Jacobian consistency loss:
        ½ ||J_pred · J_1 − I||² + ½ ||J_1 · J_pred − I||²

    Used by integratorSurrogateTrainer to enforce J_model ≈ J_true^{-1}.

    Args:
        model   : nn.Module
        X       : dataset batch with attributes .data, .nFeatures, .nSamples,
                  .storingDevice, .storingDtype
        J_1     : (B, d, d) — target inverse Jacobians

    Returns:
        scalar loss
    """
    x_sample = X.data.detach().requires_grad_(True)

    def model_fn(single_x):
        return model(single_x.unsqueeze(0), tensorInput=True).squeeze(0)

    batch_jacobian_fn = vmap(jacrev(model_fn))
    pred_jacobians = batch_jacobian_fn(x_sample)     # (B, d_out, d_in)

    left_inv = torch.bmm(pred_jacobians, J_1.data)
    right_inv = torch.bmm(J_1.data, pred_jacobians)
    I = (
        torch.eye(X.nFeatures, device=X.storingDevice, dtype=X.storingDtype)
        .unsqueeze(0)
        .expand(X.nSamples, X.nFeatures, X.nFeatures)
    )

    return 0.5 * nn.functional.mse_loss(left_inv, I) + \
           0.5 * nn.functional.mse_loss(right_inv, I)


def full_sobolev_loss(
    model,
    batch_X: torch.Tensor,
    batch_Y: torch.Tensor,
    target_jacobians: torch.Tensor,
    lamb: float = 1.0,
):
    """
    Full Sobolev loss (Czarnecki et al., 2017) computed via full batch Jacobians.

    loss = lamb * MSE(values) + (1 - lamb) * MSE(full Jacobians, directional)

    Returns:
        loss        : weighted scalar
        loss_values : detached, for logging
        loss_jac    : detached, for logging
    """
    preds = model(batch_X)
    loss_values = nn.functional.mse_loss(preds.data, batch_Y.data)

    x = batch_X.data.detach().requires_grad_(True)

    def model_fn(single_x):
        return model(single_x.unsqueeze(0), tensorInput=True).squeeze(0)

    pred_jacobians = vmap(jacrev(model_fn))(x)       # (B, d_out, d_in)

    v = (batch_Y.data - batch_X.data).unsqueeze(2)   # (B, d_in, 1)
    Jv_pred = torch.bmm(pred_jacobians, v).squeeze(2)
    Jv_true = torch.bmm(target_jacobians.data, v).squeeze(2)
    loss_jac = nn.functional.mse_loss(Jv_pred, Jv_true)

    loss = lamb * loss_values + (1.0 - lamb) * loss_jac
    return loss, loss_values.detach(), loss_jac.detach()