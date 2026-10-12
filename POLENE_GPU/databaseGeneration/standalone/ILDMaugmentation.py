"""
Génération d'états perturbés le long des modes propres de la jacobienne
chimique, dans l'esprit de la construction des ILDM de Maas & Pope,
Combust. Flame 88 (1992) 239-264.

Version 8 : identique à la version 7 (perturbation MULTIPLICATIVE en espace
log-concentration z_i = ln C_i, échelles PAR MODE via des listes, tirage avec
remise optionnel via `N_samples`, perturbation isoénergétique de T/P selon
l'équation exacte de Liu et al. 2026 avec bornes calculées automatiquement,
mode 'all' pour les modes réactifs), avec en plus :

  PERTURBATION DE PHI LOCAL (AJOUT v8)

      perturb_phi=True, scale_phi=..., phi_mode='additive'|'relative',
      phi_bounds=(phi_min, phi_max)

  Au lieu de perturber indépendamment chaque élément (modes_conserved), on
  perturbe UN SEUL paramètre : la richesse locale phi, définie à partir de la
  composition élémentaire locale b = Mu^T C :

      phi_loc = (2 b_C + b_H / 2) / b_O

  Seul le « bloc combustible » (C:1, H:4 pour CH4) varie, le comburant
  (O, N) est fixé : delta_b = (phi_new - phi_loc) * b_O / demand * f.
  La variation delta_b est répercutée sur les espèces PRÉSENTES au PREMIER
  ORDRE (pas de Newton) par moindres carrés de norme minimale :

      dz = argmin ||dz||   s.t.   Mu_z^T dz = delta_b

  dz est dans l'image de Mu_z, donc orthogonal au sous-espace réactif local :
  aucune perturbation des modes réactifs n'est introduite, et la jacobienne
  n'est pas nécessaire. Ce dz est AJOUTÉ au dz éventuel des autres modes, et
  l'exponentielle n'est appliquée qu'une seule fois (C_new > 0 garanti).

Rappel de la méthode ILDM (inchangée) :

    J_z = D^{-1} J D - diag(omega / C),   D = diag(C)

où J = domega/dC (Jacobienne espèces habituelle, à T fixé) et omega les
taux de production nets. La séparation conservé/réactif est reconstruite
localement à chaque point : Mu_z = D . Mu (vecteurs éléments locaux),
R_z = noyau de Mu_z^T (sous-espace réactif local, valide au 1er ordre).

La perturbation des espèces est appliquée comme C_new = C_used * exp(dz),
ce qui garantit C_new > 0 automatiquement (plus de clipping nécessaire) et
une somme des fractions massiques exactement égale à 1 par construction.

Perturbation d'énergie sensible (désactivée par défaut, `perturb_energy=False`) :

      e_sens* = e_sens + (c / a) * (e_sens_max - e_sens_min),   c ~ U(-1, +1)

transposition à l'énergie sensible de l'Eq. (5) de Ding, Readshaw,
Rigopoulos & Jones (2021), Combust. Flame 231, 111493, raffinée par Liu,
Kronenburg & Zirwes (2026), Section 2.2. e_sens_min / e_sens_max sont
calculés UNE SEULE FOIS, en début de calcul, sur la base de référence `TPY`
(ou un sous-échantillon via `minmax_sample_size`). La nouvelle énergie
totale est reconstruite avec la NOUVELLE composition, et T (et P en cstV)
sont recalculés via gas.HPY (cstP) ou gas.UVY (cstV).

Entrées/sorties : accepte un objet `serie` maison (avec attributs .data,
.colName, .storingDevice, .storingDtype) ou un torch.Tensor, et renvoie un
objet du même type.
"""

import cantera as ct
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from scipy.linalg import schur, null_space

from POLENE_GPU.dataHandler.serie import serie


# --- État global par worker (perturbation ILDM) ---
_worker_gas = None
_worker_Mu = None  # (n_sp, n_elem) : vecteurs éléments bruts (colonnes), state-independent


def _init_perturbation_worker(kinetics):
    global _worker_gas, _worker_Mu
    _worker_gas = ct.Solution(kinetics)

    n_sp = _worker_gas.n_species
    n_elem = _worker_gas.n_elements

    Mu = np.zeros((n_sp, n_elem))
    for k in range(n_sp):
        for e in range(n_elem):
            Mu[k, e] = _worker_gas.n_atoms(k, e)

    _worker_Mu = Mu


# --------------------------------------------------------------------
# Décomposition de Schur réelle ordonnée (algèbre linéaire pure)
# --------------------------------------------------------------------

def _schur_blocks(T, tol=1e-10):
    n = T.shape[0]
    blocks = []
    eigvals = np.zeros(n, dtype=complex)
    k = 0
    while k < n:
        is_2x2 = (k < n - 1) and (
            abs(T[k + 1, k]) > tol * (abs(T[k, k]) + abs(T[k + 1, k + 1]) + 1e-300)
        )
        if is_2x2:
            lam = np.linalg.eigvals(T[k:k + 2, k:k + 2])
            eigvals[k], eigvals[k + 1] = lam[0], lam[1]
            blocks.append((k, 2))
            k += 2
        else:
            eigvals[k] = T[k, k]
            blocks.append((k, 1))
            k += 1
    return blocks, eigvals


def _ordered_real_schur(A, tol=1e-10):
    """Schur réel ordonné par partie réelle décroissante (mode 1 = lent)."""
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))
    if n == 1:
        return A.copy(), np.eye(1)

    eigvals_raw = np.linalg.eigvals(A)
    scale = np.max(np.abs(eigvals_raw)) + 1e-300
    real_parts = eigvals_raw.real

    best_idx = int(np.argmax(real_parts))
    lam_best = eigvals_raw[best_idx]
    is_complex_best = abs(lam_best.imag) > tol * scale
    size = min(2 if is_complex_best else 1, n)

    real_sorted = np.sort(real_parts)[::-1]
    if n > size:
        threshold = 0.5 * (real_sorted[size - 1] + real_sorted[size])
    else:
        threshold = real_sorted[-1] - 1.0

    def sort_fun(x):
        return x.real > threshold

    T, Z, sdim = schur(A, output='real', sort=sort_fun)
    if sdim == 0:
        sdim = size

    T_rest, Z_rest = _ordered_real_schur(T[sdim:, sdim:], tol)

    T_full = T.copy()
    T_full[sdim:, sdim:] = T_rest
    T_full[:sdim, sdim:] = T[:sdim, sdim:] @ Z_rest

    Z_full = Z.copy()
    Z_full[:, sdim:] = Z[:, sdim:] @ Z_rest

    return T_full, Z_full


def _reactive_directions(T_r, Z_r, modes, scales, rng, tol=1e-10):
    """
    Construit une direction (norme 1, en log-espace réactif) par bloc de
    Schur sélectionné, ainsi que l'amplitude (scale) associée à cette
    direction. Chaque bloc 2x2 (paire complexe conjuguée) ne produit
    qu'UNE seule direction (combinaison aléatoire cos/sin des deux
    vecteurs de Schur), à laquelle est associée l'amplitude du PREMIER
    indice de la paire rencontré dans `modes` (le second indice de la
    paire, s'il est aussi présent dans `modes`, est ignoré).

    Retourne (directions, used_scales), deux listes de même longueur.
    """
    n_r = T_r.shape[0]
    directions = []
    used_scales = []
    processed = set()

    for m, sc in zip(modes, scales):
        if not (1 <= m <= n_r):
            raise ValueError(f"Mode réactif {m} hors de portée (1..{n_r}).")
        idx = m - 1
        if idx in processed:
            continue

        is_2x2_here = (idx < n_r - 1) and (
            abs(T_r[idx + 1, idx]) > tol * (abs(T_r[idx, idx]) + abs(T_r[idx + 1, idx + 1]) + 1e-300)
        )
        is_2x2_prev = (idx > 0) and (
            abs(T_r[idx, idx - 1]) > tol * (abs(T_r[idx, idx]) + abs(T_r[idx - 1, idx - 1]) + 1e-300)
        )

        if is_2x2_prev:
            idx0 = idx - 1
        elif is_2x2_here:
            idx0 = idx
        else:
            idx0 = None

        if idx0 is not None:
            processed.add(idx0)
            processed.add(idx0 + 1)
            q1 = Z_r[:, idx0]
            q2 = Z_r[:, idx0 + 1]
            theta = rng.uniform(0.0, 2.0 * np.pi)
            directions.append(np.cos(theta) * q1 + np.sin(theta) * q2)
        else:
            processed.add(idx)
            directions.append(Z_r[:, idx])

        used_scales.append(sc)

    return directions, used_scales


# --------------------------------------------------------------------
# Validation / normalisation des échelles par mode
# --------------------------------------------------------------------

def _broadcast_scale(modes, scale, argname):
    """
    Normalise `scale` (scalaire ou liste/array) en un np.ndarray de même
    longueur que `modes`.

    - `modes` vide/None -> retourne un array vide (scale ignoré).
    - `scale` scalaire   -> même amplitude répétée pour tous les modes.
    - `scale` liste/array -> doit avoir exactement len(modes) éléments.
    """
    if not modes:
        return np.array([], dtype=float)
    if isinstance(scale, (int, float, np.floating, np.integer)):
        return np.full(len(modes), float(scale))
    scale_arr = np.asarray(scale, dtype=float).ravel()
    if scale_arr.shape[0] != len(modes):
        raise ValueError(
            f"{argname} doit être soit un scalaire (même amplitude pour tous les "
            f"modes), soit une liste/array de même longueur que la liste de modes "
            f"correspondante (reçu {scale_arr.shape[0]} valeur(s) pour "
            f"{len(modes)} mode(s))."
        )
    return scale_arr


# --------------------------------------------------------------------
# Workers pour le calcul des bornes de l'énergie SENSIBLE (Eq. 5 de Ding,
# appliquée à h_sens ou u_sens -- raffinement de Liu). Calcul effectué UNE
# SEULE FOIS, en début de `generatePerturbedStatesILDM`.
# --------------------------------------------------------------------

_ebounds_gas = None
_ebounds_reactorType = None
_ebounds_T_ref = None


def _ebounds_worker_init(kinetics, reactorType, T_ref):
    global _ebounds_gas, _ebounds_reactorType, _ebounds_T_ref
    _ebounds_gas = ct.Solution(kinetics)
    _ebounds_reactorType = reactorType
    _ebounds_T_ref = T_ref


def _ebounds_worker_task(state):
    """Calcule l'énergie SENSIBLE (h_sens en cstP, u_sens en cstV) pour UN état."""
    global _ebounds_gas, _ebounds_reactorType, _ebounds_T_ref
    gas = _ebounds_gas
    T0, P0 = float(state[0]), float(state[1])
    Y0 = state[2:]
    gas.TPY = T0, P0, Y0
    e_tot = gas.enthalpy_mass if _ebounds_reactorType == 'cstP' else gas.int_energy_mass
    gas.TPY = _ebounds_T_ref, P0, Y0
    e_ref = gas.enthalpy_mass if _ebounds_reactorType == 'cstP' else gas.int_energy_mass
    return e_tot - e_ref  # énergie sensible


# --------------------------------------------------------------------
# Perturbation de phi local, 1er ordre (AJOUT v8)
# --------------------------------------------------------------------

def _phi_dz(gas, Mu, Mu_z, C_used, rng, scale_phi, phi_mode='additive',
            phi_bounds=None, fuel='CH4', dz_max=3.0):
    """
    Perturbation de phi local (comburant fixé), répercutée sur les espèces
    au PREMIER ORDRE : dz = argmin ||dz||  s.t.  Mu_z^T dz = delta_b.
    dz est dans l'image de Mu_z, donc orthogonal au sous-espace réactif.
    Retourne dz (n_sp,), nul si phi n'est pas défini.
    """
    f = Mu[gas.species_index(fuel)]                 # atomes du combustible
    iC, iH, iO = (gas.element_index(e) for e in ('C', 'H', 'O'))
    demand = 2.0 * f[iC] + 0.5 * f[iH]              # demande en O (CH4 -> 4)

    b = Mu.T @ C_used                               # atomes (mol/m3)
    if b[iO] <= 0:
        return np.zeros_like(C_used)
    phi = (2.0 * b[iC] + 0.5 * b[iH]) / b[iO]

    u = rng.uniform(-1.0, 1.0)
    if phi_mode == 'relative':
        phi_new = phi * np.exp(u * scale_phi)
    else:
        phi_new = phi + u * scale_phi
    if phi_bounds is not None:
        phi_new = float(np.clip(phi_new, *phi_bounds))
    phi_new = max(phi_new, 0.0)

    # seul le bloc combustible (C:1, H:4) bouge, O et N sont fixes
    delta_b = (phi_new - phi) * b[iO] / demand * f

    dz = np.linalg.lstsq(Mu_z.T, delta_b, rcond=None)[0]   # norme minimale
    return np.clip(dz, -dz_max, dz_max)             # garde-fou sur les grands pas


# --------------------------------------------------------------------
# Perturbation multiplicative (log-espace) par état, + perturbation
# isoénergétique optionnelle de T/P (équation de Liu et al. 2026)
# --------------------------------------------------------------------

def _perturb_state(
    gas, Mu, T, P, Y,
    modes_conserved, scale_conserved,
    modes_reactive, scale_reactive,
    rng,
    zero_floor_rel=1e-20,
    perturb_energy=False,
    reactorType='cstP',
    a_coef=8.0,
    d_sens_range=0.0,
    T_ref_sensible=298.15,
    perturb_phi=False,
    scale_phi=0.05,
    phi_mode='additive',
    phi_bounds=None,
):
    """
    Calcule l'état perturbé pour un unique point (T, P, Y).

    Perturbation des espèces : MULTIPLICATIVE en log-concentration,
    C_new = C_used * exp(dz), où dz est une combinaison de directions
    propres du bloc réactif local de la jacobienne en log-espace
    (+ éventuellement des directions conservées, + éventuellement une
    perturbation de phi local). Chaque mode a sa PROPRE amplitude
    (`scale_conserved[i]` / `scale_reactive[i]`).

    `modes_reactive` peut être :
      - une liste d'entiers (1 = plus lent) : Schur réelle ORDONNÉE,
      - la chaîne 'all' : Schur réelle simple SANS tri, toutes
        les directions sont perturbées avec l'amplitude scalaire
        `scale_reactive`.

    Perturbation de phi (AJOUT v8) : si `perturb_phi=True`, phi local est
    perturbé (additif ou relatif), comburant fixé, et la variation est
    répercutée sur les espèces au 1er ordre (norme minimale, orthogonale
    aux modes réactifs).

    Perturbation isoénergétique de T/P : si `perturb_energy=True`,
        e_sens* = e_sens + (c / a_coef) * d_sens_range,   c ~ U(-1, +1)
    puis reconstruction de T (et P en cstV) avec la NOUVELLE composition.

    zero_floor_rel : plancher relatif (fraction de ||C||) pour régulariser
        D = diag(C) et éviter log(0) / division par 0.

    Retourne (T_new, P_new, Y_new).
    """
    gas.TPY = T, P, Y
    n_elem = Mu.shape[1]

    # --- énergie sensible de référence, calculée sur l'état de BASE ---
    if perturb_energy:
        if reactorType == 'cstP':
            e0_total = gas.enthalpy_mass
            v0 = None
        elif reactorType == 'cstV':
            e0_total = gas.int_energy_mass
            v0 = 1.0 / gas.density
        else:
            raise ValueError(f"reactorType doit être 'cstP' ou 'cstV', reçu '{reactorType}'")

        gas.TPY = T_ref_sensible, P, Y
        e0_ref = gas.enthalpy_mass if reactorType == 'cstP' else gas.int_energy_mass
        e0_sensible = e0_total - e0_ref

        gas.TPY = T, P, Y  # on restaure l'état de base

    C = gas.concentrations
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-300:
        C_norm = 1.0

    C_reg = np.maximum(C, zero_floor_rel * C_norm)  # évite log(0) / division par 0
    D = C_reg  # vecteur diagonal de D = diag(C_reg)

    Mu_z = D[:, None] * Mu  # D . Mu  (vecteurs éléments locaux, en log-tangente)

    dz = np.zeros_like(C)

    # --- Directions conservées locales (changent la composition élémentaire) ---
    if len(modes_conserved) > 0:
        for e_idx, sc in zip(modes_conserved, scale_conserved):
            if not (1 <= e_idx <= n_elem):
                raise ValueError(f"Indice d'élément {e_idx} hors de portée (1..{n_elem}).")
            mu_z = Mu_z[:, e_idx - 1]
            norm_mu = np.linalg.norm(mu_z)
            if norm_mu < 1e-300:
                continue
            amp = rng.uniform(-1.0, 1.0) * sc
            dz += amp * (mu_z / norm_mu)

    # --- Directions réactives locales (bloc réactif en log-espace) ---
    reactive_all = isinstance(modes_reactive, str)  # 'all'
    if reactive_all or len(modes_reactive) > 0:
        R_z = null_space(Mu_z.T)  # sous-espace réactif local (log-tangente)

        J = np.array(gas.net_production_rates_ddCi)     # domega/dC, T fixé
        omega = np.array(gas.net_production_rates)

        # J_z = D^-1 J D - diag(omega / C_reg)
        J_z = (J * D[None, :]) / D[:, None] - np.diag(omega / D)
        J_z_r = R_z.T @ J_z @ R_z

        if reactive_all:
            # Pas de tri : Schur réelle simple, toutes les directions
            # (chaque bloc 2x2 -> une seule direction, cf. _reactive_directions)
            T_r, Z_r = schur(J_z_r, output='real')
            n_r = T_r.shape[0]
            modes_r = list(range(1, n_r + 1))
            scales_in = np.full(n_r, float(scale_reactive))
        else:
            T_r, Z_r = _ordered_real_schur(J_z_r)
            modes_r = modes_reactive
            scales_in = scale_reactive

        directions_r, scales_r = _reactive_directions(T_r, Z_r, modes_r, scales_in, rng)

        for d, sc in zip(directions_r, scales_r):
            d_full = R_z @ d  # direction en log-espace, norme 1
            amp = rng.uniform(-1.0, 1.0) * sc
            dz += amp * d_full

    # C_used : la vraie concentration si elle est non nulle, sinon le plancher
    C_used = np.where(C > 0, C, C_reg)

    # --- Perturbation de phi (1er ordre, orthogonale aux modes réactifs) ---
    if perturb_phi:
        dz += _phi_dz(gas, Mu, Mu_z, C_used, rng, scale_phi, phi_mode, phi_bounds)

    C_new = C_used * np.exp(dz)  # toujours > 0 par construction

    M = gas.molecular_weights
    rho_new = np.sum(C_new * M)
    if rho_new <= 0:
        return T, P, np.asarray(Y, dtype=float)

    Y_new = C_new * M / rho_new  # somme = 1 exactement, pas de clip nécessaire

    if not perturb_energy:
        return T, P, Y_new

    # --- perturbation de l'énergie SENSIBLE et reconstruction isoénergétique ---
    c_e = float(rng.uniform(-1.0, 1.0))
    e_sens_new = e0_sensible + (c_e / a_coef) * d_sens_range

    gas.TPY = T_ref_sensible, P, Y_new
    e_ref_new = gas.enthalpy_mass if reactorType == 'cstP' else gas.int_energy_mass
    e_total_new = e_sens_new + e_ref_new

    try:
        if reactorType == 'cstP':
            gas.HPY = e_total_new, P, Y_new
            P_new = P
        else:  # 'cstV'
            gas.UVY = e_total_new, v0, Y_new
            P_new = gas.P
        T_new = gas.T
    except ct.CanteraError:
        # échec de la reconstruction thermo (rare) -> on garde T, P d'origine
        return T, P, Y_new

    return T_new, P_new, Y_new


def _perturb_chunk(
    states_batch,
    modes_conserved, scale_conserved,
    modes_reactive, scale_reactive,
    seed, zero_floor_rel,
    perturb_energy, reactorType, a_coef, d_sens_range, T_ref_sensible,
    perturb_phi, scale_phi, phi_mode, phi_bounds,
):
    global _worker_gas, _worker_Mu
    rng = np.random.default_rng(seed)

    N = states_batch.shape[0]
    n_sp = states_batch.shape[1] - 2
    T_out = np.zeros(N)
    P_out = np.zeros(N)
    Y_out = np.zeros((N, n_sp))

    for k in range(N):
        T = states_batch[k, 0]
        P = states_batch[k, 1]
        Y = states_batch[k, 2:]
        T_out[k], P_out[k], Y_out[k] = _perturb_state(
            _worker_gas, _worker_Mu, T, P, Y,
            modes_conserved, scale_conserved,
            modes_reactive, scale_reactive,
            rng, zero_floor_rel,
            perturb_energy, reactorType, a_coef, d_sens_range, T_ref_sensible,
            perturb_phi, scale_phi, phi_mode, phi_bounds,
        )

    return T_out, P_out, Y_out


def _perturb_chunk_star(args):
    """Adaptateur : pool.imap ne passe qu'un seul argument (le tuple de la tâche)."""
    return _perturb_chunk(*args)


# --------------------------------------------------------------------
# Diagnostic
# --------------------------------------------------------------------

def _describe_modes(kinetics, T0, P0, Y0, modes_conserved, modes_reactive, zero_floor_rel=1e-20):
    gas = ct.Solution(kinetics)
    gas.TPY = T0, P0, Y0

    n_sp = gas.n_species
    n_elem = gas.n_elements
    species_names = gas.species_names
    element_names = gas.element_names

    reactive_all = isinstance(modes_reactive, str)  # 'all'

    Mu = np.zeros((n_sp, n_elem))
    for k in range(n_sp):
        for e in range(n_elem):
            Mu[k, e] = gas.n_atoms(k, e)

    C = gas.concentrations
    C_norm = np.linalg.norm(C) or 1.0
    C_reg = np.maximum(C, zero_floor_rel * C_norm)
    D = C_reg
    Mu_z = D[:, None] * Mu
    R_z = null_space(Mu_z.T)
    n_r = R_z.shape[1]

    print("\n=== Modes conservés locaux (valeurs propres = 0 au 1er ordre) ===")
    for e in range(1, n_elem + 1):
        marker = "  <-- sélectionné" if e in modes_conserved else ""
        print(f"  mode conservé {e:2d} -> élément '{element_names[e - 1]}'{marker}")

    order_txt = "ordre de la Schur brute" if reactive_all else "tri lent -> rapide"
    print(f"\n=== Modes réactifs locaux, log-espace (n_r={n_r}, {order_txt}) ===")
    J = np.array(gas.net_production_rates_ddCi)
    omega = np.array(gas.net_production_rates)
    J_z = (J * D[None, :]) / D[:, None] - np.diag(omega / D)
    J_z_r = R_z.T @ J_z @ R_z

    try:
        if reactive_all:
            T_r, Z_r = schur(J_z_r, output='real')
        else:
            T_r, Z_r = _ordered_real_schur(J_z_r)
    except np.linalg.LinAlgError:
        print("  [diagnostic] tri de Schur impossible sur cet état, Schur non ordonnée utilisée")
        T_r, Z_r = schur(J_z_r, output='real')

    blocks, eigvals = _schur_blocks(T_r)

    for (start, size) in blocks:
        lam = eigvals[start]
        tau = 1.0 / max(abs(lam.real), 1e-30)
        mode_ids = list(range(start + 1, start + size + 1))
        q = R_z @ Z_r[:, start]  # direction en log-espace (relative), espèce par espèce

        top = np.argsort(-np.abs(q))[:3]
        dominant = ", ".join(f"{species_names[k]} ({q[k]:+.2f})" for k in top)

        lam_str = f"{lam.real:+.3e} ± {abs(lam.imag):.3e}j" if size == 2 else f"{lam.real:+.3e}"
        marker = "  <-- sélectionné" if (reactive_all or any(m in modes_reactive for m in mode_ids)) else ""
        ids_str = ",".join(str(m) for m in mode_ids)
        print(
            f"  mode(s) {ids_str:>6s} | lambda={lam_str:>22s} (relatif, 1/s) | tau={tau:.2e} s "
            f"| espèces dominantes (relatif): {dominant}{marker}"
        )
    print()


# --------------------------------------------------------------------
# Interface entrée/sortie générique (torch.Tensor / serie)
# --------------------------------------------------------------------

def _extract_states_and_meta(TPY):
    """
    Extrait un tableau (N, 2+n_sp) (conversion interne nécessaire pour
    Cantera/SciPy) et les métadonnées nécessaires pour reconstruire un objet
    du même type en sortie. Supporte :
      - un objet `serie` maison (attributs .data, .colName, .storingDevice,
        .storingDtype),
      - un torch.Tensor.
    """
    if hasattr(TPY, "colName") and hasattr(TPY, "data"):
        states = TPY.data.cpu().detach().numpy()
        meta = {
            "type": "serie",
            "col_names": TPY.colName,
            "storingDevice": getattr(TPY, "storingDevice", None),
            "storingDtype": getattr(TPY, "storingDtype", None),
            "cls": type(TPY),
        }
    elif isinstance(TPY, torch.Tensor):
        states = TPY.cpu().detach().numpy()
        meta = {"type": "tensor", "dtype": TPY.dtype, "device": TPY.device}
    else:
        raise TypeError(
            f"Type d'entrée non supporté : {type(TPY).__name__} "
            "(attendu : serie ou torch.Tensor)."
        )
    return states, meta


def _wrap_output(arr, meta):
    if meta["type"] == "serie":
        return meta["cls"](
            arr, meta["col_names"],
            storingDevice=meta["storingDevice"],
            storingDtype=meta["storingDtype"],
        )
    return torch.as_tensor(arr, dtype=meta["dtype"], device=meta["device"])


# --------------------------------------------------------------------
# Interface principale
# --------------------------------------------------------------------

def generatePerturbedStatesILDM(
    TPY,
    kinetics,
    modes_conserved=None,
    scale_conserved=0.0,
    modes_reactive=None,
    scale_reactive=0.05,
    n_workers=None,
    chunksize=200,
    seed=0,
    zero_floor_rel=1e-30,
    N_samples=None,
    perturb_energy=False,
    reactorType='cstP',
    a_coef=8.0,
    minmax_sample_size=None,
    T_ref_sensible=298.15,
    perturb_phi=False,
    scale_phi=0.05,
    phi_mode='additive',
    phi_bounds=None,
):
    """
    Génère de nouveaux états perturbés MULTIPLICATIVEMENT (log-espace),
    le long de modes propres locaux de la jacobienne chimique, avec en
    option une perturbation de la richesse locale phi (AJOUT v8) et une
    perturbation isoénergétique de T/P suivant l'équation de Liu,
    Kronenburg & Zirwes (2026), Section 2.2 :

        e_sens* = e_sens + (c / a_coef) * (e_sens_max - e_sens_min),
        c ~ U(-1, +1)

    (e_sens_max, e_sens_min) sont calculés UNE SEULE FOIS, au début, sur la
    base de référence `TPY` (ou un sous-échantillon via `minmax_sample_size`).

    Si `N_samples` est fourni, on commence par tirer N_samples états de TPY
    AVEC REMISE (tirage uniforme), puis chaque état tiré est perturbé
    indépendamment.

    Parameters
    ----------
    TPY : serie | torch.Tensor, shape (N_in, 2 + n_sp)
        (T, P, Y_1, ..., Y_n_sp). Sert aussi de base de référence pour le
        calcul des bornes (e_sens_min, e_sens_max) si `perturb_energy=True`.
    kinetics : str
        Fichier mécanisme cantera.
    modes_conserved : list[int] ou None
        Indices d'éléments (1..n_elem) à perturber INDÉPENDAMMENT. Pour une
        perturbation de phi seul, laisser à None.
    scale_conserved : float ou list[float]
        Amplitude(s) RELATIVE(S) (log-espace) pour les modes conservés.
        Scalaire (même amplitude pour tous) ou liste de même longueur que
        `modes_conserved`.
    modes_reactive : list[int], None ou 'all'
        - list[int] : indices de modes réactifs locaux (1 = plus lent,
          n_r = plus rapide, en log-espace), Schur réelle ordonnée.
        - 'all' : perturbe TOUTES les directions réactives, avec
          une Schur réelle simple SANS tri (contourne l'erreur LAPACK
          "Leading eigenvalues do not satisfy sort condition"). Dans ce
          cas `scale_reactive` doit être un scalaire.
        - None : aucun mode réactif.
    scale_reactive : float ou list[float]
        Amplitude(s) relative(s) (log-espace) pour les modes réactifs.
        Scalaire ou liste de même longueur que `modes_reactive`. Doit être
        un scalaire si `modes_reactive='all'`.
        Pour un bloc 2x2 (paire complexe conjuguée), l'amplitude retenue
        est celle du premier indice de la paire rencontré.
    n_workers, chunksize, seed : `seed` pilote le tirage avec remise, le
        calcul des bornes d'énergie sensible et les perturbations.
    zero_floor_rel : float
        Plancher relatif (fraction de ||C||) pour régulariser les espèces
        exactement absentes.
    N_samples : int ou None
        Nombre final d'états souhaité (tirage avec remise). Si None : un
        état perturbé par état d'entrée.
    perturb_energy : bool
        Si True, perturbe aussi l'énergie sensible (h_sens en cstP, u_sens
        en cstV) et reconstruit T (et P en cstV) de façon isoénergétique.
    reactorType : str
        'cstP' (défaut) ou 'cstV'. Utilisé seulement si `perturb_energy=True`.
    a_coef : float
        Coefficient `a` de l'Eq. (5) de Ding et al. (a=8 par défaut).
    minmax_sample_size : int ou None
        Sous-échantillonnage de `TPY` pour le calcul des bornes.
    T_ref_sensible : float
        Température de référence (K) séparant énergie sensible et énergie
        de formation (298.15 K par défaut).
    perturb_phi : bool (AJOUT v8)
        Si True, perturbe la richesse locale phi (comburant fixé), avec
        répercussion sur les espèces au 1er ordre (norme minimale,
        orthogonale aux modes réactifs).
    scale_phi : float (AJOUT v8)
        Amplitude de la perturbation de phi, u ~ U(-1, +1) :
        'additive'  -> phi_new = phi + u * scale_phi
        'relative'  -> phi_new = phi * exp(u * scale_phi)
    phi_mode : str (AJOUT v8)
        'additive' (défaut) ou 'relative'.
    phi_bounds : tuple (phi_min, phi_max) ou None (AJOUT v8)
        Bornes appliquées à phi_new (par ex. le domaine d'échantillonnage
        de la base). Si None, pas de borne (phi_new >= 0 seulement).

    Returns
    -------
    Un objet du même type que `TPY` (serie / torch.Tensor),
    shape (N_samples, 2 + n_sp) (ou (N_in, 2 + n_sp) si N_samples=None).
    Y > 0 et somme à 1 toujours garanti.
    """
    # --- normalisation des modes conservés / réactifs et de leurs échelles ---
    modes_conserved = modes_conserved or []

    reactive_all = isinstance(modes_reactive, str)
    if reactive_all:
        if modes_reactive.lower() != 'all':
            raise ValueError("modes_reactive doit être une liste d'entiers, None ou 'all'.")
        if not isinstance(scale_reactive, (int, float, np.floating, np.integer)):
            raise ValueError("Avec modes_reactive='all', scale_reactive doit être un scalaire.")
        modes_reactive = 'all'
        scale_reactive_arr = float(scale_reactive)
    else:
        modes_reactive = modes_reactive or []
        scale_reactive_arr = _broadcast_scale(modes_reactive, scale_reactive, "scale_reactive")

    if perturb_energy and reactorType not in ('cstP', 'cstV'):
        raise ValueError(f"reactorType doit être 'cstP' ou 'cstV', reçu '{reactorType}'")

    if phi_mode not in ('additive', 'relative'):
        raise ValueError(f"phi_mode doit être 'additive' ou 'relative', reçu '{phi_mode}'")

    scale_conserved_arr = _broadcast_scale(modes_conserved, scale_conserved, "scale_conserved")

    states_in, meta = _extract_states_and_meta(TPY)
    N_in = states_in.shape[0]
    if N_in == 0:
        raise ValueError("TPY est vide : impossible de tirer des états.")

    if n_workers is None:
        n_workers = mp.cpu_count()

    # --- bornes de l'énergie SENSIBLE, calculées UNE SEULE FOIS sur la base
    # --- de référence `TPY` (ou un sous-échantillon), AVANT toute perturbation
    if perturb_energy:
        rng_master = np.random.default_rng(seed)
        if minmax_sample_size is not None and minmax_sample_size < N_in:
            idx_sample = rng_master.choice(N_in, size=minmax_sample_size, replace=False)
            sample_states = states_in[idx_sample]
        else:
            sample_states = states_in

        with Pool(processes=n_workers, initializer=_ebounds_worker_init,
                  initargs=(kinetics, reactorType, T_ref_sensible)) as pool:
            energies = list(tqdm(
                pool.imap(_ebounds_worker_task, list(sample_states), chunksize=chunksize),
                total=len(sample_states),
                desc="Calcul de min/max de l'énergie sensible (Eq. 5 de Ding/Liu)",
            ))
        sens_min = float(np.min(energies))
        sens_max = float(np.max(energies))
        d_sens_range = sens_max - sens_min
        energy_symbol = "h_sens" if reactorType == 'cstP' else "u_sens"
        print(
            f"[generatePerturbedStatesILDM] {energy_symbol}_min = {sens_min:.6g} J/kg, "
            f"{energy_symbol}_max = {sens_max:.6g} J/kg  (sur {len(sample_states)} états, "
            f"a_coef={a_coef}) -> e_sens* = e_sens + (c/{a_coef})*{d_sens_range:.6g}, c~U(-1,+1)"
        )
    else:
        d_sens_range = 0.0  # non utilisé

    # --- Tirage avec remise (avant toute perturbation) ---
    if N_samples is None:
        states = states_in
        N = N_in
    else:
        N = int(N_samples)
        if N <= 0:
            raise ValueError(f"N_samples doit être un entier > 0 (reçu : {N_samples}).")
        # Flux distinct de ceux des chunks (qui utilisent default_rng(seed + i))
        rng_sampling = np.random.default_rng([seed, 0xC0FFEE])
        idx = rng_sampling.integers(0, N_in, size=N)  # uniforme, avec remise
        states = states_in[idx]
        print(
            f"Tirage avec remise : {N} échantillons parmi {N_in} états "
            f"({np.unique(idx).size} états sources distincts)."
        )

    _describe_modes(
        kinetics, states[0, 0], states[0, 1], states[0, 2:],
        modes_conserved, modes_reactive, zero_floor_rel,
    )

    chunks = [states[i : i + chunksize] for i in range(0, N, chunksize)]
    tasks = [
        (chunk, modes_conserved, scale_conserved_arr, modes_reactive, scale_reactive_arr,
         seed + i, zero_floor_rel, perturb_energy, reactorType, a_coef, d_sens_range,
         T_ref_sensible, perturb_phi, scale_phi, phi_mode, phi_bounds)
        for i, chunk in enumerate(chunks)
    ]

    phi_txt = (f", + phi ({phi_mode}, scale={scale_phi}, bornes={phi_bounds})"
               if perturb_phi else "")
    print(
        f"Génération de {N} états perturbés (multiplicatif, log-espace"
        f"{phi_txt}"
        f"{', + T/P isoénergétique (Liu et al.)' if perturb_energy else ''}) "
        f"(conservés={modes_conserved}@{list(scale_conserved_arr)}, "
        f"réactifs={modes_reactive}@{np.atleast_1d(scale_reactive_arr).tolist()})..."
    )

    with Pool(
        processes=n_workers,
        initializer=_init_perturbation_worker,
        initargs=(kinetics,),
    ) as pool:
        # imap : les résultats arrivent au fil de l'eau (barre tqdm fluide).
        # L'ordre des chunks est préservé.
        results = list(
            tqdm(
                pool.imap(_perturb_chunk_star, tasks),
                total=len(tasks),
                desc="Perturbation ILDM",
                unit="chunk",
            )
        )

    T_all = np.concatenate([r[0] for r in results])
    P_all = np.concatenate([r[1] for r in results])
    Y_all = np.concatenate([r[2] for r in results], axis=0)

    TPY_arr = np.concatenate([T_all[:, None], P_all[:, None], Y_all], axis=1)

    return _wrap_output(TPY_arr, meta)