"""
Génération d'états perturbés le long des modes propres de la jacobienne
chimique, dans l'esprit de la construction des ILDM de Maas & Pope,
Combust. Flame 88 (1992) 239-264.

Version 4 : identique à la version 3 (perturbation MULTIPLICATIVE en espace
log-concentration, z_i = ln C_i), avec en plus :

  * un paramètre `N_samples` : nombre final d'états souhaité. Si il est
    fourni, N_samples états sont d'abord tirés AVEC REMISE dans TPY, puis la
    méthode de perturbation (inchangée) est appliquée à chacun d'eux.
    Si `N_samples=None`, le comportement est celui de la version 3
    (un état perturbé par état d'entrée).
  * l'utilisation effective de `_extract_states_and_meta` / `_wrap_output`,
    de sorte que serie / torch.Tensor sont réellement supportés en entrée
    et en sortie.

Rappel de la méthode :

    J_z = D^{-1} J D - diag(omega / C),   D = diag(C)

où J = domega/dC (Jacobienne espèces habituelle, à T fixé) et omega les
taux de production nets. La séparation conservé/réactif est reconstruite
localement à chaque point : Mu_z = D . Mu (vecteurs éléments locaux),
R_z = noyau de Mu_z^T (sous-espace réactif local, valide au 1er ordre).

La perturbation est appliquée comme C_new = C_used * exp(dz), ce qui
garantit C_new > 0 automatiquement (plus de clipping nécessaire) et une
somme des fractions massiques exactement égale à 1 par construction.

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


# --- État global par worker ---
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
# Décomposition de Schur réelle ordonnée (inchangé, algèbre linéaire pure)
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


def _reactive_directions(T_r, Z_r, modes, rng, tol=1e-10):
    n_r = T_r.shape[0]
    directions = []
    processed = set()

    for m in modes:
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

    return directions


# --------------------------------------------------------------------
# Perturbation multiplicative (log-espace) par état
# --------------------------------------------------------------------

def _perturb_state(
    gas, Mu, T, P, Y,
    modes_conserved, scale_conserved,
    modes_reactive, scale_reactive,
    rng,
    zero_floor_rel=1e-20,
):
    """
    Calcule Y perturbé pour un unique point (T, P, Y), par perturbation
    MULTIPLICATIVE en log-concentration :
        C_new = C_used * exp(dz)
    où dz est une combinaison de directions propres du bloc réactif local
    de la jacobienne en log-espace (+ éventuellement des directions
    conservées), avec des amplitudes DIRECTEMENT relatives (scale = 0.05
    signifie ~5% de changement relatif par mode, pas une fraction de la
    norme globale des concentrations).

    zero_floor_rel : plancher relatif (fraction de ||C||) utilisé pour
        régulariser D = diag(C) et éviter les singularités numériques
        (log(0), division par 0) pour les espèces exactement absentes.
    """
    gas.TPY = T, P, Y
    n_elem = Mu.shape[1]

    C = gas.concentrations
    C_norm = np.linalg.norm(C)
    if C_norm < 1e-300:
        C_norm = 1.0

    C_reg = np.maximum(C, zero_floor_rel * C_norm)  # évite log(0) / division par 0
    D = C_reg  # vecteur diagonal de D = diag(C_reg)

    Mu_z = D[:, None] * Mu  # D . Mu  (vecteurs éléments locaux, en log-tangente)

    dz = np.zeros_like(C)

    # --- Directions conservées locales (changent la composition élémentaire) ---
    if modes_conserved:
        for e_idx in modes_conserved:
            if not (1 <= e_idx <= n_elem):
                raise ValueError(f"Indice d'élément {e_idx} hors de portée (1..{n_elem}).")
            mu_z = Mu_z[:, e_idx - 1]
            norm_mu = np.linalg.norm(mu_z)
            if norm_mu < 1e-300:
                continue
            amp = rng.uniform(-1.0, 1.0) * scale_conserved
            dz += amp * (mu_z / norm_mu)

    # --- Directions réactives locales (bloc réactif en log-espace) ---
    if modes_reactive:
        R_z = null_space(Mu_z.T)  # sous-espace réactif local (log-tangente)

        J = np.array(gas.net_production_rates_ddCi)     # domega/dC, T fixé
        omega = np.array(gas.net_production_rates)

        # J_z = D^-1 J D - diag(omega / C_reg)
        J_z = (J * D[None, :]) / D[:, None] - np.diag(omega / D)
        J_z_r = R_z.T @ J_z @ R_z

        T_r, Z_r = _ordered_real_schur(J_z_r)
        directions_r = _reactive_directions(T_r, Z_r, modes_reactive, rng)

        for d in directions_r:
            d_full = R_z @ d  # direction en log-espace, norme 1
            amp = rng.uniform(-1.0, 1.0) * scale_reactive
            dz += amp * d_full

    # C_used : la vraie concentration si elle est non nulle, sinon le
    # plancher (permet à une espèce quasi absente d'être "ensemencée"
    # proportionnellement à sa magnitude actuelle, jamais à celle d'une
    # espèce majoritaire).
    C_used = np.where(C > 0, C, C_reg)
    C_new = C_used * np.exp(dz)  # toujours > 0 par construction

    M = gas.molecular_weights
    rho_new = np.sum(C_new * M)
    if rho_new <= 0:
        return np.asarray(Y, dtype=float)

    Y_new = C_new * M / rho_new  # somme = 1 exactement, pas de clip nécessaire
    return Y_new


def _perturb_chunk(
    states_batch,
    modes_conserved, scale_conserved,
    modes_reactive, scale_reactive,
    seed, zero_floor_rel,
):
    global _worker_gas, _worker_Mu
    rng = np.random.default_rng(seed)

    N = states_batch.shape[0]
    n_sp = states_batch.shape[1] - 2
    Y_out = np.zeros((N, n_sp))

    for k in range(N):
        T = states_batch[k, 0]
        P = states_batch[k, 1]
        Y = states_batch[k, 2:]
        Y_out[k] = _perturb_state(
            _worker_gas, _worker_Mu, T, P, Y,
            modes_conserved, scale_conserved,
            modes_reactive, scale_reactive,
            rng, zero_floor_rel,
        )

    return Y_out


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

    print(f"\n=== Modes réactifs locaux, log-espace (n_r={n_r}, tri lent -> rapide) ===")
    J = np.array(gas.net_production_rates_ddCi)
    omega = np.array(gas.net_production_rates)
    J_z = (J * D[None, :]) / D[:, None] - np.diag(omega / D)
    J_z_r = R_z.T @ J_z @ R_z
    T_r, Z_r = _ordered_real_schur(J_z_r)
    blocks, eigvals = _schur_blocks(T_r)

    for (start, size) in blocks:
        lam = eigvals[start]
        tau = 1.0 / max(abs(lam.real), 1e-30)
        mode_ids = list(range(start + 1, start + size + 1))
        q = R_z @ Z_r[:, start]  # direction en log-espace (relative), espèce par espèce

        top = np.argsort(-np.abs(q))[:3]
        dominant = ", ".join(f"{species_names[k]} ({q[k]:+.2f})" for k in top)

        lam_str = f"{lam.real:+.3e} ± {abs(lam.imag):.3e}j" if size == 2 else f"{lam.real:+.3e}"
        marker = "  <-- sélectionné" if any(m in modes_reactive for m in mode_ids) else ""
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
    zero_floor_rel=1e-20,
    N_samples=None,
):
    """
    Génère de nouveaux états perturbés MULTIPLICATIVEMENT (log-espace),
    le long de modes propres locaux de la jacobienne chimique.

    Si `N_samples` est fourni, on commence par tirer N_samples états de TPY
    AVEC REMISE (tirage uniforme), puis chaque état tiré est perturbé
    indépendamment (amplitudes/signes/angles tirés à part pour chaque ligne,
    y compris pour les doublons issus du tirage avec remise).

    Parameters
    ----------
    TPY : serie | torch.Tensor, shape (N_in, 2 + n_sp)
        (T, P, Y_1, ..., Y_n_sp).
    kinetics : str
        Fichier mécanisme cantera.
    modes_conserved : list[int] ou None
        Indices d'éléments (1..n_elem) à perturber.
    scale_conserved : float
        Amplitude RELATIVE (log-espace) pour les modes conservés
        (0.05 ~ 5% de changement relatif par mode).
    modes_reactive : list[int] ou None
        Indices de modes réactifs locaux (1 = plus lent, n_r = plus
        rapide, calculés en log-espace).
    scale_reactive : float
        Amplitude relative (log-espace) pour les modes réactifs.
    n_workers, chunksize, seed : cf. versions précédentes. `seed` pilote à
        la fois le tirage avec remise et les perturbations (flux
        aléatoires distincts).
    zero_floor_rel : float
        Plancher relatif (fraction de ||C||) pour régulariser les espèces
        exactement absentes (évite log(0) / division par 0).
    N_samples : int ou None
        Nombre final d'états souhaité. Peut être inférieur ou supérieur à
        N_in (tirage avec remise). Si None : pas de rééchantillonnage, un
        état perturbé par état d'entrée (N_in en sortie).

    Returns
    -------
    Un objet du même type que `TPY` (serie / torch.Tensor),
    shape (N_samples, 2 + n_sp) (ou (N_in, 2 + n_sp) si N_samples=None),
    avec T, P inchangés par rapport aux états tirés et Y > 0 sommant à 1.
    """
    modes_conserved = modes_conserved or []
    modes_reactive = modes_reactive or []

    states_in, meta = _extract_states_and_meta(TPY)
    N_in = states_in.shape[0]
    if N_in == 0:
        raise ValueError("TPY est vide : impossible de tirer des états.")

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

    if n_workers is None:
        n_workers = mp.cpu_count()

    _describe_modes(
        kinetics, states[0, 0], states[0, 1], states[0, 2:],
        modes_conserved, modes_reactive, zero_floor_rel,
    )

    chunks = [states[i : i + chunksize] for i in range(0, N, chunksize)]
    tasks = [
        (chunk, modes_conserved, scale_conserved, modes_reactive, scale_reactive,
         seed + i, zero_floor_rel)
        for i, chunk in enumerate(chunks)
    ]

    print(
        f"Génération de {N} états perturbés (multiplicatif, log-espace) "
        f"(conservés={modes_conserved}@{scale_conserved}, "
        f"réactifs={modes_reactive}@{scale_reactive})..."
    )

    with Pool(
        processes=n_workers,
        initializer=_init_perturbation_worker,
        initargs=(kinetics,),
    ) as pool:
        # imap (et non starmap) : les résultats arrivent au fil de l'eau, donc
        # la barre tqdm avance pendant le calcul. L'ordre des chunks est
        # préservé, donc la concaténation reste cohérente avec `states`.
        results = list(
            tqdm(
                pool.imap(_perturb_chunk_star, tasks),
                total=len(tasks),
                desc="Perturbation ILDM",
                unit="chunk",
            )
        )

    Y_all = np.concatenate(results, axis=0)
    TPY_arr = np.concatenate([states[:, 0:2], Y_all], axis=1)

    return _wrap_output(TPY_arr, meta)
