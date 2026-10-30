"""
Zonage d'une base de données chimique par analyse des modes de la jacobienne
du réacteur (esprit CEMA, Lu et al. 2010 / CSP, Lam & Goussis / ILDM).

À placer dans POLENE_GPU/misc/jacobianZoning.py.

En chaque état y = (T, C_1..C_n), on forme la jacobienne analytique J = df/dy du
système dy/dt = f(y) du réacteur (cstV ou cstP, mêmes équations que
`computeConditionNumberParallel`), puis on diagonalise J. On en déduit, pour
chaque point, des features DIMENSIONNELLEMENT HOMOGÈNES :

  lam_e_dt  : lambda_e * dt, lambda_e = max Re(lambda_i)     (mode explosif si > 0)
  phi_e     : symlog(lam_e_dt) = sign * log10(1 + |x| / lambda_scale)
  log_stiff : log10( max |Re lambda_i| * dt )                 (raideur / mode le plus rapide)
  n_exh     : nb de modes "épuisés" sur un pas : exp(Re lambda_i dt) < exh_thr
  a_fast    : ||P_fast f|| / ||f||, P_fast = projecteur spectral sur les modes
              épuisés, f = champ de vitesse (échelles relatives). ~1 : l'état
              est encore en train de relaxer vers la variété lente ; ~0 : il est
              sur la variété lente.
  log_act   : log10( ||f_relatif|| * dt )  (variation relative par pas)

Les valeurs propres sont invariantes par changement d'échelle des variables
(similitude D^-1 J D) : `scale_floor_rel` n'influence que a_fast et log_act.

Approximation : dans la jacobienne, on néglige la dépendance en T des capacités
calorifiques molaires dans rho*c (terme d(cp)/dT), standard en CEMA.
"""

import numpy as np
import cantera as ct
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

try:
    import torch
except ImportError:  # le module reste utilisable avec des ndarray seuls
    torch = None

FEATURE_KEYS = ('phi_e', 'log_stiff', 'n_exh', 'a_fast', 'log_act')

_worker_gas = None


def _init_modes_worker(kinetics):
    global _worker_gas
    try:  # évite la sur-souscription BLAS quand on a beaucoup de workers
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass
    _worker_gas = ct.Solution(kinetics)


# --------------------------------------------------------------------
# Jacobienne analytique du réacteur, variables (T, C_1..C_n)
# --------------------------------------------------------------------

def _reactor_jacobian(sa, reactor_type):
    omega = sa.net_production_rates              # (N, n) kmol/m3/s
    A = sa.net_production_rates_ddCi             # (N, n, n)  d omega_k / d C_j
    b = sa.net_production_rates_ddT              # (N, n)     d omega_k / d T
    C = sa.concentrations
    T = sa.T
    N, n = C.shape

    if reactor_type == 'cstV':
        e = sa.partial_molar_int_energies                       # J/kmol
        c = sa.partial_molar_cp - ct.gas_constant               # cv molaire
    elif reactor_type == 'cstP':
        e = sa.partial_molar_enthalpies
        c = sa.partial_molar_cp
    else:
        raise ValueError("reactor_type must be 'cstV' or 'cstP'")

    rhoc = np.sum(C * c, axis=1)                                # rho*cv ou rho*cp
    Tdot = -np.sum(e * omega, axis=1) / rhoc
    dTdot_dC = (-np.einsum('nk,nkj->nj', e, A) / rhoc[:, None]
                - Tdot[:, None] * c / rhoc[:, None])
    dTdot_dT = -(np.sum(c * omega, axis=1) + np.sum(e * b, axis=1)) / rhoc

    if reactor_type == 'cstP':
        S_om = omega.sum(axis=1)
        S_C = C.sum(axis=1)
        v = Tdot / T + S_om / S_C                               # Vdot / V
        dv_dC = (dTdot_dC / T[:, None] + A.sum(axis=1) / S_C[:, None]
                 - (S_om / S_C ** 2)[:, None])
        dv_dT = dTdot_dT / T - Tdot / T ** 2 + b.sum(axis=1) / S_C
    else:
        v = np.zeros(N)
        dv_dC = np.zeros((N, n))
        dv_dT = np.zeros(N)

    Cdot = omega - C * v[:, None]
    f = np.concatenate([Tdot[:, None], Cdot], axis=1)

    J = np.zeros((N, n + 1, n + 1))
    J[:, 0, 0] = dTdot_dT
    J[:, 0, 1:] = dTdot_dC
    J[:, 1:, 0] = b - C * dv_dT[:, None]
    J[:, 1:, 1:] = A - C[:, :, None] * dv_dC[:, None, :]
    idx = np.arange(1, n + 1)
    J[:, idx, idx] -= v[:, None]
    return J, f, T, C


def _batched_eig(A):
    N, m, _ = A.shape
    try:
        lam, V = np.linalg.eig(A)
        return lam, V, np.ones(N, dtype=bool)
    except np.linalg.LinAlgError:
        lam = np.zeros((N, m), dtype=complex)
        V = np.tile(np.eye(m, dtype=complex), (N, 1, 1))
        ok = np.zeros(N, dtype=bool)
        for i in range(N):
            try:
                lam[i], V[i] = np.linalg.eig(A[i])
                ok[i] = True
            except np.linalg.LinAlgError:
                pass
        return lam, V, ok


def _symlog(x, x0):
    return np.sign(x) * np.log10(1.0 + np.abs(x) / x0)


def _modes_chunk(states, reactor_type, dt, exh_thr, lambda_scale, scale_floor_rel, act_floor):
    gas = _worker_gas
    N = states.shape[0]
    sa = ct.SolutionArray(gas, shape=(N,))
    sa.TPY = states[:, 0], states[:, 1], states[:, 2:]

    J, f, T, C = _reactor_jacobian(sa, reactor_type)

    # échelles : T et max(C_k, plancher relatif) -> jacobienne en variables relatives
    d = np.empty_like(f)
    d[:, 0] = T
    d[:, 1:] = np.maximum(C, scale_floor_rel * C.sum(axis=1, keepdims=True))
    Js = J * d[:, None, :] / d[:, :, None]
    fs = f / d

    valid = np.isfinite(Js).all(axis=(1, 2)) & np.isfinite(fs).all(axis=1)
    Js[~valid] = 0.0
    fs[~valid] = 0.0

    lam, V, ok = _batched_eig(Js)
    valid &= ok
    lr = lam.real

    # projection de f sur les modes épuisés : coefficients modaux c = V^-1 f
    rhs = fs[:, :, None].astype(complex)
    try:
        coef = np.linalg.solve(V, rhs)[:, :, 0]
    except np.linalg.LinAlgError:
        coef = (np.linalg.pinv(V) @ rhs)[:, :, 0]
    fast = (lr * dt) < np.log(exh_thr)
    f_fast = np.einsum('nij,nj->ni', V, coef * fast).real
    nf = np.linalg.norm(fs, axis=1)
    a_fast = np.linalg.norm(f_fast, axis=1) / np.maximum(nf, 1e-300)
    # point fixe (||f|| dt ~ 0) : la direction de f n'a plus de sens (bruit numérique)
    a_fast[np.log10(np.maximum(nf * dt, 1e-30)) < act_floor] = 0.0

    lam_e_dt = lr.max(axis=1) * dt
    stiff_dt = np.abs(lr).max(axis=1) * dt

    return dict(
        lam_e_dt=lam_e_dt,
        phi_e=_symlog(lam_e_dt, lambda_scale),
        log_stiff=np.log10(np.maximum(stiff_dt, 1e-12)),
        n_exh=fast.sum(axis=1).astype(np.float64),
        a_fast=a_fast,
        log_act=np.log10(np.maximum(nf * dt, 1e-30)),
        valid=valid,
    )


def _modes_chunk_star(args):
    return _modes_chunk(*args)


def _extract_states(TPY):
    if isinstance(TPY, np.ndarray):
        return TPY
    if torch is not None and isinstance(TPY, torch.Tensor):
        return TPY.cpu().detach().numpy()
    return TPY.data.cpu().detach().numpy()


# --------------------------------------------------------------------
# Interface 1 : features spectrales
# --------------------------------------------------------------------

def computeJacobianModeFeatures(
    TPY, kinetics, reactor_type='cstP', dt=1e-7,
    exh_thr=0.5, lambda_scale=1e-5, scale_floor_rel=1e-12, act_floor=-8.0,
    n_workers=None, chunksize=300,
):
    """
    TPY : serie / torch.Tensor / ndarray (N, 2 + n_sp), colonnes [T, P, Y_1..Y_n].
    exh_thr : un mode est "épuisé" si exp(Re(lambda) dt) < exh_thr (0.5 -> lambda*dt < -0.69).
              Valeurs plus basses = seuls les modes ultra-rapides comptent (a_fast ne voit plus
              l'équilibrage des mineurs aux premiers pas).
    lambda_scale : échelle x0 de la symlog appliquée à lambda_e * dt (1e-5 : sépare lambda_e*dt
              de pré-flamme, 1e-6..1e-3, du bruit numérique post-flamme, <1e-7).
    act_floor : log10(||f|| dt) en dessous duquel a_fast est forcé à 0 (point fixe).
    scale_floor_rel : plancher relatif de C_k pour les échelles (n'affecte que a_fast, log_act).

    Retourne un dict d'arrays (N,) : lam_e_dt, phi_e, log_stiff, n_exh, a_fast, log_act, valid.
    """
    states = _extract_states(TPY)
    N = states.shape[0]
    if n_workers is None:
        n_workers = mp.cpu_count()

    chunks = [states[i:i + chunksize] for i in range(0, N, chunksize)]
    tasks = [(c, reactor_type, dt, exh_thr, lambda_scale, scale_floor_rel, act_floor) for c in chunks]

    print(f"Modes de la jacobienne ({reactor_type}) sur {N} points...")
    with Pool(processes=n_workers, initializer=_init_modes_worker, initargs=(kinetics,)) as pool:
        results = list(tqdm(pool.imap(_modes_chunk_star, tasks), total=len(tasks),
                            desc="Jacobian modes"))

    feat = {k: np.concatenate([r[k] for r in results]) for k in results[0]}
    n_bad = int((~feat['valid']).sum())
    if n_bad:
        print(f"  [warning] {n_bad} points invalides (NaN / eig non convergé), label -1")
    return feat


# --------------------------------------------------------------------
# Interface 2 : features -> zones
# --------------------------------------------------------------------

ZONE_ORDER_DOC = "0 équilibrage | 1 induction (mode explosif faible) | 2 flamme | 3 post-flamme"


def _name_clusters(centroids):
    """Associe 4 clusters aux zones 0..3 d'après leurs centroïdes (unités d'origine)."""
    k = centroids.shape[0]
    if k != 4:
        return {i: i for i in range(k)}
    ia, ie = FEATURE_KEYS.index('a_fast'), FEATURE_KEYS.index('phi_e')
    rem = list(range(4))
    c_eq = max(rem, key=lambda c: centroids[c, ia]); rem.remove(c_eq)
    c_fl = max(rem, key=lambda c: centroids[c, ie]); rem.remove(c_fl)
    c_in = max(rem, key=lambda c: centroids[c, ie]); rem.remove(c_in)
    return {c_eq: 0, c_in: 1, c_fl: 2, rem[0]: 3}


def zonesFromFeatures(
    feat, method='gmm', n_clusters=4, fit_sample=100_000, seed=0,
    lam_eps=1e-5, lam_flame=0.02, a_fast_thr=0.5, verbose=True,
):
    """
    method='gmm'   : GMM (covariance pleine) sur les features standardisées ; les 4 clusters
                     sont nommés d'après leurs centroïdes (a_fast max -> équilibrage,
                     phi_e max -> flamme, puis induction / post-flamme).
    method='rules' : règles fondées sur la physique des modes (seuils sur lambda_e*dt) :
                       lam_e_dt > lam_flame           -> 2 flamme
                       sinon a_fast > a_fast_thr      -> 0 équilibrage
                       sinon lam_e_dt > lam_eps       -> 1 induction (mode explosif)
                       sinon                          -> 3 post-flamme  (Re lambda_max <= 0)

    Retourne (zone (N,) int64 avec -1 pour les points invalides, info dict).
    """
    valid = feat['valid']
    zone = np.full(valid.shape[0], -1, dtype=np.int64)
    info = {}

    if method == 'rules':
        lam = feat['lam_e_dt']
        z = np.full(valid.shape[0], 3, dtype=np.int64)
        z[lam > lam_eps] = 1
        z[feat['a_fast'] > a_fast_thr] = 0
        z[lam > lam_flame] = 2
        zone[valid] = z[valid]

    elif method == 'gmm':
        from sklearn.mixture import GaussianMixture
        F = np.column_stack([feat[k] for k in FEATURE_KEYS])[valid]
        mu, sd = F.mean(axis=0), F.std(axis=0) + 1e-12
        Zs = (F - mu) / sd
        rng = np.random.default_rng(seed)
        sub = rng.choice(len(Zs), size=min(fit_sample, len(Zs)), replace=False)
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full',
                              n_init=3, random_state=seed).fit(Zs[sub])
        raw = gmm.predict(Zs)
        mapping = _name_clusters(gmm.means_ * sd + mu)
        zone[valid] = np.vectorize(mapping.get)(raw)
        info.update(gmm=gmm, mu=mu, sd=sd, mapping=mapping)
    else:
        raise ValueError("method doit valoir 'gmm' ou 'rules'")

    if verbose:
        print(f"\nZonage '{method}' ({ZONE_ORDER_DOC})")
        hdr = "  zone |  n pts   | " + " ".join(f"{k:>9s}" for k in FEATURE_KEYS)
        print(hdr)
        for zz in range(n_clusters if method == 'gmm' else 4):
            m = zone == zz
            if m.sum() == 0:
                print(f"  {zz:4d} | {0:8d} |")
                continue
            vals = " ".join(f"{feat[k][m].mean():9.3f}" for k in FEATURE_KEYS)
            print(f"  {zz:4d} | {m.sum():8d} | {vals}")
    return zone, info


def zoneQuality(feat, zone, keys=('log_stiff', 'phi_e', 'a_fast')):
    """
    Part de variance expliquée par le zonage (R^2 = 1 - variance intra / totale) pour
    chaque feature : plus c'est proche de 1, plus les zones sont homogènes en structure
    spectrale. Utile pour comparer deux zonages sur les MÊMES features.
    """
    m = (zone >= 0) & feat['valid']
    out = {}
    for k in keys:
        v, z = feat[k][m], zone[m]
        tot = ((v - v.mean()) ** 2).sum()
        within = sum(((v[z == c] - v[z == c].mean()) ** 2).sum() for c in np.unique(z))
        out[k] = float(1.0 - within / max(tot, 1e-300))
    return out