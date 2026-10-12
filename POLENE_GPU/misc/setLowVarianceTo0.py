from POLENE_GPU.dataHandler.serie import serie

import torch

def zero_low_variance_cols(X, Ys, threshold):
    """
    Zero out columns in X (and corresponding columns in Ys) whose std < threshold,
    then renormalize the species columns so that their sum is 1 for each sample.

    Columns 0 and 1 (P and T) are never modified.

    Parameters
    ----------
    X         : serie (N, F)
    Ys        : list[serie]
    threshold : float

    Returns
    -------
    X_clean, Ys_clean : serie, list[serie]
    """

    # ------------------------------------------------------------------
    # 1. Identify low-variance columns
    # ------------------------------------------------------------------
    low_var = X.data.std(dim=0) < threshold

    # Never modify P and T
    low_var[:2] = False

    # ------------------------------------------------------------------
    # 2. Zero low-variance columns in X
    # ------------------------------------------------------------------
    X_clean = X.data.clone()
    X_clean[:, low_var] = 0.0

    # ------------------------------------------------------------------
    # 3. Zero low-variance columns in Ys
    # ------------------------------------------------------------------
    Ys_clean = []

    for y in Ys:
        Y = y.data.clone()
        Y[:, low_var] = 0.0
        Ys_clean.append(Y)

    # ------------------------------------------------------------------
    # 4. Renormalize species (columns 2:)
    # ------------------------------------------------------------------
    X_species_sum = X_clean[:, 2:].sum(dim=1, keepdim=True)

    # Évite une division par zéro si toutes les espèces valent 0
    X_species_sum = torch.where(
        X_species_sum > 0,
        X_species_sum,
        torch.ones_like(X_species_sum)
    )

    X_clean[:, 2:] /= X_species_sum

    # Même chose pour chaque Y
    for Y in Ys_clean:
        Y_species_sum = Y[:, 2:].sum(dim=1, keepdim=True)

        Y_species_sum = torch.where(
            Y_species_sum > 0,
            Y_species_sum,
            torch.ones_like(Y_species_sum)
        )

        Y[:, 2:] /= Y_species_sum

    # ------------------------------------------------------------------
    # 5. Information
    # ------------------------------------------------------------------
    print(
        f"Zeroed {low_var.sum().item()} column(s): "
        f"{[name for name, flag in zip(X.colName, low_var) if flag]}"
    )

    return (
        serie(
            X_clean,
            X.colName,
            storingDevice=X.storingDevice,
            storingDtype=X.storingDtype
        ),
        [
            serie(
                Y_clean,
                y.colName,
                storingDevice=y.storingDevice,
                storingDtype=y.storingDtype
            )
            for Y_clean, y in zip(Ys_clean, Ys)
        ],
    )