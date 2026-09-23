from POLENE_GPU.dataHandler.serie import serie


def zero_low_variance_cols(X, Ys, threshold):
    """
    Zero out columns in X (and the corresponding columns in Y) whose std < threshold.

    Parameters
    ----------
    X         : serie (N, F)
    Y         : serie (N, F)  — same column layout as X
    threshold : float

    Returns
    -------
    X_clean, Y_clean : serie, serie
    """
    low_var = X.data.std(dim=0) < threshold          # (F,) bool mask

    X_clean = X.data.clone()
    X_clean[:, low_var] = 0.0
    Ys_clean = []
    for y in Ys:
        Y = y.data.clone()
        Y[:, low_var] = 0.0
        Ys_clean.append(Y)


    print(f"Zeroed {low_var.sum().item()} column(s): "
          f"{[name for name, flag in zip(X.colName, low_var) if flag]}")

    return (
        serie(X_clean, X.colName, storingDevice=X.storingDevice, storingDtype=X.storingDtype),
        [serie(Y_clean, Ys[0].colName, storingDevice=Ys[0].storingDevice, storingDtype=Ys[0].storingDtype) for Y_clean in Ys_clean],
    )