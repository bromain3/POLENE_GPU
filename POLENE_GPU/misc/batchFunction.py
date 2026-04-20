import torch

def batched_apply(fn, *tensors, batch_size, device, dtype=None):
    """
    Applique fn par batch sur GPU, stocke les résultats sur CPU.

    fn         : callable torch
    *tensors   : tenseurs d'entrée (sur CPU ou GPU)
    batch_size : taille des batchs
    device     : device de calcul (ex: torch.device('cuda:0'))
    dtype      : cast optionnel avant calcul
    """
    N = tensors[0].shape[0]
    results = []

    for start in range(0, N, batch_size):
        batch_inputs = []
        print(start/N)
        for t in tensors:
            b = t[start:start + batch_size].to(device)
            if dtype is not None:
                b = b.to(dtype)
            batch_inputs.append(b)

        with torch.no_grad():
            out = fn(*batch_inputs)

        results.append(out.cpu())

    return torch.cat(results, dim=0)