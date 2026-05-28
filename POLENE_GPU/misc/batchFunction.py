import torch
from tqdm import tqdm

def batched_apply(fn, *tensors, batch_size=None, device=None, dtype=None):
    N = tensors[0].shape[0]

    def process_batch(start, end):
        batch_inputs = [t[start:end].to(device) for t in tensors]
        if dtype is not None:
            batch_inputs = [b.to(dtype) for b in batch_inputs]
        with torch.no_grad():
            return fn(*batch_inputs).cpu()

    if batch_size is None:
        return process_batch(0, N)

    results = [process_batch(start, start + batch_size) for start in tqdm(range(0, N, batch_size))]
    return torch.cat(results, dim=0)
