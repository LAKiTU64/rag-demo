#!/usr/bin/env python3
"""Simple CUDA workload for profiling.

Generates several matrix multiplications on GPU to create CUDA kernels for nsys/ncu.
"""
import time
try:
    import torch
except Exception as e:
    raise SystemExit(f"torch not available: {e}")

def workload(iterations: int = 12, size: int = 2048):
    device = 'cuda'
    if not torch.cuda.is_available():
        raise SystemExit('CUDA not available in this environment.')
    for i in range(iterations):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = a @ b  # GEMM kernel
        c = torch.relu(c)
        # Force synchronization so durations are captured
        torch.cuda.synchronize()
        if (i+1) % 3 == 0:
            print(f"Iteration {i+1}/{iterations} done")
    # Small sleep to ensure timeline separation
    time.sleep(0.5)

if __name__ == '__main__':
    workload()