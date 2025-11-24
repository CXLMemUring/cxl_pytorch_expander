#!/usr/bin/env python3
"""
Simple CXL Memory Expander Test

A minimal test to verify basic functionality.
Run this first before running more complex tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def main():
    print("=" * 50)
    print("Simple CXL Memory Expander Test")
    print("=" * 50)

    # Import our module
    try:
        from python.cxl_tensor import CXLTensorManager, CXLTensor
        print("[OK] Module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return 1

    # Initialize manager
    manager = CXLTensorManager.get_instance()
    success = manager.initialize(buffer_size_mb=64)
    if success:
        print("[OK] CXL Manager initialized")
    else:
        print("[FAIL] CXL Manager initialization failed")
        return 1

    # Get CXL info
    info = manager.get_info()
    print(f"[INFO] CXL Info: {info}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("[INFO] Using CPU (CUDA not available)")

    # Create test tensor
    print("\n--- Test: Tensor Offload/Restore ---")
    tensor = torch.randn(1000, 1000, device=device, dtype=torch.float32)
    original_sum = tensor.sum().item()
    print(f"[INFO] Created tensor: shape={tensor.shape}, sum={original_sum:.4f}")

    # Wrap in CXL tensor
    cxl_tensor = CXLTensor(tensor, manager)
    print(f"[INFO] CXL Tensor: {cxl_tensor}")

    # Offload to CXL
    cxl_tensor.offload_to_cxl()
    print(f"[INFO] After offload: {cxl_tensor}")

    # Free GPU memory
    del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"[INFO] GPU memory: {torch.cuda.memory_allocated()/1024/1024:.1f}MB")

    # Restore from CXL
    restored = cxl_tensor.to_gpu(device)
    restored_sum = restored.sum().item()
    print(f"[INFO] Restored tensor: shape={restored.shape}, sum={restored_sum:.4f}")

    # Verify
    diff = abs(original_sum - restored_sum)
    print(f"[INFO] Difference: {diff:.6f}")

    if diff < 0.01:
        print("\n[PASS] Test completed successfully!")
        return 0
    else:
        print("\n[FAIL] Data verification failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
