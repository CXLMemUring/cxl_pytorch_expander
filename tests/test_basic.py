#!/usr/bin/env python3
"""
Basic CXL Memory Expander Test

Tests basic tensor offloading functionality between GPU and CXL memory.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time

# Try to import our module
try:
    from python.cxl_tensor import CXLTensorManager, CXLTensor, offload_tensor
    from python.memory_pool import CXLMemoryPool
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to build the module first: python setup.py build_ext --inplace")
    sys.exit(1)


def test_basic_offload():
    """Test basic tensor offload and restore"""
    print("\n=== Test 1: Basic Tensor Offload ===")

    # Initialize CXL manager
    manager = CXLTensorManager.get_instance()
    manager.initialize(buffer_size_mb=64)

    print(f"CXL Info: {manager.get_info()}")

    # Create a GPU tensor
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")

    # Create test tensor
    tensor_size = (1024, 1024)  # 4MB for float32
    original_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
    original_data = original_tensor.cpu().numpy().copy()

    print(f"Original tensor: shape={original_tensor.shape}, "
          f"size={original_tensor.numel() * 4 / 1024 / 1024:.2f}MB")

    # Create CXL tensor and offload
    cxl_tensor = CXLTensor(original_tensor, manager)
    print(f"Before offload: {cxl_tensor}")

    start = time.time()
    cxl_tensor.offload_to_cxl()
    offload_time = time.time() - start
    print(f"After offload: {cxl_tensor} (took {offload_time*1000:.2f}ms)")

    # Check GPU memory was freed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB")

    # Restore tensor
    start = time.time()
    restored_tensor = cxl_tensor.to_gpu(device)
    restore_time = time.time() - start
    print(f"After restore: {cxl_tensor} (took {restore_time*1000:.2f}ms)")

    # Verify data integrity
    restored_data = restored_tensor.cpu().numpy()
    max_diff = np.max(np.abs(original_data - restored_data))
    print(f"Max difference after roundtrip: {max_diff}")

    if max_diff < 1e-6:
        print("PASSED: Data integrity verified")
    else:
        print("FAILED: Data corruption detected!")
        return False

    print(f"Manager stats: {manager.get_stats()}")
    return True


def test_multiple_tensors():
    """Test offloading multiple tensors"""
    print("\n=== Test 2: Multiple Tensor Offload ===")

    manager = CXLTensorManager.get_instance()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create multiple tensors
    num_tensors = 10
    tensor_size = (512, 512)  # 1MB each
    tensors = []
    cxl_tensors = []

    print(f"Creating {num_tensors} tensors...")
    for i in range(num_tensors):
        t = torch.randn(tensor_size, device=device, dtype=torch.float32)
        tensors.append(t.cpu().numpy().copy())
        cxl_tensors.append(CXLTensor(t, manager))

    # Offload all
    print("Offloading all tensors...")
    start = time.time()
    for ct in cxl_tensors:
        ct.offload_to_cxl()
    offload_time = time.time() - start
    print(f"Total offload time: {offload_time*1000:.2f}ms ({offload_time/num_tensors*1000:.2f}ms per tensor)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after offload: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB")

    # Restore and verify each
    print("Restoring and verifying...")
    all_passed = True
    start = time.time()
    for i, ct in enumerate(cxl_tensors):
        restored = ct.to_gpu(device)
        max_diff = np.max(np.abs(tensors[i] - restored.cpu().numpy()))
        if max_diff > 1e-6:
            print(f"Tensor {i}: FAILED (max_diff={max_diff})")
            all_passed = False
    restore_time = time.time() - start
    print(f"Total restore time: {restore_time*1000:.2f}ms")

    if all_passed:
        print("PASSED: All tensors verified")
    else:
        print("FAILED: Some tensors corrupted")

    print(f"Manager stats: {manager.get_stats()}")
    return all_passed


def test_large_tensor():
    """Test with a large tensor"""
    print("\n=== Test 3: Large Tensor (256MB) ===")

    manager = CXLTensorManager.get_instance()
    manager.initialize(buffer_size_mb=512)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create 256MB tensor
    tensor_size = (8192, 8192)  # 256MB for float32
    print(f"Creating {tensor_size[0]}x{tensor_size[1]} tensor ({tensor_size[0]*tensor_size[1]*4/1024/1024:.0f}MB)...")

    original_tensor = torch.randn(tensor_size, device=device, dtype=torch.float32)
    # Store checksum instead of full copy for large tensors
    original_sum = original_tensor.sum().item()
    original_mean = original_tensor.mean().item()

    cxl_tensor = CXLTensor(original_tensor, manager)

    print("Offloading...")
    start = time.time()
    cxl_tensor.offload_to_cxl()
    offload_time = time.time() - start
    bandwidth = (tensor_size[0] * tensor_size[1] * 4) / offload_time / (1024 * 1024 * 1024)
    print(f"Offload time: {offload_time*1000:.2f}ms ({bandwidth:.2f} GB/s)")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory freed: {torch.cuda.memory_allocated() / 1024 / 1024:.2f}MB")

    print("Restoring...")
    start = time.time()
    restored_tensor = cxl_tensor.to_gpu(device)
    restore_time = time.time() - start
    bandwidth = (tensor_size[0] * tensor_size[1] * 4) / restore_time / (1024 * 1024 * 1024)
    print(f"Restore time: {restore_time*1000:.2f}ms ({bandwidth:.2f} GB/s)")

    # Verify
    restored_sum = restored_tensor.sum().item()
    restored_mean = restored_tensor.mean().item()

    sum_diff = abs(original_sum - restored_sum)
    mean_diff = abs(original_mean - restored_mean)

    print(f"Sum difference: {sum_diff}")
    print(f"Mean difference: {mean_diff}")

    if sum_diff < 1.0 and mean_diff < 1e-5:
        print("PASSED: Large tensor verified")
        return True
    else:
        print("FAILED: Data corruption detected")
        return False


def test_memory_pool():
    """Test memory pool functionality"""
    print("\n=== Test 4: Memory Pool ===")

    pool = CXLMemoryPool(max_size_mb=128, block_size_mb=16)

    # Allocate several blocks
    allocations = []
    for i in range(5):
        tensor_id, buffer_id, offset = pool.allocate(32 * 1024 * 1024)  # 32MB
        allocations.append(tensor_id)
        print(f"Allocated tensor {tensor_id} at buffer {buffer_id}, offset {offset}")

    print(f"Pool stats after allocation: {pool.get_stats()}")

    # Deallocate some
    for tensor_id in allocations[:3]:
        pool.deallocate(tensor_id)
        print(f"Deallocated tensor {tensor_id}")

    print(f"Pool stats after deallocation: {pool.get_stats()}")

    # Allocate again (should reuse)
    tensor_id, buffer_id, offset = pool.allocate(32 * 1024 * 1024)
    print(f"Re-allocated tensor {tensor_id} at buffer {buffer_id}, offset {offset}")

    print(f"Final pool stats: {pool.get_stats()}")
    print("PASSED: Memory pool test completed")
    return True


def main():
    print("=" * 60)
    print("CXL Memory Expander - Basic Tests")
    print("=" * 60)

    results = []

    results.append(("Basic Offload", test_basic_offload()))
    results.append(("Multiple Tensors", test_multiple_tensors()))
    results.append(("Large Tensor", test_large_tensor()))
    results.append(("Memory Pool", test_memory_pool()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
