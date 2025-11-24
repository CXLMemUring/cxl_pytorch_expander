#!/usr/bin/env python3
"""
vLLM-style KV Cache Offloading Test

Simulates KV cache offloading for LLM inference, which is a key use case
for CXL memory expansion in AI workloads.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from python.cxl_tensor import CXLTensorManager, CXLTensor


@dataclass
class KVCacheConfig:
    """Configuration for KV cache"""
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    max_seq_len: int = 2048
    dtype: torch.dtype = torch.float16


class KVCacheBlock:
    """A block of KV cache that can be offloaded"""

    def __init__(self, config: KVCacheConfig, block_size: int = 16):
        self.config = config
        self.block_size = block_size

        # Shape: (2, num_layers, block_size, num_heads, head_dim)
        # 2 for K and V
        self.shape = (2, config.num_layers, block_size, config.num_heads, config.head_dim)
        self.dtype = config.dtype

        self._gpu_tensor: Optional[torch.Tensor] = None
        self._cxl_tensor: Optional[CXLTensor] = None
        self._location = 'none'

    @property
    def size_bytes(self) -> int:
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * (2 if self.dtype == torch.float16 else 4)

    def allocate_gpu(self, device: torch.device):
        """Allocate on GPU"""
        self._gpu_tensor = torch.zeros(self.shape, dtype=self.dtype, device=device)
        self._location = 'gpu'

    def fill_with_data(self, data: torch.Tensor):
        """Fill with KV data"""
        if self._location != 'gpu':
            raise RuntimeError("Block must be on GPU to fill")
        self._gpu_tensor.copy_(data)

    def offload_to_cxl(self, manager: CXLTensorManager):
        """Offload to CXL memory"""
        if self._location != 'gpu':
            return

        self._cxl_tensor = CXLTensor(self._gpu_tensor, manager)
        self._cxl_tensor.offload_to_cxl()
        self._gpu_tensor = None
        self._location = 'cxl'

    def load_to_gpu(self, device: torch.device) -> torch.Tensor:
        """Load back to GPU"""
        if self._location == 'gpu':
            return self._gpu_tensor

        if self._location == 'cxl':
            self._gpu_tensor = self._cxl_tensor.to_gpu(device)
            self._cxl_tensor = None
            self._location = 'gpu'
            return self._gpu_tensor

        raise RuntimeError("Block not initialized")

    @property
    def location(self) -> str:
        return self._location


class PagedKVCache:
    """
    Paged KV cache with CXL offloading support.

    Similar to vLLM's paged attention KV cache, but with the ability
    to offload cold pages to CXL memory.
    """

    def __init__(self, config: KVCacheConfig, device: torch.device,
                 num_gpu_blocks: int = 100, num_cxl_blocks: int = 1000):
        self.config = config
        self.device = device
        self.block_size = 16  # Tokens per block

        self.num_gpu_blocks = num_gpu_blocks
        self.num_cxl_blocks = num_cxl_blocks

        # Initialize CXL manager
        self.cxl_manager = CXLTensorManager.get_instance()
        block_size_mb = KVCacheBlock(config, self.block_size).size_bytes / (1024 * 1024)
        buffer_size_mb = int(num_cxl_blocks * block_size_mb * 1.5)
        self.cxl_manager.initialize(buffer_size_mb=max(buffer_size_mb, 64))

        # Block pools
        self.gpu_blocks: Dict[int, KVCacheBlock] = {}
        self.cxl_blocks: Dict[int, KVCacheBlock] = {}
        self.free_gpu_blocks: List[int] = []
        self.free_cxl_blocks: List[int] = list(range(num_cxl_blocks))

        # Pre-allocate GPU blocks
        print(f"Pre-allocating {num_gpu_blocks} GPU blocks "
              f"({num_gpu_blocks * block_size_mb:.1f} MB)...")
        for i in range(num_gpu_blocks):
            block = KVCacheBlock(config, self.block_size)
            block.allocate_gpu(device)
            self.gpu_blocks[i] = block
            self.free_gpu_blocks.append(i)

        # Sequence to block mapping
        self.seq_to_blocks: Dict[int, List[int]] = {}

        # Statistics
        self.stats = {
            'gpu_hits': 0,
            'cxl_hits': 0,
            'offloads': 0,
            'loads': 0,
        }

    def allocate_for_sequence(self, seq_id: int, num_tokens: int) -> List[int]:
        """Allocate blocks for a sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        allocated_blocks = []

        for _ in range(num_blocks_needed):
            if self.free_gpu_blocks:
                block_id = self.free_gpu_blocks.pop(0)
                allocated_blocks.append(block_id)
            else:
                # Need to evict to CXL and allocate
                evicted_block_id = self._evict_lru_to_cxl()
                if evicted_block_id is not None:
                    allocated_blocks.append(evicted_block_id)
                else:
                    raise RuntimeError("No blocks available")

        self.seq_to_blocks[seq_id] = allocated_blocks
        return allocated_blocks

    def _evict_lru_to_cxl(self) -> Optional[int]:
        """Evict least recently used block to CXL"""
        # Simple LRU: evict first used block found
        for seq_id, blocks in self.seq_to_blocks.items():
            if blocks:
                block_id = blocks[0]
                if block_id in self.gpu_blocks:
                    block = self.gpu_blocks[block_id]
                    if block.location == 'gpu':
                        block.offload_to_cxl(self.cxl_manager)
                        self.cxl_blocks[block_id] = block
                        del self.gpu_blocks[block_id]
                        self.stats['offloads'] += 1

                        # Create new GPU block
                        new_block = KVCacheBlock(self.config, self.block_size)
                        new_block.allocate_gpu(self.device)
                        new_block_id = max(self.gpu_blocks.keys()) + 1 if self.gpu_blocks else 0
                        self.gpu_blocks[new_block_id] = new_block
                        return new_block_id
        return None

    def get_block(self, block_id: int) -> torch.Tensor:
        """Get a block, loading from CXL if needed"""
        if block_id in self.gpu_blocks:
            self.stats['gpu_hits'] += 1
            return self.gpu_blocks[block_id].load_to_gpu(self.device)

        if block_id in self.cxl_blocks:
            self.stats['cxl_hits'] += 1
            self.stats['loads'] += 1
            block = self.cxl_blocks[block_id]
            tensor = block.load_to_gpu(self.device)
            # Move block back to GPU pool
            self.gpu_blocks[block_id] = block
            del self.cxl_blocks[block_id]
            return tensor

        raise KeyError(f"Block {block_id} not found")

    def free_sequence(self, seq_id: int):
        """Free blocks for a sequence"""
        if seq_id in self.seq_to_blocks:
            blocks = self.seq_to_blocks.pop(seq_id)
            for block_id in blocks:
                if block_id in self.gpu_blocks:
                    self.free_gpu_blocks.append(block_id)
                elif block_id in self.cxl_blocks:
                    self.free_cxl_blocks.append(block_id)
                    del self.cxl_blocks[block_id]

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'gpu_blocks_used': len(self.gpu_blocks) - len(self.free_gpu_blocks),
            'cxl_blocks_used': len(self.cxl_blocks),
            **self.stats
        }


def simulate_llm_inference():
    """Simulate LLM inference with KV cache offloading"""
    print("\n=== Simulating LLM Inference with KV Cache Offloading ===\n")

    # Configuration similar to LLaMA-7B
    config = KVCacheConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        max_seq_len=2048,
        dtype=torch.float16
    )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Calculate memory requirements
    block_size = 16
    block = KVCacheBlock(config, block_size)
    block_mem_mb = block.size_bytes / (1024 * 1024)
    print(f"KV cache block size: {block_mem_mb:.2f} MB")

    # Create cache with limited GPU blocks to force offloading
    num_gpu_blocks = 50
    num_cxl_blocks = 500

    print(f"GPU blocks: {num_gpu_blocks} ({num_gpu_blocks * block_mem_mb:.1f} MB)")
    print(f"CXL blocks: {num_cxl_blocks} ({num_cxl_blocks * block_mem_mb:.1f} MB)")

    cache = PagedKVCache(config, device, num_gpu_blocks, num_cxl_blocks)

    # Simulate multiple sequences (like batch inference)
    num_sequences = 100
    avg_seq_len = 512

    print(f"\nSimulating {num_sequences} sequences with avg length {avg_seq_len}...")

    start_time = time.time()
    active_sequences = []

    for seq_id in range(num_sequences):
        # Allocate for new sequence
        seq_len = avg_seq_len + (seq_id % 256) - 128  # Vary length
        try:
            blocks = cache.allocate_for_sequence(seq_id, seq_len)
            active_sequences.append(seq_id)
        except RuntimeError as e:
            print(f"  Sequence {seq_id}: {e}")

        # Simulate accessing some blocks (like attention)
        if seq_id > 0 and seq_id % 10 == 0:
            # Access random previous sequence
            access_seq = active_sequences[seq_id // 2] if active_sequences else 0
            if access_seq in cache.seq_to_blocks:
                for block_id in cache.seq_to_blocks[access_seq][:2]:
                    try:
                        _ = cache.get_block(block_id)
                    except:
                        pass

        # Complete some sequences (free memory)
        if seq_id > 20 and seq_id % 5 == 0:
            old_seq = active_sequences.pop(0)
            cache.free_sequence(old_seq)

        if seq_id % 20 == 0:
            stats = cache.get_stats()
            print(f"  Progress: {seq_id}/{num_sequences}, "
                  f"GPU: {stats['gpu_blocks_used']}, CXL: {stats['cxl_blocks_used']}, "
                  f"Offloads: {stats['offloads']}, Loads: {stats['loads']}")

    elapsed = time.time() - start_time

    print(f"\nSimulation completed in {elapsed:.2f}s")
    print(f"\nFinal Statistics:")
    stats = cache.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nCXL Manager Stats: {cache.cxl_manager.get_stats()}")

    return True


def test_kv_block_roundtrip():
    """Test KV cache block offload and restore"""
    print("\n=== Test: KV Cache Block Roundtrip ===\n")

    config = KVCacheConfig(num_layers=4, num_heads=8, head_dim=64)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    manager = CXLTensorManager.get_instance()
    manager.initialize(buffer_size_mb=64)

    # Create and fill block
    block = KVCacheBlock(config, block_size=16)
    block.allocate_gpu(device)

    # Fill with test data
    test_data = torch.randn(block.shape, dtype=config.dtype, device=device)
    block.fill_with_data(test_data)
    original_sum = test_data.sum().item()

    print(f"Block size: {block.size_bytes / 1024:.1f} KB")
    print(f"Original data sum: {original_sum:.4f}")

    # Offload
    start = time.time()
    block.offload_to_cxl(manager)
    offload_time = time.time() - start
    print(f"Offload time: {offload_time*1000:.2f}ms")
    print(f"Block location: {block.location}")

    # Load back
    start = time.time()
    restored = block.load_to_gpu(device)
    load_time = time.time() - start
    print(f"Load time: {load_time*1000:.2f}ms")
    print(f"Block location: {block.location}")

    # Verify
    restored_sum = restored.sum().item()
    print(f"Restored data sum: {restored_sum:.4f}")

    diff = abs(original_sum - restored_sum)
    print(f"Difference: {diff:.6f}")

    if diff < 1.0:
        print("PASSED: KV cache block roundtrip successful")
        return True
    else:
        print("FAILED: Data corruption detected")
        return False


def main():
    print("=" * 60)
    print("CXL Memory Expander - KV Cache Tests")
    print("=" * 60)

    results = []

    results.append(("KV Block Roundtrip", test_kv_block_roundtrip()))
    results.append(("LLM Inference Simulation", simulate_llm_inference()))

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
