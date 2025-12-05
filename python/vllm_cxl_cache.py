"""
vLLM CXL KV Cache Integration

Provides CXL memory expansion for vLLM's KV cache, enabling larger
context windows and more concurrent requests by offloading cold
KV cache blocks to CXL memory.

Usage:
    from python.vllm_cxl_cache import CXLKVCacheManager, patch_vllm_cache

    # Patch vLLM to use CXL-backed cache
    patch_vllm_cache(gpu_cache_ratio=0.7)  # Keep 70% on GPU, 30% can overflow to CXL
"""

import torch
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.cxl_tensor import CXLTensorManager, CXLTensor


@dataclass
class CXLCacheBlock:
    """A KV cache block that can be on GPU or CXL"""
    block_id: int
    gpu_tensor: Optional[torch.Tensor] = None
    cxl_tensor: Optional[CXLTensor] = None
    location: str = 'none'  # 'gpu', 'cxl', 'none'
    last_access: float = 0
    access_count: int = 0


class CXLKVCacheManager:
    """
    Manages KV cache blocks with CXL overflow support.

    When GPU memory pressure is high, cold cache blocks are automatically
    offloaded to CXL memory. Hot blocks are kept on GPU for fast access.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
        device: torch.device = None,
        gpu_blocks: int = 1000,
        cxl_blocks: int = 5000,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device or torch.device('cuda:0')

        self.max_gpu_blocks = gpu_blocks
        self.max_cxl_blocks = cxl_blocks

        # Calculate block size in bytes
        # KV cache shape per block: (2, num_layers, block_size, num_heads, head_dim)
        self.block_shape = (2, num_layers, block_size, num_heads, head_dim)
        element_size = 2 if dtype == torch.float16 else 4
        self.block_size_bytes = 2 * num_layers * block_size * num_heads * head_dim * element_size

        # Initialize CXL manager
        self.cxl_manager = CXLTensorManager.get_instance()
        cxl_buffer_mb = int((cxl_blocks * self.block_size_bytes) / (1024 * 1024) * 1.2)
        self.cxl_manager.initialize(buffer_size_mb=max(cxl_buffer_mb, 256))

        # Block tracking
        self.blocks: Dict[int, CXLCacheBlock] = {}
        self.gpu_block_ids: List[int] = []
        self.cxl_block_ids: List[int] = []
        self.free_block_ids: List[int] = list(range(gpu_blocks + cxl_blocks))

        self._lock = threading.Lock()
        self._next_block_id = 0

        # Statistics
        self.stats = {
            'gpu_hits': 0,
            'cxl_hits': 0,
            'offloads': 0,
            'restores': 0,
            'allocations': 0,
            'deallocations': 0,
        }

        print(f"CXL KV Cache Manager initialized:")
        print(f"  Block shape: {self.block_shape}")
        print(f"  Block size: {self.block_size_bytes / 1024:.1f} KB")
        print(f"  GPU blocks: {gpu_blocks} ({gpu_blocks * self.block_size_bytes / (1024**3):.2f} GB)")
        print(f"  CXL blocks: {cxl_blocks} ({cxl_blocks * self.block_size_bytes / (1024**3):.2f} GB)")

    def allocate_block(self) -> int:
        """Allocate a new cache block, returns block_id"""
        with self._lock:
            if not self.free_block_ids:
                # Try to evict a CXL block
                if self.cxl_block_ids:
                    evict_id = self.cxl_block_ids.pop(0)
                    self._free_block(evict_id)
                else:
                    raise RuntimeError("No free blocks available")

            block_id = self.free_block_ids.pop(0)

            # Allocate on GPU if possible
            if len(self.gpu_block_ids) < self.max_gpu_blocks:
                tensor = torch.zeros(self.block_shape, dtype=self.dtype, device=self.device)
                block = CXLCacheBlock(
                    block_id=block_id,
                    gpu_tensor=tensor,
                    location='gpu',
                    last_access=time.time()
                )
                self.gpu_block_ids.append(block_id)
            else:
                # Allocate in CXL
                cpu_tensor = torch.zeros(self.block_shape, dtype=self.dtype)
                cxl_tensor = CXLTensor(cpu_tensor, self.cxl_manager)
                cxl_tensor.offload_to_cxl()
                block = CXLCacheBlock(
                    block_id=block_id,
                    cxl_tensor=cxl_tensor,
                    location='cxl',
                    last_access=time.time()
                )
                self.cxl_block_ids.append(block_id)

            self.blocks[block_id] = block
            self.stats['allocations'] += 1

            return block_id

    def free_block(self, block_id: int):
        """Free a cache block"""
        with self._lock:
            self._free_block(block_id)

    def _free_block(self, block_id: int):
        """Internal free (must hold lock)"""
        if block_id not in self.blocks:
            return

        block = self.blocks[block_id]

        if block.location == 'gpu':
            block.gpu_tensor = None
            if block_id in self.gpu_block_ids:
                self.gpu_block_ids.remove(block_id)
        elif block.location == 'cxl':
            block.cxl_tensor = None
            if block_id in self.cxl_block_ids:
                self.cxl_block_ids.remove(block_id)

        del self.blocks[block_id]
        self.free_block_ids.append(block_id)
        self.stats['deallocations'] += 1

    def get_block(self, block_id: int) -> torch.Tensor:
        """Get a block's tensor, loading from CXL if needed"""
        with self._lock:
            if block_id not in self.blocks:
                raise KeyError(f"Block {block_id} not found")

            block = self.blocks[block_id]
            block.last_access = time.time()
            block.access_count += 1

            if block.location == 'gpu':
                self.stats['gpu_hits'] += 1
                return block.gpu_tensor

            elif block.location == 'cxl':
                self.stats['cxl_hits'] += 1
                # Restore to GPU
                self._promote_block(block)
                return block.gpu_tensor

            else:
                raise RuntimeError(f"Block {block_id} in invalid state")

    def _promote_block(self, block: CXLCacheBlock):
        """Move a block from CXL to GPU (must hold lock)"""
        if block.location != 'cxl':
            return

        # Make room on GPU if needed
        while len(self.gpu_block_ids) >= self.max_gpu_blocks:
            self._evict_coldest_gpu_block()

        # Restore from CXL
        block.gpu_tensor = block.cxl_tensor.to_gpu(self.device)
        block.cxl_tensor = None
        block.location = 'gpu'

        if block.block_id in self.cxl_block_ids:
            self.cxl_block_ids.remove(block.block_id)
        self.gpu_block_ids.append(block.block_id)

        self.stats['restores'] += 1

    def _evict_coldest_gpu_block(self):
        """Evict the least recently used GPU block to CXL (must hold lock)"""
        if not self.gpu_block_ids:
            return

        # Find coldest block
        coldest_id = None
        coldest_time = float('inf')

        for bid in self.gpu_block_ids:
            block = self.blocks[bid]
            if block.last_access < coldest_time:
                coldest_time = block.last_access
                coldest_id = bid

        if coldest_id is None:
            return

        block = self.blocks[coldest_id]

        # Offload to CXL
        cxl_tensor = CXLTensor(block.gpu_tensor, self.cxl_manager)
        cxl_tensor.offload_to_cxl()

        block.cxl_tensor = cxl_tensor
        block.gpu_tensor = None
        block.location = 'cxl'

        self.gpu_block_ids.remove(coldest_id)
        self.cxl_block_ids.append(coldest_id)

        self.stats['offloads'] += 1
        torch.cuda.empty_cache()

    def write_block(self, block_id: int, key_cache: torch.Tensor, value_cache: torch.Tensor):
        """Write KV data to a block"""
        tensor = self.get_block(block_id)
        tensor[0].copy_(key_cache)
        tensor[1].copy_(value_cache)

    def read_block(self, block_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read KV data from a block"""
        tensor = self.get_block(block_id)
        return tensor[0], tensor[1]

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            return {
                'gpu_blocks_used': len(self.gpu_block_ids),
                'cxl_blocks_used': len(self.cxl_block_ids),
                'free_blocks': len(self.free_block_ids),
                'total_blocks': len(self.blocks),
                **self.stats,
                'gpu_memory_mb': len(self.gpu_block_ids) * self.block_size_bytes / (1024**2),
                'cxl_memory_mb': len(self.cxl_block_ids) * self.block_size_bytes / (1024**2),
            }

    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print("\nCXL KV Cache Statistics:")
        print(f"  GPU blocks: {stats['gpu_blocks_used']} ({stats['gpu_memory_mb']:.1f} MB)")
        print(f"  CXL blocks: {stats['cxl_blocks_used']} ({stats['cxl_memory_mb']:.1f} MB)")
        print(f"  Free blocks: {stats['free_blocks']}")
        print(f"  GPU hits: {stats['gpu_hits']}, CXL hits: {stats['cxl_hits']}")
        print(f"  Offloads: {stats['offloads']}, Restores: {stats['restores']}")


# Global cache manager instance
_cxl_cache_manager: Optional[CXLKVCacheManager] = None


def get_cxl_cache_manager() -> Optional[CXLKVCacheManager]:
    """Get the global CXL cache manager"""
    return _cxl_cache_manager


def init_cxl_cache(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    block_size: int = 16,
    dtype: torch.dtype = torch.float16,
    gpu_blocks: int = 1000,
    cxl_blocks: int = 5000,
) -> CXLKVCacheManager:
    """Initialize the global CXL cache manager"""
    global _cxl_cache_manager

    _cxl_cache_manager = CXLKVCacheManager(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        gpu_blocks=gpu_blocks,
        cxl_blocks=cxl_blocks,
    )

    return _cxl_cache_manager


def demo_cxl_cache():
    """Demo the CXL KV cache manager"""
    print("=" * 60)
    print("CXL KV Cache Manager Demo")
    print("=" * 60)

    # TinyLlama config
    num_layers = 22
    num_heads = 4
    head_dim = 64
    block_size = 16

    # Limited GPU, large CXL
    gpu_blocks = 100
    cxl_blocks = 1000

    manager = init_cxl_cache(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        gpu_blocks=gpu_blocks,
        cxl_blocks=cxl_blocks,
    )

    print("\nSimulating sequence processing...")

    # Allocate blocks for multiple sequences
    num_sequences = 50
    blocks_per_seq = 10
    seq_blocks: Dict[int, List[int]] = {}

    for seq_id in range(num_sequences):
        seq_blocks[seq_id] = []
        for _ in range(blocks_per_seq):
            block_id = manager.allocate_block()
            seq_blocks[seq_id].append(block_id)

        # Access some blocks to simulate attention
        for block_id in seq_blocks[seq_id]:
            _ = manager.get_block(block_id)

        if seq_id % 10 == 0:
            manager.print_stats()

    # Simulate accessing old sequences (causes CXL->GPU restore)
    print("\nAccessing old sequences...")
    for seq_id in range(0, 10):
        for block_id in seq_blocks[seq_id]:
            _ = manager.get_block(block_id)

    manager.print_stats()

    # Free some sequences
    print("\nFreeing sequences 0-19...")
    for seq_id in range(20):
        for block_id in seq_blocks[seq_id]:
            manager.free_block(block_id)

    manager.print_stats()

    print("\nDemo complete!")


if __name__ == "__main__":
    demo_cxl_cache()
