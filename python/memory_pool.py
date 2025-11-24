"""
CXL Memory Pool for PyTorch

Provides a memory pool abstraction for managing CXL memory allocation,
similar to PyTorch's caching allocator but for CXL memory.
"""

import torch
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class MemoryBlock:
    """A block of memory in the CXL pool"""
    buffer_id: int
    offset: int
    size: int
    in_use: bool = False
    tensor_id: Optional[int] = None
    last_access: float = 0


class CXLMemoryPool:
    """
    Memory pool for CXL memory with LRU eviction.

    Features:
    - Automatic memory allocation and deallocation
    - LRU-based eviction for memory pressure
    - Coalescing of free blocks
    - Statistics tracking
    """

    def __init__(self, max_size_mb: int = 1024, block_size_mb: int = 64):
        self._max_size = max_size_mb * 1024 * 1024
        self._block_size = block_size_mb * 1024 * 1024
        self._allocated_size = 0

        self._blocks: Dict[int, MemoryBlock] = {}  # block_id -> MemoryBlock
        self._free_blocks: List[int] = []  # List of free block IDs
        self._tensor_to_block: Dict[int, int] = {}  # tensor_id -> block_id

        self._next_block_id = 0
        self._next_tensor_id = 0

        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            'allocations': 0,
            'deallocations': 0,
            'evictions': 0,
            'hits': 0,
            'misses': 0,
        }

    def allocate(self, size: int) -> Tuple[int, int, int]:
        """
        Allocate memory from the pool.

        Returns: (tensor_id, buffer_id, offset)
        """
        with self._lock:
            # Round up to block size
            aligned_size = ((size + self._block_size - 1) // self._block_size) * self._block_size

            # Try to find a free block
            for block_id in self._free_blocks:
                block = self._blocks[block_id]
                if block.size >= aligned_size and not block.in_use:
                    block.in_use = True
                    block.last_access = time.time()
                    tensor_id = self._next_tensor_id
                    self._next_tensor_id += 1
                    block.tensor_id = tensor_id
                    self._tensor_to_block[tensor_id] = block_id
                    self._free_blocks.remove(block_id)
                    self._stats['hits'] += 1
                    self._stats['allocations'] += 1
                    return tensor_id, block.buffer_id, block.offset

            # Need to allocate new block
            self._stats['misses'] += 1

            # Check if we need to evict
            if self._allocated_size + aligned_size > self._max_size:
                self._evict_lru(aligned_size)

            # Allocate new block
            block_id = self._next_block_id
            self._next_block_id += 1

            tensor_id = self._next_tensor_id
            self._next_tensor_id += 1

            block = MemoryBlock(
                buffer_id=block_id,  # In real impl, this would be CXL buffer ID
                offset=0,
                size=aligned_size,
                in_use=True,
                tensor_id=tensor_id,
                last_access=time.time()
            )

            self._blocks[block_id] = block
            self._tensor_to_block[tensor_id] = block_id
            self._allocated_size += aligned_size
            self._stats['allocations'] += 1

            return tensor_id, block.buffer_id, block.offset

    def deallocate(self, tensor_id: int):
        """Return memory to the pool"""
        with self._lock:
            if tensor_id not in self._tensor_to_block:
                return

            block_id = self._tensor_to_block.pop(tensor_id)
            block = self._blocks[block_id]
            block.in_use = False
            block.tensor_id = None
            self._free_blocks.append(block_id)
            self._stats['deallocations'] += 1

    def _evict_lru(self, needed_size: int):
        """Evict least recently used blocks to free space"""
        # Sort blocks by last access time
        used_blocks = [
            (bid, b) for bid, b in self._blocks.items()
            if b.in_use
        ]
        used_blocks.sort(key=lambda x: x[1].last_access)

        freed = 0
        for block_id, block in used_blocks:
            if freed >= needed_size:
                break

            # Evict this block
            if block.tensor_id in self._tensor_to_block:
                del self._tensor_to_block[block.tensor_id]

            block.in_use = False
            block.tensor_id = None
            self._free_blocks.append(block_id)
            freed += block.size
            self._stats['evictions'] += 1

    def get_stats(self) -> Dict:
        """Get pool statistics"""
        with self._lock:
            used_size = sum(b.size for b in self._blocks.values() if b.in_use)
            return {
                'total_allocated_mb': self._allocated_size / (1024 * 1024),
                'used_mb': used_size / (1024 * 1024),
                'free_mb': (self._allocated_size - used_size) / (1024 * 1024),
                'num_blocks': len(self._blocks),
                'num_free_blocks': len(self._free_blocks),
                **self._stats
            }

    def clear(self):
        """Clear all allocations"""
        with self._lock:
            self._blocks.clear()
            self._free_blocks.clear()
            self._tensor_to_block.clear()
            self._allocated_size = 0


class GPUCXLTieredMemory:
    """
    Tiered memory system with GPU as fast tier and CXL as capacity tier.

    Automatically moves cold tensors to CXL and hot tensors to GPU.
    """

    def __init__(self, gpu_budget_mb: int = 4096, cxl_budget_mb: int = 16384):
        from .cxl_tensor import CXLTensorManager, CXLTensor

        self._gpu_budget = gpu_budget_mb * 1024 * 1024
        self._cxl_budget = cxl_budget_mb * 1024 * 1024

        self._cxl_manager = CXLTensorManager.get_instance()
        self._cxl_pool = CXLMemoryPool(max_size_mb=cxl_budget_mb)

        # Track tensors
        self._gpu_tensors: Dict[int, torch.Tensor] = {}
        self._cxl_tensors: Dict[int, 'CXLTensor'] = {}
        self._access_count: Dict[int, int] = defaultdict(int)
        self._next_id = 0
        self._lock = threading.Lock()

    def register_tensor(self, tensor: torch.Tensor) -> int:
        """Register a tensor with the tiered memory system"""
        with self._lock:
            tensor_id = self._next_id
            self._next_id += 1

            if tensor.is_cuda:
                self._gpu_tensors[tensor_id] = tensor
            else:
                # Offload CPU tensors to CXL
                from .cxl_tensor import CXLTensor
                cxl_tensor = CXLTensor(tensor.cuda())
                cxl_tensor.offload_to_cxl()
                self._cxl_tensors[tensor_id] = cxl_tensor

            return tensor_id

    def get_tensor(self, tensor_id: int, device: torch.device = None) -> torch.Tensor:
        """Get a tensor, potentially moving it between tiers"""
        with self._lock:
            self._access_count[tensor_id] += 1

            if tensor_id in self._gpu_tensors:
                return self._gpu_tensors[tensor_id]

            if tensor_id in self._cxl_tensors:
                # Promote to GPU
                cxl_tensor = self._cxl_tensors[tensor_id]
                gpu_tensor = cxl_tensor.to_gpu(device)

                # Check if we need to evict something from GPU
                self._maybe_evict_from_gpu()

                self._gpu_tensors[tensor_id] = gpu_tensor
                del self._cxl_tensors[tensor_id]

                return gpu_tensor

            raise KeyError(f"Tensor {tensor_id} not found")

    def _maybe_evict_from_gpu(self):
        """Evict cold tensors from GPU to CXL if needed"""
        gpu_used = sum(t.numel() * t.element_size()
                      for t in self._gpu_tensors.values())

        if gpu_used <= self._gpu_budget:
            return

        # Find coldest tensors
        tensors_by_access = sorted(
            self._gpu_tensors.items(),
            key=lambda x: self._access_count[x[0]]
        )

        from .cxl_tensor import CXLTensor

        for tensor_id, tensor in tensors_by_access:
            if gpu_used <= self._gpu_budget * 0.8:  # Keep 20% headroom
                break

            # Offload to CXL
            cxl_tensor = CXLTensor(tensor)
            cxl_tensor.offload_to_cxl()
            self._cxl_tensors[tensor_id] = cxl_tensor
            del self._gpu_tensors[tensor_id]

            gpu_used -= tensor.numel() * tensor.element_size()
            torch.cuda.empty_cache()

    def get_stats(self) -> Dict:
        """Get memory statistics"""
        with self._lock:
            gpu_used = sum(t.numel() * t.element_size()
                          for t in self._gpu_tensors.values())
            cxl_count = len(self._cxl_tensors)

            return {
                'gpu_tensors': len(self._gpu_tensors),
                'cxl_tensors': cxl_count,
                'gpu_used_mb': gpu_used / (1024 * 1024),
                'gpu_budget_mb': self._gpu_budget / (1024 * 1024),
            }
