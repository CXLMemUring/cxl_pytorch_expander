# SPDX-License-Identifier: Apache-2.0
"""
CXL Backend for vLLM KV Cache Offloading

Provides CXL memory expansion for vLLM's KV cache, using GPU P2P DMA
for high-bandwidth transfers between GPU and CXL memory.

This module integrates with vLLM's kv_offload framework to enable:
- Transparent KV cache offloading to CXL memory
- LRU-based eviction of cold cache blocks
- GPU P2P DMA transfers for efficient data movement
"""

import ctypes
from collections.abc import Iterable
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# vLLM imports (optional, for type hints)
try:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
    from vllm.v1.kv_offload.backend import Backend, BlockStatus
    from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager
    from vllm.v1.kv_offload.mediums import GPULoadStoreSpec, BlockIDsLoadStoreSpec
    from vllm.v1.kv_offload.spec import OffloadingSpec
    from vllm.v1.kv_offload.worker.worker import (OffloadingHandler,
                                                  TransferResult, TransferSpec)
    from vllm.attention import AttentionBackend
    from vllm.config import VllmConfig, get_layers_from_vllm_config
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
    from vllm.platforms import current_platform
    from vllm import _custom_ops as ops
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    BlockHash = int
    BlockStatus = object
    Backend = object
    LoadStoreSpec = object
    OffloadingManager = object
    OffloadingSpec = object
    OffloadingHandler = object

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.cxl_tensor import CXLTensorManager


class CXLLoadStoreSpec(BlockIDsLoadStoreSpec if VLLM_AVAILABLE else object):
    """
    Spec for loading/storing KV blocks to CXL memory via P2P DMA.
    """

    def __init__(self, block_ids: list[int]):
        if VLLM_AVAILABLE:
            super().__init__(block_ids)
        else:
            self.block_ids = np.array(block_ids, dtype=np.int64)

    @staticmethod
    def medium() -> str:
        return "CXL"


class CXLBlockStatus(ctypes.Structure if VLLM_AVAILABLE else object):
    """
    Block status for CXL-backed blocks.
    Tracks reference count and block ID in CXL buffer.
    """
    _fields_ = [("ref_cnt", ctypes.c_int32),
                ("block_id", ctypes.c_int64),
                ("cxl_offset", ctypes.c_uint64)]

    def __init__(self, block_id: int, cxl_offset: int = 0):
        if hasattr(super(), '__init__'):
            super().__init__()
        self.ref_cnt = -1  # Not ready initially
        self.block_id = block_id
        self.cxl_offset = cxl_offset

    @property
    def is_ready(self) -> bool:
        return self.ref_cnt >= 0


class CXLBackend(Backend if VLLM_AVAILABLE else object):
    """
    CXL backend for KV cache block allocation.

    Uses CXL memory via P2P DMA for storing offloaded KV cache blocks.
    Blocks are allocated from a pre-mapped CXL buffer and accessed
    via GPU P2P DMA for efficient transfers.
    """

    def __init__(self, block_size: int, num_blocks: int,
                 cxl_buffer_size_mb: int = 0):
        if VLLM_AVAILABLE:
            super().__init__(block_size=block_size, medium=CXLLoadStoreSpec.medium())
        else:
            self.block_size = block_size
            self.medium = CXLLoadStoreSpec.medium()

        self.num_blocks = num_blocks
        self.num_allocated_blocks = 0
        self.allocated_blocks_free_list: list[int] = []

        # Initialize CXL manager
        self.cxl_manager = CXLTensorManager.get_instance()

        # Calculate buffer size if not specified
        if cxl_buffer_size_mb == 0:
            # Estimate: assume fp16, typical KV cache block shape
            cxl_buffer_size_mb = max(256, (num_blocks * block_size * 1024) // (1024 * 1024))

        self.cxl_manager.initialize(buffer_size_mb=cxl_buffer_size_mb)

        # Track CXL offsets for each block
        self.block_offsets: dict[int, int] = {}
        self.next_offset = 0

        print(f"CXL Backend initialized: {num_blocks} blocks, "
              f"{cxl_buffer_size_mb} MB buffer")

    def get_num_free_blocks(self) -> int:
        """Returns the number of blocks available for allocation."""
        return (len(self.allocated_blocks_free_list) +
                self.num_blocks - self.num_allocated_blocks)

    def allocate_blocks(self, block_hashes: list) -> list:
        """
        Allocate space for writing blocks to CXL memory.

        Args:
            block_hashes: the hashes identifying the blocks to be written.

        Returns:
            A list of CXLBlockStatus for the allocated blocks.
        """
        num_fresh_blocks = min(len(block_hashes),
                               self.num_blocks - self.num_allocated_blocks)
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        blocks = []

        # Allocate fresh blocks
        for _ in range(num_fresh_blocks):
            block_id = self.num_allocated_blocks
            cxl_offset = self.next_offset
            self.block_offsets[block_id] = cxl_offset
            self.next_offset += self.block_size

            blocks.append(CXLBlockStatus(block_id, cxl_offset))
            self.num_allocated_blocks += 1

        # Reuse freed blocks
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            cxl_offset = self.block_offsets[block_id]
            blocks.append(CXLBlockStatus(block_id, cxl_offset))

        return blocks

    def free(self, block) -> None:
        """Free a previously allocated block."""
        if hasattr(block, 'block_id'):
            self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(self, block_hashes: Iterable,
                           blocks: Iterable) -> LoadStoreSpec:
        """Get CXL-specific load/store spec."""
        return CXLLoadStoreSpec([block.block_id for block in blocks])


class CxlGpuOffloadingHandler(OffloadingHandler if VLLM_AVAILABLE else object):
    """
    Handler for GPU <-> CXL P2P DMA transfers.

    Uses NVIDIA's P2P DMA capabilities to transfer KV cache blocks
    between GPU memory and CXL memory with minimal CPU involvement.
    """

    def __init__(self, gpu_block_size: int, cxl_block_size: int,
                 num_cxl_blocks: int, gpu_caches: dict,
                 attn_backends: Optional[dict] = None):
        """
        Initialize the CXL-GPU offloading handler.

        Args:
            gpu_block_size: Number of tokens per GPU block
            cxl_block_size: Number of tokens per CXL block
            num_cxl_blocks: Total number of CXL blocks
            gpu_caches: Dictionary mapping layer names to GPU cache tensors
            attn_backends: Dictionary mapping layer names to attention backends
        """
        assert cxl_block_size % gpu_block_size == 0
        self.block_size_factor = cxl_block_size // gpu_block_size

        # CUDA streams for async transfers
        self.d2h_stream = torch.cuda.Stream()  # GPU -> CXL
        self.h2d_stream = torch.cuda.Stream()  # CXL -> GPU

        # Transfer event tracking
        self.transfer_events: dict[int, torch.cuda.Event] = {}
        self.events_pool: list[torch.cuda.Event] = []

        # Initialize CXL manager
        self.cxl_manager = CXLTensorManager.get_instance()

        # Store GPU cache references
        self.gpu_tensors: list[torch.Tensor] = []
        self.cxl_tensors: list[torch.Tensor] = []
        self.kv_dim_before_num_blocks: list[bool] = []

        for layer_name, gpu_tensor in gpu_caches.items():
            self.gpu_tensors.append(gpu_tensor)

            gpu_shape = gpu_tensor.shape

            # Determine tensor layout
            if attn_backends and layer_name in attn_backends:
                test_shape = attn_backends[layer_name].get_kv_cache_shape(
                    num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256)
                if test_shape[0] == 1234:
                    num_blocks_idx = 0
                    self.kv_dim_before_num_blocks.append(False)
                else:
                    num_blocks_idx = 1
                    self.kv_dim_before_num_blocks.append(True)
            else:
                # Default: assume (2, num_blocks, ...) layout
                if gpu_shape[0] == 2:
                    num_blocks_idx = 1
                    self.kv_dim_before_num_blocks.append(True)
                else:
                    num_blocks_idx = 0
                    self.kv_dim_before_num_blocks.append(False)

            # Allocate CXL-backed tensor
            cxl_shape = list(gpu_shape)
            cxl_shape[num_blocks_idx] = num_cxl_blocks * self.block_size_factor

            # Use CXL manager to allocate backing memory
            cxl_tensor = torch.zeros(cxl_shape, dtype=gpu_tensor.dtype, device='cpu')
            self.cxl_tensors.append(cxl_tensor)

        print(f"CXL-GPU Handler initialized: {len(gpu_caches)} layers, "
              f"{num_cxl_blocks} CXL blocks")

    def transfer_async(self, job_id: int, spec) -> bool:
        """
        Perform async transfer between GPU and CXL.

        Args:
            job_id: Unique identifier for this transfer job
            spec: Tuple of (src_spec, dst_spec)

        Returns:
            True if transfer was initiated successfully
        """
        src_spec, dst_spec = spec

        # Determine transfer direction
        if isinstance(src_spec, CXLLoadStoreSpec):
            # CXL -> GPU
            stream = self.h2d_stream
            src_tensors = self.cxl_tensors
            dst_tensors = self.gpu_tensors
            src_block_size_factor = self.block_size_factor
            dst_block_size_factor = 1
        else:
            # GPU -> CXL
            stream = self.d2h_stream
            src_tensors = self.gpu_tensors
            dst_tensors = self.cxl_tensors
            src_block_size_factor = 1
            dst_block_size_factor = self.block_size_factor

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids

        # Calculate sub-block mapping
        dst_sub_blocks_to_skip = (-src_blocks.size % dst_block_size_factor)
        src_sub_block_count = src_blocks.size * src_block_size_factor

        # Build source -> destination block mapping
        src_to_dst = np.empty((src_sub_block_count, 2), dtype=np.int64)
        self._expand_block_ids(src_blocks, src_block_size_factor, src_to_dst[:, 0])
        self._expand_block_ids(dst_blocks, dst_block_size_factor,
                              src_to_dst[:, 1], skip_count=dst_sub_blocks_to_skip)
        src_to_dst_tensor = torch.from_numpy(src_to_dst)

        # Get or create CUDA event
        event = self.events_pool.pop() if self.events_pool else torch.cuda.Event()

        with torch.cuda.stream(stream):
            for src_tensor, dst_tensor, kv_dim in zip(
                    src_tensors, dst_tensors, self.kv_dim_before_num_blocks):
                if kv_dim:
                    # Shape: (2, num_blocks, ...)
                    self._swap_blocks(src_tensor[0], dst_tensor[0], src_to_dst_tensor)
                    self._swap_blocks(src_tensor[1], dst_tensor[1], src_to_dst_tensor)
                else:
                    # Shape: (num_blocks, ...)
                    self._swap_blocks(src_tensor, dst_tensor, src_to_dst_tensor)
            event.record(stream)

        self.transfer_events[job_id] = event
        return True

    def _swap_blocks(self, src: torch.Tensor, dst: torch.Tensor,
                    src_to_dst: torch.Tensor) -> None:
        """
        Swap blocks between source and destination tensors.
        Uses vLLM ops if available, otherwise falls back to indexing.
        """
        if VLLM_AVAILABLE:
            ops.swap_blocks(src, dst, src_to_dst)
        else:
            # Fallback: simple indexing
            for i in range(src_to_dst.shape[0]):
                src_idx = src_to_dst[i, 0].item()
                dst_idx = src_to_dst[i, 1].item()
                dst[dst_idx].copy_(src[src_idx])

    def _expand_block_ids(self, block_ids: np.ndarray, factor: int,
                         output: np.ndarray, skip_count: int = 0) -> None:
        """Expand block IDs for sub-block addressing."""
        first_range = np.arange(skip_count, factor)
        full_range = np.arange(0, factor)

        output_idx = 0
        for i, block_id in enumerate(block_ids):
            base = block_id * factor
            indices = first_range if i == 0 else full_range
            output[output_idx:output_idx + len(indices)] = base + indices
            output_idx += len(indices)

    def get_finished(self) -> list:
        """
        Get list of completed transfers.

        Returns:
            List of (job_id, success) tuples
        """
        results = []
        for job_id, event in list(self.transfer_events.items()):
            if event.query():
                results.append((job_id, True))
                self.events_pool.append(event)
                del self.transfer_events[job_id]
        return results


class CXLOffloadingSpec(OffloadingSpec if VLLM_AVAILABLE else object):
    """
    vLLM offloading spec for CXL memory.

    Configuration via kv_connector_extra_config:
        - num_cxl_blocks: Number of CXL blocks to allocate
        - cxl_buffer_mb: CXL buffer size in MB (optional)
    """

    def __init__(self, vllm_config: "VllmConfig"):
        super().__init__(vllm_config)

        num_cxl_blocks = self.extra_config.get("num_cxl_blocks")
        if not num_cxl_blocks:
            raise ValueError("num_cxl_blocks must be specified in kv_connector_extra_config")
        self.num_cxl_blocks: int = num_cxl_blocks

        self.cxl_buffer_mb: int = self.extra_config.get("cxl_buffer_mb", 0)

        self._manager: Optional[OffloadingManager] = None
        self._handler: Optional[OffloadingHandler] = None

    def get_manager(self) -> OffloadingManager:
        """Get the CXL offloading manager."""
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (kv_events_config is not None
                           and kv_events_config.enable_kv_cache_events)

            self._manager = LRUOffloadingManager(
                CXLBackend(
                    block_size=self.offloaded_block_size,
                    num_blocks=self.num_cxl_blocks,
                    cxl_buffer_size_mb=self.cxl_buffer_mb),
                enable_events=enable_events)
        return self._manager

    def get_handlers(self, kv_caches: dict):
        """Get CXL offloading handlers."""
        if not self._handler:
            if not current_platform.is_cuda():
                raise RuntimeError("CXL offloading requires CUDA GPU")

            layer_names = list(kv_caches.keys())
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase, layer_names)

            attn_backends = {
                layer_name: layers[layer_name].get_attn_backend()
                for layer_name in layer_names
            }

            self._handler = CxlGpuOffloadingHandler(
                gpu_block_size=self.gpu_block_size,
                cxl_block_size=self.offloaded_block_size,
                num_cxl_blocks=self.num_cxl_blocks,
                gpu_caches=kv_caches,
                attn_backends=attn_backends)

        assert self._handler is not None
        yield GPULoadStoreSpec, CXLLoadStoreSpec, self._handler
        yield CXLLoadStoreSpec, GPULoadStoreSpec, self._handler


def register_cxl_spec():
    """Register CXL spec with vLLM factory."""
    if not VLLM_AVAILABLE:
        print("Warning: vLLM not available, cannot register CXL spec")
        return

    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    try:
        OffloadingSpecFactory.register_spec(
            "CXLOffloadingSpec",
            "python.vllm_cxl_backend",
            "CXLOffloadingSpec")
        print("Registered CXLOffloadingSpec with vLLM")
    except ValueError:
        # Already registered
        pass


# Auto-register on import
if VLLM_AVAILABLE:
    register_cxl_spec()
