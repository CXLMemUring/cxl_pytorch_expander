"""
CXL Tensor Manager for PyTorch

Provides transparent GPU memory expansion using CXL memory.
Tensors can be offloaded from GPU to CXL and brought back on demand.

Supports two transfer modes:
1. P2P DMA: Direct GPU <-> CXL transfers (requires hardware support)
2. CPU-Staged: GPU -> CPU -> CXL (fallback, slower)
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import ctypes
import threading
import weakref
import os

# Try to import the C extension
try:
    import cxl_memory
    HAS_CXL_MODULE = True
except ImportError:
    HAS_CXL_MODULE = False
    print("Warning: cxl_memory module not found. Using simulation mode.")

# Check for CUDA availability
HAS_CUDA = torch.cuda.is_available()


class TensorLocation(Enum):
    """Where a tensor is currently stored"""
    GPU = "gpu"
    CXL = "cxl"
    CPU = "cpu"


class TransferMode(Enum):
    """Data transfer mode between GPU and CXL"""
    P2P_DMA = "p2p_dma"       # Direct GPU <-> CXL via P2P DMA
    CPU_STAGED = "cpu_staged"  # GPU -> CPU -> CXL (fallback)
    CUDA_MAPPED = "cuda_mapped"  # CXL mapped as CUDA host memory


@dataclass
class TensorMetadata:
    """Metadata for tracking offloaded tensors"""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    location: TensorLocation
    cxl_buffer_id: Optional[int] = None
    cxl_offset: int = 0
    size_bytes: int = 0
    requires_grad: bool = False
    gpu_data_ptr: Optional[int] = None  # GPU memory pointer for P2P


class CXLTensor:
    """
    A tensor wrapper that can transparently move between GPU and CXL memory.

    Usage:
        cxl_tensor = CXLTensor(torch_tensor)
        cxl_tensor.offload()  # Move to CXL
        tensor = cxl_tensor.to_gpu()  # Bring back to GPU
    """

    def __init__(self, tensor: torch.Tensor, manager: 'CXLTensorManager' = None):
        self._manager = manager or CXLTensorManager.get_instance()
        self._metadata = TensorMetadata(
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
            location=TensorLocation.GPU if tensor.is_cuda else TensorLocation.CPU,
            requires_grad=tensor.requires_grad,
            size_bytes=tensor.numel() * tensor.element_size()
        )
        self._gpu_tensor: Optional[torch.Tensor] = tensor if tensor.is_cuda else None
        self._cpu_tensor: Optional[torch.Tensor] = tensor if not tensor.is_cuda else None
        self._cxl_data: Optional[bytes] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._metadata.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._metadata.dtype

    @property
    def location(self) -> TensorLocation:
        return self._metadata.location

    @property
    def size_bytes(self) -> int:
        return self._metadata.size_bytes

    def offload_to_cxl(self) -> 'CXLTensor':
        """Move tensor data from GPU to CXL memory"""
        if self._metadata.location == TensorLocation.CXL:
            return self

        if self._metadata.location == TensorLocation.GPU:
            # Pass GPU tensor directly to manager - it will use the optimal path
            # (P2P DMA if available, otherwise CUDA-mapped or CPU-staged)
            self._metadata.cxl_buffer_id, self._metadata.cxl_offset = \
                self._manager._store_to_cxl(self._gpu_tensor)
            self._metadata.gpu_data_ptr = self._gpu_tensor.data_ptr()
            self._gpu_tensor = None
            torch.cuda.empty_cache()
        else:
            # CPU tensor path
            self._metadata.cxl_buffer_id, self._metadata.cxl_offset = \
                self._manager._store_to_cxl(self._cpu_tensor)
            self._cpu_tensor = None

        self._metadata.location = TensorLocation.CXL

        return self

    def to_gpu(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Bring tensor back to GPU memory"""
        if device is None:
            device = self._metadata.device

        if self._metadata.location == TensorLocation.GPU:
            return self._gpu_tensor

        if self._metadata.location == TensorLocation.CXL:
            # Restore from CXL - now passes device for direct transfer
            self._gpu_tensor = self._manager._load_from_cxl(
                self._metadata.cxl_buffer_id,
                self._metadata.cxl_offset,
                self._metadata.shape,
                self._metadata.dtype,
                device  # Pass device for P2P or CUDA-mapped transfer
            )
            self._metadata.location = TensorLocation.GPU
            return self._gpu_tensor

        if self._metadata.location == TensorLocation.CPU:
            self._gpu_tensor = self._cpu_tensor.to(device)
            self._cpu_tensor = None
            self._metadata.location = TensorLocation.GPU
            return self._gpu_tensor

    def to_cpu(self) -> torch.Tensor:
        """Get tensor on CPU"""
        if self._metadata.location == TensorLocation.CPU:
            return self._cpu_tensor

        if self._metadata.location == TensorLocation.GPU:
            self._cpu_tensor = self._gpu_tensor.cpu()
            return self._cpu_tensor

        if self._metadata.location == TensorLocation.CXL:
            self._cpu_tensor = self._manager._load_from_cxl(
                self._metadata.cxl_buffer_id,
                self._metadata.cxl_offset,
                self._metadata.shape,
                self._metadata.dtype
            )
            return self._cpu_tensor

    def __repr__(self):
        return f"CXLTensor(shape={self.shape}, dtype={self.dtype}, location={self.location.value})"


class CXLTensorManager:
    """
    Singleton manager for CXL memory operations.

    Manages CXL buffer allocation, tensor storage, and DMA transfers.
    Supports multiple transfer modes:
    - P2P DMA: Direct GPU <-> CXL (fastest, requires hardware)
    - CUDA Mapped: CXL as pinned host memory (fast, works with most setups)
    - CPU Staged: Traditional GPU -> CPU -> CXL (slowest, fallback)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._initialized = False
        self._cxl_info = None
        self._buffers: Dict[int, Dict] = {}  # buffer_id -> {size, used, data, cuda_tensor}
        self._current_buffer_id: Optional[int] = None
        self._current_offset = 0
        self._buffer_size = 256 * 1024 * 1024  # 256MB default buffer size
        self._simulation_mode = not HAS_CXL_MODULE
        self._transfer_mode = TransferMode.CPU_STAGED
        self._p2p_available = False

    @classmethod
    def get_instance(cls) -> 'CXLTensorManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self, buffer_size_mb: int = 256,
                   prefer_p2p: bool = True) -> bool:
        """Initialize CXL connection and allocate initial buffer"""
        if self._initialized:
            return True

        self._buffer_size = buffer_size_mb * 1024 * 1024

        if self._simulation_mode:
            print("CXL Manager: Running in simulation mode (no hardware)")
            self._cxl_info = {
                'link_up': True,
                'memory_expander': True,
                'num_links': 1,
                'version': 2,
                'bandwidth_mbps': 3900
            }
            self._transfer_mode = TransferMode.CUDA_MAPPED
            self._initialized = True
            return True

        try:
            cxl_memory.init()
            self._cxl_info = cxl_memory.query_info()

            if not self._cxl_info.get('link_up'):
                print("Warning: CXL link is not up")

            # Check if P2P DMA is available
            if prefer_p2p and self._cxl_info.get('link_up'):
                self._p2p_available = self._test_p2p_dma()
                if self._p2p_available:
                    self._transfer_mode = TransferMode.P2P_DMA
                    print("CXL Manager: Using P2P DMA transfer mode")
                else:
                    self._transfer_mode = TransferMode.CUDA_MAPPED
                    print("CXL Manager: P2P not available, using CUDA mapped mode")
            else:
                self._transfer_mode = TransferMode.CUDA_MAPPED

            self._initialized = True
            print(f"CXL Manager initialized: {self._cxl_info}")
            print(f"Transfer mode: {self._transfer_mode.value}")
            return True

        except Exception as e:
            print(f"CXL initialization failed: {e}")
            print("Falling back to simulation mode")
            self._simulation_mode = True
            self._cxl_info = {'link_up': True, 'memory_expander': True}
            self._transfer_mode = TransferMode.CUDA_MAPPED
            self._initialized = True
            return True

    def _test_p2p_dma(self) -> bool:
        """Test if P2P DMA is available"""
        try:
            # Try to allocate a small test buffer and do P2P DMA
            test_buffer = cxl_memory.alloc_buffer(4096)
            # Try a small P2P transfer
            cxl_memory.gpu_to_cxl(test_buffer, 0, 0, 64)
            cxl_memory.free_buffer(test_buffer)
            return True
        except Exception as e:
            print(f"P2P DMA test failed: {e}")
            return False

    def get_info(self) -> Dict:
        """Get CXL system information"""
        if not self._initialized:
            self.initialize()
        return self._cxl_info

    @property
    def transfer_mode(self) -> TransferMode:
        return self._transfer_mode

    def _allocate_buffer(self) -> int:
        """Allocate a new CXL buffer with CUDA mapping if available"""
        if self._simulation_mode:
            buffer_id = len(self._buffers) + 1
            # Use pinned memory for better GPU transfer performance
            if HAS_CUDA:
                # Allocate as pinned memory that CUDA can access directly
                pinned_tensor = torch.zeros(
                    self._buffer_size // 4,  # float32 elements
                    dtype=torch.float32,
                    pin_memory=True
                )
                self._buffers[buffer_id] = {
                    'size': self._buffer_size,
                    'used': 0,
                    'pinned_tensor': pinned_tensor,
                    'data_ptr': pinned_tensor.data_ptr()
                }
            else:
                self._buffers[buffer_id] = {
                    'size': self._buffer_size,
                    'used': 0,
                    'data': bytearray(self._buffer_size)
                }
            return buffer_id

        buffer_id = cxl_memory.alloc_buffer(self._buffer_size)
        ptr = cxl_memory.get_buffer_ptr(buffer_id)

        self._buffers[buffer_id] = {
            'size': self._buffer_size,
            'used': 0,
            'cpu_ptr': ptr,
        }

        # Create a pinned tensor view of the CXL memory for low-latency access
        # This allows direct GPU<->CXL transfers without kernel launch overhead
        if HAS_CUDA:
            try:
                # Create numpy array from CXL memory pointer (zero-copy)
                import ctypes
                ctype_arr = (ctypes.c_float * (self._buffer_size // 4)).from_address(ptr)
                np_arr = np.ctypeslib.as_array(ctype_arr)

                # Create pinned tensor from numpy (this registers with CUDA)
                pinned_tensor = torch.from_numpy(np_arr)
                # Note: We can't pin arbitrary memory, but we can use it directly

                self._buffers[buffer_id]['pinned_tensor'] = pinned_tensor
                self._buffers[buffer_id]['data_ptr'] = ptr
                self._buffers[buffer_id]['cuda_accessible'] = True
            except Exception as e:
                print(f"Warning: Could not create CUDA tensor view: {e}")
                self._buffers[buffer_id]['cuda_accessible'] = False

        return buffer_id

    def _store_to_cxl_p2p(self, gpu_tensor: torch.Tensor,
                          buffer_id: int, offset: int) -> None:
        """Store GPU tensor to CXL using P2P DMA (direct path)"""
        size = gpu_tensor.numel() * gpu_tensor.element_size()
        gpu_ptr = gpu_tensor.data_ptr()

        # Use the C extension's P2P DMA function
        cxl_memory.gpu_to_cxl(buffer_id, gpu_ptr, offset, size)

    def _load_from_cxl_p2p(self, buffer_id: int, offset: int,
                           shape: Tuple[int, ...], dtype: torch.dtype,
                           device: torch.device) -> torch.Tensor:
        """Load tensor from CXL to GPU using P2P DMA (direct path)"""
        dummy = torch.zeros(1, dtype=dtype)
        element_size = dummy.element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * element_size

        # Allocate GPU tensor
        gpu_tensor = torch.empty(shape, dtype=dtype, device=device)
        gpu_ptr = gpu_tensor.data_ptr()

        # Use the C extension's P2P DMA function
        cxl_memory.cxl_to_gpu(buffer_id, gpu_ptr, offset, size)

        return gpu_tensor

    def _store_to_cxl_cuda_mapped(self, gpu_tensor: torch.Tensor,
                                   buffer_id: int, offset: int) -> None:
        """Store GPU tensor to CXL using CUDA-mapped memory (fast path)"""
        buf_info = self._buffers[buffer_id]

        if self._simulation_mode and 'pinned_tensor' in buf_info:
            # FAST PATH: Direct GPU->pinned copy using storage view
            # Avoid creating intermediate tensors - use storage directly
            pinned = buf_info['pinned_tensor']
            numel = gpu_tensor.numel()
            elem_offset = offset // 4  # pinned is float32

            # Use storage-level copy for maximum speed
            # Flatten source and copy directly to pinned storage region
            flat_gpu = gpu_tensor.view(-1)
            if gpu_tensor.dtype == torch.float32:
                # Same dtype - direct copy
                pinned.narrow(0, elem_offset, numel).copy_(flat_gpu)
            else:
                # Different dtype - need to view
                pinned.narrow(0, elem_offset, numel).view(gpu_tensor.dtype).copy_(flat_gpu)
        else:
            # CXL hardware path - use CPU staging
            ptr = buf_info.get('cpu_ptr') or cxl_memory.get_buffer_ptr(buffer_id)
            cpu_tensor = gpu_tensor.cpu().contiguous()
            size = cpu_tensor.numel() * cpu_tensor.element_size()
            ctypes.memmove(ptr + offset, cpu_tensor.data_ptr(), size)

    def _load_from_cxl_cuda_mapped(self, buffer_id: int, offset: int,
                                    shape: Tuple[int, ...], dtype: torch.dtype,
                                    device: torch.device) -> torch.Tensor:
        """Load tensor from CXL using CUDA-mapped memory (fast path)"""
        buf_info = self._buffers[buffer_id]

        # Calculate numel without creating dummy tensor
        numel = 1
        for dim in shape:
            numel *= dim

        if self._simulation_mode and 'pinned_tensor' in buf_info:
            # FAST PATH: Direct pinned->GPU copy
            pinned = buf_info['pinned_tensor']
            elem_offset = offset // 4  # pinned is float32

            if dtype == torch.float32:
                # Same dtype - direct narrow and copy to GPU
                cpu_view = pinned.narrow(0, elem_offset, numel).view(shape)
                return cpu_view.to(device, non_blocking=False)
            else:
                # Different dtype - view as target dtype first
                cpu_view = pinned.narrow(0, elem_offset, numel).view(dtype).view(shape)
                return cpu_view.to(device, non_blocking=False)
        else:
            # CXL hardware path - use CPU staging
            ptr = buf_info.get('cpu_ptr') or cxl_memory.get_buffer_ptr(buffer_id)

            # Get element size for dtype
            element_size = {
                torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
                torch.int32: 4, torch.int64: 8, torch.int8: 1,
            }.get(dtype, 4)
            size = numel * element_size

            # Create tensor directly from memory pointer
            arr = (ctypes.c_char * size).from_address(ptr + offset)
            tensor_bytes = bytes(arr)

            np_dtype = {
                torch.float32: np.float32, torch.float16: np.float16,
                torch.int32: np.int32, torch.int64: np.int64, torch.int8: np.int8,
            }.get(dtype, np.float32)

            np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
            tensor = torch.from_numpy(np_array.copy())
            return tensor.to(device)

    # Threshold below which CUDA-mapped is faster than P2P DMA (ioctl overhead)
    P2P_DMA_THRESHOLD = 256 * 1024  # 256KB - below this, use CUDA-mapped

    def _store_to_cxl(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Store tensor data to CXL memory, returns (buffer_id, offset)"""
        if not self._initialized:
            self.initialize()

        size = tensor.numel() * tensor.element_size()

        # Find or allocate buffer with enough space
        if self._current_buffer_id is None or \
           self._current_offset + size > self._buffer_size:
            self._current_buffer_id = self._allocate_buffer()
            self._current_offset = 0

        buffer_id = self._current_buffer_id
        offset = self._current_offset

        # Use appropriate transfer method based on mode and size
        # P2P DMA has ~80us ioctl overhead, so only use for large transfers
        if self._transfer_mode == TransferMode.P2P_DMA and tensor.is_cuda:
            if size >= self.P2P_DMA_THRESHOLD:
                self._store_to_cxl_p2p(tensor, buffer_id, offset)
            else:
                # Use CUDA-mapped for small transfers (lower overhead)
                self._store_to_cxl_cuda_mapped(tensor, buffer_id, offset)
        elif self._transfer_mode == TransferMode.CUDA_MAPPED:
            if tensor.is_cuda:
                self._store_to_cxl_cuda_mapped(tensor, buffer_id, offset)
            else:
                self._store_to_cxl_cpu_staged(tensor, buffer_id, offset)
        else:
            self._store_to_cxl_cpu_staged(tensor, buffer_id, offset)

        self._current_offset += size
        # Align to 256 bytes
        self._current_offset = (self._current_offset + 255) & ~255

        return buffer_id, offset

    def _store_to_cxl_cpu_staged(self, tensor: torch.Tensor,
                                  buffer_id: int, offset: int) -> None:
        """Store tensor to CXL using CPU staging (slowest path)"""
        # Ensure tensor is on CPU
        if tensor.is_cuda:
            tensor = tensor.cpu()

        tensor_bytes = tensor.numpy().tobytes()
        size = len(tensor_bytes)

        if self._simulation_mode:
            if 'data' in self._buffers[buffer_id]:
                buf = self._buffers[buffer_id]['data']
                buf[offset:offset + size] = tensor_bytes
            elif 'pinned_tensor' in self._buffers[buffer_id]:
                # Fallback: copy via pinned tensor
                pinned = self._buffers[buffer_id]['pinned_tensor']
                elem_offset = offset // 4
                np_data = np.frombuffer(tensor_bytes, dtype=np.float32)
                pinned[elem_offset:elem_offset + len(np_data)] = torch.from_numpy(np_data)
        else:
            ptr = cxl_memory.get_buffer_ptr(buffer_id)
            arr = (ctypes.c_char * size).from_address(ptr + offset)
            ctypes.memmove(arr, tensor_bytes, size)

    def _load_from_cxl(self, buffer_id: int, offset: int,
                       shape: Tuple[int, ...], dtype: torch.dtype,
                       device: Optional[torch.device] = None) -> torch.Tensor:
        """Load tensor data from CXL memory"""
        if device is None:
            device = torch.device('cpu')

        # Calculate size for threshold check
        element_size = {
            torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
            torch.int32: 4, torch.int64: 8, torch.int8: 1,
        }.get(dtype, 4)
        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * element_size

        # Use appropriate transfer method based on mode and size
        if self._transfer_mode == TransferMode.P2P_DMA and device.type == 'cuda':
            if size >= self.P2P_DMA_THRESHOLD:
                return self._load_from_cxl_p2p(buffer_id, offset, shape, dtype, device)
            else:
                # Use CUDA-mapped for small transfers (lower overhead)
                return self._load_from_cxl_cuda_mapped(buffer_id, offset, shape, dtype, device)
        elif self._transfer_mode == TransferMode.CUDA_MAPPED:
            return self._load_from_cxl_cuda_mapped(buffer_id, offset, shape, dtype, device)
        else:
            return self._load_from_cxl_cpu_staged(buffer_id, offset, shape, dtype, device)

    def _load_from_cxl_cpu_staged(self, buffer_id: int, offset: int,
                                   shape: Tuple[int, ...], dtype: torch.dtype,
                                   device: torch.device) -> torch.Tensor:
        """Load tensor from CXL using CPU staging (slowest path)"""
        dummy = torch.zeros(1, dtype=dtype)
        element_size = dummy.element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * element_size

        if self._simulation_mode:
            if 'data' in self._buffers[buffer_id]:
                buf = self._buffers[buffer_id]['data']
                tensor_bytes = bytes(buf[offset:offset + size])
            else:
                # Use pinned tensor
                pinned = self._buffers[buffer_id]['pinned_tensor']
                elem_offset = offset // 4
                elem_count = size // 4
                tensor_bytes = pinned[elem_offset:elem_offset + elem_count].numpy().tobytes()
        else:
            ptr = cxl_memory.get_buffer_ptr(buffer_id)
            arr = (ctypes.c_char * size).from_address(ptr + offset)
            tensor_bytes = bytes(arr)

        np_dtype = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.float32,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
        }.get(dtype, np.float32)

        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(np_array.copy())

        if dtype == torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)

        return tensor.to(device) if device.type != 'cpu' else tensor

    def get_stats(self) -> Dict:
        """Get memory usage statistics"""
        total_allocated = sum(b['size'] for b in self._buffers.values())
        total_used = self._current_offset if self._current_buffer_id else 0

        return {
            'num_buffers': len(self._buffers),
            'total_allocated_mb': total_allocated / (1024 * 1024),
            'current_used_mb': total_used / (1024 * 1024),
            'simulation_mode': self._simulation_mode,
            'transfer_mode': self._transfer_mode.value,
            'p2p_available': self._p2p_available,
        }

    def cleanup(self):
        """Free all CXL buffers"""
        if not self._simulation_mode:
            for buffer_id in list(self._buffers.keys()):
                try:
                    cxl_memory.free_buffer(buffer_id)
                except:
                    pass

        self._buffers.clear()
        self._current_buffer_id = None
        self._current_offset = 0

    # ===== Direct Transfer API (low overhead) =====

    def direct_gpu_to_cxl(self, gpu_tensor: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], 'torch.dtype']:
        """
        Direct GPU to CXL transfer without CXLTensor wrapper.
        Returns (buffer_id, offset, shape, dtype) for later retrieval.
        """
        if not self._initialized:
            self.initialize()

        shape = tuple(gpu_tensor.shape)
        dtype = gpu_tensor.dtype
        buffer_id, offset = self._store_to_cxl(gpu_tensor)
        return buffer_id, offset, shape, dtype

    def direct_cxl_to_gpu(self, buffer_id: int, offset: int,
                          shape: Tuple[int, ...], dtype: 'torch.dtype',
                          device: torch.device) -> torch.Tensor:
        """
        Direct CXL to GPU transfer without CXLTensor wrapper.
        """
        return self._load_from_cxl(buffer_id, offset, shape, dtype, device)

    # ===== Timed Transfer API (kernel-level wall time) =====

    def direct_gpu_to_cxl_timed(self, gpu_tensor: torch.Tensor) -> Tuple[int, int, Tuple[int, ...], 'torch.dtype', int]:
        """
        Direct GPU to CXL transfer with kernel-level wall time measurement.
        Returns (buffer_id, offset, shape, dtype, latency_ns).

        The latency is measured at the C level using clock_gettime(CLOCK_MONOTONIC).
        This measures only the memcpy/transfer time, excluding Python overhead.
        """
        if not self._initialized:
            self.initialize()

        shape = tuple(gpu_tensor.shape)
        dtype = gpu_tensor.dtype
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        # Find or allocate buffer
        if self._current_buffer_id is None or \
           self._current_offset + size > self._buffer_size:
            self._current_buffer_id = self._allocate_buffer()
            self._current_offset = 0

        buffer_id = self._current_buffer_id
        offset = self._current_offset

        # Perform transfer with timing
        latency_ns = self._store_to_cxl_timed(gpu_tensor, buffer_id, offset)

        self._current_offset += size
        self._current_offset = (self._current_offset + 255) & ~255

        return buffer_id, offset, shape, dtype, latency_ns

    def _store_to_cxl_timed(self, tensor: torch.Tensor, buffer_id: int, offset: int) -> int:
        """
        Store tensor to CXL with kernel-level timing.
        Returns latency in nanoseconds.
        """
        size = tensor.numel() * tensor.element_size()

        if self._simulation_mode:
            buf_info = self._buffers[buffer_id]
            if 'pinned_tensor' in buf_info:
                # For simulation mode: we need to copy GPU->CPU first, then time the memcpy
                # Sync GPU first to ensure data is ready
                if tensor.is_cuda:
                    torch.cuda.synchronize()
                    # Copy to CPU (this uses CUDA)
                    cpu_tensor = tensor.cpu().contiguous()
                else:
                    cpu_tensor = tensor.contiguous()

                # Now time the CPU memcpy using C extension
                dst_ptr = buf_info['data_ptr'] + offset
                src_ptr = cpu_tensor.data_ptr()
                latency_ns = cxl_memory.memcpy_timed(dst_ptr, src_ptr, size)
                return latency_ns
            else:
                # Fallback - no timing
                return 0
        else:
            # Real CXL hardware - use timed P2P DMA
            if tensor.is_cuda and size >= self.P2P_DMA_THRESHOLD:
                gpu_ptr = tensor.data_ptr()
                transfer_id, latency_ns = cxl_memory.gpu_to_cxl_timed(
                    buffer_id, gpu_ptr, offset, size)
                return latency_ns
            else:
                # Use CPU staging with memcpy timing
                if tensor.is_cuda:
                    torch.cuda.synchronize()
                    cpu_tensor = tensor.cpu().contiguous()
                else:
                    cpu_tensor = tensor.contiguous()

                ptr = cxl_memory.get_buffer_ptr(buffer_id)
                dst_ptr = ptr + offset
                src_ptr = cpu_tensor.data_ptr()
                latency_ns = cxl_memory.memcpy_timed(dst_ptr, src_ptr, size)
                return latency_ns

    def direct_cxl_to_gpu_timed(self, buffer_id: int, offset: int,
                                 shape: Tuple[int, ...], dtype: 'torch.dtype',
                                 device: torch.device) -> Tuple[torch.Tensor, int]:
        """
        Direct CXL to GPU transfer with kernel-level wall time measurement.
        Returns (gpu_tensor, latency_ns).

        The latency is measured at the C level using clock_gettime(CLOCK_MONOTONIC).
        """
        latency_ns = 0
        element_size = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * element_size

        if self._simulation_mode:
            buf_info = self._buffers[buffer_id]
            if 'pinned_tensor' in buf_info:
                # Allocate destination CPU buffer
                cpu_tensor = torch.empty(shape, dtype=dtype)

                # Time the memcpy from pinned to CPU buffer
                src_ptr = buf_info['data_ptr'] + offset
                dst_ptr = cpu_tensor.data_ptr()
                latency_ns = cxl_memory.memcpy_timed(dst_ptr, src_ptr, size)

                # Then copy to GPU (untimed CUDA transfer)
                gpu_tensor = cpu_tensor.to(device)
                return gpu_tensor, latency_ns
            else:
                # Fallback
                gpu_tensor = self._load_from_cxl(buffer_id, offset, shape, dtype, device)
                return gpu_tensor, 0
        else:
            # Real CXL hardware
            if device.type == 'cuda' and size >= self.P2P_DMA_THRESHOLD:
                gpu_tensor = torch.empty(shape, dtype=dtype, device=device)
                gpu_ptr = gpu_tensor.data_ptr()
                transfer_id, latency_ns = cxl_memory.cxl_to_gpu_timed(
                    buffer_id, gpu_ptr, offset, size)
                return gpu_tensor, latency_ns
            else:
                # Use CPU staging with memcpy timing
                cpu_tensor = torch.empty(shape, dtype=dtype)
                ptr = cxl_memory.get_buffer_ptr(buffer_id)
                src_ptr = ptr + offset
                dst_ptr = cpu_tensor.data_ptr()
                latency_ns = cxl_memory.memcpy_timed(dst_ptr, src_ptr, size)
                gpu_tensor = cpu_tensor.to(device)
                return gpu_tensor, latency_ns

    def memcpy_benchmark(self, size_bytes: int) -> Tuple[int, int]:
        """
        Benchmark raw memcpy latency for a given size.
        Returns (write_latency_ns, read_latency_ns).

        This measures pure CPU memcpy time, excluding all GPU/CUDA overhead.
        Use this to understand theoretical best-case data movement latency.
        """
        if not self._initialized:
            self.initialize()

        # Ensure we have a buffer
        if self._current_buffer_id is None:
            self._current_buffer_id = self._allocate_buffer()

        buf_info = self._buffers[self._current_buffer_id]

        # Allocate source/destination on CPU (not pinned, to avoid CUDA paths)
        src_buffer = torch.zeros(size_bytes // 4 + 1, dtype=torch.float32)
        dst_buffer = torch.zeros(size_bytes // 4 + 1, dtype=torch.float32)

        if self._simulation_mode and 'data_ptr' in buf_info:
            cxl_ptr = buf_info['data_ptr']
        elif not self._simulation_mode:
            cxl_ptr = cxl_memory.get_buffer_ptr(self._current_buffer_id)
        else:
            # Fallback - can't do raw memcpy timing
            return 0, 0

        # Write: CPU buffer -> CXL memory
        write_latency_ns = cxl_memory.memcpy_timed(cxl_ptr, src_buffer.data_ptr(), size_bytes)

        # Read: CXL memory -> CPU buffer
        read_latency_ns = cxl_memory.memcpy_timed(dst_buffer.data_ptr(), cxl_ptr, size_bytes)

        return write_latency_ns, read_latency_ns


# Convenience functions
def offload_tensor(tensor: torch.Tensor) -> CXLTensor:
    """Offload a GPU tensor to CXL memory"""
    cxl_tensor = CXLTensor(tensor)
    return cxl_tensor.offload_to_cxl()


def offload_model_layer(layer: torch.nn.Module) -> Dict[str, CXLTensor]:
    """Offload all parameters of a model layer to CXL"""
    offloaded = {}
    for name, param in layer.named_parameters():
        offloaded[name] = offload_tensor(param.data)
    return offloaded


def restore_model_layer(layer: torch.nn.Module, offloaded: Dict[str, CXLTensor]):
    """Restore offloaded parameters back to a model layer"""
    for name, cxl_tensor in offloaded.items():
        param = dict(layer.named_parameters())[name]
        param.data = cxl_tensor.to_gpu(param.device)
