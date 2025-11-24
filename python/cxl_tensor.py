"""
CXL Tensor Manager for PyTorch

Provides transparent GPU memory expansion using CXL memory.
Tensors can be offloaded from GPU to CXL and brought back on demand.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import ctypes
import threading
import weakref

# Try to import the C extension
try:
    import cxl_memory
    HAS_CXL_MODULE = True
except ImportError:
    HAS_CXL_MODULE = False
    print("Warning: cxl_memory module not found. Using simulation mode.")


class TensorLocation(Enum):
    """Where a tensor is currently stored"""
    GPU = "gpu"
    CXL = "cxl"
    CPU = "cpu"


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
            # First copy to CPU, then to CXL
            cpu_tensor = self._gpu_tensor.cpu()
            self._gpu_tensor = None
            torch.cuda.empty_cache()
        else:
            cpu_tensor = self._cpu_tensor

        # Store in CXL buffer through manager
        self._metadata.cxl_buffer_id, self._metadata.cxl_offset = \
            self._manager._store_to_cxl(cpu_tensor)

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
            # Restore from CXL
            cpu_tensor = self._manager._load_from_cxl(
                self._metadata.cxl_buffer_id,
                self._metadata.cxl_offset,
                self._metadata.shape,
                self._metadata.dtype
            )
            self._gpu_tensor = cpu_tensor.to(device)
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
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self._initialized = False
        self._cxl_info = None
        self._buffers: Dict[int, Dict] = {}  # buffer_id -> {size, used, data}
        self._current_buffer_id: Optional[int] = None
        self._current_offset = 0
        self._buffer_size = 256 * 1024 * 1024  # 256MB default buffer size
        self._simulation_mode = not HAS_CXL_MODULE

    @classmethod
    def get_instance(cls) -> 'CXLTensorManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self, buffer_size_mb: int = 256) -> bool:
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
            self._initialized = True
            return True

        try:
            cxl_memory.init()
            self._cxl_info = cxl_memory.query_info()

            if not self._cxl_info.get('link_up'):
                print("Warning: CXL link is not up")

            self._initialized = True
            print(f"CXL Manager initialized: {self._cxl_info}")
            return True

        except Exception as e:
            print(f"CXL initialization failed: {e}")
            print("Falling back to simulation mode")
            self._simulation_mode = True
            self._cxl_info = {'link_up': True, 'memory_expander': True}
            self._initialized = True
            return True

    def get_info(self) -> Dict:
        """Get CXL system information"""
        if not self._initialized:
            self.initialize()
        return self._cxl_info

    def _allocate_buffer(self) -> int:
        """Allocate a new CXL buffer"""
        if self._simulation_mode:
            buffer_id = len(self._buffers) + 1
            self._buffers[buffer_id] = {
                'size': self._buffer_size,
                'used': 0,
                'data': bytearray(self._buffer_size)
            }
            return buffer_id

        buffer_id = cxl_memory.alloc_buffer(self._buffer_size)
        self._buffers[buffer_id] = {
            'size': self._buffer_size,
            'used': 0,
        }
        return buffer_id

    def _store_to_cxl(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """Store tensor data to CXL memory, returns (buffer_id, offset)"""
        if not self._initialized:
            self.initialize()

        # Get tensor bytes
        tensor_bytes = tensor.numpy().tobytes()
        size = len(tensor_bytes)

        # Find or allocate buffer with enough space
        if self._current_buffer_id is None or \
           self._current_offset + size > self._buffer_size:
            self._current_buffer_id = self._allocate_buffer()
            self._current_offset = 0

        buffer_id = self._current_buffer_id
        offset = self._current_offset

        if self._simulation_mode:
            # Copy to simulation buffer
            buf = self._buffers[buffer_id]['data']
            buf[offset:offset + size] = tensor_bytes
        else:
            # Use CXL DMA (for now, just use CPU copy via mmap)
            ptr = cxl_memory.get_buffer_ptr(buffer_id)
            # Create ctypes array and copy
            arr = (ctypes.c_char * size).from_address(ptr + offset)
            ctypes.memmove(arr, tensor_bytes, size)

        self._current_offset += size
        # Align to 256 bytes
        self._current_offset = (self._current_offset + 255) & ~255

        return buffer_id, offset

    def _load_from_cxl(self, buffer_id: int, offset: int,
                       shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """Load tensor data from CXL memory"""
        # Calculate size
        dummy = torch.zeros(1, dtype=dtype)
        element_size = dummy.element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * element_size

        if self._simulation_mode:
            # Read from simulation buffer
            buf = self._buffers[buffer_id]['data']
            tensor_bytes = bytes(buf[offset:offset + size])
        else:
            # Read from CXL buffer
            ptr = cxl_memory.get_buffer_ptr(buffer_id)
            arr = (ctypes.c_char * size).from_address(ptr + offset)
            tensor_bytes = bytes(arr)

        # Convert back to tensor
        np_dtype = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.bfloat16: np.float32,  # numpy doesn't support bfloat16
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int8: np.int8,
        }.get(dtype, np.float32)

        np_array = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
        tensor = torch.from_numpy(np_array.copy())

        if dtype == torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)

        return tensor

    def get_stats(self) -> Dict:
        """Get memory usage statistics"""
        total_allocated = sum(b['size'] for b in self._buffers.values())
        total_used = self._current_offset if self._current_buffer_id else 0

        return {
            'num_buffers': len(self._buffers),
            'total_allocated_mb': total_allocated / (1024 * 1024),
            'current_used_mb': total_used / (1024 * 1024),
            'simulation_mode': self._simulation_mode,
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
