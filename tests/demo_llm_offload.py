#!/usr/bin/env python3
"""
LLM Weight Offloading Demo with CXL Memory Expander

Demonstrates how to use CXL memory to expand GPU memory for large LLMs:
1. Load model weights to GPU
2. Offload inactive layers to CXL
3. Dynamically swap layers during inference

This enables running larger models than would fit in GPU memory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from python.cxl_tensor import CXLTensorManager, CXLTensor, offload_tensor


@dataclass
class ModelConfig:
    """Configuration for a transformer-style model"""
    num_layers: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    vocab_size: int = 50257


class SimpleFFN(nn.Module):
    """Simple feed-forward network layer"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))


class SimpleAttention(nn.Module):
    """Simple multi-head attention"""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """Single transformer layer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = SimpleAttention(config.hidden_size, config.num_heads)
        self.ffn = SimpleFFN(config.hidden_size, config.intermediate_size)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class CXLModelOffloader:
    """
    Manages offloading of model layers to CXL memory.

    Keeps active layers on GPU while offloading inactive ones to CXL,
    dynamically swapping as needed during inference.
    """

    def __init__(self, model: nn.Module, max_gpu_layers: int = 4):
        self.model = model
        self.max_gpu_layers = max_gpu_layers
        self.device = next(model.parameters()).device

        # Initialize CXL manager
        self.cxl_manager = CXLTensorManager.get_instance()
        self.cxl_manager.initialize(buffer_size_mb=512)

        # Track which layers are where
        self.layer_states: Dict[int, str] = {}  # layer_idx -> 'gpu' or 'cxl'
        self.offloaded_params: Dict[int, Dict[str, CXLTensor]] = {}
        self.param_shapes: Dict[int, Dict[str, tuple]] = {}  # Store original shapes

        # Stats
        self.stats = {
            'offloads': 0,
            'restores': 0,
            'offload_time_ms': 0,
            'restore_time_ms': 0,
        }

    def get_layer_size_mb(self, layer_idx: int) -> float:
        """Get size of a layer in MB"""
        layer = self.model.layers[layer_idx]
        size = sum(p.numel() * p.element_size() for p in layer.parameters())
        return size / (1024 * 1024)

    def _get_param_by_name(self, layer, name: str):
        """Navigate to parameter by dotted name"""
        parts = name.split('.')
        obj = layer
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _set_param_data(self, layer, name: str, data: torch.Tensor):
        """Set parameter data by dotted name"""
        parts = name.split('.')
        obj = layer
        for part in parts[:-1]:
            obj = getattr(obj, part)
        param = getattr(obj, parts[-1])
        param.data = data

    def offload_layer(self, layer_idx: int):
        """Offload a layer to CXL memory"""
        if self.layer_states.get(layer_idx) == 'cxl':
            return  # Already offloaded

        start = time.time()
        layer = self.model.layers[layer_idx]

        # Store and offload each parameter (except small ones like LayerNorm)
        self.offloaded_params[layer_idx] = {}
        self.param_shapes[layer_idx] = {}

        for name, param in layer.named_parameters():
            # Skip small parameters (e.g., LayerNorm weights/biases)
            if param.numel() < 1000:
                continue

            # Store shape before offloading
            self.param_shapes[layer_idx][name] = param.data.shape

            cxl_tensor = CXLTensor(param.data.clone(), self.cxl_manager)
            cxl_tensor.offload_to_cxl()
            self.offloaded_params[layer_idx][name] = cxl_tensor

            # Replace parameter with small placeholder to free GPU memory
            param.data = torch.zeros(1, device=self.device, dtype=param.dtype)

        self.layer_states[layer_idx] = 'cxl'
        self.stats['offloads'] += 1
        self.stats['offload_time_ms'] += (time.time() - start) * 1000
        torch.cuda.empty_cache()

    def restore_layer(self, layer_idx: int):
        """Restore a layer from CXL memory"""
        state = self.layer_states.get(layer_idx)
        if state != 'cxl':
            print(f'restore_layer({layer_idx}): state is {state}, not restoring')
            return  # Not offloaded

        if layer_idx not in self.offloaded_params:
            print(f'restore_layer({layer_idx}): NOT in offloaded_params! Keys: {list(self.offloaded_params.keys())}')
            return  # Nothing to restore

        print(f'restore_layer({layer_idx}): restoring {len(self.offloaded_params[layer_idx])} params')
        start = time.time()
        layer = self.model.layers[layer_idx]

        # Restore each parameter
        for name, cxl_tensor in self.offloaded_params[layer_idx].items():
            restored_data = cxl_tensor.to_gpu(self.device)
            if layer_idx == 0:
                print(f'  CXL->GPU {name}: returned shape={restored_data.shape}, device={restored_data.device}')
            self._set_param_data(layer, name, restored_data)
            # Debug: verify restoration
            current = self._get_param_by_name(layer, name)
            if layer_idx == 0:
                print(f'  After set {name}: param.data.shape={current.data.shape}')
            if current.data.shape != restored_data.shape:
                print(f'WARNING: Layer {layer_idx} {name} shape mismatch after restore: {current.data.shape} vs {restored_data.shape}')

        del self.offloaded_params[layer_idx]
        if layer_idx in self.param_shapes:
            del self.param_shapes[layer_idx]

        self.layer_states[layer_idx] = 'gpu'
        self.stats['restores'] += 1
        self.stats['restore_time_ms'] += (time.time() - start) * 1000

    def ensure_layer_on_gpu(self, layer_idx: int):
        """Ensure a specific layer is on GPU, potentially offloading others"""
        state = self.layer_states.get(layer_idx)
        if state != 'cxl':
            return  # Already on GPU

        # Debug: check layer 0 before we do anything
        if layer_idx == 1:
            print(f'ensure(1): BEFORE, layer0.weight.shape = {self.model.layers[0].attn.qkv_proj.weight.shape}')

        # Count current GPU layers
        gpu_layers = [i for i, s in self.layer_states.items() if s == 'gpu']

        # If at capacity, offload the oldest GPU layer
        while len(gpu_layers) >= self.max_gpu_layers:
            oldest = gpu_layers.pop(0)
            if layer_idx == 1:
                print(f'ensure(1): offloading layer {oldest}')
            self.offload_layer(oldest)

        # Debug: check layer 0 after offloading
        if layer_idx == 1:
            print(f'ensure(1): after offloading, layer0.weight.shape = {self.model.layers[0].attn.qkv_proj.weight.shape}')

        # Restore the requested layer
        self.restore_layer(layer_idx)

        # Debug: check layer 0 after restoring layer 1
        if layer_idx == 1:
            print(f'ensure(1): after restore(1), layer0.weight.shape = {self.model.layers[0].attn.qkv_proj.weight.shape}')

    def offload_all_but_active(self, active_range: range):
        """Offload all layers except those in the active range"""
        num_layers = len(self.model.layers)
        for i in range(num_layers):
            if i not in active_range:
                self.offload_layer(i)
            else:
                self.layer_states[i] = 'gpu'

    _forward_count = 0

    def forward_with_offloading(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic layer offloading.

        Keeps a sliding window of layers on GPU, swapping as needed.
        """
        x = self.model.embed(input_ids)

        num_layers = len(self.model.layers)

        # Reset: offload all layers except first max_gpu_layers to start fresh
        for i in range(num_layers):
            if i < self.max_gpu_layers:
                if self.layer_states.get(i) == 'cxl':
                    self.restore_layer(i)
            else:
                if self.layer_states.get(i) == 'gpu':
                    self.offload_layer(i)

        for i in range(num_layers):
            # Ensure this layer is on GPU
            if self.layer_states.get(i) == 'cxl':
                # Need to make room - offload an old layer first
                oldest_gpu = min((j for j, s in self.layer_states.items() if s == 'gpu'), default=None)
                if oldest_gpu is not None and oldest_gpu < i - 1:  # Keep at least the previous layer
                    self.offload_layer(oldest_gpu)
                self.restore_layer(i)

            # Run the layer
            x = self.model.layers[i](x)

            # Offload layers that are far behind (keep sliding window)
            if i >= self.max_gpu_layers:
                old_layer = i - self.max_gpu_layers
                if self.layer_states.get(old_layer) == 'gpu':
                    self.offload_layer(old_layer)

        x = self.model.ln_f(x)
        return self.model.lm_head(x)

    def get_stats(self) -> Dict:
        """Get offloading statistics"""
        gpu_layers = sum(1 for s in self.layer_states.values() if s == 'gpu')
        cxl_layers = sum(1 for s in self.layer_states.values() if s == 'cxl')

        return {
            'gpu_layers': gpu_layers,
            'cxl_layers': cxl_layers,
            **self.stats,
            'cxl_manager': self.cxl_manager.get_stats(),
        }


def demo_model_offloading():
    """Demonstrate LLM weight offloading to CXL"""
    print("\n" + "=" * 60)
    print("LLM Weight Offloading Demo with CXL Memory")
    print("=" * 60 + "\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create a model (GPT-2 small size for demo)
    config = ModelConfig(
        num_layers=12,
        hidden_size=768,
        intermediate_size=3072,
        num_heads=12,
        vocab_size=50257
    )

    print(f"\nModel Configuration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Intermediate size: {config.intermediate_size}")
    print(f"  Vocab size: {config.vocab_size}")

    # Create and move model to GPU
    print("\nCreating model...")
    model = SimpleTransformer(config).to(device).half()  # FP16 for memory efficiency

    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    layer_size_mb = sum(p.numel() * p.element_size()
                       for p in model.layers[0].parameters()) / (1024 * 1024)

    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Size per layer: {layer_size_mb:.2f} MB")

    # Create offloader - keep only 4 layers on GPU at a time
    max_gpu_layers = 4
    print(f"\nCXL Offloader Configuration:")
    print(f"  Max GPU layers: {max_gpu_layers}")
    print(f"  GPU memory budget: {max_gpu_layers * layer_size_mb:.1f} MB")
    print(f"  CXL offload capacity: {(config.num_layers - max_gpu_layers) * layer_size_mb:.1f} MB")

    offloader = CXLModelOffloader(model, max_gpu_layers=max_gpu_layers)

    # Initial offload - keep only first few layers on GPU
    print("\nInitial offload to CXL...")
    offloader.offload_all_but_active(range(max_gpu_layers))

    stats = offloader.get_stats()
    print(f"  Layers on GPU: {stats['gpu_layers']}")
    print(f"  Layers on CXL: {stats['cxl_layers']}")
    print(f"  CXL Manager: {stats['cxl_manager']}")

    # Run inference with dynamic offloading
    print("\nRunning inference with dynamic layer swapping...")
    batch_size = 4
    seq_len = 128

    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Debug: check layer 0 before warmup
    print("Layer 0 params before warmup:")
    layer0 = model.layers[0]
    for name, param in layer0.named_parameters():
        if param.numel() >= 1000:
            print(f"  {name}: shape={param.data.shape}, numel={param.data.numel()}")

    # Warmup
    print("\nStarting warmup...")
    with torch.no_grad():
        _ = offloader.forward_with_offloading(input_ids)
    print("Warmup done.")

    # Debug: check layer 0 after warmup
    print("\nLayer 0 params after warmup:")
    for name, param in layer0.named_parameters():
        if param.numel() >= 1000:
            print(f"  {name}: shape={param.data.shape}, numel={param.data.numel()}")

    # Timed run
    torch.cuda.synchronize()
    start = time.time()

    num_iterations = 5
    for i in range(num_iterations):
        with torch.no_grad():
            output = offloader.forward_with_offloading(input_ids)

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"\nInference Results:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Per iteration: {elapsed*1000/num_iterations:.1f} ms")
    print(f"  Output shape: {output.shape}")

    # Final statistics
    stats = offloader.get_stats()
    print(f"\nOffloading Statistics:")
    print(f"  Total offloads: {stats['offloads']}")
    print(f"  Total restores: {stats['restores']}")
    print(f"  Offload time: {stats['offload_time_ms']:.1f} ms")
    print(f"  Restore time: {stats['restore_time_ms']:.1f} ms")
    print(f"  Final GPU layers: {stats['gpu_layers']}")
    print(f"  Final CXL layers: {stats['cxl_layers']}")

    # Memory comparison
    print(f"\nMemory Savings:")
    print(f"  Without CXL: {total_size_mb:.1f} MB GPU required")
    print(f"  With CXL: {max_gpu_layers * layer_size_mb:.1f} MB GPU + "
          f"{(config.num_layers - max_gpu_layers) * layer_size_mb:.1f} MB CXL")
    print(f"  GPU Memory Reduction: {(1 - max_gpu_layers/config.num_layers)*100:.1f}%")

    return True


def demo_kv_cache_offload():
    """Quick demo of KV cache offloading"""
    print("\n" + "=" * 60)
    print("KV Cache Offloading Demo")
    print("=" * 60 + "\n")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Simulate a KV cache for one layer
    batch_size = 8
    num_heads = 32
    seq_len = 2048
    head_dim = 128

    print(f"KV Cache Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dimension: {head_dim}")

    # Create KV cache tensors
    k_cache = torch.randn(batch_size, num_heads, seq_len, head_dim,
                         dtype=torch.float16, device=device)
    v_cache = torch.randn(batch_size, num_heads, seq_len, head_dim,
                         dtype=torch.float16, device=device)

    cache_size_mb = (k_cache.numel() + v_cache.numel()) * 2 / (1024 * 1024)
    print(f"  Total cache size: {cache_size_mb:.1f} MB")

    # Offload to CXL
    manager = CXLTensorManager.get_instance()
    manager.initialize(buffer_size_mb=256)

    print("\nOffloading KV cache to CXL...")
    start = time.time()

    k_cxl = CXLTensor(k_cache, manager)
    k_cxl.offload_to_cxl()
    v_cxl = CXLTensor(v_cache, manager)
    v_cxl.offload_to_cxl()

    offload_time = time.time() - start
    print(f"  Offload time: {offload_time*1000:.1f} ms")
    print(f"  Throughput: {cache_size_mb/offload_time:.1f} MB/s")

    # Restore from CXL
    print("\nRestoring KV cache from CXL...")
    start = time.time()

    k_restored = k_cxl.to_gpu(device)
    v_restored = v_cxl.to_gpu(device)

    restore_time = time.time() - start
    print(f"  Restore time: {restore_time*1000:.1f} ms")
    print(f"  Throughput: {cache_size_mb/restore_time:.1f} MB/s")

    # Verify data
    k_diff = (k_cache - k_restored).abs().max().item()
    v_diff = (v_cache - v_restored).abs().max().item()
    print(f"\nData Integrity:")
    print(f"  K cache max diff: {k_diff}")
    print(f"  V cache max diff: {v_diff}")

    if k_diff < 1e-5 and v_diff < 1e-5:
        print("  PASSED: Data integrity verified")
        return True
    else:
        print("  FAILED: Data corruption detected")
        return False


def main():
    print("=" * 60)
    print("CXL Memory Expander - LLM Demo")
    print("=" * 60)

    results = []

    results.append(("KV Cache Offloading", demo_kv_cache_offload()))
    results.append(("Model Weight Offloading", demo_model_offloading()))

    print("\n" + "=" * 60)
    print("Demo Results Summary")
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
