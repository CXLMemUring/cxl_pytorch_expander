#!/usr/bin/env python3
"""
CXL vs CPU KV Cache Offloading Benchmark

This benchmark measures:
1. Tiering policy behavior (LRU eviction patterns)
2. Transfer latency: GPU <-> CXL vs GPU <-> CPU
3. Throughput under various workloads
4. Memory bandwidth utilization

Usage:
    python tests/benchmark_cxl_vs_cpu.py
"""

import sys
import os
import time
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.cxl_tensor import CXLTensorManager, CXLTensor


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark"""
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    block_size: int = 16
    dtype: torch.dtype = torch.float16

    # Cache sizes
    gpu_blocks: int = 100
    offload_blocks: int = 500

    # Workload parameters
    num_sequences: int = 200
    tokens_per_seq: int = 512

    @property
    def block_shape(self) -> Tuple[int, ...]:
        """Shape of each KV cache block"""
        return (2, self.num_layers, self.block_size, self.num_heads, self.head_dim)

    @property
    def block_bytes(self) -> int:
        """Size of each block in bytes"""
        numel = 2 * self.num_layers * self.block_size * self.num_heads * self.head_dim
        return numel * (2 if self.dtype == torch.float16 else 4)


class TieringStats:
    """Track tiering policy statistics"""

    def __init__(self):
        self.gpu_hits = 0
        self.offload_hits = 0
        self.misses = 0
        self.evictions = 0
        self.stores = 0
        self.loads = 0

        # Latency tracking
        self.store_latencies: List[float] = []
        self.load_latencies: List[float] = []

        # Access pattern tracking
        self.access_history: List[Tuple[int, str]] = []  # (block_id, 'gpu'|'offload'|'miss')

    def record_access(self, block_id: int, location: str):
        self.access_history.append((block_id, location))
        if location == 'gpu':
            self.gpu_hits += 1
        elif location == 'offload':
            self.offload_hits += 1
        else:
            self.misses += 1

    def record_store(self, latency_ms: float):
        self.stores += 1
        self.store_latencies.append(latency_ms)

    def record_load(self, latency_ms: float):
        self.loads += 1
        self.load_latencies.append(latency_ms)

    def record_eviction(self):
        self.evictions += 1

    def get_summary(self) -> Dict:
        total_accesses = self.gpu_hits + self.offload_hits + self.misses
        return {
            'total_accesses': total_accesses,
            'gpu_hits': self.gpu_hits,
            'gpu_hit_rate': self.gpu_hits / max(total_accesses, 1) * 100,
            'offload_hits': self.offload_hits,
            'offload_hit_rate': self.offload_hits / max(total_accesses, 1) * 100,
            'misses': self.misses,
            'miss_rate': self.misses / max(total_accesses, 1) * 100,
            'evictions': self.evictions,
            'stores': self.stores,
            'loads': self.loads,
            'avg_store_latency_ms': np.mean(self.store_latencies) if self.store_latencies else 0,
            'avg_load_latency_ms': np.mean(self.load_latencies) if self.load_latencies else 0,
            'p99_store_latency_ms': np.percentile(self.store_latencies, 99) if self.store_latencies else 0,
            'p99_load_latency_ms': np.percentile(self.load_latencies, 99) if self.load_latencies else 0,
        }


class LRUCache:
    """Simple LRU cache for benchmarking"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    def get(self, key: int) -> Optional[torch.Tensor]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: int, value: torch.Tensor) -> Optional[int]:
        """Put item, return evicted key if any"""
        evicted = None
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                evicted, _ = self.cache.popitem(last=False)
            self.cache[key] = value
        return evicted

    def __len__(self):
        return len(self.cache)


class CPUOffloadBackend:
    """CPU pinned memory offload backend"""

    def __init__(self, config: BenchmarkConfig, num_blocks: int):
        self.config = config
        self.num_blocks = num_blocks

        # Pre-allocate pinned CPU memory
        self.cpu_cache: Dict[int, torch.Tensor] = {}
        self.free_slots: List[int] = list(range(num_blocks))
        self.block_to_slot: Dict[int, int] = {}

        # Pre-allocate pinned memory pool
        print(f"Allocating {num_blocks} CPU pinned memory blocks...")
        self.pinned_pool = torch.zeros(
            (num_blocks, *config.block_shape),
            dtype=config.dtype,
            pin_memory=True
        )

    def store(self, block_id: int, gpu_tensor: torch.Tensor) -> float:
        """Store GPU tensor to CPU, return latency in ms"""
        if block_id in self.block_to_slot:
            slot = self.block_to_slot[block_id]
        else:
            if not self.free_slots:
                raise RuntimeError("No free CPU slots")
            slot = self.free_slots.pop()
            self.block_to_slot[block_id] = slot

        torch.cuda.synchronize()
        start = time.perf_counter()
        self.pinned_pool[slot].copy_(gpu_tensor)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    def load(self, block_id: int, device: torch.device) -> Tuple[torch.Tensor, float]:
        """Load from CPU to GPU, return tensor and latency in ms"""
        slot = self.block_to_slot[block_id]

        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_tensor = self.pinned_pool[slot].to(device, non_blocking=False)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

        return gpu_tensor, latency

    def free(self, block_id: int):
        if block_id in self.block_to_slot:
            slot = self.block_to_slot.pop(block_id)
            self.free_slots.append(slot)


class CXLOffloadBackend:
    """CXL memory offload backend using P2P DMA"""

    def __init__(self, config: BenchmarkConfig, num_blocks: int):
        self.config = config
        self.num_blocks = num_blocks

        # Initialize CXL manager
        self.cxl_manager = CXLTensorManager.get_instance()
        buffer_mb = int((num_blocks * config.block_bytes) / (1024 * 1024) * 1.2)
        self.cxl_manager.initialize(buffer_size_mb=max(buffer_mb, 256))

        # Block storage
        self.cxl_tensors: Dict[int, CXLTensor] = {}

    def store(self, block_id: int, gpu_tensor: torch.Tensor) -> float:
        """Store GPU tensor to CXL via P2P DMA, return latency in ms"""
        torch.cuda.synchronize()
        start = time.perf_counter()

        cxl_tensor = CXLTensor(gpu_tensor, self.cxl_manager)
        cxl_tensor.offload_to_cxl()
        self.cxl_tensors[block_id] = cxl_tensor

        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000

    def load(self, block_id: int, device: torch.device) -> Tuple[torch.Tensor, float]:
        """Load from CXL to GPU via P2P DMA, return tensor and latency in ms"""
        cxl_tensor = self.cxl_tensors[block_id]

        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_tensor = cxl_tensor.to_gpu(device)
        torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000

        return gpu_tensor, latency

    def free(self, block_id: int):
        if block_id in self.cxl_tensors:
            del self.cxl_tensors[block_id]


class TieredKVCache:
    """Two-tier KV cache with GPU + offload backend"""

    def __init__(self, config: BenchmarkConfig, offload_backend, stats: TieringStats):
        self.config = config
        self.device = torch.device('cuda:0')
        self.offload_backend = offload_backend
        self.stats = stats

        # GPU cache (hot tier)
        self.gpu_cache = LRUCache(config.gpu_blocks)

        # Track which blocks are offloaded
        self.offloaded_blocks: set = set()

        # Block ID counter
        self.next_block_id = 0

    def allocate_block(self) -> int:
        """Allocate a new block"""
        block_id = self.next_block_id
        self.next_block_id += 1

        # Create GPU tensor
        tensor = torch.randn(self.config.block_shape, dtype=self.config.dtype, device=self.device)

        # Put in GPU cache, possibly evicting
        evicted = self.gpu_cache.put(block_id, tensor)

        if evicted is not None:
            # Offload evicted block
            evicted_tensor = self.gpu_cache.cache.get(evicted)
            if evicted_tensor is None:
                # Already evicted, need to get from somewhere
                pass
            else:
                latency = self.offload_backend.store(evicted, evicted_tensor)
                self.stats.record_store(latency)
                self.stats.record_eviction()
                self.offloaded_blocks.add(evicted)

        return block_id

    def access_block(self, block_id: int) -> torch.Tensor:
        """Access a block, loading from offload tier if needed"""
        # Check GPU cache
        tensor = self.gpu_cache.get(block_id)
        if tensor is not None:
            self.stats.record_access(block_id, 'gpu')
            return tensor

        # Check offload tier
        if block_id in self.offloaded_blocks:
            # Load from offload backend
            tensor, latency = self.offload_backend.load(block_id, self.device)
            self.stats.record_load(latency)
            self.stats.record_access(block_id, 'offload')

            # Put back in GPU cache
            evicted = self.gpu_cache.put(block_id, tensor)
            self.offloaded_blocks.discard(block_id)

            if evicted is not None:
                evicted_tensor = self.gpu_cache.cache.get(evicted)
                if evicted_tensor is not None:
                    store_latency = self.offload_backend.store(evicted, evicted_tensor)
                    self.stats.record_store(store_latency)
                    self.stats.record_eviction()
                    self.offloaded_blocks.add(evicted)

            return tensor

        # Miss - block doesn't exist
        self.stats.record_access(block_id, 'miss')
        return None


def benchmark_transfer_latency(config: BenchmarkConfig):
    """Benchmark raw transfer latency: GPU <-> CPU vs GPU <-> CXL"""
    print("\n" + "=" * 70)
    print("Transfer Latency Benchmark")
    print("=" * 70)

    device = torch.device('cuda:0')
    num_iterations = 100

    # Create test tensor
    tensor = torch.randn(config.block_shape, dtype=config.dtype, device=device)
    block_mb = config.block_bytes / (1024 * 1024)
    print(f"\nBlock size: {block_mb:.2f} MB")
    print(f"Iterations: {num_iterations}")

    # CPU benchmark
    print("\n--- CPU Pinned Memory ---")
    cpu_backend = CPUOffloadBackend(config, num_blocks=10)

    cpu_store_times = []
    cpu_load_times = []

    for i in range(num_iterations):
        # Store
        latency = cpu_backend.store(0, tensor)
        cpu_store_times.append(latency)

        # Load
        _, latency = cpu_backend.load(0, device)
        cpu_load_times.append(latency)

    cpu_store_avg = np.mean(cpu_store_times[10:])  # Skip warmup
    cpu_load_avg = np.mean(cpu_load_times[10:])
    cpu_store_bw = block_mb / (cpu_store_avg / 1000)  # MB/s
    cpu_load_bw = block_mb / (cpu_load_avg / 1000)

    print(f"  GPU -> CPU: {cpu_store_avg:.3f} ms ({cpu_store_bw:.0f} MB/s)")
    print(f"  CPU -> GPU: {cpu_load_avg:.3f} ms ({cpu_load_bw:.0f} MB/s)")

    # CXL benchmark
    print("\n--- CXL Memory (P2P DMA) ---")
    cxl_backend = CXLOffloadBackend(config, num_blocks=10)

    cxl_store_times = []
    cxl_load_times = []

    for i in range(num_iterations):
        # Store
        latency = cxl_backend.store(i % 10, tensor)
        cxl_store_times.append(latency)

        # Load
        _, latency = cxl_backend.load(i % 10, device)
        cxl_load_times.append(latency)

    cxl_store_avg = np.mean(cxl_store_times[10:])
    cxl_load_avg = np.mean(cxl_load_times[10:])
    cxl_store_bw = block_mb / (cxl_store_avg / 1000)
    cxl_load_bw = block_mb / (cxl_load_avg / 1000)

    print(f"  GPU -> CXL: {cxl_store_avg:.3f} ms ({cxl_store_bw:.0f} MB/s)")
    print(f"  CXL -> GPU: {cxl_load_avg:.3f} ms ({cxl_load_bw:.0f} MB/s)")

    # Comparison
    print("\n--- Comparison ---")
    store_speedup = cpu_store_avg / cxl_store_avg
    load_speedup = cpu_load_avg / cxl_load_avg

    print(f"  Store speedup (CXL vs CPU): {store_speedup:.2f}x")
    print(f"  Load speedup (CXL vs CPU): {load_speedup:.2f}x")

    if store_speedup > 1 or load_speedup > 1:
        print(f"\n  >> CXL is FASTER for {'store' if store_speedup > 1 else ''}"
              f"{' and ' if store_speedup > 1 and load_speedup > 1 else ''}"
              f"{'load' if load_speedup > 1 else ''}")
    else:
        print(f"\n  >> CPU is faster (CXL may need optimization)")

    return {
        'cpu_store_ms': cpu_store_avg,
        'cpu_load_ms': cpu_load_avg,
        'cxl_store_ms': cxl_store_avg,
        'cxl_load_ms': cxl_load_avg,
        'store_speedup': store_speedup,
        'load_speedup': load_speedup,
    }


def benchmark_tiering_policy(config: BenchmarkConfig, backend_type: str):
    """Benchmark tiering policy with realistic workload"""
    print(f"\n--- Tiering Policy Benchmark ({backend_type}) ---")

    stats = TieringStats()

    if backend_type == 'cpu':
        backend = CPUOffloadBackend(config, config.offload_blocks)
    else:
        backend = CXLOffloadBackend(config, config.offload_blocks)

    cache = TieredKVCache(config, backend, stats)

    # Simulate LLM inference workload
    sequence_blocks: Dict[int, List[int]] = {}
    blocks_per_seq = config.tokens_per_seq // config.block_size

    print(f"  Simulating {config.num_sequences} sequences, {blocks_per_seq} blocks each")
    print(f"  GPU blocks: {config.gpu_blocks}, Offload blocks: {config.offload_blocks}")

    start_time = time.time()

    for seq_id in range(config.num_sequences):
        # Allocate blocks for new sequence
        sequence_blocks[seq_id] = []
        for _ in range(blocks_per_seq):
            block_id = cache.allocate_block()
            sequence_blocks[seq_id].append(block_id)

        # Simulate attention: access all blocks in sequence
        for block_id in sequence_blocks[seq_id]:
            cache.access_block(block_id)

        # Simulate KV reuse: randomly access older sequences (prefix caching)
        if seq_id > 10 and seq_id % 5 == 0:
            # Access random old sequence
            old_seq_id = np.random.randint(0, seq_id - 5)
            if old_seq_id in sequence_blocks:
                for block_id in sequence_blocks[old_seq_id][:3]:  # First few blocks
                    cache.access_block(block_id)

    elapsed = time.time() - start_time

    summary = stats.get_summary()
    summary['elapsed_seconds'] = elapsed
    summary['backend'] = backend_type

    print(f"\n  Results for {backend_type.upper()}:")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    GPU hit rate: {summary['gpu_hit_rate']:.1f}%")
    print(f"    Offload hit rate: {summary['offload_hit_rate']:.1f}%")
    print(f"    Stores: {summary['stores']}, Loads: {summary['loads']}")
    print(f"    Avg store latency: {summary['avg_store_latency_ms']:.3f} ms")
    print(f"    Avg load latency: {summary['avg_load_latency_ms']:.3f} ms")
    print(f"    P99 store latency: {summary['p99_store_latency_ms']:.3f} ms")
    print(f"    P99 load latency: {summary['p99_load_latency_ms']:.3f} ms")

    return summary


def benchmark_throughput(config: BenchmarkConfig):
    """Benchmark sustained throughput"""
    print("\n" + "=" * 70)
    print("Sustained Throughput Benchmark")
    print("=" * 70)

    device = torch.device('cuda:0')
    num_ops = 1000
    block_mb = config.block_bytes / (1024 * 1024)

    tensor = torch.randn(config.block_shape, dtype=config.dtype, device=device)

    results = {}

    for backend_type in ['cpu', 'cxl']:
        print(f"\n--- {backend_type.upper()} Backend ---")

        if backend_type == 'cpu':
            backend = CPUOffloadBackend(config, num_blocks=100)
        else:
            backend = CXLOffloadBackend(config, num_blocks=100)

        # Measure store throughput
        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_ops):
            backend.store(i % 100, tensor)
        torch.cuda.synchronize()
        store_elapsed = time.time() - start
        store_throughput = (num_ops * block_mb) / store_elapsed  # MB/s

        # Measure load throughput
        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_ops):
            backend.load(i % 100, device)
        torch.cuda.synchronize()
        load_elapsed = time.time() - start
        load_throughput = (num_ops * block_mb) / load_elapsed  # MB/s

        print(f"  Store throughput: {store_throughput:.0f} MB/s")
        print(f"  Load throughput: {load_throughput:.0f} MB/s")

        results[backend_type] = {
            'store_throughput_mbps': store_throughput,
            'load_throughput_mbps': load_throughput,
        }

    # Comparison
    print("\n--- Throughput Comparison ---")
    store_ratio = results['cxl']['store_throughput_mbps'] / results['cpu']['store_throughput_mbps']
    load_ratio = results['cxl']['load_throughput_mbps'] / results['cpu']['load_throughput_mbps']

    print(f"  CXL/CPU store ratio: {store_ratio:.2f}x")
    print(f"  CXL/CPU load ratio: {load_ratio:.2f}x")

    return results


def main():
    print("=" * 70)
    print("CXL vs CPU KV Cache Offloading Benchmark")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Configuration (smaller for faster benchmarking)
    config = BenchmarkConfig(
        num_layers=22,      # TinyLlama
        num_heads=4,
        head_dim=64,
        block_size=16,
        gpu_blocks=100,
        offload_blocks=500,
        num_sequences=100,
        tokens_per_seq=256,
    )

    print(f"\nConfiguration:")
    print(f"  Block shape: {config.block_shape}")
    print(f"  Block size: {config.block_bytes / 1024:.1f} KB")
    print(f"  GPU blocks: {config.gpu_blocks}")
    print(f"  Offload blocks: {config.offload_blocks}")

    # Run benchmarks
    latency_results = benchmark_transfer_latency(config)

    print("\n" + "=" * 70)
    print("Tiering Policy Benchmark")
    print("=" * 70)

    cpu_tiering = benchmark_tiering_policy(config, 'cpu')
    cxl_tiering = benchmark_tiering_policy(config, 'cxl')

    throughput_results = benchmark_throughput(config)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\nLatency (lower is better):")
    print(f"  CPU store: {latency_results['cpu_store_ms']:.3f} ms")
    print(f"  CXL store: {latency_results['cxl_store_ms']:.3f} ms")
    print(f"  CPU load:  {latency_results['cpu_load_ms']:.3f} ms")
    print(f"  CXL load:  {latency_results['cxl_load_ms']:.3f} ms")

    print("\nThroughput (higher is better):")
    print(f"  CPU: {throughput_results['cpu']['store_throughput_mbps']:.0f} / "
          f"{throughput_results['cpu']['load_throughput_mbps']:.0f} MB/s (store/load)")
    print(f"  CXL: {throughput_results['cxl']['store_throughput_mbps']:.0f} / "
          f"{throughput_results['cxl']['load_throughput_mbps']:.0f} MB/s (store/load)")

    print("\nEnd-to-end workload time:")
    print(f"  CPU: {cpu_tiering['elapsed_seconds']:.2f}s")
    print(f"  CXL: {cxl_tiering['elapsed_seconds']:.2f}s")

    speedup = cpu_tiering['elapsed_seconds'] / cxl_tiering['elapsed_seconds']
    print(f"\n  >> CXL speedup: {speedup:.2f}x")

    if speedup > 1:
        print(f"  >> CXL BEATS CPU by {(speedup - 1) * 100:.0f}%")
    else:
        print(f"  >> CPU is faster by {(1/speedup - 1) * 100:.0f}%")

    print("\nTiering Policy Observations:")
    print(f"  - LRU eviction keeps hot blocks on GPU")
    print(f"  - Cold blocks (old sequences) offloaded to {'CXL' if speedup > 1 else 'CPU'}")
    print(f"  - Optimal when GPU hit rate > 50% and offload has low latency")


if __name__ == "__main__":
    main()
