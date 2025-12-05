#!/usr/bin/env python3
"""
GPU to CXL vs GPU to CPU DRAM Latency Benchmark

Tests data transfer latency between:
1. GPU <-> CXL Memory (via P2P DMA)
2. GPU <-> CPU Local DRAM (via PCIe)

IO sizes tested: 64B to 128MB

Usage:
    python tests/benchmark_gpu_latency.py
    python tests/benchmark_gpu_latency.py --iterations 200
    python tests/benchmark_gpu_latency.py --warmup 20 --iterations 100
"""

import sys
import os
import time
import argparse
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.cxl_tensor import CXLTensorManager, CXLTensor

# IO sizes to test (bytes)
IO_SIZES = [
    64,                    # 64B
    256,                   # 256B
    1024,                  # 1KB
    4 * 1024,              # 4KB
    16 * 1024,             # 16KB
    64 * 1024,             # 64KB
    256 * 1024,            # 256KB
    1024 * 1024,           # 1MB
    4 * 1024 * 1024,       # 4MB
    16 * 1024 * 1024,      # 16MB
    64 * 1024 * 1024,      # 64MB
    128 * 1024 * 1024,     # 128MB
]


def format_size(size_bytes: int) -> str:
    """Format byte size to human readable string"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)}MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f}GB"


@dataclass
class LatencyStats:
    """Statistics for latency measurements"""
    latencies_us: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.latencies_us)

    @property
    def avg_us(self) -> float:
        return np.mean(self.latencies_us) if self.latencies_us else 0.0

    @property
    def min_us(self) -> float:
        return np.min(self.latencies_us) if self.latencies_us else 0.0

    @property
    def max_us(self) -> float:
        return np.max(self.latencies_us) if self.latencies_us else 0.0

    @property
    def std_us(self) -> float:
        return np.std(self.latencies_us) if self.latencies_us else 0.0

    @property
    def p50_us(self) -> float:
        return np.percentile(self.latencies_us, 50) if self.latencies_us else 0.0

    @property
    def p99_us(self) -> float:
        return np.percentile(self.latencies_us, 99) if self.latencies_us else 0.0

    def add(self, latency_us: float):
        self.latencies_us.append(latency_us)

    def bandwidth_mbps(self, size_bytes: int) -> float:
        """Calculate bandwidth in MB/s"""
        if self.avg_us <= 0:
            return 0.0
        # size_bytes / (avg_us * 1e-6) = bytes/sec
        # divide by 1e6 to get MB/s
        return (size_bytes / (self.avg_us * 1e-6)) / (1024 * 1024)


@dataclass
class BenchmarkResult:
    """Results for a single IO size benchmark"""
    size_bytes: int
    gpu_to_cpu_stats: LatencyStats
    cpu_to_gpu_stats: LatencyStats
    gpu_to_cxl_stats: LatencyStats
    cxl_to_gpu_stats: LatencyStats


class CPUDRAMBackend:
    """CPU Pinned DRAM backend for GPU transfers"""

    def __init__(self, max_size: int):
        self.max_size = max_size
        # Pre-allocate pinned memory for best performance
        self.pinned_buffer = torch.zeros(max_size // 4 + 1, dtype=torch.float32, pin_memory=True)
        # Pre-create CUDA events for accurate timing (avoids sync overhead)
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def allocate_tensor(self, size_bytes: int, dtype=torch.float32) -> torch.Tensor:
        """Allocate a pinned CPU tensor of given size"""
        numel = size_bytes // 4  # float32 = 4 bytes
        if numel == 0:
            numel = 1
        return torch.zeros(numel, dtype=dtype, pin_memory=True)

    def gpu_to_cpu(self, gpu_tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Transfer GPU tensor to CPU pinned memory.
        Returns (cpu_tensor, latency_us)
        """
        # Use CUDA events for accurate GPU timing (eliminates sync overhead)
        self.start_event.record()
        cpu_tensor = gpu_tensor.cpu()
        self.end_event.record()
        self.end_event.synchronize()
        latency_us = self.start_event.elapsed_time(self.end_event) * 1000  # ms to us
        return cpu_tensor, latency_us

    def cpu_to_gpu(self, cpu_tensor: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, float]:
        """
        Transfer CPU tensor to GPU.
        Returns (gpu_tensor, latency_us)
        """
        # Use CUDA events for accurate GPU timing
        self.start_event.record()
        gpu_tensor = cpu_tensor.to(device, non_blocking=False)
        self.end_event.record()
        self.end_event.synchronize()
        latency_us = self.start_event.elapsed_time(self.end_event) * 1000  # ms to us
        return gpu_tensor, latency_us


class CXLBackend:
    """CXL Memory backend using P2P DMA, CUDA-mapped memory, or CPU-staged"""

    def __init__(self, buffer_size_mb: int = 512):
        # Reset the singleton to get fresh state for each benchmark run
        CXLTensorManager._instance = None

        self.manager = CXLTensorManager.get_instance()

        try:
            self.manager.initialize(buffer_size_mb=buffer_size_mb)
            # Test if CXL/simulation works by attempting a small transfer
            test_tensor = torch.randn(64, dtype=torch.float32, device='cuda:0')
            cxl_test = CXLTensor(test_tensor, self.manager)
            cxl_test.offload_to_cxl()
            result = cxl_test.to_gpu()
            # Verify data integrity
            assert result.shape == test_tensor.shape
            del cxl_test, result
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"CXL initialization test failed: {e}")
            raise

        self.cxl_tensors: Dict[int, tuple] = {}
        self._next_id = 0
        # Pre-create CUDA events for accurate timing
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    @property
    def simulation_mode(self) -> bool:
        return self.manager._simulation_mode

    @property
    def transfer_mode(self) -> str:
        return self.manager.transfer_mode.value if hasattr(self.manager, 'transfer_mode') else 'unknown'

    def gpu_to_cxl(self, gpu_tensor: torch.Tensor, use_kernel_timing: bool = False) -> Tuple[int, float]:
        """
        Transfer GPU tensor to CXL memory.
        Uses direct API for minimal overhead.
        Returns (tensor_id, latency_us)

        If use_kernel_timing=True, uses C-level wall clock timing for memcpy only.
        Otherwise uses CUDA event timing (includes full transfer).
        """
        tensor_id = self._next_id
        self._next_id += 1

        if use_kernel_timing:
            # Use kernel-level timing (measures memcpy only, not CUDA overhead)
            buffer_id, offset, shape, dtype, latency_ns = self.manager.direct_gpu_to_cxl_timed(gpu_tensor)
            self.cxl_tensors[tensor_id] = (buffer_id, offset, shape, dtype)
            latency_us = latency_ns / 1000.0  # ns to us
        else:
            # Use CUDA events for accurate GPU timing
            self.start_event.record()
            buffer_id, offset, shape, dtype = self.manager.direct_gpu_to_cxl(gpu_tensor)
            self.cxl_tensors[tensor_id] = (buffer_id, offset, shape, dtype)
            self.end_event.record()
            self.end_event.synchronize()
            latency_us = self.start_event.elapsed_time(self.end_event) * 1000  # ms to us

        return tensor_id, latency_us

    def cxl_to_gpu(self, tensor_id: int, device: torch.device, use_kernel_timing: bool = False) -> Tuple[torch.Tensor, float]:
        """
        Transfer CXL tensor back to GPU.
        Uses direct API for minimal overhead.
        Returns (gpu_tensor, latency_us)

        If use_kernel_timing=True, uses C-level wall clock timing for memcpy only.
        """
        buffer_id, offset, shape, dtype = self.cxl_tensors[tensor_id]

        if use_kernel_timing:
            # Use kernel-level timing (measures memcpy only)
            gpu_tensor, latency_ns = self.manager.direct_cxl_to_gpu_timed(buffer_id, offset, shape, dtype, device)
            latency_us = latency_ns / 1000.0  # ns to us
        else:
            # Use CUDA events for accurate GPU timing
            self.start_event.record()
            gpu_tensor = self.manager.direct_cxl_to_gpu(buffer_id, offset, shape, dtype, device)
            self.end_event.record()
            self.end_event.synchronize()
            latency_us = self.start_event.elapsed_time(self.end_event) * 1000  # ms to us

        return gpu_tensor, latency_us

    def raw_memcpy_benchmark(self, size_bytes: int) -> Tuple[float, float]:
        """
        Benchmark raw CPU memcpy latency (no CUDA overhead).
        Returns (write_latency_us, read_latency_us)
        """
        write_ns, read_ns = self.manager.memcpy_benchmark(size_bytes)
        return write_ns / 1000.0, read_ns / 1000.0

    def cleanup(self, tensor_id: int):
        """Clean up a CXL tensor"""
        if tensor_id in self.cxl_tensors:
            del self.cxl_tensors[tensor_id]


def create_gpu_tensor(size_bytes: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Create a GPU tensor with given size in bytes"""
    element_size = torch.tensor([], dtype=dtype).element_size()
    numel = size_bytes // element_size
    if numel == 0:
        numel = 1
    return torch.randn(numel, dtype=dtype, device=device)


def benchmark_single_size(
    size_bytes: int,
    device: torch.device,
    cpu_backend: CPUDRAMBackend,
    cxl_backend: CXLBackend,
    warmup_iterations: int,
    test_iterations: int,
    verbose: bool = False,
    use_kernel_timing: bool = False
) -> BenchmarkResult:
    """Run benchmark for a single IO size"""

    gpu_to_cpu_stats = LatencyStats()
    cpu_to_gpu_stats = LatencyStats()
    gpu_to_cxl_stats = LatencyStats()
    cxl_to_gpu_stats = LatencyStats()

    size_str = format_size(size_bytes)
    total_iterations = warmup_iterations + test_iterations

    if verbose:
        print(f"  Testing {size_str}...")

    # Run warmup + test iterations
    for i in range(total_iterations):
        is_warmup = i < warmup_iterations

        # Create fresh GPU tensor for each test
        gpu_tensor = create_gpu_tensor(size_bytes, device)

        # ===== GPU to CPU DRAM =====
        cpu_tensor, latency = cpu_backend.gpu_to_cpu(gpu_tensor)
        if not is_warmup:
            gpu_to_cpu_stats.add(latency)

        # ===== CPU DRAM to GPU =====
        _, latency = cpu_backend.cpu_to_gpu(cpu_tensor, device)
        if not is_warmup:
            cpu_to_gpu_stats.add(latency)

        del cpu_tensor

        # Create fresh GPU tensor for CXL test
        gpu_tensor = create_gpu_tensor(size_bytes, device)

        # ===== GPU to CXL =====
        tensor_id, latency = cxl_backend.gpu_to_cxl(gpu_tensor, use_kernel_timing=use_kernel_timing)
        if not is_warmup:
            gpu_to_cxl_stats.add(latency)

        # ===== CXL to GPU =====
        _, latency = cxl_backend.cxl_to_gpu(tensor_id, device, use_kernel_timing=use_kernel_timing)
        if not is_warmup:
            cxl_to_gpu_stats.add(latency)

        cxl_backend.cleanup(tensor_id)

        # Clear GPU memory periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()

    return BenchmarkResult(
        size_bytes=size_bytes,
        gpu_to_cpu_stats=gpu_to_cpu_stats,
        cpu_to_gpu_stats=cpu_to_gpu_stats,
        gpu_to_cxl_stats=gpu_to_cxl_stats,
        cxl_to_gpu_stats=cxl_to_gpu_stats
)


def benchmark_raw_memcpy(
    io_sizes: List[int],
    cxl_backend: CXLBackend,
    warmup_iterations: int,
    test_iterations: int
) -> Dict[int, Tuple[LatencyStats, LatencyStats]]:
    """
    Benchmark raw CPU memcpy latency (no CUDA overhead).
    Returns {size_bytes: (write_stats, read_stats)}
    """
    results = {}

    for size_bytes in io_sizes:
        write_stats = LatencyStats()
        read_stats = LatencyStats()

        for i in range(warmup_iterations + test_iterations):
            is_warmup = i < warmup_iterations

            write_us, read_us = cxl_backend.raw_memcpy_benchmark(size_bytes)

            if not is_warmup:
                write_stats.add(write_us)
                read_stats.add(read_us)

        results[size_bytes] = (write_stats, read_stats)

    return results


def print_raw_memcpy_results(results: Dict[int, Tuple[LatencyStats, LatencyStats]]):
    """Print raw memcpy benchmark results"""
    print("\n" + "=" * 100)
    print("RAW MEMCPY LATENCY (nanoseconds) - No CUDA/PyTorch overhead")
    print("This shows the theoretical best-case data movement latency")
    print("=" * 100)

    print(f"\n{'Size':<10} | {'Write (ns)':<40} | {'Read (ns)':<40}")
    print(f"{'':10} | {'Avg':>10} {'Min':>10} {'Max':>10} | {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("-" * 100)

    for size_bytes in sorted(results.keys()):
        write_stats, read_stats = results[size_bytes]
        size_str = format_size(size_bytes)

        # Convert to nanoseconds for display
        write_avg_ns = write_stats.avg_us * 1000
        write_min_ns = write_stats.min_us * 1000
        write_max_ns = write_stats.max_us * 1000
        read_avg_ns = read_stats.avg_us * 1000
        read_min_ns = read_stats.min_us * 1000
        read_max_ns = read_stats.max_us * 1000

        print(f"{size_str:<10} | "
              f"{write_avg_ns:>10.0f} {write_min_ns:>10.0f} {write_max_ns:>10.0f} | "
              f"{read_avg_ns:>10.0f} {read_min_ns:>10.0f} {read_max_ns:>10.0f}")


def print_header():
    """Print benchmark header"""
    print("=" * 120)
    print("GPU to CXL vs GPU to CPU DRAM Latency Benchmark")
    print("=" * 120)


def print_system_info(device: torch.device, cxl_backend: Optional['CXLBackend'] = None):
    """Print system information"""
    print("\nSystem Information:")
    print("-" * 40)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"  GPU: {gpu_name}")
    print(f"  GPU Memory: {gpu_mem:.1f} GB")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")

    # CXL info
    manager = CXLTensorManager.get_instance()
    try:
        manager.initialize()
        cxl_info = manager.get_info()
    except:
        cxl_info = {}

    print(f"\n  CXL Info:")
    print(f"    Link Up: {cxl_info.get('link_up', 'N/A')}")
    print(f"    Memory Expander: {cxl_info.get('memory_expander', 'N/A')}")
    print(f"    CXL Version: {cxl_info.get('version', 'N/A')}")
    print(f"    Bandwidth: {cxl_info.get('bandwidth_mbps', 'N/A')} MB/s")

    if cxl_backend:
        sim_mode = cxl_backend.simulation_mode
        transfer_mode = cxl_backend.transfer_mode
    else:
        sim_mode = manager._simulation_mode
        transfer_mode = manager.transfer_mode.value if hasattr(manager, 'transfer_mode') else 'unknown'

    print(f"    Simulation Mode: {sim_mode}")
    print(f"    Transfer Mode: {transfer_mode}")
    if sim_mode:
        print(f"    NOTE: CXL hardware not available, using CUDA-mapped pinned memory")


def print_latency_table(results: List[BenchmarkResult]):
    """Print latency results in table format"""
    print("\n" + "=" * 120)
    print("LATENCY RESULTS (microseconds)")
    print("=" * 120)

    # Header
    print(f"\n{'Size':<10} | {'GPU->CPU':<45} | {'GPU->CXL':<45}")
    print(f"{'':10} | {'Avg':>10} {'Min':>10} {'Max':>10} {'P99':>10} | {'Avg':>10} {'Min':>10} {'Max':>10} {'P99':>10}")
    print("-" * 120)

    for r in results:
        size_str = format_size(r.size_bytes)
        print(f"{size_str:<10} | "
              f"{r.gpu_to_cpu_stats.avg_us:>10.1f} "
              f"{r.gpu_to_cpu_stats.min_us:>10.1f} "
              f"{r.gpu_to_cpu_stats.max_us:>10.1f} "
              f"{r.gpu_to_cpu_stats.p99_us:>10.1f} | "
              f"{r.gpu_to_cxl_stats.avg_us:>10.1f} "
              f"{r.gpu_to_cxl_stats.min_us:>10.1f} "
              f"{r.gpu_to_cxl_stats.max_us:>10.1f} "
              f"{r.gpu_to_cxl_stats.p99_us:>10.1f}")

    print("\n" + "-" * 120)
    print(f"\n{'Size':<10} | {'CPU->GPU':<45} | {'CXL->GPU':<45}")
    print(f"{'':10} | {'Avg':>10} {'Min':>10} {'Max':>10} {'P99':>10} | {'Avg':>10} {'Min':>10} {'Max':>10} {'P99':>10}")
    print("-" * 120)

    for r in results:
        size_str = format_size(r.size_bytes)
        print(f"{size_str:<10} | "
              f"{r.cpu_to_gpu_stats.avg_us:>10.1f} "
              f"{r.cpu_to_gpu_stats.min_us:>10.1f} "
              f"{r.cpu_to_gpu_stats.max_us:>10.1f} "
              f"{r.cpu_to_gpu_stats.p99_us:>10.1f} | "
              f"{r.cxl_to_gpu_stats.avg_us:>10.1f} "
              f"{r.cxl_to_gpu_stats.min_us:>10.1f} "
              f"{r.cxl_to_gpu_stats.max_us:>10.1f} "
              f"{r.cxl_to_gpu_stats.p99_us:>10.1f}")


def print_bandwidth_table(results: List[BenchmarkResult]):
    """Print bandwidth results in table format"""
    print("\n" + "=" * 120)
    print("BANDWIDTH RESULTS (MB/s)")
    print("=" * 120)

    print(f"\n{'Size':<10} | {'GPU->CPU':>12} | {'CPU->GPU':>12} | {'GPU->CXL':>12} | {'CXL->GPU':>12} | {'Speedup (Write)':>15} | {'Speedup (Read)':>15}")
    print("-" * 120)

    for r in results:
        size_str = format_size(r.size_bytes)

        gpu_to_cpu_bw = r.gpu_to_cpu_stats.bandwidth_mbps(r.size_bytes)
        cpu_to_gpu_bw = r.cpu_to_gpu_stats.bandwidth_mbps(r.size_bytes)
        gpu_to_cxl_bw = r.gpu_to_cxl_stats.bandwidth_mbps(r.size_bytes)
        cxl_to_gpu_bw = r.cxl_to_gpu_stats.bandwidth_mbps(r.size_bytes)

        # Speedup: CXL vs CPU (>1 means CXL is faster)
        write_speedup = gpu_to_cxl_bw / gpu_to_cpu_bw if gpu_to_cpu_bw > 0 else 0
        read_speedup = cxl_to_gpu_bw / cpu_to_gpu_bw if cpu_to_gpu_bw > 0 else 0

        write_speedup_str = f"{write_speedup:.2f}x" if write_speedup >= 1 else f"{write_speedup:.2f}x (slower)"
        read_speedup_str = f"{read_speedup:.2f}x" if read_speedup >= 1 else f"{read_speedup:.2f}x (slower)"

        print(f"{size_str:<10} | "
              f"{gpu_to_cpu_bw:>12.1f} | "
              f"{cpu_to_gpu_bw:>12.1f} | "
              f"{gpu_to_cxl_bw:>12.1f} | "
              f"{cxl_to_gpu_bw:>12.1f} | "
              f"{write_speedup_str:>15} | "
              f"{read_speedup_str:>15}")


def print_comparison_summary(results: List[BenchmarkResult]):
    """Print comparison summary"""
    print("\n" + "=" * 120)
    print("COMPARISON SUMMARY")
    print("=" * 120)

    # Find crossover points where CXL becomes faster
    write_crossover = None
    read_crossover = None

    for r in results:
        if r.gpu_to_cxl_stats.avg_us < r.gpu_to_cpu_stats.avg_us and write_crossover is None:
            write_crossover = r.size_bytes
        if r.cxl_to_gpu_stats.avg_us < r.cpu_to_gpu_stats.avg_us and read_crossover is None:
            read_crossover = r.size_bytes

    print("\nLatency Comparison (lower is better):")
    print("-" * 60)

    # Small IO comparison (64B - 4KB)
    small_results = [r for r in results if r.size_bytes <= 4096]
    if small_results:
        avg_cpu_write = np.mean([r.gpu_to_cpu_stats.avg_us for r in small_results])
        avg_cxl_write = np.mean([r.gpu_to_cxl_stats.avg_us for r in small_results])
        avg_cpu_read = np.mean([r.cpu_to_gpu_stats.avg_us for r in small_results])
        avg_cxl_read = np.mean([r.cxl_to_gpu_stats.avg_us for r in small_results])

        print(f"\n  Small IO (64B-4KB):")
        print(f"    Write: CPU={avg_cpu_write:.1f}us, CXL={avg_cxl_write:.1f}us")
        print(f"    Read:  CPU={avg_cpu_read:.1f}us, CXL={avg_cxl_read:.1f}us")
        winner_write = "CXL" if avg_cxl_write < avg_cpu_write else "CPU"
        winner_read = "CXL" if avg_cxl_read < avg_cpu_read else "CPU"
        print(f"    Winner: Write={winner_write}, Read={winner_read}")

    # Large IO comparison (16MB+)
    large_results = [r for r in results if r.size_bytes >= 16 * 1024 * 1024]
    if large_results:
        avg_cpu_write = np.mean([r.gpu_to_cpu_stats.avg_us for r in large_results])
        avg_cxl_write = np.mean([r.gpu_to_cxl_stats.avg_us for r in large_results])
        avg_cpu_read = np.mean([r.cpu_to_gpu_stats.avg_us for r in large_results])
        avg_cxl_read = np.mean([r.cxl_to_gpu_stats.avg_us for r in large_results])

        print(f"\n  Large IO (16MB+):")
        print(f"    Write: CPU={avg_cpu_write/1000:.2f}ms, CXL={avg_cxl_write/1000:.2f}ms")
        print(f"    Read:  CPU={avg_cpu_read/1000:.2f}ms, CXL={avg_cxl_read/1000:.2f}ms")
        winner_write = "CXL" if avg_cxl_write < avg_cpu_write else "CPU"
        winner_read = "CXL" if avg_cxl_read < avg_cpu_read else "CPU"
        print(f"    Winner: Write={winner_write}, Read={winner_read}")

    print(f"\nCrossover Points (where CXL becomes faster than CPU):")
    print("-" * 60)
    if write_crossover:
        print(f"  Write crossover: {format_size(write_crossover)}")
    else:
        print(f"  Write crossover: Not found (CPU always faster for writes)")

    if read_crossover:
        print(f"  Read crossover: {format_size(read_crossover)}")
    else:
        print(f"  Read crossover: Not found (CPU always faster for reads)")

    # Peak bandwidth
    print(f"\nPeak Bandwidth:")
    print("-" * 60)

    max_cpu_write_bw = max(r.gpu_to_cpu_stats.bandwidth_mbps(r.size_bytes) for r in results)
    max_cpu_read_bw = max(r.cpu_to_gpu_stats.bandwidth_mbps(r.size_bytes) for r in results)
    max_cxl_write_bw = max(r.gpu_to_cxl_stats.bandwidth_mbps(r.size_bytes) for r in results)
    max_cxl_read_bw = max(r.cxl_to_gpu_stats.bandwidth_mbps(r.size_bytes) for r in results)

    print(f"  CPU DRAM Write: {max_cpu_write_bw:.0f} MB/s ({max_cpu_write_bw/1024:.1f} GB/s)")
    print(f"  CPU DRAM Read:  {max_cpu_read_bw:.0f} MB/s ({max_cpu_read_bw/1024:.1f} GB/s)")
    print(f"  CXL Write:      {max_cxl_write_bw:.0f} MB/s ({max_cxl_write_bw/1024:.1f} GB/s)")
    print(f"  CXL Read:       {max_cxl_read_bw:.0f} MB/s ({max_cxl_read_bw/1024:.1f} GB/s)")


def export_csv(results: List[BenchmarkResult], filename: str):
    """Export results to CSV file"""
    with open(filename, 'w') as f:
        # Header
        f.write("Size_Bytes,Size_Human,"
                "GPU_to_CPU_Avg_us,GPU_to_CPU_Min_us,GPU_to_CPU_Max_us,GPU_to_CPU_P99_us,"
                "CPU_to_GPU_Avg_us,CPU_to_GPU_Min_us,CPU_to_GPU_Max_us,CPU_to_GPU_P99_us,"
                "GPU_to_CXL_Avg_us,GPU_to_CXL_Min_us,GPU_to_CXL_Max_us,GPU_to_CXL_P99_us,"
                "CXL_to_GPU_Avg_us,CXL_to_GPU_Min_us,CXL_to_GPU_Max_us,CXL_to_GPU_P99_us,"
                "GPU_to_CPU_BW_MBps,CPU_to_GPU_BW_MBps,GPU_to_CXL_BW_MBps,CXL_to_GPU_BW_MBps\n")

        for r in results:
            f.write(f"{r.size_bytes},{format_size(r.size_bytes)},"
                    f"{r.gpu_to_cpu_stats.avg_us:.2f},{r.gpu_to_cpu_stats.min_us:.2f},"
                    f"{r.gpu_to_cpu_stats.max_us:.2f},{r.gpu_to_cpu_stats.p99_us:.2f},"
                    f"{r.cpu_to_gpu_stats.avg_us:.2f},{r.cpu_to_gpu_stats.min_us:.2f},"
                    f"{r.cpu_to_gpu_stats.max_us:.2f},{r.cpu_to_gpu_stats.p99_us:.2f},"
                    f"{r.gpu_to_cxl_stats.avg_us:.2f},{r.gpu_to_cxl_stats.min_us:.2f},"
                    f"{r.gpu_to_cxl_stats.max_us:.2f},{r.gpu_to_cxl_stats.p99_us:.2f},"
                    f"{r.cxl_to_gpu_stats.avg_us:.2f},{r.cxl_to_gpu_stats.min_us:.2f},"
                    f"{r.cxl_to_gpu_stats.max_us:.2f},{r.cxl_to_gpu_stats.p99_us:.2f},"
                    f"{r.gpu_to_cpu_stats.bandwidth_mbps(r.size_bytes):.2f},"
                    f"{r.cpu_to_gpu_stats.bandwidth_mbps(r.size_bytes):.2f},"
                    f"{r.gpu_to_cxl_stats.bandwidth_mbps(r.size_bytes):.2f},"
                    f"{r.cxl_to_gpu_stats.bandwidth_mbps(r.size_bytes):.2f}\n")

    print(f"\nResults exported to: {filename}")


def main():
    parser = argparse.ArgumentParser(description="GPU to CXL/CPU Latency Benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations (default: 10)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of test iterations (default: 100)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--csv", type=str, default=None, help="Export results to CSV file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--sizes", type=str, default=None,
                        help="Comma-separated list of sizes to test (e.g., '64,1024,1048576')")
    parser.add_argument("--kernel-timing", action="store_true",
                        help="Use kernel-level wall clock timing (measures memcpy only, excludes CUDA overhead)")
    parser.add_argument("--raw-memcpy", action="store_true",
                        help="Also run raw CPU memcpy benchmark (shows theoretical best-case latency)")
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU required for this benchmark")
        sys.exit(1)

    device = torch.device(f'cuda:{args.device}')

    print_header()

    # Parse custom sizes if provided
    if args.sizes:
        io_sizes = [int(s.strip()) for s in args.sizes.split(',')]
    else:
        io_sizes = IO_SIZES

    # Initialize backends first (needed for system info)
    print("\nInitializing backends...")
    max_size = max(io_sizes)
    cpu_backend = CPUDRAMBackend(max_size)
    cxl_backend = CXLBackend(buffer_size_mb=max(512, (max_size * 10) // (1024 * 1024)))

    print_system_info(device, cxl_backend)

    print(f"\nBenchmark Configuration:")
    print("-" * 40)
    print(f"  Warmup iterations: {args.warmup}")
    print(f"  Test iterations: {args.iterations}")
    print(f"  IO sizes: {len(io_sizes)} ({format_size(min(io_sizes))} to {format_size(max(io_sizes))})")
    print(f"  Kernel timing: {args.kernel_timing}")
    print(f"  Raw memcpy benchmark: {args.raw_memcpy}")

    # Run raw memcpy benchmark first if requested
    if args.raw_memcpy:
        print("\nRunning raw memcpy benchmark...")
        print("-" * 40)
        raw_results = benchmark_raw_memcpy(io_sizes, cxl_backend, args.warmup, args.iterations)
        print_raw_memcpy_results(raw_results)

    print("\nRunning GPU transfer benchmarks...")
    print("-" * 40)

    results = []
    total_sizes = len(io_sizes)

    for idx, size_bytes in enumerate(io_sizes):
        print(f"[{idx+1}/{total_sizes}] Testing {format_size(size_bytes)}...", end=" ", flush=True)

        result = benchmark_single_size(
            size_bytes=size_bytes,
            device=device,
            cpu_backend=cpu_backend,
            cxl_backend=cxl_backend,
            warmup_iterations=args.warmup,
            test_iterations=args.iterations,
            verbose=args.verbose,
            use_kernel_timing=args.kernel_timing
        )
        results.append(result)

        # Print quick summary
        print(f"CPU: {result.gpu_to_cpu_stats.avg_us:.1f}/{result.cpu_to_gpu_stats.avg_us:.1f}us, "
              f"CXL: {result.gpu_to_cxl_stats.avg_us:.1f}/{result.cxl_to_gpu_stats.avg_us:.1f}us (w/r)")

        # Clear GPU memory between sizes
        torch.cuda.empty_cache()

    # Print results
    print_latency_table(results)
    print_bandwidth_table(results)
    print_comparison_summary(results)

    # Export CSV if requested
    if args.csv:
        export_csv(results, args.csv)

    print("\n" + "=" * 120)
    print("Benchmark complete!")
    print("=" * 120)


if __name__ == "__main__":
    main()
