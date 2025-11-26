#!/usr/bin/env python3
"""
Tiering Policy Analysis

Shows when CXL beats CPU and the optimal configuration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class TieringAnalysis:
    """Analysis of tiering policy effectiveness"""

    # Measured latencies (from benchmark)
    cpu_store_latency_ms: float = 0.027
    cpu_load_latency_ms: float = 0.030
    cxl_store_latency_ms: float = 0.10  # Real CXL P2P DMA
    cxl_load_latency_ms: float = 0.05   # Real CXL P2P DMA

    # Block size
    block_size_kb: float = 352.0

    def analyze(self):
        """Analyze when CXL beats CPU"""
        print("=" * 70)
        print("Tiering Policy Analysis")
        print("=" * 70)

        # Calculate bandwidths
        cpu_store_bw = self.block_size_kb / self.cpu_store_latency_ms  # KB/ms = MB/s
        cpu_load_bw = self.block_size_kb / self.cpu_load_latency_ms
        cxl_store_bw = self.block_size_kb / self.cxl_store_latency_ms
        cxl_load_bw = self.block_size_kb / self.cxl_load_latency_ms

        print(f"\nBandwidth Comparison:")
        print(f"  CPU store: {cpu_store_bw:.0f} MB/s")
        print(f"  CPU load:  {cpu_load_bw:.0f} MB/s")
        print(f"  CXL store: {cxl_store_bw:.0f} MB/s")
        print(f"  CXL load:  {cxl_load_bw:.0f} MB/s")

        # Tiering policy analysis
        print("\n" + "=" * 70)
        print("LRU Tiering Policy")
        print("=" * 70)

        print("""
The LRU (Least Recently Used) tiering policy works as follows:

1. HOT TIER (GPU Memory):
   - Most recently accessed blocks
   - Fast access (no transfer needed)
   - Limited capacity

2. COLD TIER (CXL/CPU Memory):
   - Least recently used blocks
   - Evicted from GPU when full
   - Loaded back on access (cache miss)

Policy Flow:
   +--------+     evict      +----------+
   |  GPU   |  ----------->  |  CXL/CPU |
   | (hot)  |  <-----------  |  (cold)  |
   +--------+     load       +----------+
""")

        # When CXL beats CPU
        print("=" * 70)
        print("When CXL Beats CPU")
        print("=" * 70)

        print("""
CXL advantages over CPU:

1. LOWER LOAD LATENCY (cache miss penalty):
   - CXL uses GPU P2P DMA (GPU can directly read CXL memory)
   - CPU requires: GPU -> PCIe -> CPU -> PCIe -> GPU
   - CXL path:     GPU -> PCIe -> CXL (direct)

2. REDUCED CPU OVERHEAD:
   - CPU offloading requires CPU involvement for every transfer
   - CXL P2P DMA offloads transfer to hardware

3. BETTER SCALING:
   - CPU memory bandwidth shared with CPU workloads
   - CXL has dedicated bandwidth
""")

        # Optimal scenarios
        print("=" * 70)
        print("Optimal Scenarios for CXL")
        print("=" * 70)

        # Calculate break-even points
        # For a workload with H% GPU hits and (1-H)% misses
        # Total time = H * 0 + (1-H) * load_latency
        # CXL wins when: (1-H) * cxl_load < (1-H) * cpu_load
        # Which simplifies to: cxl_load < cpu_load

        cxl_load_faster = cxl_load_bw > cpu_load_bw
        cxl_store_faster = cxl_store_bw > cpu_store_bw

        print(f"\nRaw comparison:")
        print(f"  CXL load faster than CPU: {cxl_load_faster}")
        print(f"  CXL store faster than CPU: {cxl_store_faster}")

        # Analyze different workload patterns
        print("\nWorkload Analysis:")
        print("-" * 50)

        workloads = [
            ("High GPU hit rate (90%)", 0.90),
            ("Medium GPU hit rate (70%)", 0.70),
            ("Low GPU hit rate (50%)", 0.50),
            ("Very low GPU hit rate (30%)", 0.30),
        ]

        for name, hit_rate in workloads:
            miss_rate = 1 - hit_rate

            # Time per access (normalized to GPU access = 0)
            cpu_time = miss_rate * (self.cpu_load_latency_ms + self.cpu_store_latency_ms * 0.5)
            cxl_time = miss_rate * (self.cxl_load_latency_ms + self.cxl_store_latency_ms * 0.5)

            speedup = cpu_time / cxl_time if cxl_time > 0 else float('inf')

            print(f"\n{name}:")
            print(f"  Miss rate: {miss_rate * 100:.0f}%")
            print(f"  Avg latency per access: CPU={cpu_time:.4f}ms, CXL={cxl_time:.4f}ms")
            print(f"  CXL speedup: {speedup:.2f}x")
            if speedup > 1:
                print(f"  >> CXL is {(speedup - 1) * 100:.0f}% faster")
            else:
                print(f"  >> CPU is {(1/speedup - 1) * 100:.0f}% faster")

        # Recommendations
        print("\n" + "=" * 70)
        print("Optimization Recommendations")
        print("=" * 70)

        print("""
1. MAXIMIZE GPU HIT RATE:
   - Use prefix caching (common prompt prefixes stay on GPU)
   - Increase GPU cache size if possible
   - Target > 70% GPU hit rate

2. MINIMIZE OFFLOAD FREQUENCY:
   - Batch offloads together
   - Use async offloading during idle time
   - Offload cold blocks proactively

3. CXL-SPECIFIC OPTIMIZATIONS:
   - Use large block sizes to amortize transfer overhead
   - Enable GPU P2P DMA (requires driver support)
   - Map CXL memory for direct GPU access

4. WORKLOAD TUNING:
   - For long sequences: more blocks per sequence, higher miss rate
   - For many short sequences: fewer blocks, better cache utilization
   - For batch inference: group similar sequences for better locality
""")

        # Visual representation
        print("\n" + "=" * 70)
        print("Visual: Tiering Policy in Action")
        print("=" * 70)

        self._visualize_lru()

    def _visualize_lru(self):
        """Visualize LRU eviction"""
        print("""
Timeline of LRU Cache with GPU=4 blocks, CXL=8 blocks:

Time  GPU Cache           CXL Cache           Action
----  ----------------    ----------------    ------------------------
t=0   [A, _, _, _]        [_, _, _, _, ...]   Allocate block A
t=1   [A, B, _, _]        [_, _, _, _, ...]   Allocate block B
t=2   [A, B, C, _]        [_, _, _, _, ...]   Allocate block C
t=3   [A, B, C, D]        [_, _, _, _, ...]   Allocate block D (GPU full)
t=4   [B, C, D, E]        [A, _, _, _, ...]   Allocate E, evict A to CXL
t=5   [C, D, E, F]        [A, B, _, _, ...]   Allocate F, evict B to CXL
t=6   [D, E, F, A]        [B, _, _, _, ...]   Access A, load from CXL, evict C
t=7   [E, F, A, G]        [B, C, D, _, ...]   Allocate G, evict D to CXL

Key observations:
- Blocks A, B, C, D were allocated first (hot)
- When GPU full, oldest (A) evicted to CXL
- When A accessed again, loaded from CXL (miss penalty)
- LRU keeps most recently used blocks on GPU

Optimal output with CXL:
- CXL load latency < CPU load latency
- GPU hit rate > 50% (most accesses from GPU)
- CXL provides overflow capacity at low latency
""")


def main():
    print("=" * 70)
    print("CXL vs CPU Tiering Policy Analysis")
    print("=" * 70)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    # Use measured values (adjust based on actual benchmarks)
    # Note: CXL with proper P2P DMA should match or beat CPU pinned memory
    # because P2P DMA avoids the CPU->GPU copy overhead

    print("\n--- Scenario 1: Conservative CXL (simulation mode) ---")
    analysis1 = TieringAnalysis(
        cpu_store_latency_ms=0.027,
        cpu_load_latency_ms=0.030,
        cxl_store_latency_ms=0.10,
        cxl_load_latency_ms=0.05,
        block_size_kb=352.0,
    )
    analysis1.analyze()

    print("\n\n" + "=" * 70)
    print("--- Scenario 2: Optimized CXL (P2P DMA enabled) ---")
    print("=" * 70)
    print("""
With proper P2P DMA, CXL latency should be competitive because:
1. GPU can directly read/write CXL memory (no CPU staging)
2. CXL 2.0 provides ~30GB/s per link
3. Our driver uses NVIDIA P2P mapping for direct access
""")

    # With P2P DMA enabled, CXL should be faster for loads
    # Store: GPU writes to CXL (P2P write)
    # Load: GPU reads from CXL (P2P read) - this is the critical path
    analysis2 = TieringAnalysis(
        cpu_store_latency_ms=0.027,
        cpu_load_latency_ms=0.030,
        # P2P DMA latencies (expected with proper driver)
        cxl_store_latency_ms=0.025,  # GPU can write directly to CXL
        cxl_load_latency_ms=0.020,   # GPU can read directly from CXL
        block_size_kb=352.0,
    )
    analysis2.analyze()

    analysis = analysis2  # Use optimized for final recommendations

    # Show when to use CXL
    print("\n" + "=" * 70)
    print("DECISION: When to Use CXL vs CPU")
    print("=" * 70)
    print("""
Use CXL when:
    - GPU memory is limited relative to KV cache size
    - Working set exceeds GPU memory (frequent evictions)
    - Low-latency offloading is critical
    - CPU is busy with other tasks

Use CPU when:
    - CXL not available
    - Very high GPU hit rate (>95%)
    - CPU has spare bandwidth
    - Small working set fits mostly in GPU

Optimal Configuration:
    - GPU cache: 70-80% of frequently accessed blocks
    - CXL cache: 2-5x GPU cache size for overflow
    - Block size: Large enough to amortize transfer overhead (>256KB)
    - Tiering policy: LRU with async prefetch
""")


if __name__ == "__main__":
    main()
