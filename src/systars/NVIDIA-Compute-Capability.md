# NVIDIA SIMT Compute Capability

## Marketing numbers

|               |       CUDA Cores           |            Tensor Cores                    |
| ARCHITECTURE  |  FP64 | FP32 | FP16 | INT8 | | FP64 | TF32 | FP16 | INT8 | INT4 |  INT1 |
|---------------|-------|------|------|------|-|------|------|------|------|------|-------|
|   Volta       |  32   |  64  |  128 |  256 | |    - |   -  |  512 |  -   | -    |   -   |
|  Turing       |   2   |  64  |  128 |  256 | |    - |   -  |  512 | 1024 | 2048 |  8192 |
| Ampere A100   |  32   |  64  |  256 |  256 | |   64 |  512 | 1024 | 2048 | 4096 | 16384 |
| Ampere Sparse |       |      |      |      | |    - | 1024 | 2048 | 4096 | 8192 |   -   |

This is a common source of confusion. The table shows **throughput (ops/SM/cycle)**, not physical core counts. NVIDIA's marketing conflates these, which muddies the architectural picture.

## The Mapping

| Table Value | What It Actually Means |
|-------------|------------------------|
| FP32: 64 | 64 FP32 operations per SM per cycle |
| FP64: 32 | 32 FP64 operations per SM per cycle |
| Tensor FP16: 1024 | 1024 FP16 FMA ops per SM per cycle (via Tensor Cores) |

For simple ALU operations, throughput equals physical core count because each core retires one op per cycle. So:

**GA100 (A100) physical layout per SM:**
- 4 partitions × 16 FP32 cores = 64 → matches table
- 4 partitions × 8 FP64 cores = 32 → matches table
- 4 partitions × 16 INT32 cores = 64

## The GA100 vs GA10x Discrepancy

Here's the critical issue: **GA100 (datacenter) and GA10x (consumer) have different SM designs**, despite both being "Ampere."

```
GA100 (A100, CC 8.0):
┌────────────────────────┐
│ Partition                       │
│  16× FP32 (dedicated)           │
│  16× INT32 (dedicated)          │
│   8× FP64                       │
│   1× Tensor Core                │
└────────────────────────┘
→ 64 FP32 ops/cycle, 64 INT32 ops/cycle (parallel)

GA10x (RTX 30xx, CC 8.6):
┌────────────────────────┐
│ Partition                       │
│  16× FP32 (dedicated)           │
│  16× FP32/INT32 (dual-purpose)  │
│   2× FP64 (minimal)             │
│   1× Tensor Core                │
└────────────────────────┘
→ 128 FP32 ops/cycle OR 64 FP32 + 64 INT32 (trade-off)
```

Your original `cores_per_partition = 32` was correct **for GA10x**, not GA100.

## Corrected Parameterized Config

```cpp
#include <cstdint>

enum class ComputeCapability : uint32_t {
    SM_80 = 800,  // GA100: A100
    SM_86 = 860,  // GA10x: RTX 30xx, A40, etc.
};

template <ComputeCapability CC>
struct SmConfig;

template <>
struct SmConfig<ComputeCapability::SM_80> {
    // Physical cores per partition
    static constexpr int fp32_cores_per_partition  = 16;
    static constexpr int int32_cores_per_partition = 16;  // Dedicated, parallel with FP32
    static constexpr int fp64_cores_per_partition  = 8;
    
    // Throughput per SM per cycle
    static constexpr int fp32_ops_per_cycle  = 64;   // 4 × 16
    static constexpr int int32_ops_per_cycle = 64;   // 4 × 16, concurrent with FP32
    static constexpr int fp64_ops_per_cycle  = 32;   // 4 × 8
    
    static constexpr int num_partitions       = 4;
    static constexpr int warp_size            = 32;
    static constexpr int max_warps_per_sm     = 64;
    static constexpr int max_blocks_per_sm    = 32;
    static constexpr int tensor_cores_per_sm  = 4;
};

template <>
struct SmConfig<ComputeCapability::SM_86> {
    // Physical cores per partition
    static constexpr int fp32_dedicated_per_partition = 16;
    static constexpr int fp32_int32_shared_per_partition = 16;  // Dual-purpose datapath
    static constexpr int fp64_cores_per_partition = 2;          // Minimal, for compatibility
    
    // Throughput per SM per cycle (mutually exclusive modes)
    static constexpr int fp32_max_ops_per_cycle = 128;  // 4 × 32, if no INT32 on shared path
    static constexpr int fp32_with_int32_ops    = 64;   // 4 × 16, if INT32 also running
    static constexpr int int32_ops_per_cycle    = 64;   // 4 × 16
    static constexpr int fp64_ops_per_cycle     = 8;    // 4 × 2
    
    static constexpr int num_partitions       = 4;
    static constexpr int warp_size            = 32;
    static constexpr int max_warps_per_sm     = 48;
    static constexpr int max_blocks_per_sm    = 16;
    static constexpr int tensor_cores_per_sm  = 4;
};
```

## Why NVIDIA Does This

The marketing term "CUDA Core" maps 1:1 with ops/cycle for FP32, letting them advertise "10,752 CUDA Cores" for an RTX 3080 (84 SMs × 128). It's technically accurate but architecturally misleading since those aren't independent physical units—they're throughput capacity on shared datapaths.

For Tensor Cores, the numbers are even more abstract: they represent MMA throughput for specific matrix tile sizes, not anything resembling a "core count."