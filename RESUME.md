# Session 10: Profile-guided optimization targets

**Session 9 result:** Tile-in-L2 matmul dispatch implemented in pool.rs but had **neutral impact** on lm_head (reads each weight row exactly once — no L2 reuse benefit). No regressions across all 6 models. Minor 4B Q1_0 improvement (+6.4%) likely system variance.

**Key insight from Session 9:** Weight-stationary matmul reads each row exactly once per forward — tiling worker chunks to fit L2 provides no reuse benefit. The remaining bottlenecks need different approaches.

---

## Next optimization targets (from timing profiles)

Session 9 warm 1.7B Q1_0 profile (~18ms forward):
- ffn_gate_up_matmul: ~35% (6300µs)
- ffn_down_matmul: ~20% (3700µs)
- qkv_matmul: ~12% (2200µs)
- lm_head_matmul: ~14% (2700µs)
- attention: ~6% (1100µs)
- attn_output_matmul: ~7% (1300µs)

### Target 1: Fuse Q8_0 quant into first matmul row
The `quantize_act()` call is ~0.5% of forward time (160µs), but fusing it into the first matmul row avoids the separate quantize pass entirely. Since `matmul()` already receives `x_q8: Option<&[u8]>`, the Q/K/V and FFN matmuls after the first per-layer matmul reuse the cached Q8 buffer. The gain is bounded to ~0.5% but removes a pass over `x` data in memory.

### Target 2: Check Q1_0_kernel (d=4096 8B) for 8B Q1_0 improvement
8B Q1_0 lags at 13.5 tok/s vs 8.2 ref = 1.65×. 8B has d=4096 leading to 576-byte rows in Q1_0. The shuffle kernel reads 16 block headers (scales) then 16×16-byte sign groups. At d=4096, a dot product iterates 32 blocks = 2 internal loops. Consider whether widening to 8-wide SIMD for sign accumulation would help.

### Target 3: Investigate persistent 8B Q2_0 gap
8B Q2_0 at 7.0 tok/s vs 1.5 ref = 4.67×. This model is completely bandwidth-bound (2.1GB model on DDR5). Single-core ref achieves 1.5 tok/s scaling linearly to 8-core = 12 tok/s. Hearth at 7.0 tok/s is only 58% of linear scaling. Investigate memory contention, cache line bouncing, or TLB misses from the Q2V lookup table.

### Target 4: Continue investigating AVX-512 (if hardware available)
The Ryzen 7 8840HS has AVX-512 (Zen 4). Currently enabled via `target-cpu=native`. Could write AVX-512 kernels for Q1_0 and Q2_0 dot products. LLVM auto-vectorization may not be optimal for the shuffle kernel pattern.

---

## Key files
- `crates/hearth-llm/src/pool.rs` — `WorkParams` now has `tile_size` field; workers tile within chunks
- `crates/hearth-llm/src/model/matmul.rs` — matmul dispatch (no changes needed for tiling)
- `crates/hearth-llm/src/model/mod.rs` — forward pass, lm_head timing

## Session 9 baselines (50-token, 10-prompt warm avg)
| Model | tok/s | avg_cpu_overhead (µs/tok) |
|---|---|---|
| 1.7B Q1_0 | 49.5 | 20,202 |
| 1.7B Q2_0 | 28.7 | 34,843 |
| 4B Q1_0 | 23.5 | 42,553 |
| 4B Q2_0 | 13.0 | 76,923 |
| 8B Q1_0 | 13.4 | 74,626 |
| 8B Q2_0 | 7.1 | 140,845 |
