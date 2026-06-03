# Session 9: LM head tile-in-L2 matmul

**Goal:** Improve lm_head matmul throughput by dispatching cache-friendly tiles instead of contiguous row chunks. The lm_head is 151669 × 2048 = 310M elements = 43.7MB packed Q1_0. Current pool dispatches 18959-row contiguous chunks per worker — each chunk is 5.5MB, far exceeding the 1MB L2 cache.

By processing ~3555-row tiles per worker (tile size = 1MB L2 / 288 bytes per row), each tile stays hot in L2, reducing cache misses from ~80% to near-zero per tile.

**Estimated impact:** 10-20% on 1.7B Q1_0 lm_head (currently ~5% of forward time), 5-10% on 4B. 8B already bandwidth-limited from L3.

---

## Implementation plan

### Step 1 — Read current pool.rs and matmul.rs

The `par_dot_rows` function in `pool.rs` dispatches contiguous chunks. Change the per-worker iteration to loop over tiles within its chunk:

Current:
```rust
for row in begin..end {
    *out[row] = dot_fn(w_base + row * row_bytes, a_ptr, n_cols);
}
```

Target:
```rust
let tile_rows = 3555;  // ~1MB of weight data
let r_begin = begin;
let r_end = end;
let mut r = r_begin;
while r < r_end {
    let tile_end = (r + tile_rows).min(r_end);
    for row in r..tile_end {
        *out[row] = dot_fn(w_base + row * row_bytes, a_ptr, n_cols);
    }
    r = tile_end;
}
```

The tile loop introduces no memory overhead — just changes the access pattern so each tile's weight data is hot in L2 before moving to the next tile.

**Important:** The tile size must be tuned. 1MB L2 / 288 bytes = 3555 rows for Q1_0. For Q2_0: 1MB / 544 bytes = 1925 rows.

### Step 2 — Add tile_size parameter to WorkParams

In `pool.rs`:
- Add `tile_size: usize` to `WorkParams` (default = `usize::MAX` = no tiling)
- In `par_dot_rows`, pass the tile_size
- Workers use tile_size for inner loop

### Step 3 — Set tile_size in matmul.rs

When dispatching lm_head matmuls, set `tile_size = l2_cache_per_core / row_bytes`:
- For Q1_0: 1024*1024 / (2048/128*18) ≈ 3700 rows/tile
- For Q2_0: 1024*1024 / (2048/128*34) ≈ 1900 rows/tile
- For Q8_0: 1024*1024 / (2048/32*34) ≈ 470 rows/tile

### Step 4 — Measure

Compare tok/s for `lm_head_matmul` timing before and after. The rest of the forward pass should be unchanged.

---

## Benchmark

1. Build: `cargo build --release`
2. Run 1.7B Q1_0 and 1.7B Q2_0 (they have the largest lm_head relative to total model size)
3. Check the `lm_head_matmul` timing section
4. Run all 6 models to verify no regression

**Session 8 baselines (50-token, warm):**
| Model | tok/s | avg_cpu_overhead (µs/tok) |
|---|---|---|
| 1.7B Q1_0 | 47.9 | 20,835 |
| 1.7B Q2_0 | 27.8 | 36,006 |
| 4B Q1_0 | 21.9 | 45,805 |
| 4B Q2_0 | 12.8 | 77,919 |
| 8B Q1_0 | 13.1 | 76,114 |
| 8B Q2_0 | 7.1 | 140,533 |

## Key files
- `crates/hearth-llm/src/pool.rs` — `WorkParams`, `par_dot_rows`, worker loop
- `crates/hearth-llm/src/model/matmul.rs` — lm_head dispatch in matmul() `GgmlDType::Q1_0_G128` / `GgmlDType::Q1_0` branches
