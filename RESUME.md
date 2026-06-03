# Session 8: Pre-expand Q1_0 weight rows to i8 sign arrays

**Goal:** Eliminate bit-unpacking from the Q1_0 dot-product hot loop by expanding weights at load time. Q1_0 packs 128 1-bit signs into 18 bytes per block. Every matmul call unpacks these on-the-fly. Pre-expanding to 128 i8 bytes (128B/block) turns the kernel into a trivial `i8×i8→i32` dot product — no bit manipulation, no shuffle, just multiply-accumulate.

**Estimated impact:** 15-25% tok/s on small models (1.7B, 4B). 8B may show less or regress from bandwidth pressure (7× more weight data to read).

---

## Implementation plan

### Step 1 — Understand current Q1_0 block format

Read `crates/hearth-quant/src/q1_0g128.rs`. The Q1_0_G128 block is 18 bytes for 128 elements:
- Bytes 0-1: f16 scale (same as Q8_0)
- Byte 2: 8 sign bits for elements 0-7 (bit=0 → +1, bit=1 → -1)
- ...continues for all 128 elements packed into 16 sign bytes (128 bits total)

The current `dot_q1_0g128_q8_0` function iterates each byte, extracts 8 sign bits, looks up in `Q1V[256]` (a 2KB LUT), then multiplies with Q8_0 values. This shuffle + LUT pattern is the hot loop.

### Step 2 — Add expanded weight storage

In `crates/hearth-llm/src/model/mod.rs`, add to `LlamaModel`:
```rust
expanded_q1_weights: HashMap<String, Vec<i8>>,
```

Populate during `load_model()` (in the tensor-loading loop around line 150-200): for each Q1_0 or Q1_0_G128 tensor, unpack the entire tensor into `Vec<i8>` (one `i8` per weight element, value -1 or +1). Store keyed by tensor name.

Memory: each Q1_0 weight grows from 18B/128el (0.14B/el) to 128B/128el (1B/el) = 7.1×. For 1.7B: ~37MB → ~265MB. For 8B: ~1.1GB → ~7.8GB (may not fit). **Conditional flag**: only expand if `d_model < 4096` (skip 8B).

### Step 3 — Write the expanded dot kernel

In `crates/hearth-quant/src/q1_0g128.rs`, add:
```rust
pub fn dot_q1_0g128_q8_0_expanded(w: &[i8], a: &[u8], n: usize) -> f32
```
That's just: for each Q8_0 block, `sum += w[i] * q8_val[i] * scale`. No bit unpacking needed — `w[i]` is already -1 or +1.

Add a raw-pointer variant `dot_q1_0g128_q8_0_expanded_ptr` for the thread-pool dispatch.

### Step 4 — Wire into matmul path

In `crates/hearth-llm/src/model/matmul.rs`:

In `matmul()` under the `Q1_0_G128 | Q1_0` branches: if `self.expanded_q1_weights` contains the weight name, use the expanded dot kernel with the pre-expanded row pointer instead of the packed row pointer. Pass the expanded row's raw pointer to `par_dot_rows`.

**Important:** The expanded row pointer needs to work with `par_dot_rows` which passes the same `row_bytes` (stride) to the worker functions. With expanded weights, each row is `n_cols` bytes (1 i8 per element), not the packed `n_cols/128*18` bytes. You'll need to either:
- Pass the expanded data with `row_bytes = n_cols` and let the dot kernel iterate n_cols elements, OR
- Create a separate dispatch path for expanded weights

The simplest approach: a separate `par_dot_rows_expanded` variant in `pool.rs` that iterates with `row_bytes = n_cols` and calls the expanded dot kernel.

### Step 5 — Update `matmul_batch` similarly

If expanded weights exist for a tensor, use the expanded dot kernel in the batch path too.

---

## Files to modify

| File | Changes |
|---|---|
| `crates/hearth-quant/src/q1_0g128.rs` | Add `dot_q1_0g128_q8_0_expanded` / `_ptr` functions |
| `crates/hearth-quant/src/lib.rs` | Export the new functions |
| `crates/hearth-llm/src/pool.rs` | Add `par_dot_rows_expanded` or parameterize `row_stride` |
| `crates/hearth-llm/src/model/mod.rs` | Add `expanded_q1_weights: HashMap<String, Vec<i8>>`, populate at load time |
| `crates/hearth-llm/src/model/matmul.rs` | Dispatch to expanded kernel when weights exist |

---

## Benchmark & verify

1. Build: `cargo build --release`
2. Run all 6 models one at a time (warm, at least 2 runs each):
   - `& ".\target\release\hearth-chat-cli.exe" "$model" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw`
3. Compare avg_cpu_overhead tok/s to Session 7 baseline below.
4. **If any model degrades, revert.** Particularly watch 8B Q1_0 — the 7× memory increase may cause swapping or bandwidth collapse.

**Session 7 baselines (20-token, warm, second run):**
| Model | avg_cpu_overhead (µs/tok) | tok/s |
|---|---|---|
| 1.7B Q1_0 | 21,778 | 45.9 |
| 1.7B Q2_0 | 35,863 | 27.9 |
| 4B Q1_0 | 42,736 | 23.4 |
| 4B Q2_0 | — | ~11 (BUG_TRACKER) |
| 8B Q1_0 | 73,736 | 13.6 |
| 8B Q2_0 | — | ~5 (BUG_TRACKER) |

---

## Key references
- `crates/hearth-quant/src/q1_0g128.rs` — current packed dot kernel (study the sign-bit extraction pattern)
- `crates/hearth-quant/src/q8_0.rs` — simple reference: Q8_0 dot is `i8×i8×scale×scale`, no bit packing. Expanded Q1_0 should look similar.
- `crates/hearth-llm/src/pool.rs` — `par_dot_rows` with `row_bytes` stride parameter
