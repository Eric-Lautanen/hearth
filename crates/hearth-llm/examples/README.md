# hearth-llm examples

Run with `cargo run --release -p hearth-llm --example <name>`.

## Tools

| Example | Purpose |
| --- | --- |
| `dump_gguf` | Print GGUF metadata and tensor shapes/dtypes (use it to inspect a model before loading). |
| `benchk` | Raw throughput of the quantized dot kernels (GB/s per thread) for one layer's weights. |
| `gen` | Headless text generation through `Model::run`: `--example gen -- <model.gguf> [prompt] [max_tokens] [temp]`. Prints the text plus tok/s; smoke-tests every model without the UI. |

## Diagnostic probes

These were used to isolate numerical bugs in the PrismML `qwen35` path by
comparing single-forward-pass activations against a reference build of the
PrismML llama.cpp fork. Several expect reference dumps (`refx.<tensor>.f32`) in
`%TEMP%`; produce them by patching that fork's eval callback to write named
tensors to disk, then run it as
`REFCHK=1 REFCHK_DUMP=<prefix> ./oracle <model.gguf> TOK <token_id> 1`.

| Example | Purpose |
| --- | --- |
| `probe_embed` | Brute-forces which (token, rotation variant) reproduces the reference's layer-0 `attn_norm`. This is what proved the embedding inverse is `R⁻¹ = H·D` (WHT then signs), not `R`. |
| `probe_matmul_rot` | Identifies the transform a folded weight needs: `signs→WHT` for the activation (matches `mm`). |
| `probe_ssm_out` | Verifies the `ssm_out` head permutation + rotation and, at the end of the file, checks whether a residual FFN error is a real math bug or activation-quantization precision. |
| `probe_matmul_cmp` | Compares `dequant(row)·x` between the fork's dequantizer and ours. |
| `probe_weight_stats` | Mean/max of dequantized tensors, ours vs the fork's. |
| `probe_quant` | Round-trip and dot-kernel sanity checks for PTQ1_0 / PQ2_0. |
| `probe_signs` | Scans embedding rows against the Hadamard sign groups. |
| `probe_q35_dbg` | Tokenizer / embedding-lookup probe (used to catch the BPE pre-tokenization bug). |

The runtime itself can dump matching activations with
`HEARTH_DUMP_TENSORS=<dir-or-prefix>` (layer 0, position 0) and print stage
statistics with `HEARTH_Q35_DEBUG=1`.
