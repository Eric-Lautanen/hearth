//! Software throughput of the hand-rolled primitives, for calibration.
//!
//! Run with `cargo run --release -p hearth-tls --example crypto_bench`.
//!
//! These are the numbers that decide whether the hardware-acceleration slice
//! (AES-NI + PCLMULQDQ) is a nicety or a requirement: the downloader will push
//! multi-gigabyte model files through the record layer, so the interesting
//! figure is bytes per second, not cycles per block.
//!
//! Sequential by construction — one measurement at a time, no threads — because
//! this box's results move 10-15% under any other load.

use std::time::Instant;

use hearth_tls::{x25519, Aes128, Aes128Gcm};

fn mb_per_second(bytes: usize, seconds: f64) -> f64 {
    (bytes as f64 / (1024.0 * 1024.0)) / seconds
}

fn main() {
    // ---- AES-128 block function ------------------------------------------
    let aes = Aes128::new(&[0x42; 16]);
    let mut block = [0u8; 16];
    let blocks = 2_000_000u64;
    let start = Instant::now();
    for _ in 0..blocks {
        aes.encrypt_block(&mut block);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "AES-128 block:      {:>8.1} MB/s   ({:.0} ns/block, {:>6.1} M blocks/s)",
        mb_per_second((blocks * 16) as usize, elapsed),
        elapsed / blocks as f64 * 1e9,
        blocks as f64 / elapsed / 1e6
    );

    // The same function with the hardware path ruled out, so the effect is
    // measured in one process rather than inferred across builds.
    let start = Instant::now();
    for _ in 0..blocks {
        aes.encrypt_block_software(&mut block);
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "  AES-128 software: {:>8.1} MB/s   (aes-ni={})",
        mb_per_second((blocks * 16) as usize, elapsed),
        aes.uses_hardware()
    );

    // ---- AES-128-GCM ------------------------------------------------------
    let gcm = Aes128Gcm::new(&[0x42; 16]);
    let nonce = [0u8; 12];
    let aad = [0u8; 13];
    let (aes_ni, pclmul) = gcm.accelerated();
    let forced_off = std::env::var_os("HEARTH_TLS_NO_HW").is_some();
    println!(
        "dispatch: aes-ni={aes_ni} pclmul={pclmul} (HEARTH_TLS_NO_HW {})",
        if forced_off { "set" } else { "unset" }
    );

    for (name, len, iters) in [
        ("GCM 16 KiB record", 16 * 1024usize, 2000u32),
        ("GCM 1 MiB", 1024 * 1024, 32),
    ] {
        let plaintext: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        let start = Instant::now();
        let mut total = 0usize;
        for _ in 0..iters {
            let (ct, tag) = gcm.seal(&nonce, &aad, &plaintext);
            total += ct.len();
            assert_eq!(tag.len(), 16);
        }
        let elapsed = start.elapsed().as_secs_f64();
        println!("{name}:  {:>8.1} MB/s", mb_per_second(total, elapsed));
    }

    // Seal-and-open, i.e. the real cost of a record in each direction.
    let plaintext: Vec<u8> = (0..16 * 1024).map(|i| (i % 251) as u8).collect();
    let (ct, tag) = gcm.seal(&nonce, &aad, &plaintext);
    let start = Instant::now();
    for _ in 0..2000 {
        gcm.open(&nonce, &aad, &ct, &tag).expect("authentic");
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "GCM open 16 KiB:    {:>8.1} MB/s",
        mb_per_second(2000 * 16 * 1024, elapsed)
    );

    // ---- X25519 -----------------------------------------------------------
    let scalar = [0x77u8; 32];
    let u = [0x09u8; 32];
    let ops = 200u32;
    let start = Instant::now();
    for _ in 0..ops {
        std::hint::black_box(x25519(&scalar, &u));
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "X25519:             {:>8.0} ops/s ({:.0} us/op)",
        ops as f64 / elapsed,
        elapsed / ops as f64 * 1e6
    );

    // ---- What that means for a model download ----------------------------
    let gcm_bytes_per_s = {
        let plaintext: Vec<u8> = vec![0x5a; 1024 * 1024];
        let start = Instant::now();
        for _ in 0..32 {
            std::hint::black_box(gcm.seal(&nonce, &aad, &plaintext));
        }
        (32.0 * 1024.0 * 1024.0) / start.elapsed().as_secs_f64()
    };
    for size_gib in [1.0f64, 4.0, 8.0] {
        let seconds = size_gib * 1024.0 * 1024.0 * 1024.0 / gcm_bytes_per_s;
        println!("  {size_gib:.0} GiB through this build's GCM: {seconds:.0} s of CPU");
    }
    // CPU time is not added latency. A 100 Mbps link delivers about 12 MB/s, so
    // while the decrypt path sustains more than that it keeps up even though it
    // is burning a core; below the link rate it becomes the bottleneck itself.
    println!(
        "  (that is CPU time, not latency: at {:.0} MB/s this path stays ahead of a \
         100 Mbps link's ~12 MB/s, so the cost is a busy core. On a gigabit link, or below \
         ~12 MB/s, it caps the download instead.)",
        gcm_bytes_per_s / (1024.0 * 1024.0)
    );
}
