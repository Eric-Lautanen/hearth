//! AES-128 (FIPS-197) — the block cipher underneath AES-128-GCM.
//!
//! Only the **forward** direction is implemented. GCM is a counter mode plus a
//! universal hash, so it never needs the inverse cipher; adding
//! `decrypt_block` would be dead code with its own InvSubBytes/InvMixColumns
//! tables to get wrong.
//!
//! # Why the S-box is computed instead of transcribed
//!
//! A hand-typed 256-byte S-box is 256 chances to make a one-character mistake
//! that no amount of reading the diff reliably catches. Instead the table is
//! derived here at compile time from its definition in FIPS-197 §5.1.1: take
//! the multiplicative inverse in GF(2^8) (via log/exp tables built from the
//! generator 3) and apply the affine transformation. `tests` pins the result
//! against known S-box entries, so a broken derivation cannot pass silently.
//!
//! # Timing
//!
//! The software S-box is a 256-byte table, i.e. four cache lines, so a local
//! attacker who can observe this process's cache at that granularity can in
//! principle recover key material (Bernstein 2005; Osvik–Shamir–Tromer 2006).
//! Every other primitive in this crate is written without data-dependent
//! branches or lookups (constant-time GHASH, constant-time tag compare,
//! constant-time X25519) and this one now has a hardware path too: on x86_64
//! with AES-NI, [`Aes128::encrypt_block`] dispatches to `crate::aesni`, which
//! has no table at all. [`Aes128::uses_hardware`] reports which one ran, so the
//! caveat above applies only where it is true.
//!
//! Apple silicon's crypto extensions would remove the caveat on that target as
//! well; until then macOS runs the software path here.

/// Round constants (FIPS-197 §5.2), one per round key addition after the first.
const RCON: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];

/// Multiplication by `x` in GF(2^8) modulo the AES polynomial
/// `x^8 + x^4 + x^3 + x + 1` (0x11b). Branchless so it stays constant-time.
#[inline]
const fn xtime(b: u8) -> u8 {
    let reduce = ((b >> 7) & 1).wrapping_neg() & 0x1b;
    (b << 1) ^ reduce
}

/// Carry-less-free schoolbook multiply in GF(2^8). Only used at compile time.
const fn gfmul(mut a: u8, mut b: u8) -> u8 {
    let mut acc = 0u8;
    let mut i = 0;
    while i < 8 {
        let mask = (b & 1).wrapping_neg();
        acc ^= a & mask;
        a = xtime(a);
        b >>= 1;
        i += 1;
    }
    acc
}

/// Log/exp tables over the generator 3, plus a wrapped `exp[255] = 1`.
///
/// `exp[i] = 3^i mod 0x11b` and `log` is its inverse, which makes the inverse of
/// any `x` the single lookup `exp[255 - log[x]]`.
const fn build_tables() -> ([u8; 256], [u8; 256]) {
    let mut exp = [0u8; 256];
    let mut log = [0u8; 256];
    let mut x = 1u8;
    let mut i = 0;
    while i < 255 {
        exp[i] = x;
        log[x as usize] = i as u8;
        x = gfmul(x, 3);
        i += 1;
    }
    // 3^255 == 1, so index 255 is the same element as index 0; having it
    // available is what lets `255 - log[1] == 255` be a valid index.
    exp[255] = 1;
    (exp, log)
}

/// The AES S-box (FIPS-197 §5.1.1), derived rather than transcribed.
const fn build_sbox() -> [u8; 256] {
    let (exp, log) = build_tables();
    let mut sbox = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        // 0 has no multiplicative inverse and maps to 0 by definition.
        let inv = if i == 0 {
            0
        } else {
            exp[255 - log[i] as usize]
        };
        sbox[i] = inv
            ^ inv.rotate_left(1)
            ^ inv.rotate_left(2)
            ^ inv.rotate_left(3)
            ^ inv.rotate_left(4)
            ^ 0x63;
        i += 1;
    }
    sbox
}

/// Substitution box, built at compile time.
pub(crate) const SBOX: [u8; 256] = build_sbox();

/// AES-128 with a precomputed key schedule.
#[derive(Clone)]
pub struct Aes128 {
    /// 11 round keys of 16 bytes each (FIPS-197 §5.2).
    round_keys: [u8; 176],
    /// Set once from the runtime probe in [`Aes128::new`], so the per-block
    /// decision is a predictable branch instead of a CPUID query, and so this
    /// field is the single justification for the `unsafe` call below.
    aes_ni: bool,
}

impl Aes128 {
    /// Expands a 128-bit key into the 11 round keys.
    pub fn new(key: &[u8; 16]) -> Self {
        let mut round_keys = [0u8; 176];
        round_keys[..16].copy_from_slice(key);

        let mut rcon_index = 0;
        let mut i = 16;
        while i < 176 {
            let mut t = [
                round_keys[i - 4],
                round_keys[i - 3],
                round_keys[i - 2],
                round_keys[i - 1],
            ];

            // Every fourth word gets the RotWord/SubWord/Rcon transform.
            if i % 16 == 0 {
                let first = t[0];
                t[0] = SBOX[t[1] as usize];
                t[1] = SBOX[t[2] as usize];
                t[2] = SBOX[t[3] as usize];
                t[3] = SBOX[first as usize];
                t[0] ^= RCON[rcon_index];
                rcon_index += 1;
            }

            let mut j = 0;
            while j < 4 {
                round_keys[i + j] = round_keys[i - 16 + j] ^ t[j];
                j += 1;
            }
            i += 4;
        }

        Self {
            round_keys,
            aes_ni: aes_ni_available(),
        }
    }

    /// Whether block encryption is using AES-NI rather than the software
    /// S-box path. Always `false` off x86_64.
    pub fn uses_hardware(&self) -> bool {
        self.aes_ni
    }

    /// The 176-byte key schedule, exposed for the FIPS-197 Appendix A vector.
    pub fn round_keys(&self) -> &[u8; 176] {
        &self.round_keys
    }

    /// Encrypts one 16-byte block in place, via AES-NI where available.
    pub fn encrypt_block(&self, block: &mut [u8; 16]) {
        if self.aes_ni {
            // SAFETY: `aes_ni` is only ever true when the runtime probe in
            // `new` reported the `aes` feature, which is exactly this call's
            // requirement.
            unsafe { crate::aesni::encrypt_block(&self.round_keys, block) };
            return;
        }
        self.encrypt_block_software(block);
    }

    /// The portable path: same round keys, same rounds, S-box in software.
    /// Kept as the reference the accelerated path is tested against.
    pub fn encrypt_block_software(&self, block: &mut [u8; 16]) {
        add_round_key(block, &self.round_keys[..16]);

        let mut round = 1;
        while round < 10 {
            sub_bytes(block);
            shift_rows(block);
            mix_columns(block);
            add_round_key(block, &self.round_keys[round * 16..round * 16 + 16]);
            round += 1;
        }

        // Final round has no MixColumns (FIPS-197 §5.1).
        sub_bytes(block);
        shift_rows(block);
        add_round_key(block, &self.round_keys[160..176]);
    }
}

/// Whether AES-NI is usable, or `false` on other architectures.
#[inline]
fn aes_ni_available() -> bool {
    if crate::hardware_disabled() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        crate::aesni::available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline]
fn add_round_key(state: &mut [u8; 16], rk: &[u8]) {
    for (s, k) in state.iter_mut().zip(rk) {
        *s ^= *k;
    }
}

#[inline]
fn sub_bytes(state: &mut [u8; 16]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

/// Row `r` is rotated left by `r`, where the state is column-major:
/// `state[r + 4*c]` (FIPS-197 §3.4).
#[inline]
fn shift_rows(state: &mut [u8; 16]) {
    let src = *state;
    for r in 0..4 {
        for c in 0..4 {
            state[r + 4 * c] = src[r + 4 * ((c + r) & 3)];
        }
    }
}

/// One MixColumns transform per column (FIPS-197 §5.1.3), written in the
/// `a ^ all ^ xtime(a ^ b)` form so there is no branch per byte.
#[inline]
fn mix_columns(state: &mut [u8; 16]) {
    for c in 0..4 {
        let i = 4 * c;
        let (a0, a1, a2, a3) = (state[i], state[i + 1], state[i + 2], state[i + 3]);
        let all = a0 ^ a1 ^ a2 ^ a3;
        state[i] = a0 ^ all ^ xtime(a0 ^ a1);
        state[i + 1] = a1 ^ all ^ xtime(a1 ^ a2);
        state[i + 2] = a2 ^ all ^ xtime(a2 ^ a3);
        state[i + 3] = a3 ^ all ^ xtime(a3 ^ a0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            cleaned.len().is_multiple_of(2),
            "odd hex length: {}",
            cleaned.len()
        );
        (0..cleaned.len() / 2)
            .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
            .collect()
    }

    /// If the derived S-box were wrong, every GCM vector below would fail in a
    /// way that is tedious to localise. These entries are the known values from
    /// FIPS-197 Figure 7, including the two fixed points of the affine map.
    #[test]
    fn derived_sbox_matches_known_entries() {
        assert_eq!(SBOX[0x00], 0x63);
        assert_eq!(SBOX[0x01], 0x7c);
        assert_eq!(SBOX[0x10], 0xca);
        assert_eq!(SBOX[0x53], 0xed);
        assert_eq!(SBOX[0x7f], 0xd2);
        assert_eq!(SBOX[0xff], 0x16);
        // The S-box is a permutation: every byte appears exactly once.
        let mut seen = [0usize; 256];
        for b in SBOX {
            seen[b as usize] += 1;
        }
        assert!(seen.iter().all(|&n| n == 1), "S-box is not a permutation");
    }

    /// The inverse relation the derivation depends on: `sbox[x]` is the affine
    /// transform of `x^-1`, so `inv(x) * x == 1` must hold for every non-zero x.
    #[test]
    fn sbox_inputs_are_field_inverses() {
        let (exp, log) = build_tables();
        for x in 1u16..=255 {
            let x = x as u8;
            let inv = exp[255 - log[x as usize] as usize];
            assert_eq!(gfmul(x, inv), 1, "x = {}", x);
        }
        assert_eq!(gfmul(0, 0), 0);
    }

    /// FIPS-197 Appendix A.1: the first two expanded words after the originals.
    #[test]
    fn key_expansion_matches_fips_197_a1() {
        let key: [u8; 16] = h("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
        let aes = Aes128::new(&key);
        let rk = aes.round_keys();
        assert_eq!(&rk[..16], &key[..]);
        assert_eq!(rk[16..32], h("d6aa74fdd2af72fadaa678f1d6ab76fe")[..]);
        // Words 8..11 are deliberately not asserted from memory: the Appendix C.1
        // cipher vector below validates the whole schedule far more strongly,
        // and a mis-transcribed expectation here would only obscure a real bug.
    }

    /// FIPS-197 Appendix C.1, the AES-128 known-answer test.
    #[test]
    fn encrypts_fips_197_c1_vector() {
        let key: [u8; 16] = h("000102030405060708090a0b0c0d0e0f").try_into().unwrap();
        let mut block: [u8; 16] = h("00112233445566778899aabbccddeeff").try_into().unwrap();
        Aes128::new(&key).encrypt_block(&mut block);
        assert_eq!(block[..], h("69c4e0d86a7b0430d8cdb78070b4c55a")[..]);
    }

    /// The all-zero key/block case, where a mis-set Rcon often shows up first.
    #[test]
    fn encrypts_the_zero_block() {
        let aes = Aes128::new(&[0u8; 16]);
        let mut block = [0u8; 16];
        aes.encrypt_block(&mut block);
        assert_eq!(block[..], h("66e94bd4ef8a2c3b884cfa59ca342b2e")[..]);
    }

    /// Every distinct key must give a distinct permutation of the block.
    #[test]
    fn distinct_keys_differ() {
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        Aes128::new(&[0u8; 16]).encrypt_block(&mut a);
        let mut key = [0u8; 16];
        key[15] = 1;
        Aes128::new(&key).encrypt_block(&mut b);
        assert_ne!(a, b);
    }
}
