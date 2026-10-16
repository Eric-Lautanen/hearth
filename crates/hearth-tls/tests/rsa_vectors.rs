//! RSA verification against vectors from independent implementations.
//!
//! Nothing in this file was produced by this crate. The modular-exponentiation
//! vectors come from CPython's arbitrary-precision `pow(s, e, n)`; the PSS
//! vectors come from OpenSSL through the `cryptography` package, which chooses
//! the salt, masks the encoding and assembles `EM` itself; and the primes for the
//! one odd-sized key — which OpenSSL's generator will not produce, because it
//! always sets the top bit — come from `sympy.randprime`. The hex below was
//! copied into this file by a script that read the generator's output, not by
//! hand, because a constant that is *nearly* right is a failure mode this
//! repository has already been bitten by once.
//!
//! `scripts/tls_rsa_vectors.py` is that generator, committed. Re-running it
//! reproduces the **modpow** block here byte for byte (the RNG is seeded) and
//! emits *new* PSS vectors, since OpenSSL seeds itself from the OS; that
//! difference is expected and is documented in the script. Use it to add a limb
//! count or a scheme, not to diff the file after a re-run.
//!
//! Three properties are checked, each of them the kind that passes for the wrong
//! reason if it is only checked loosely:
//!
//! 1. `s^e mod n` agrees with an independent implementation at 4, 8, 9, 16, 24 and
//!    32 limbs, and with exponent 3 as well as 65537.
//! 2. A real `rsa_pss_rsae_sha256` signature verifies, both for a 2048-bit key
//!    where `emLen == k` and for a **1025-bit** key where `emLen == k - 1`. The
//!    second is the case that catches an implementation using the modulus length
//!    where the standard says the encoding length.
//! 3. Everything that should be refused is refused, and for the reason named.

use hearth_tls::{RsaError, RsaPublicKey};

/// The vectors, as `(name, hex)` pairs. Kept as strings so the hex reads exactly
/// as it was generated; `bytes` is the only thing that turns them into numbers.
const VECTORS: &[(&str, &str)] = &[
    (
        "RSAVP1_256_N",
        "\
            8c39d2ee690383a8ae5b7a7da9f7e03c \
            83c9e5db8f89697fba6dd33e22266a0b",
    ),
    (
        "RSAVP1_256_S",
        "\
            5ba1bd9878db4c1e9a066965e4811b6a \
            be89d0ff00d38174afd524fb0fbbc1bb",
    ),
    (
        "RSAVP1_256_M",
        "\
            696ea68bb84a2a4e77d5457902d6a37d \
            85213eca27cffaf0f91390fa88bc0535",
    ),
    (
        "RSAVP1_512_N",
        "\
            8f74a8c358e4b89f6baf298fa2fda818 \
            6e5b33891ed995067762b5c964f7585a \
            97876a865c181ab0a230a4b0f3d71cea \
            a43916b9aa13107968eaed9e903a586d",
    ),
    (
        "RSAVP1_512_S",
        "\
            23356714c3a2453625c06752c25316a9 \
            eb41c4ff504d65af8271925f8e540a7f \
            39279a1979952ee7073c953cb490044e \
            a92fa52b3b41f8b59a9bf59280381de6",
    ),
    (
        "RSAVP1_512_M",
        "\
            502887f1cbbcfba86f3067f4636b1d37 \
            7176a27af1cf734576ab10f67953e047 \
            00bc1cd50b3766dab9423b0a8da5cf76 \
            62e85a1f497534fcac31d96a1bd72594",
    ),
    (
        "RSAVP1_576_N",
        "\
            90ef852ce214ac260dc06a71a09b9fad \
            9af9ea03990ccf81587e95517700c5c9 \
            1c4c0673a0f6cf045786b560a16efc06 \
            4e2f360ac32a33d528baa50e1f371e21 \
            dca7640d230441d5",
    ),
    (
        "RSAVP1_576_S",
        "\
            686dbd4e20bbfbcef155611bcbc30030 \
            10a03bfeb1398005aff4cd19b6f51682 \
            62c9c99910c215a0dbcf6107f7a42ef8 \
            8ca450a6101d63fd5963dbe61768cdfd \
            fae6aa9c52cebe1f",
    ),
    (
        "RSAVP1_576_M",
        "\
            831803c74bee85d22a30729cd4efc136 \
            13e36b9a9b01a10b671e42889e904f3e \
            ad4464bd47ae178673191b1f805c7c8d \
            097223b5b96464258d73601dee5b8f1e \
            09954a4f58bf1f82",
    ),
    (
        "RSAVP1_1024_N",
        "\
            e4eef00c105af476e2a4ce797d19920e \
            7352c62d068716bfe6049f0ca5fc4b20 \
            ea227953793e2c94d1d58ff1353abf5d \
            540902119bd42dfc70de6e8198e4f64c \
            d2c6e996bc33684a82dba0402016e37c \
            102a888270b451f352fe96be512c6635 \
            3f9c5bc89dcab95c4f4e02eb2f4a4a6f \
            b5c46fe31d9133cf81d82ac7ed2749ab",
    ),
    (
        "RSAVP1_1024_S",
        "\
            0687c784d919a719322ab863bf3c85db \
            dccf0e905004e4818753797da5685ff5 \
            88cb2d7ff8b9beb3676697dc674364c0 \
            f1cbdfd9ee4ddc8dbdccf2697a5f2c17 \
            159233acea65052a6b1fbd11ff6d8a54 \
            a7e365cbf512a75bb0d9251a4f4b155b \
            fd0463a4ae25d321d4271eede7bae8ac \
            afd66aa10a50bd826eb074d5ca21f5a0",
    ),
    (
        "RSAVP1_1024_M",
        "\
            723ad16626e7631ddbfa2ec1c73dfe28 \
            750f86f5b5752364a450107e9ac3f04d \
            af500008e8db4daff5b788179a223167 \
            f59c7878b846c2a87fd7376e8d766b07 \
            32e866eca2dd09b59c121c26cdf0cd36 \
            0b5bde49c2f4df4e4625d74ff77c19a8 \
            82ee3660eeae6d2789509dad2215687e \
            b2376adea372377e200d9bbc11556f0c",
    ),
    (
        "RSAVP1_1536_N",
        "\
            e79785bf4353b868c66bd4452c47d789 \
            cc2fc79f2d3180d4230c9732a73961eb \
            00d07aa8c9acfc8eebde172caa28dfcd \
            0c858323a89bab9bda98cdb2299954de \
            cf4d3174d8d03042abf8d62b1bd0affb \
            fe287778655e84dabbb559a631aff2f9 \
            8d54bf1c6c7664f7ef01c06e1a9c8a71 \
            8b793740353614a55f70f21edcbd98cd \
            e0143571b52eca885587dc1ad3910b4f \
            f868a291ecc0e727bf1c2da4236f4c9d \
            066859b99bd6495bc8e262ae10e35000 \
            e3be227071625686e998fb54ec37f3b3",
    ),
    (
        "RSAVP1_1536_S",
        "\
            1e61dba46cf697627efe4799f6f9a967 \
            fa9c6d9d81b3d1565d8b7c6d19f7ebd2 \
            b1ce35d475afebaf42e2cf4d7e523c62 \
            e1c40d61588184a89c2911968a688d58 \
            e418c629f2e895a36319a54a05cafccb \
            94700b43255418c0bf70dbac6a9ee68d \
            211f9202eb0e401eb780cf5ea347e58b \
            63bf159d406e1e3387388a479d7e61e1 \
            9db5aab6786133e5cff017134f7ecd4d \
            3f01124f8b68e2f154ec75ea91895d47 \
            e71a2ccaa30ac469714399935fc8b3ef \
            d3a22b618f448290e87a81ad5de73d9d",
    ),
    (
        "RSAVP1_1536_M",
        "\
            e4a96413da6590142d6ae9594ac2c659 \
            7e9219ed06452291b8b1e6a44df1eb81 \
            a53612136626fc5f158c855897860e32 \
            3e96120830ceed9775a941564f80599f \
            619f5f6a72644bb98ac11c4841118d2f \
            7653204eba0ec309fad042d2b45c2e7a \
            159e937fe2bfb646a786b7a9ae2c18c1 \
            e47ef7e421f8d9f12921ef97e1c88390 \
            22770d5f899793f409b552ab84c1dbcb \
            3b32696799ab22becfcf08a69f51d11e \
            9daba9adae17ae42ae0c5f342b7de7a7 \
            cdc9c1182e8ba7c5ac1d943b8a861745",
    ),
    (
        "RSAVP1_2048_N",
        "\
            8cf14d1664eb86f119d299d10a768c38 \
            6540bae49aad608b67647bda93cc5dfc \
            fe31ad0588ac83a81787e2785db8f08f \
            45d5c49e01ddc4e0ef3a02fe6625431b \
            643bcab65baaa022bdce3c9051e9873e \
            f7ccef8459280b603010972c41050de4 \
            690ddba0ba0f1a5b19bc2c35cc94fee8 \
            3b1627db1fe29d4577b6e651cc70f63e \
            830fd156a014af61b1e85ce4e3b6c3b1 \
            5b8a4ed449657a6bc518221e2c8d538c \
            b4463d20a71e438af49c556a077ed088 \
            12efd7bc091192c0a739d5e3d75d5583 \
            cdb27dec9fc6858e007cfe56ee31f210 \
            1bc9db6163ba6c0e29233d81ef8899ed \
            d777f59d0982ddb218c65c15e7f824ea \
            7d9774bf4cbb9fea9d066ccb970b3f5d",
    ),
    (
        "RSAVP1_2048_S",
        "\
            2d4a9570ed64039b10d412f8050f8e1e \
            be8a866066aeb4f53ed6ac14a3643289 \
            8ffdfbe5b8b8e5609941d14486abe5e6 \
            2505a62584bdfac3c29213d6879d3172 \
            57667cf8cc18f39583f7349536360ec0 \
            a239ba412eedcf73c48c7cf8440b50bd \
            439b97526c11a71660e00c2f9c6819a5 \
            b6340adf2cbd159d10b264ae472c1887 \
            a775d80fef971ffe23ab25114f8cb806 \
            c1ddfab8fac726dc27373e42edb70494 \
            7a97e3f646af4b2928c18ccbffe0072c \
            6b3d3e8dd8f356d6a959ae036a7924d0 \
            d0ce85d1c605206c1014d73a5c6aebdf \
            6e37a630edeab94b0692a9e06284f64c \
            0d4ab6fbc6e984d29a16bec1919f1219 \
            3340c3227d996e72db289e2e6d3ee4ab",
    ),
    (
        "RSAVP1_2048_M",
        "\
            789a9a4abbf62b66178a88181fd750bb \
            cbdf5484ecf8fc73c049153ec9fbe24b \
            48c31dac87f38a7f3aa0bf84330ec5dd \
            cafb45e7da7b8205d6887cdaaf0504d8 \
            53384e25625c7dbaeca617ce0d33a05a \
            be9bcb1fae7ea5ecaab236f2809c4759 \
            a21919d57b88af419e0047d68d595b9c \
            4e1c58b746e90b9667dda6130326f5b8 \
            d5a623e016c2a197f2521193d3a00b99 \
            aadade17390173272a6b4a152dc22a6c \
            ab8c95f5ae714185c757306caba7bab9 \
            c533a725912d7498438be2c55d03b382 \
            c6ffbd8798b16467fae7a0dea92c4617 \
            8145328310241d2441a94d6056e030a1 \
            610a8ed42f71ad6d4fa34338fbc7c095 \
            c0abec8efa7809f58b395861714bb722",
    ),
    (
        "PSS_2048_N",
        "\
            bd86d4e9954410db4d8371838db8f19f \
            63a95c8f4eeabb3b868fdf019c915e53 \
            bef90823d231bc4b7ce796ded7cf1f32 \
            734ace1387371949f73eea0403f9bba3 \
            2981c84d079549d1b3c68bd6b683ce47 \
            647a0d1e55f6fa9ebde0a4a3b0e88366 \
            1ea4f81a067b3fb698f586a2f7550991 \
            c80341f92312ad40e4072057f7cac16f \
            82085c665f60e251998dbdc74736d339 \
            60c4d02358856fedf939854833988592 \
            8f9ff0beed360291eb1a071cf13a48bc \
            f5d995e7b87c06c6fea52e95857b046c \
            993ae5a12b83a22095e6025f6e3a25f3 \
            146ae1186f20a39b56a352988f64c0c2 \
            9b773eced8e5023c539817e9d69f33e7 \
            51586f136e2acb6c0c05abda5cab4d5f",
    ),
    (
        "PSS_2048_SIG",
        "\
            b11ab335ff58e30592ac3093d5977f2a \
            e3c1a93d1da61e5b788dce2fcd7f3eee \
            b5abcc24f6e39e5b76dff06352c84856 \
            98073fba559fa34376cdf8b3bd11991e \
            66272b99128ae13b5025140ea389dede \
            84e79408693e9f422e0979ca042155a3 \
            07b75b9a16245cfff46813e260e42d03 \
            d2b9dd4539dea3dd954e3483cb132838 \
            74ceb719cff3e959eddba9126cdeaa10 \
            458299ddd726e774705582cf8ba7986f \
            ba3bd2051774cc76acb69eec9203fe28 \
            426eebc9c7ce71c9a41daa681922ba88 \
            ea482160ecf3d9d33b745fe6d0381f36 \
            78a250f81b68f91ef77b136cf18f0580 \
            f1f7c5974b5a253e4d179ec5aaed0f30 \
            4b35f3df98f5deb19ea728d9b7b7128c",
    ),
    (
        "PSS_1025_N",
        "\
            01adb4f93dd9efe8b0a41e0599a90ab0 \
            5a6e640f41990fa68e7f9ea7cc123b4b \
            168633fa41cecb818a59914269b4e3fc \
            03d7925d68a32acd4b94afcd73baf127 \
            d9317dd64d6eac900607344c6168896c \
            bf20f93d55930f6d468a3d90c0ac69cf \
            2ccc8e8f9ed9cd7e963ee257b192e07f \
            2a6095f85a6d8d67d8f0439ff52cf85a \
            e1",
    ),
    (
        "PSS_1025_SIG",
        "\
            01439589d0a92786e4177771fc2bfc00 \
            6869a394d3b42d18b547eded68e21619 \
            c623927d93d3287718be9dd8c694f735 \
            5589126a0fdcda86ac456b3840da5399 \
            0a16c2309cdf73a5eb4692ca9e437eeb \
            db4c0eee542bb1e0d32724d35a139626 \
            3f2b92ebc657b99ed65ccff43687cc12 \
            56bf9c6224c33b9dd20ff0f42c727cbe \
            b5",
    ),
    (
        "PSS_2048_SHORT_SALT_N",
        "\
            bc2a38915d277832ae3f180578cbb637 \
            26a6eb806df93403d86be7da092f4870 \
            b6492708584d99502d5375dc3103f0ef \
            ebc03ebc13732831076df53da401e8dd \
            2ceddbafaee581470885f7a03905a0a4 \
            7aa67d1721817eaf219f500bf2129520 \
            783c18e53d5e530c9ba274b97e6947fc \
            1077b662663853cba8d0c7cf50404df0 \
            75200357ac8ff6eb2644f0f7f1487e23 \
            e80b4ddff104f9916e99a34a7abea2b0 \
            6cac96cb6a487f531e3c353a2b1c5ee1 \
            690e5e62ec96ca8797ee4dd60d81616f \
            7759d77d986d0d342b12d4e99eb582b6 \
            c938cbd93ba839dcd3040542f452ea52 \
            b56f53bb8e55acf3cdf0ae651164a05f \
            3ce696d65ab2121957043ac259a8ceb9",
    ),
    (
        "PSS_2048_SHORT_SALT_SIG",
        "\
            2d9f2082c09c104b8a4b5e87da3a4b5a \
            fa3f9edf9e33916bb7104f0fc15247e9 \
            cc6a78ca66facdf86c418cce713bedd7 \
            ddb2349215629b28ee1349baa3d847c8 \
            a9ca9c6e2657198a93da072403d38bff \
            455d8a0df8d38447a2d348409f11d136 \
            9f18ce1d57abf284353196130d968186 \
            9b64854f35a64ab22dc00538bd79ec57 \
            c4af0a40b5be39cb922b2949b4fe70b1 \
            3815f44c35dc929d2b8de39145a37b4b \
            69896a58243e37d370d1fb37e0b41694 \
            8efc71b0ed69b889f7441e3171742a53 \
            8d0d3f247f2b84c1c0efce9e2b9388b8 \
            5fc811bdfdf65297e521896e69cbb1e5 \
            05d40c0dcb18b983247272269f4fb780 \
            1a2f7e9949ee259c9837b6719100340a",
    ),
];

fn bytes(text: &str) -> Vec<u8> {
    let cleaned: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(cleaned.len().is_multiple_of(2), "odd hex length");
    (0..cleaned.len() / 2)
        .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

fn get(name: &str) -> String {
    VECTORS
        .iter()
        .find(|(key, _)| *key == name)
        .unwrap_or_else(|| panic!("no vector named {name}"))
        .1
        .to_string()
}

fn key(modulus: &str, exponent: u64) -> RsaPublicKey {
    RsaPublicKey::from_be_bytes(&bytes(modulus), &exponent.to_be_bytes()).expect("a usable key")
}

/// Left-pads a big-endian value to `width` bytes, which is what `I2OSP(m, k)`
/// does to the shorter `emLen`-byte encoding.
fn padded(value: &[u8], width: usize) -> Vec<u8> {
    let mut out = vec![0u8; width - value.len()];
    out.extend_from_slice(value);
    out
}

/// `value + 1`, as a big-endian byte string of the same width.
fn plus_one(value: &[u8]) -> Vec<u8> {
    let mut out = value.to_vec();
    for byte in out.iter_mut().rev() {
        let (next, carried) = byte.overflowing_add(1);
        *byte = next;
        if !carried {
            return out;
        }
    }
    panic!("the value was all 0xff, so +1 does not fit");
}

/// One modular-exponentiation vector: the modulus, the signature, the exponent
/// and the expected `s^e mod n`.
fn modpow(name: &str, exponent: u64, bits: usize) {
    let key = key(&get(&format!("RSAVP1_{name}_N")), exponent);
    assert_eq!(key.key_bits(), bits, "{name}: the modulus's own size");
    assert_eq!(key.exponent(), exponent);
    let result = key
        .rsa_vp1(&bytes(&get(&format!("RSAVP1_{name}_S"))))
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    // RSAVP1 renders m in k bytes; the generator wrote it in emLen, which is one
    // byte shorter whenever the modulus is 1 mod 8 bits wide.
    let expected = padded(
        &bytes(&get(&format!("RSAVP1_{name}_M"))),
        key.modulus_bytes(),
    );
    assert_eq!(result, expected, "{name}: s^e mod n");
}

#[test]
fn the_public_key_operation_agrees_with_an_independent_implementation() {
    modpow("256", 65537, 256);
    modpow("512", 65537, 512);
    modpow("576", 65537, 576);
    modpow("1024", 65537, 1024);
    modpow("1536", 3, 1536);
    modpow("2048", 65537, 2048);
}

#[test]
fn a_2048_bit_pss_signature_verifies() {
    let key = key(&get("PSS_2048_N"), 65537);
    assert_eq!(key.key_bits(), 2048);
    assert_eq!(key.modulus_bytes(), 256, "k");
    let signature = bytes(&get("PSS_2048_SIG"));
    assert_eq!(signature.len(), 256);
    assert_eq!(
        key.verify_pss_sha256(
            b"hearth tls 1.3 certificate verify, 2048-bit key",
            &signature
        ),
        Ok(())
    );
}

#[test]
fn a_1025_bit_pss_signature_verifies_even_though_its_encoding_is_a_byte_shorter() {
    let key = key(&get("PSS_1025_N"), 65537);
    assert_eq!(key.key_bits(), 1025);
    assert_eq!(key.modulus_bytes(), 129, "k, in 17 limbs");
    let signature = bytes(&get("PSS_1025_SIG"));
    assert_eq!(signature.len(), 129);
    // 1025 bits is 1 mod 8, so emLen is 128 while k is 129. An implementation
    // that used k where emLen belongs fails here and virtually nowhere else -- a
    // real 2048-bit key would not notice.
    assert_eq!(
        key.verify_pss_sha256(
            b"hearth tls 1.3 certificate verify, 1025-bit key",
            &signature
        ),
        Ok(())
    );
    // The same key must also refuse a signature one byte short, since k is 129.
    let mut short = signature.clone();
    short.remove(0);
    assert_eq!(
        key.verify_pss_sha256(b"hearth tls 1.3 certificate verify, 1025-bit key", &short),
        Err(RsaError::SignatureLength {
            expected: 129,
            found: 128
        })
    );
}

#[test]
fn a_salt_that_is_not_the_digest_length_is_refused() {
    // RFC 8446 §4.2.3: for the PSS schemes, "the length of the Salt MUST be equal
    // to the length of the output of the digest algorithm". OpenSSL will happily
    // sign with a 20-byte salt, so this is a real signature by a real key that
    // TLS has to refuse.
    let key = key(&get("PSS_2048_SHORT_SALT_N"), 65537);
    let signature = bytes(&get("PSS_2048_SHORT_SALT_SIG"));
    assert_eq!(signature.len(), 256);
    assert_eq!(
        key.verify_pss_sha256(b"hearth tls 1.3 certificate verify, salt 20", &signature),
        Err(RsaError::Inconsistent(
            "the separator byte is not 0x01 where a 32-byte salt requires it"
        ))
    );
}

#[test]
fn every_single_bit_of_the_signature_matters() {
    let key = key(&get("PSS_2048_N"), 65537);
    let signature = bytes(&get("PSS_2048_SIG"));
    let message = b"hearth tls 1.3 certificate verify, 2048-bit key";
    assert_eq!(key.verify_pss_sha256(message, &signature), Ok(()));

    // A signature is a permutation applied to the encoding, so changing any of
    // its bits changes `s^e mod n`, and the decoder must notice. Checking a
    // spread of bytes rather than all 256 keeps the test quick; the point is that
    // no part of the input is read past.
    for byte in (0..signature.len()).step_by(7) {
        for bit in 0..8 {
            let mut broken = signature.clone();
            broken[byte] ^= 1 << bit;
            assert!(
                key.verify_pss_sha256(message, &broken).is_err(),
                "byte {byte} bit {bit} was accepted"
            );
        }
    }
}

#[test]
fn a_different_message_is_refused() {
    let key = key(&get("PSS_2048_N"), 65537);
    let signature = bytes(&get("PSS_2048_SIG"));
    let message = b"hearth tls 1.3 certificate verify, 2048-bit key";
    let mut other = message.to_vec();
    other[0] ^= 1;
    assert_eq!(
        key.verify_pss_sha256(&other, &signature),
        Err(RsaError::Inconsistent(
            "the message hash does not match the one the signature encodes"
        ))
    );
    assert!(key.verify_pss_sha256(b"", &signature).is_err());
    // And the trailing context of the real signed data cannot be dropped either.
    assert!(key
        .verify_pss_sha256(&message[..message.len() - 1], &signature)
        .is_err());
}

#[test]
fn a_signature_representative_at_or_above_the_modulus_is_refused() {
    let key = key(&get("PSS_2048_N"), 65537);
    let message = b"hearth tls 1.3 certificate verify, 2048-bit key";
    let n = bytes(&get("PSS_2048_N"));

    // RSAVP1 is only defined for 0 <= s < n: s == n and s == n + 1 are not the
    // output of any signing operation, and accepting them would mean accepting a
    // value that no signature could produce.
    assert_eq!(key.rsa_vp1(&n), Err(RsaError::SignatureOutOfRange));
    assert_eq!(
        key.rsa_vp1(&plus_one(&n)),
        Err(RsaError::SignatureOutOfRange)
    );
    assert_eq!(
        key.verify_pss_sha256(message, &n),
        Err(RsaError::SignatureOutOfRange)
    );

    // All zeros is in range, and is refused because its encoding is not PSS.
    assert!(matches!(
        key.verify_pss_sha256(message, &[0u8; 256]),
        Err(RsaError::Inconsistent(_))
    ));
}

#[test]
fn a_key_too_small_for_the_encoding_says_so_before_doing_arithmetic() {
    // 512 bits is emLen 64 while hLen + sLen + 2 is 66, so PSS with a 32-byte salt
    // cannot fit at all. The error names the sizes rather than reporting a
    // decoding failure, because nothing was decoded.
    let key = key(&get("RSAVP1_512_N"), 65537);
    let error = key
        .verify_pss_sha256(b"message", &vec![1u8; key.modulus_bytes()])
        .expect_err("512 bits cannot carry PSS/SHA-256");
    assert_eq!(
        error,
        RsaError::KeyTooSmall {
            bits: 512,
            needed_bits: 522
        }
    );
    assert!(error.to_string().contains("522"), "{error}");
}
