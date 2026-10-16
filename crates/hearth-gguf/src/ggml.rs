/// The GGML/GGUF tensor type ids this build understands.
///
/// Variants are spelled the way rustc's `non_camel_case_types` wants them, which is *not* the
/// way the GGUF spec spells them: the spec writes `Q2_K`, `IQ4_NL`, `Q1_0_G128`, and the ids
/// whose parts are all digits around the underscore (`Q1_0`, `Q2_0`, `Q4_0_4_4`) keep their spec
/// spelling because nothing cased sits next to an underscore. The spec spelling is what
/// [`GgmlDType::name`] returns, so logs, the UI and error messages still read `Q2_K`; only the
/// Rust identifier is camel-cased.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlDType {
    F32,      // 0
    F16,      // 1
    Q4_0,     // 2
    Q4_1,     // 3
    Q5_0,     // 6
    Q5_1,     // 7
    Q8_0,     // 8
    Q8_1,     // 9
    Q2K,      // 10
    Q3K,      // 11
    Q4K,      // 12
    Q5K,      // 13
    Q6K,      // 14
    Q8K,      // 15
    IQ2XXS,   // 16
    IQ2XS,    // 17
    IQ3XXS,   // 18
    IQ1S,     // 19
    IQ4NL,    // 20
    IQ3S,     // 21
    IQ2S,     // 22
    IQ4XS,    // 23
    I8,       // 24
    I16,      // 25
    I32,      // 26
    I64,      // 27
    F64,      // 28
    IQ1M,     // 29
    BF16,     // 30
    Q4_0_4_4, // 31
    Q4_0_4_8, // 32
    Q4_0_8_8, // 33
    TQ1_0,    // 34
    TQ2_0,    // 35
    IQ4NL4x4, // 36
    IQ4NL4x8, // 37
    IQ4NL8x8, // 38
    // 39-41: reserved (some tools used 39 for an early Q1_0_G128)
    Q1_0,    // 41: PRISM Q1_0 1-bit binary, 128-element blocks, 18 bytes
    Q2_0,    // 42: PRISM Q2_0 2-bit ternary {-1,0,+1}, 128-element blocks, 34 bytes
    Q10G128, // 43: PRISM Q1_0_G128 1-bit binary, 128-element blocks, 18 bytes
    PQ2_0,   // 142: Prism 2-bit ternary g128, 34-byte blocks, Hadamard-rotated basis
    PTQ1_0,  // 143: Prism dense-trit ternary g128, 28-byte blocks, Hadamard-rotated basis
}

impl GgmlDType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::IQ2XXS,
            17 => Self::IQ2XS,
            18 => Self::IQ3XXS,
            19 => Self::IQ1S,
            20 => Self::IQ4NL,
            21 => Self::IQ3S,
            22 => Self::IQ2S,
            23 => Self::IQ4XS,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::IQ1M,
            30 => Self::BF16,
            31 => Self::Q4_0_4_4,
            32 => Self::Q4_0_4_8,
            33 => Self::Q4_0_8_8,
            34 => Self::TQ1_0,
            35 => Self::TQ2_0,
            36 => Self::IQ4NL4x4,
            37 => Self::IQ4NL4x8,
            38 => Self::IQ4NL8x8,
            39 | 43 => Self::Q10G128, // 39 = legacy, 43 = PRISM standard
            41 => Self::Q1_0,         // PRISM Q1_0 (1-bit binary)
            42 => Self::Q2_0,         // PRISM Q2_0 (2-bit ternary {-1,0,+1})
            142 => Self::PQ2_0,       // Prism PQ2_0 (2-bit ternary g128, rotated basis)
            143 => Self::PTQ1_0,      // Prism PTQ1_0 (dense trits g128, rotated basis)
            _ => return None,
        })
    }

    pub fn block_size(&self) -> usize {
        match self {
            Self::F32
            | Self::F16
            | Self::BF16
            | Self::F64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
            Self::IQ2XXS | Self::IQ2XS | Self::IQ2S => 256,
            Self::IQ3XXS | Self::IQ3S => 256,
            Self::IQ1S | Self::IQ1M => 256,
            Self::IQ4XS => 256,
            Self::IQ4NL => 32,
            Self::IQ4NL4x4 | Self::IQ4NL4x8 | Self::IQ4NL8x8 => 32,
            Self::Q4_0_4_4 | Self::Q4_0_4_8 | Self::Q4_0_8_8 => 32,
            Self::Q2_0 => 128,
            Self::PQ2_0 => 128,
            Self::PTQ1_0 => 128,
            Self::TQ1_0 | Self::TQ2_0 => 32,
            Self::Q1_0 | Self::Q10G128 => 128,
        }
    }

    pub fn block_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
            Self::Q8K => 272,
            Self::IQ2XXS => 66,
            Self::IQ2XS => 74,
            Self::IQ2S => 82,
            Self::IQ3XXS => 98,
            Self::IQ3S => 110,
            Self::IQ1S => 50,
            Self::IQ1M => 54,
            Self::IQ4NL => 84,
            Self::IQ4XS => 290,
            Self::IQ4NL4x4 | Self::IQ4NL4x8 | Self::IQ4NL8x8 => 84,
            Self::Q4_0_4_4 | Self::Q4_0_4_8 | Self::Q4_0_8_8 => 18,
            Self::TQ1_0 => 34,
            Self::TQ2_0 => 34,
            Self::Q2_0 => 34,
            Self::PQ2_0 => 34,
            Self::PTQ1_0 => 28,
            Self::Q1_0 | Self::Q10G128 => 18,
        }
    }

    pub fn byte_size(&self, n: usize) -> usize {
        let blocks = n.div_ceil(self.block_size());
        blocks * self.block_bytes()
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::F64 => "F64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::IQ2XXS => "IQ2_XXS",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ3XXS => "IQ3_XXS",
            Self::IQ1S => "IQ1_S",
            Self::IQ4NL => "IQ4_NL",
            Self::IQ3S => "IQ3_S",
            Self::IQ2S => "IQ2_S",
            Self::IQ4XS => "IQ4_XS",
            Self::IQ1M => "IQ1_M",
            Self::Q4_0_4_4 => "Q4_0_4_4",
            Self::Q4_0_4_8 => "Q4_0_4_8",
            Self::Q4_0_8_8 => "Q4_0_8_8",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::IQ4NL4x4 => "IQ4_NL_4x4",
            Self::IQ4NL4x8 => "IQ4_NL_4x8",
            Self::IQ4NL8x8 => "IQ4_NL_8x8",
            Self::Q1_0 => "Q1_0",
            Self::Q2_0 => "Q2_0",
            Self::Q10G128 => "Q1_0_G128",
            Self::PQ2_0 => "PQ2_0",
            Self::PTQ1_0 => "PTQ1_0",
        }
    }

    pub fn is_quantized(&self) -> bool {
        !matches!(
            self,
            Self::F32
                | Self::F16
                | Self::BF16
                | Self::F64
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
        )
    }

    pub fn is_q1(&self) -> bool {
        matches!(self, Self::Q1_0 | Self::Q10G128 | Self::Q2_0)
    }
}
