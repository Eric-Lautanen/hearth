#include <stdint.h>

typedef uint16_t ggml_half;

static inline float GGML_CPU_FP16_TO_FP32(ggml_half x) {
    const uint32_t e = (x & 0x7C00) >> 10;
    const uint32_t m = (x & 0x03FF) << 13;
    const uint32_t v = ((x & 0x8000u) ? 0x80000000u : 0u)
                     | ((e == 0) ? (m >> (e ? 0 : 14)) : ((e + 112) << 23 | m))
                     | (e == 0x1F ? 255u << 23 : 0u);
    union { uint32_t u; float f; } u;
    u.u = v;
    return u.f;
}

#define QK1_0 128
typedef struct {
    ggml_half d;
    uint8_t qs[QK1_0 / 8];
} block_q1_0;

#define QK8_0 32
typedef struct {
    ggml_half d;
    int8_t qs[QK8_0];
} block_q8_0;

float dot_q1_0_q8_0_msvc(const void *vx, const void *vy, int n) {
    const int qk = QK1_0;
    const int nb = n / qk;
    const block_q1_0 *x = (const block_q1_0 *)vx;
    const block_q8_0 *y = (const block_q8_0 *)vy;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d0 = GGML_CPU_FP16_TO_FP32(x[i].d);
        float sumi = 0.0f;
        for (int k = 0; k < 4; k++) {
            const block_q8_0 *yb = &y[i * 4 + k];
            const float d1 = GGML_CPU_FP16_TO_FP32(yb->d);
            int sumi_block = 0;
            const uint8_t *bits = &x[i].qs[k * 4];
            const int8_t *qy = yb->qs;
            for (int b = 0; b < 4; ++b, qy += 8) {
                const unsigned mask = bits[b];
                sumi_block += ((mask & 0x01) ? qy[0] : -qy[0])
                           +  ((mask & 0x02) ? qy[1] : -qy[1])
                           +  ((mask & 0x04) ? qy[2] : -qy[2])
                           +  ((mask & 0x08) ? qy[3] : -qy[3])
                           +  ((mask & 0x10) ? qy[4] : -qy[4])
                           +  ((mask & 0x20) ? qy[5] : -qy[5])
                           +  ((mask & 0x40) ? qy[6] : -qy[6])
                           +  ((mask & 0x80) ? qy[7] : -qy[7]);
            }
            sumi += d1 * sumi_block;
        }
        sumf += d0 * sumi;
    }
    return sumf;
}
