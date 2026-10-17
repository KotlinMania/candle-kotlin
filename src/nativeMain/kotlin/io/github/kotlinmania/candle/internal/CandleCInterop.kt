// port-lint: source cpu/neon.rs
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.candle.internal

import candle.cinterop.candle_accelerate_dgemm
import candle.cinterop.candle_accelerate_sgemm
import candle.cinterop.candle_accelerate_vdsp_vadd
import candle.cinterop.candle_accelerate_vdsp_vaddd
import candle.cinterop.candle_accelerate_vdsp_vdiv
import candle.cinterop.candle_accelerate_vdsp_vdivd
import candle.cinterop.candle_accelerate_vdsp_vmax
import candle.cinterop.candle_accelerate_vdsp_vmaxd
import candle.cinterop.candle_accelerate_vdsp_vmin
import candle.cinterop.candle_accelerate_vdsp_vmind
import candle.cinterop.candle_accelerate_vdsp_vmul
import candle.cinterop.candle_accelerate_vdsp_vmuld
import candle.cinterop.candle_accelerate_vdsp_vsub
import candle.cinterop.candle_accelerate_vdsp_vsubd
import candle.cinterop.candle_accelerate_vvcos
import candle.cinterop.candle_accelerate_vvcosf
import candle.cinterop.candle_accelerate_vvexp
import candle.cinterop.candle_accelerate_vvexpf
import candle.cinterop.candle_accelerate_vvlog
import candle.cinterop.candle_accelerate_vvlogf
import candle.cinterop.candle_accelerate_vvsin
import candle.cinterop.candle_accelerate_vvsinf
import candle.cinterop.candle_accelerate_vvsqrt
import candle.cinterop.candle_accelerate_vvsqrtf
import candle.cinterop.candle_accelerate_vvtanh
import candle.cinterop.candle_accelerate_vvtanhf
import candle.cinterop.candle_avx_add_ps
import candle.cinterop.candle_avx_cvtneps_pbh
import candle.cinterop.candle_avx_cvtpbh_ps
import candle.cinterop.candle_avx_cvtph_ps
import candle.cinterop.candle_avx_cvtps_ph
import candle.cinterop.candle_avx_fma_ps
import candle.cinterop.candle_avx_loadu_ps
import candle.cinterop.candle_avx_reduce_one_ps
import candle.cinterop.candle_avx_set1_ps
import candle.cinterop.candle_avx_storeu_ps
import candle.cinterop.candle_avx_zero
import candle.cinterop.candle_neon_bf16_load_shll
import candle.cinterop.candle_neon_bf16_store_shrn
import candle.cinterop.candle_neon_bf16_vec_add
import candle.cinterop.candle_neon_bf16_vec_fma
import candle.cinterop.candle_neon_f16_from_f32
import candle.cinterop.candle_neon_f16_load_fcvtl
import candle.cinterop.candle_neon_f16_reduce_one
import candle.cinterop.candle_neon_f16_store_fcvtn
import candle.cinterop.candle_neon_f16_vec_add
import candle.cinterop.candle_neon_f16_vec_fma
import candle.cinterop.candle_neon_f16_vec_store
import candle.cinterop.candle_neon_f16_zero
import candle.cinterop.candle_neon_reduce_one_arm_f32
import candle.cinterop.candle_neon_reduce_one_f32
import candle.cinterop.candle_neon_vaddq_f32
import candle.cinterop.candle_neon_vdupq_n_f32
import candle.cinterop.candle_neon_vfmaq_f32
import candle.cinterop.candle_neon_vld1q_f32
import candle.cinterop.candle_neon_vst1q_f32
import candle.cinterop.candle_qavx_hadd_epi16
import candle.cinterop.candle_qavx_hadd_epi32
import candle.cinterop.candle_qavx_madd_epi16
import candle.cinterop.candle_qavx_maddubs_epi16
import candle.cinterop.candle_qavx_mm_madd_epi16
import candle.cinterop.candle_qavx_sign_epi8
import candle.cinterop.candle_qneon_vaddvq_s32
import candle.cinterop.candle_qneon_vdotq_s32
import candle.cinterop.candle_qneon_vmlaq_n_f32
import candle.cinterop.candle_qneon_vmull_s16
import candle.cinterop.candle_qneon_vmull_s8
import candle.cinterop.candle_qneon_vpaddlq_s16
import candle.cinterop.candle_qneon_vpaddq_s16
import candle.cinterop.candle_qsimd128_dot_i16x8
import candle.cinterop.candle_simd128_add_f32
import candle.cinterop.candle_simd128_fma_f32
import candle.cinterop.candle_simd128_load_f32
import candle.cinterop.candle_simd128_mul_f32
import candle.cinterop.candle_simd128_reduce_one_f32
import candle.cinterop.candle_simd128_splat_f32
import candle.cinterop.candle_simd128_store_f32
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned

internal object CandleCInterop {
    fun neonReduceOneF32(x: FloatArray): Float = x.usePinned { pinned -> candle_neon_reduce_one_f32(pinned.addressOf(0)) }

    fun neonReduceOneArmF32(
        x: FloatArray,
        offset: Int,
    ): Float = x.usePinned { pinned -> candle_neon_reduce_one_arm_f32(pinned.addressOf(offset)) }

    fun neonVdupqNF32(v: Float): FloatArray {
        val out = FloatArray(4)
        out.usePinned { pinned -> candle_neon_vdupq_n_f32(v, pinned.addressOf(0)) }
        return out
    }

    fun neonVld1qF32(
        memAddr: FloatArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_neon_vld1q_f32(src.addressOf(offset), dst.addressOf(0))
            }
        }
        return out
    }

    fun neonVaddqF32(
        a: FloatArray,
        b: FloatArray,
    ): FloatArray {
        val out = FloatArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_neon_vaddq_f32(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun neonVfmaqF32(
        a: FloatArray,
        b: FloatArray,
        c: FloatArray,
    ): FloatArray {
        val out = FloatArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    out.usePinned { pout ->
                        candle_neon_vfmaq_f32(pa.addressOf(0), pb.addressOf(0), pc.addressOf(0), pout.addressOf(0))
                    }
                }
            }
        }
        return out
    }

    fun neonVst1qF32(
        memAddr: FloatArray,
        offset: Int,
        a: FloatArray,
    ) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_vst1q_f32(dst.addressOf(offset), src.addressOf(0))
            }
        }
    }

    fun neonF16LoadFcvtl(
        memAddr: ShortArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_neon_f16_load_fcvtl(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun neonF16StoreFcvtn(
        memAddr: ShortArray,
        offset: Int,
        a: FloatArray,
    ) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_f16_store_fcvtn(dst.addressOf(offset).reinterpret(), src.addressOf(0))
            }
        }
    }

    fun neonF16ReduceOne(x: ShortArray): Float = x.usePinned { pinned -> candle_neon_f16_reduce_one(pinned.addressOf(0).reinterpret()) }

    fun neonF16Zero(): ShortArray {
        val out = ShortArray(8)
        out.usePinned { pinned -> candle_neon_f16_zero(pinned.addressOf(0).reinterpret()) }
        return out
    }

    fun neonF16VecAdd(
        a: ShortArray,
        b: ShortArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_neon_f16_vec_add(
                        pa.addressOf(0).reinterpret(),
                        pb.addressOf(0).reinterpret(),
                        pout.addressOf(0).reinterpret(),
                    )
                }
            }
        }
        return out
    }

    fun neonF16VecFma(
        a: ShortArray,
        b: ShortArray,
        c: ShortArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    out.usePinned { pout ->
                        candle_neon_f16_vec_fma(
                            pa.addressOf(0).reinterpret(),
                            pb.addressOf(0).reinterpret(),
                            pc.addressOf(0).reinterpret(),
                            pout.addressOf(0).reinterpret(),
                        )
                    }
                }
            }
        }
        return out
    }

    fun neonF16FromF32(v: Float): ShortArray {
        val out = ShortArray(8)
        out.usePinned { pout -> candle_neon_f16_from_f32(v, pout.addressOf(0).reinterpret()) }
        return out
    }

    fun neonF16VecStore(
        memAddr: ShortArray,
        offset: Int,
        a: ShortArray,
    ) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_f16_vec_store(dst.addressOf(offset).reinterpret(), src.addressOf(0).reinterpret())
            }
        }
    }

    fun neonBf16LoadShll(
        memAddr: ShortArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_neon_bf16_load_shll(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun neonBf16StoreShrn(
        memAddr: ShortArray,
        offset: Int,
        a: FloatArray,
    ) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_bf16_store_shrn(dst.addressOf(offset).reinterpret(), src.addressOf(0))
            }
        }
    }

    fun neonBf16VecAdd(
        a: ShortArray,
        b: ShortArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_neon_bf16_vec_add(
                        pa.addressOf(0).reinterpret(),
                        pb.addressOf(0).reinterpret(),
                        pout.addressOf(0).reinterpret(),
                    )
                }
            }
        }
        return out
    }

    fun neonBf16VecFma(
        a: ShortArray,
        b: ShortArray,
        c: ShortArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    out.usePinned { pout ->
                        candle_neon_bf16_vec_fma(
                            pa.addressOf(0).reinterpret(),
                            pb.addressOf(0).reinterpret(),
                            pc.addressOf(0).reinterpret(),
                            pout.addressOf(0).reinterpret(),
                        )
                    }
                }
            }
        }
        return out
    }

    fun avxZero(): FloatArray {
        val out = FloatArray(8)
        out.usePinned { pout -> candle_avx_zero(pout.addressOf(0)) }
        return out
    }

    fun avxSet1Ps(v: Float): FloatArray {
        val out = FloatArray(8)
        out.usePinned { pout -> candle_avx_set1_ps(v, pout.addressOf(0)) }
        return out
    }

    fun avxLoaduPs(
        memAddr: FloatArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(8)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_avx_loadu_ps(src.addressOf(offset), dst.addressOf(0))
            }
        }
        return out
    }

    fun avxStoreuPs(
        memAddr: FloatArray,
        offset: Int,
        a: FloatArray,
    ) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_avx_storeu_ps(dst.addressOf(offset), src.addressOf(0))
            }
        }
    }

    fun avxReduceOnePs(x: FloatArray): Float = x.usePinned { pinned -> candle_avx_reduce_one_ps(pinned.addressOf(0)) }

    fun avxAddPs(
        a: FloatArray,
        b: FloatArray,
    ): FloatArray {
        val out = FloatArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_avx_add_ps(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun avxFmaPs(
        a: FloatArray,
        b: FloatArray,
        c: FloatArray,
    ): FloatArray {
        val out = FloatArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    out.usePinned { pout ->
                        candle_avx_fma_ps(pa.addressOf(0), pb.addressOf(0), pc.addressOf(0), pout.addressOf(0))
                    }
                }
            }
        }
        return out
    }

    fun avxCvtphPs(
        memAddr: ShortArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(8)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_avx_cvtph_ps(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun avxCvtpsPh(
        a: FloatArray,
        memAddr: ShortArray,
        offset: Int,
    ) {
        a.usePinned { src ->
            memAddr.usePinned { dst ->
                candle_avx_cvtps_ph(src.addressOf(0), dst.addressOf(offset).reinterpret())
            }
        }
    }

    fun avxCvtpbhPs(
        memAddr: ShortArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(8)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_avx_cvtpbh_ps(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun avxCvtnepsPbh(
        a: FloatArray,
        memAddr: ShortArray,
        offset: Int,
    ) {
        a.usePinned { src ->
            memAddr.usePinned { dst ->
                candle_avx_cvtneps_pbh(src.addressOf(0), dst.addressOf(offset).reinterpret())
            }
        }
    }

    fun simd128SplatF32(v: Float): FloatArray {
        val out = FloatArray(4)
        out.usePinned { pout -> candle_simd128_splat_f32(v, pout.addressOf(0)) }
        return out
    }

    fun simd128LoadF32(
        memAddr: FloatArray,
        offset: Int,
    ): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_simd128_load_f32(src.addressOf(offset), dst.addressOf(0))
            }
        }
        return out
    }

    fun simd128AddF32(
        a: FloatArray,
        b: FloatArray,
    ): FloatArray {
        val out = FloatArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_simd128_add_f32(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun simd128MulF32(
        a: FloatArray,
        b: FloatArray,
    ): FloatArray {
        val out = FloatArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_simd128_mul_f32(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun simd128FmaF32(
        a: FloatArray,
        b: FloatArray,
        c: FloatArray,
    ): FloatArray {
        val out = FloatArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    out.usePinned { pout ->
                        candle_simd128_fma_f32(pa.addressOf(0), pb.addressOf(0), pc.addressOf(0), pout.addressOf(0))
                    }
                }
            }
        }
        return out
    }

    fun simd128StoreF32(
        memAddr: FloatArray,
        offset: Int,
        a: FloatArray,
    ) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_simd128_store_f32(dst.addressOf(offset), src.addressOf(0))
            }
        }
    }

    fun simd128ReduceOneF32(x: FloatArray): Float = x.usePinned { px -> candle_simd128_reduce_one_f32(px.addressOf(0)) }

    fun accelerateSgemm(
        transa: Byte,
        transb: Byte,
        m: Int,
        n: Int,
        k: Int,
        alpha: Float,
        a: FloatArray,
        lda: Int,
        b: FloatArray,
        ldb: Int,
        beta: Float,
        c: FloatArray,
        ldc: Int,
    ) {
        if (m <= 0 || n <= 0 || k <= 0) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_sgemm(
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        pa.addressOf(0),
                        lda,
                        pb.addressOf(0),
                        ldb,
                        beta,
                        pc.addressOf(0),
                        ldc,
                    )
                }
            }
        }
    }

    fun accelerateDgemm(
        transa: Byte,
        transb: Byte,
        m: Int,
        n: Int,
        k: Int,
        alpha: Double,
        a: DoubleArray,
        lda: Int,
        b: DoubleArray,
        ldb: Int,
        beta: Double,
        c: DoubleArray,
        ldc: Int,
    ) {
        if (m <= 0 || n <= 0 || k <= 0) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_dgemm(
                        transa,
                        transb,
                        m,
                        n,
                        k,
                        alpha,
                        pa.addressOf(0),
                        lda,
                        pb.addressOf(0),
                        ldb,
                        beta,
                        pc.addressOf(0),
                        ldc,
                    )
                }
            }
        }
    }

    fun accelerateVvexpf(
        dst: FloatArray,
        src: FloatArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvexpf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvexp(
        dst: DoubleArray,
        src: DoubleArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvexp(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvsqrtf(
        dst: FloatArray,
        src: FloatArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvsqrtf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvsqrt(
        dst: DoubleArray,
        src: DoubleArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvsqrt(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvsinf(
        dst: FloatArray,
        src: FloatArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvsinf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvsin(
        dst: DoubleArray,
        src: DoubleArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvsin(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvcosf(
        dst: FloatArray,
        src: FloatArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvcosf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvcos(
        dst: DoubleArray,
        src: DoubleArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvcos(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvlogf(
        dst: FloatArray,
        src: FloatArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvlogf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvlog(
        dst: DoubleArray,
        src: DoubleArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvlog(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvtanhf(
        dst: FloatArray,
        src: FloatArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvtanhf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvtanh(
        dst: DoubleArray,
        src: DoubleArray,
        len: Int,
    ) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvtanh(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVdspVadd(
        a: FloatArray,
        ia: Long,
        b: FloatArray,
        ib: Long,
        c: FloatArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vadd(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVaddD(
        a: DoubleArray,
        ia: Long,
        b: DoubleArray,
        ib: Long,
        c: DoubleArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vaddd(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVsub(
        a: FloatArray,
        ia: Long,
        b: FloatArray,
        ib: Long,
        c: FloatArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vsub(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVsubD(
        a: DoubleArray,
        ia: Long,
        b: DoubleArray,
        ib: Long,
        c: DoubleArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vsubd(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVmul(
        a: FloatArray,
        ia: Long,
        b: FloatArray,
        ib: Long,
        c: FloatArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vmul(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVmulD(
        a: DoubleArray,
        ia: Long,
        b: DoubleArray,
        ib: Long,
        c: DoubleArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vmuld(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVdiv(
        a: FloatArray,
        ia: Long,
        b: FloatArray,
        ib: Long,
        c: FloatArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vdiv(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVdivD(
        a: DoubleArray,
        ia: Long,
        b: DoubleArray,
        ib: Long,
        c: DoubleArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vdivd(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVmin(
        a: FloatArray,
        ia: Long,
        b: FloatArray,
        ib: Long,
        c: FloatArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vmin(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVminD(
        a: DoubleArray,
        ia: Long,
        b: DoubleArray,
        ib: Long,
        c: DoubleArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vmind(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVmax(
        a: FloatArray,
        ia: Long,
        b: FloatArray,
        ib: Long,
        c: FloatArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vmax(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun accelerateVdspVmaxD(
        a: DoubleArray,
        ia: Long,
        b: DoubleArray,
        ib: Long,
        c: DoubleArray,
        ic: Long,
        n: ULong,
    ) {
        if (n == 0uL) return
        a.usePinned { pa ->
            b.usePinned { pb ->
                c.usePinned { pc ->
                    candle_accelerate_vdsp_vmaxd(pa.addressOf(0), ia, pb.addressOf(0), ib, pc.addressOf(0), ic, n)
                }
            }
        }
    }

    fun qneonVdotqS32(
        a: ByteArray,
        b: ByteArray,
    ): IntArray {
        val out = IntArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qneon_vdotq_s32(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qneonVmullS8(
        a: ByteArray,
        b: ByteArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qneon_vmull_s8(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qneonVmullS16(
        a: ShortArray,
        b: ShortArray,
    ): IntArray {
        val out = IntArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qneon_vmull_s16(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qneonVpaddlqS16(a: ShortArray): IntArray {
        val out = IntArray(4)
        a.usePinned { pa ->
            out.usePinned { pout ->
                candle_qneon_vpaddlq_s16(pa.addressOf(0), pout.addressOf(0))
            }
        }
        return out
    }

    fun qneonVpaddqS16(
        a: ShortArray,
        b: ShortArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qneon_vpaddq_s16(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qneonVmlaqNF32(
        acc: FloatArray,
        v: FloatArray,
        s: Float,
    ): FloatArray {
        val out = FloatArray(4)
        acc.usePinned { pacc ->
            v.usePinned { pv ->
                out.usePinned { pout ->
                    candle_qneon_vmlaq_n_f32(pacc.addressOf(0), pv.addressOf(0), s, pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qneonVaddvqS32(a: IntArray): Int = a.usePinned { pa -> candle_qneon_vaddvq_s32(pa.addressOf(0)) }

    fun qavxMaddEpi16(
        a: ShortArray,
        b: ShortArray,
    ): IntArray {
        val out = IntArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qavx_madd_epi16(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qavxMmMaddEpi16(
        a: ShortArray,
        b: ShortArray,
    ): IntArray {
        val out = IntArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qavx_mm_madd_epi16(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qavxMaddubsEpi16(
        ax: ByteArray,
        sy: ByteArray,
    ): ShortArray {
        val out = ShortArray(16)
        ax.usePinned { pax ->
            sy.usePinned { psy ->
                out.usePinned { pout ->
                    candle_qavx_maddubs_epi16(pax.addressOf(0), psy.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qavxSignEpi8(
        a: ByteArray,
        b: ByteArray,
    ): ByteArray {
        val out = ByteArray(32)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qavx_sign_epi8(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qavxHaddEpi16(
        a: ShortArray,
        b: ShortArray,
    ): ShortArray {
        val out = ShortArray(8)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qavx_hadd_epi16(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qavxHaddEpi32(
        a: IntArray,
        b: IntArray,
    ): IntArray {
        val out = IntArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qavx_hadd_epi32(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }

    fun qsimd128DotI16x8(
        a: ShortArray,
        b: ShortArray,
    ): IntArray {
        val out = IntArray(4)
        a.usePinned { pa ->
            b.usePinned { pb ->
                out.usePinned { pout ->
                    candle_qsimd128_dot_i16x8(pa.addressOf(0), pb.addressOf(0), pout.addressOf(0))
                }
            }
        }
        return out
    }
}
