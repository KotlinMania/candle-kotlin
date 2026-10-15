// port-lint: source cpu/neon.rs
@file:OptIn(kotlinx.cinterop.ExperimentalForeignApi::class)

package io.github.kotlinmania.candle.internal

import candle.cinterop.*
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned

internal object CandleCInterop {
    fun neonReduceOneF32(x: FloatArray): Float =
        x.usePinned { pinned -> candle_neon_reduce_one_f32(pinned.addressOf(0)) }

    fun neonReduceOneArmF32(x: FloatArray, offset: Int): Float =
        x.usePinned { pinned -> candle_neon_reduce_one_arm_f32(pinned.addressOf(offset)) }

    fun neonVdupqNF32(v: Float): FloatArray {
        val out = FloatArray(4)
        out.usePinned { pinned -> candle_neon_vdupq_n_f32(v, pinned.addressOf(0)) }
        return out
    }

    fun neonVld1qF32(memAddr: FloatArray, offset: Int): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_neon_vld1q_f32(src.addressOf(offset), dst.addressOf(0))
            }
        }
        return out
    }

    fun neonVaddqF32(a: FloatArray, b: FloatArray): FloatArray {
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

    fun neonVfmaqF32(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray {
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

    fun neonVst1qF32(memAddr: FloatArray, offset: Int, a: FloatArray) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_vst1q_f32(dst.addressOf(offset), src.addressOf(0))
            }
        }
    }

    fun neonF16LoadFcvtl(memAddr: ShortArray, offset: Int): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_neon_f16_load_fcvtl(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun neonF16StoreFcvtn(memAddr: ShortArray, offset: Int, a: FloatArray) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_f16_store_fcvtn(dst.addressOf(offset).reinterpret(), src.addressOf(0))
            }
        }
    }

    fun neonF16ReduceOne(x: ShortArray): Float =
        x.usePinned { pinned -> candle_neon_f16_reduce_one(pinned.addressOf(0).reinterpret()) }

    fun neonF16Zero(): ShortArray {
        val out = ShortArray(8)
        out.usePinned { pinned -> candle_neon_f16_zero(pinned.addressOf(0).reinterpret()) }
        return out
    }

    fun neonF16VecAdd(a: ShortArray, b: ShortArray): ShortArray {
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

    fun neonF16VecFma(a: ShortArray, b: ShortArray, c: ShortArray): ShortArray {
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

    fun neonF16VecStore(memAddr: ShortArray, offset: Int, a: ShortArray) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_f16_vec_store(dst.addressOf(offset).reinterpret(), src.addressOf(0).reinterpret())
            }
        }
    }

    fun neonBf16LoadShll(memAddr: ShortArray, offset: Int): FloatArray {
        val out = FloatArray(4)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_neon_bf16_load_shll(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun neonBf16StoreShrn(memAddr: ShortArray, offset: Int, a: FloatArray) {
        memAddr.usePinned { dst ->
            a.usePinned { src ->
                candle_neon_bf16_store_shrn(dst.addressOf(offset).reinterpret(), src.addressOf(0))
            }
        }
    }

    fun neonBf16VecAdd(a: ShortArray, b: ShortArray): ShortArray {
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

    fun neonBf16VecFma(a: ShortArray, b: ShortArray, c: ShortArray): ShortArray {
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

    fun avxReduceOnePs(x: FloatArray): Float =
        x.usePinned { pinned -> candle_avx_reduce_one_ps(pinned.addressOf(0)) }

    fun avxAddPs(a: FloatArray, b: FloatArray): FloatArray {
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

    fun avxFmaPs(a: FloatArray, b: FloatArray, c: FloatArray): FloatArray {
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

    fun avxCvtphPs(memAddr: ShortArray, offset: Int): FloatArray {
        val out = FloatArray(8)
        memAddr.usePinned { src ->
            out.usePinned { dst ->
                candle_avx_cvtph_ps(src.addressOf(offset).reinterpret(), dst.addressOf(0))
            }
        }
        return out
    }

    fun avxCvtpsPh(a: FloatArray, memAddr: ShortArray, offset: Int) {
        a.usePinned { src ->
            memAddr.usePinned { dst ->
                candle_avx_cvtps_ph(src.addressOf(0), dst.addressOf(offset).reinterpret())
            }
        }
    }

    fun accelerateVvexpf(dst: FloatArray, src: FloatArray, len: Int) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvexpf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvsqrtf(dst: FloatArray, src: FloatArray, len: Int) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvsqrtf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun accelerateVvtanhf(dst: FloatArray, src: FloatArray, len: Int) {
        if (len <= 0) return
        dst.usePinned { pdst ->
            src.usePinned { psrc ->
                candle_accelerate_vvtanhf(pdst.addressOf(0), psrc.addressOf(0), len)
            }
        }
    }

    fun qneonVdotqS32(a: ByteArray, b: ByteArray): IntArray {
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

    fun qneonVmullS8(a: ByteArray, b: ByteArray): ShortArray {
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

    fun qneonVmullS16(a: ShortArray, b: ShortArray): IntArray {
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

    fun qneonVpaddqS16(a: ShortArray, b: ShortArray): ShortArray {
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

    fun qneonVmlaqNF32(acc: FloatArray, v: FloatArray, s: Float): FloatArray {
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

    fun qneonVaddvqS32(a: IntArray): Int =
        a.usePinned { pa -> candle_qneon_vaddvq_s32(pa.addressOf(0)) }

    fun qavxMaddEpi16(a: ShortArray, b: ShortArray): IntArray {
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

    fun qavxMmMaddEpi16(a: ShortArray, b: ShortArray): IntArray {
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

    fun qavxMaddubsEpi16(ax: ByteArray, sy: ByteArray): ShortArray {
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

    fun qavxSignEpi8(a: ByteArray, b: ByteArray): ByteArray {
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

    fun qavxHaddEpi16(a: ShortArray, b: ShortArray): ShortArray {
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

    fun qavxHaddEpi32(a: IntArray, b: IntArray): IntArray {
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

    fun qsimd128DotI16x8(a: ShortArray, b: ShortArray): IntArray {
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
