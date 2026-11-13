// port-lint: source cuda_backend/utils.rs
package io.github.kotlinmania.candle.cudabackend.utils

import io.github.kotlinmania.candle.cudabackend.CudaStorageSlice
import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.cudabackend.error.CudaError
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.DeviceRepr
import io.github.kotlinmania.cudarc.driver.ValidAsZeroBits
import io.github.kotlinmania.rustlang.core.F16

public typealias S = CudaStorageSlice

public interface Map1 {
    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits

    public fun map(s: S, d: CudaDevice, l: Layout): Result<S> {
        val out =
            when (s) {
                is S.U8 -> {
                    val (s1) = s
                    S.u8(
                        when (val tryResult = this.f(s1, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.U32 -> {
                    val (s2) = s
                    S.u32(
                        when (val tryResult = this.f(s2, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.I16 -> {
                    val (s3) = s
                    S.i16(
                        when (val tryResult = this.f(s3, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.I32 -> {
                    val (s4) = s
                    S.i32(
                        when (val tryResult = this.f(s4, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.I64 -> {
                    val (s5) = s
                    S.i64(
                        when (val tryResult = this.f(s5, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.Bf16 -> {
                    val (s6) = s
                    S.bf16(
                        when (val tryResult = this.f(s6, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.F16 -> {
                    val (s7) = s
                    S.f16(
                        when (val tryResult = this.f(s7, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.F32 -> {
                    val (s8) = s
                    S.f32(
                        when (val tryResult = this.f(s8, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.F64 -> {
                    val (s9) = s
                    S.f64(
                        when (val tryResult = this.f(s9, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.F8e4m3 -> {
                    val (s10) = s
                    S.f8e4m3(
                        when (val tryResult = this.f(s10, d, l)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                is S.F4 -> {
                    bail("Map1 does not uspport this dtype.")
                }
                is S.F6e2m3 -> {
                    bail("Map1 does not uspport this dtype.")
                }
                is S.F6e3m2 -> {
                    bail("Map1 does not uspport this dtype.")
                }
                is S.F8e8m0 -> {
                    bail("Map1 does not uspport this dtype.")
                }
            }
        return Result.Ok(out)
    }
}

public interface Map2 {
    public fun <T> f(src1: CudaSlice<T>, layout1: Layout, src2: CudaSlice<T>, layout2: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, d: CudaDevice): Result<S> {
        val out =
            run {
                val matchVal = Pair(s1, s2)
                when {
                    matchVal.first is S.U8 && matchVal.second is S.U8 -> {
                        val (s11) = matchVal.first as S.U8
                        val (s21) = matchVal.second as S.U8
                        S.u8(
                            when (val tryResult = this.f(s11, l1, s21, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.U32 && matchVal.second is S.U32 -> {
                        val (s12) = matchVal.first as S.U32
                        val (s22) = matchVal.second as S.U32
                        S.u32(
                            when (val tryResult = this.f(s12, l1, s22, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.I16 && matchVal.second is S.I16 -> {
                        val (s13) = matchVal.first as S.I16
                        val (s23) = matchVal.second as S.I16
                        S.i16(
                            when (val tryResult = this.f(s13, l1, s23, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.I32 && matchVal.second is S.I32 -> {
                        val (s14) = matchVal.first as S.I32
                        val (s24) = matchVal.second as S.I32
                        S.i32(
                            when (val tryResult = this.f(s14, l1, s24, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.I64 && matchVal.second is S.I64 -> {
                        val (s15) = matchVal.first as S.I64
                        val (s25) = matchVal.second as S.I64
                        S.i64(
                            when (val tryResult = this.f(s15, l1, s25, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.Bf16 && matchVal.second is S.Bf16 -> {
                        val (s16) = matchVal.first as S.Bf16
                        val (s26) = matchVal.second as S.Bf16
                        S.bf16(
                            when (val tryResult = this.f(s16, l1, s26, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F16 && matchVal.second is S.F16 -> {
                        val (s17) = matchVal.first as S.F16
                        val (s27) = matchVal.second as S.F16
                        S.f16(
                            when (val tryResult = this.f(s17, l1, s27, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F32 && matchVal.second is S.F32 -> {
                        val (s18) = matchVal.first as S.F32
                        val (s28) = matchVal.second as S.F32
                        S.f32(
                            when (val tryResult = this.f(s18, l1, s28, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F64 && matchVal.second is S.F64 -> {
                        val (s19) = matchVal.first as S.F64
                        val (s29) = matchVal.second as S.F64
                        S.f64(
                            when (val tryResult = this.f(s19, l1, s29, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 -> {
                        val (s110) = matchVal.first as S.F8e4m3
                        val (s210) = matchVal.second as S.F8e4m3
                        S.f8e4m3(
                            when (val tryResult = this.f(s110, l1, s210, l2, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    else ->
                        when (val tryResult = Result.Err(CudaError.InternalError("dtype mismatch in binary op"))) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                }
            }
        return Result.Ok(out)
    }
}

public interface Map3 {
    public fun <T> f(src1: CudaSlice<T>, layout1: Layout, src2: CudaSlice<T>, layout2: Layout, src3: CudaSlice<T>, layout3: Layout, dev: CudaDevice): Result<CudaSlice<T>> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, s3: S, l3: Layout, d: CudaDevice): Result<S> {
        val out =
            run {
                val matchVal = Triple(s1, s2, s3)
                when {
                    matchVal.first is S.U8 && matchVal.second is S.U8 && matchVal.third is S.U8 -> {
                        val (s11) = matchVal.first as S.U8
                        val (s21) = matchVal.second as S.U8
                        val (s31) = matchVal.third as S.U8
                        S.u8(
                            when (val tryResult = this.f(s11, l1, s21, l2, s31, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.U32 && matchVal.second is S.U32 && matchVal.third is S.U32 -> {
                        val (s12) = matchVal.first as S.U32
                        val (s22) = matchVal.second as S.U32
                        val (s32) = matchVal.third as S.U32
                        S.u32(
                            when (val tryResult = this.f(s12, l1, s22, l2, s32, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.I64 && matchVal.second is S.I64 && matchVal.third is S.I64 -> {
                        val (s13) = matchVal.first as S.I64
                        val (s23) = matchVal.second as S.I64
                        val (s33) = matchVal.third as S.I64
                        S.i64(
                            when (val tryResult = this.f(s13, l1, s23, l2, s33, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.Bf16 && matchVal.second is S.Bf16 && matchVal.third is S.Bf16 -> {
                        val (s14) = matchVal.first as S.Bf16
                        val (s24) = matchVal.second as S.Bf16
                        val (s34) = matchVal.third as S.Bf16
                        S.bf16(
                            when (val tryResult = this.f(s14, l1, s24, l2, s34, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F16 && matchVal.second is S.F16 && matchVal.third is S.F16 -> {
                        val (s15) = matchVal.first as S.F16
                        val (s25) = matchVal.second as S.F16
                        val (s35) = matchVal.third as S.F16
                        S.f16(
                            when (val tryResult = this.f(s15, l1, s25, l2, s35, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F32 && matchVal.second is S.F32 && matchVal.third is S.F32 -> {
                        val (s16) = matchVal.first as S.F32
                        val (s26) = matchVal.second as S.F32
                        val (s36) = matchVal.third as S.F32
                        S.f32(
                            when (val tryResult = this.f(s16, l1, s26, l2, s36, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F64 && matchVal.second is S.F64 && matchVal.third is S.F64 -> {
                        val (s17) = matchVal.first as S.F64
                        val (s27) = matchVal.second as S.F64
                        val (s37) = matchVal.third as S.F64
                        S.f64(
                            when (val tryResult = this.f(s17, l1, s27, l2, s37, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 && matchVal.third is S.F8e4m3 -> {
                        val (s18) = matchVal.first as S.F8e4m3
                        val (s28) = matchVal.second as S.F8e4m3
                        val (s38) = matchVal.third as S.F8e4m3
                        S.f8e4m3(
                            when (val tryResult = this.f(s18, l1, s28, l2, s38, l3, d)) {
                                is Result.Ok<*> -> tryResult.value
                                is Result.Err<*> -> return Result.Err(tryResult.error)
                            },
                        )
                    }
                    else ->
                        when (val tryResult = Result.Err(CudaError.InternalError("dtype mismatch in ternary op"))) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                }
            }
        return Result.Ok(out)
    }
}

public interface Map2InPlace {
    public fun <T> f(dst: CudaSlice<T>, dstL: Layout, src: CudaSlice<T>, srcL: Layout, dev: CudaDevice): Result<Unit> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits

    public fun map(dst: S, dstL: Layout, src: S, srcL: Layout, d: CudaDevice): Result<Unit> {
        return run {
            val matchVal = Pair(dst, src)
            when {
                matchVal.first is S.U8 && matchVal.second is S.U8 -> {
                    val (dst1) = matchVal.first as S.U8
                    val (src1) = matchVal.second as S.U8
                    this.f(dst1, dstL, src1, srcL, d)
                }
                matchVal.first is S.U32 && matchVal.second is S.U32 -> {
                    val (dst2) = matchVal.first as S.U32
                    val (src2) = matchVal.second as S.U32
                    this.f(dst2, dstL, src2, srcL, d)
                }
                matchVal.first is S.I16 && matchVal.second is S.I16 -> {
                    val (dst3) = matchVal.first as S.I16
                    val (src3) = matchVal.second as S.I16
                    this.f(dst3, dstL, src3, srcL, d)
                }
                matchVal.first is S.I32 && matchVal.second is S.I32 -> {
                    val (dst4) = matchVal.first as S.I32
                    val (src4) = matchVal.second as S.I32
                    this.f(dst4, dstL, src4, srcL, d)
                }
                matchVal.first is S.I64 && matchVal.second is S.I64 -> {
                    val (dst5) = matchVal.first as S.I64
                    val (src5) = matchVal.second as S.I64
                    this.f(dst5, dstL, src5, srcL, d)
                }
                matchVal.first is S.Bf16 && matchVal.second is S.Bf16 -> {
                    val (dst6) = matchVal.first as S.Bf16
                    val (src6) = matchVal.second as S.Bf16
                    this.f(dst6, dstL, src6, srcL, d)
                }
                matchVal.first is S.F16 && matchVal.second is S.F16 -> {
                    val (dst7) = matchVal.first as S.F16
                    val (src7) = matchVal.second as S.F16
                    this.f(dst7, dstL, src7, srcL, d)
                }
                matchVal.first is S.F32 && matchVal.second is S.F32 -> {
                    val (dst8) = matchVal.first as S.F32
                    val (src8) = matchVal.second as S.F32
                    this.f(dst8, dstL, src8, srcL, d)
                }
                matchVal.first is S.F64 && matchVal.second is S.F64 -> {
                    val (dst9) = matchVal.first as S.F64
                    val (src9) = matchVal.second as S.F64
                    this.f(dst9, dstL, src9, srcL, d)
                }
                matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 -> {
                    val (dst10) = matchVal.first as S.F8e4m3
                    val (src10) = matchVal.second as S.F8e4m3
                    this.f(dst10, dstL, src10, srcL, d)
                }
                else ->
                    when (val tryResult = Result.Err(CudaError.InternalError("dtype mismatch in binary op"))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
            }
        }
    }
}

public interface Map1Any {
    public fun <T> f(src: CudaSlice<T>, dev: CudaDevice, layout: Layout, wrap: (CudaSlice<T>) -> S): Result<S> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits

    public fun map(s: S, d: CudaDevice, l: Layout): Result<S> {
        val out =
            when (s) {
                is S.U8 -> {
                    val (s1) = s
                    when (val tryResult = this.f(s1, d, l, S.U8)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.U32 -> {
                    val (s2) = s
                    when (val tryResult = this.f(s2, d, l, S.U32)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.I16 -> {
                    val (s3) = s
                    when (val tryResult = this.f(s3, d, l, S.I16)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.I32 -> {
                    val (s4) = s
                    when (val tryResult = this.f(s4, d, l, S.I32)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.I64 -> {
                    val (s5) = s
                    when (val tryResult = this.f(s5, d, l, S.I64)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.Bf16 -> {
                    val (s6) = s
                    when (val tryResult = this.f(s6, d, l, S.BF16)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.F16 -> {
                    val (s7) = s
                    when (val tryResult = this.f(s7, d, l, S.F16)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.F32 -> {
                    val (s8) = s
                    when (val tryResult = this.f(s8, d, l, S.F32)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.F64 -> {
                    val (s9) = s
                    when (val tryResult = this.f(s9, d, l, S.F64)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.F8e4m3 -> {
                    val (s10) = s
                    when (val tryResult = this.f(s10, d, l, S.F8E4M3)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                is S.F4 -> {
                    bail("Map1 does not uspport this dtype.")
                }
                is S.F6e2m3 -> {
                    bail("Map1 does not uspport this dtype.")
                }
                is S.F6e3m2 -> {
                    bail("Map1 does not uspport this dtype.")
                }
                is S.F8e8m0 -> {
                    bail("Map1 does not uspport this dtype.")
                }
            }
        return Result.Ok(out)
    }
}

public interface Map2Any {
    public fun <T> f(src1: CudaSlice<T>, layout1: Layout, src2: CudaSlice<T>, layout2: Layout, dev: CudaDevice): Result<S> where T : DeviceRepr, T : WithDType, T : ValidAsZeroBits

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, d: CudaDevice): Result<S> {
        val out =
            run {
                val matchVal = Pair(s1, s2)
                when {
                    matchVal.first is S.U8 && matchVal.second is S.U8 -> {
                        val (s11) = matchVal.first as S.U8
                        val (s21) = matchVal.second as S.U8
                        when (val tryResult = this.f(s11, l1, s21, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.U32 && matchVal.second is S.U32 -> {
                        val (s12) = matchVal.first as S.U32
                        val (s22) = matchVal.second as S.U32
                        when (val tryResult = this.f(s12, l1, s22, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.I64 && matchVal.second is S.I64 -> {
                        val (s13) = matchVal.first as S.I64
                        val (s23) = matchVal.second as S.I64
                        when (val tryResult = this.f(s13, l1, s23, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.Bf16 && matchVal.second is S.Bf16 -> {
                        val (s14) = matchVal.first as S.Bf16
                        val (s24) = matchVal.second as S.Bf16
                        when (val tryResult = this.f(s14, l1, s24, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.F16 && matchVal.second is S.F16 -> {
                        val (s15) = matchVal.first as S.F16
                        val (s25) = matchVal.second as S.F16
                        when (val tryResult = this.f(s15, l1, s25, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.F32 && matchVal.second is S.F32 -> {
                        val (s16) = matchVal.first as S.F32
                        val (s26) = matchVal.second as S.F32
                        when (val tryResult = this.f(s16, l1, s26, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.F64 && matchVal.second is S.F64 -> {
                        val (s17) = matchVal.first as S.F64
                        val (s27) = matchVal.second as S.F64
                        when (val tryResult = this.f(s17, l1, s27, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    matchVal.first is S.F8e4m3 && matchVal.second is S.F8e4m3 -> {
                        val (s18) = matchVal.first as S.F8e4m3
                        val (s28) = matchVal.second as S.F8e4m3
                        when (val tryResult = this.f(s18, l1, s28, l2, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    }
                    else ->
                        when (val tryResult = Result.Err(CudaError.InternalError("dtype mismatch in binary op")).w()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                }
            }
        return Result.Ok(out)
    }
}
