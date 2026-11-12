// port-lint: source cpu_backend/utils.rs
package io.github.kotlinmania.candle.cpubackend.utils

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.nditer.NdIter
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.candle.utils.barrierPool
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.core.result.fromResidual

public typealias C = CpuStorage

public interface Map1 {
    public fun <T : WithDType> f(vs: List<T>, layout: Layout): Result<List<T>>

    public fun map(vs: C, layout: Layout): Result<C> {
        return when (vs) {
            is C.U8 -> {
                val (vs1) = vs
                Ok(
                    C.u8(
                        when (val tryResult = this.f(vs1, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.U32 -> {
                val (vs2) = vs
                Ok(
                    C.u32(
                        when (val tryResult = this.f(vs2, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.I16 -> {
                val (vs3) = vs
                Ok(
                    C.i16(
                        when (val tryResult = this.f(vs3, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.I32 -> {
                val (vs4) = vs
                Ok(
                    C.i32(
                        when (val tryResult = this.f(vs4, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.I64 -> {
                val (vs5) = vs
                Ok(
                    C.i64(
                        when (val tryResult = this.f(vs5, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.Bf16 -> {
                val (vs6) = vs
                Ok(
                    C.bf16(
                        when (val tryResult = this.f(vs6, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.F16 -> {
                val (vs7) = vs
                Ok(
                    C.f16(
                        when (val tryResult = this.f(vs7, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.F32 -> {
                val (vs8) = vs
                Ok(
                    C.f32(
                        when (val tryResult = this.f(vs8, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.F64 -> {
                val (vs9) = vs
                Ok(
                    C.f64(
                        when (val tryResult = this.f(vs9, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is C.F8e4m3 -> {
                val (vs10) = vs
                Ok(
                    C.f8e4m3(
                        when (val tryResult = this.f(vs10, layout)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            // Dummy types don't support Map1 operations
            is C.F6e2m3 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
            is C.F6e3m2 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
            is C.F4 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
            is C.F8e8m0 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
        }
    }
}

public interface Map1Any {
    public fun <T : WithDType> f(vs: List<T>, layout: Layout, wrap: (List<T>) -> C): Result<C>

    public fun map(vs: C, layout: Layout): Result<C> {
        return when (vs) {
            is C.U8 -> {
                val (vs1) = vs
                Ok(
                    when (val tryResult = this.f(vs1, layout, C.U8)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.U32 -> {
                val (vs2) = vs
                Ok(
                    when (val tryResult = this.f(vs2, layout, C.U32)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.I16 -> {
                val (vs3) = vs
                Ok(
                    when (val tryResult = this.f(vs3, layout, C.I16)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.I32 -> {
                val (vs4) = vs
                Ok(
                    when (val tryResult = this.f(vs4, layout, C.I32)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.I64 -> {
                val (vs5) = vs
                Ok(
                    when (val tryResult = this.f(vs5, layout, C.I64)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.Bf16 -> {
                val (vs6) = vs
                Ok(
                    when (val tryResult = this.f(vs6, layout, C.BF16)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.F16 -> {
                val (vs7) = vs
                Ok(
                    when (val tryResult = this.f(vs7, layout, C.F16)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.F32 -> {
                val (vs8) = vs
                Ok(
                    when (val tryResult = this.f(vs8, layout, C.F32)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.F64 -> {
                val (vs9) = vs
                Ok(
                    when (val tryResult = this.f(vs9, layout, C.F64)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is C.F8e4m3 -> {
                val (vs10) = vs
                Ok(
                    when (val tryResult = this.f(vs10, layout, C.F8E4M3)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            // Dummy types don't support Map1Any operations
            is C.F6e2m3 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
            is C.F6e3m2 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
            is C.F4 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
            is C.F8e8m0 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
        }
    }
}

public interface Map2 {
    public val OP: String

    public fun <T : WithDType> f(v1: List<T>, l1: Layout, v2: List<T>, l2: Layout): Result<List<T>>

    public fun map(v1: C, l1: Layout, v2: C, l2: Layout): Result<C> {
        return run {
            val matchVal = Pair(v1, v2)
            when {
                matchVal.first is C.U8 && matchVal.second is C.U8 -> {
                    val (v11) = matchVal.first as C.U8
                    val (v21) = matchVal.second as C.U8
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v11, l1, v21, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.U32 && matchVal.second is C.U32 -> {
                    val (v12) = matchVal.first as C.U32
                    val (v22) = matchVal.second as C.U32
                    Ok(
                        C.u32(
                            when (val tryResult = this.f(v12, l1, v22, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.I16 && matchVal.second is C.I16 -> {
                    val (v13) = matchVal.first as C.I16
                    val (v23) = matchVal.second as C.I16
                    Ok(
                        C.i16(
                            when (val tryResult = this.f(v13, l1, v23, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.I32 && matchVal.second is C.I32 -> {
                    val (v14) = matchVal.first as C.I32
                    val (v24) = matchVal.second as C.I32
                    Ok(
                        C.i32(
                            when (val tryResult = this.f(v14, l1, v24, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.I64 && matchVal.second is C.I64 -> {
                    val (v15) = matchVal.first as C.I64
                    val (v25) = matchVal.second as C.I64
                    Ok(
                        C.i64(
                            when (val tryResult = this.f(v15, l1, v25, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.Bf16 && matchVal.second is C.Bf16 -> {
                    val (v16) = matchVal.first as C.Bf16
                    val (v26) = matchVal.second as C.Bf16
                    Ok(
                        C.bf16(
                            when (val tryResult = this.f(v16, l1, v26, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F16 && matchVal.second is C.F16 -> {
                    val (v17) = matchVal.first as C.F16
                    val (v27) = matchVal.second as C.F16
                    Ok(
                        C.f16(
                            when (val tryResult = this.f(v17, l1, v27, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F32 && matchVal.second is C.F32 -> {
                    val (v18) = matchVal.first as C.F32
                    val (v28) = matchVal.second as C.F32
                    Ok(
                        C.f32(
                            when (val tryResult = this.f(v18, l1, v28, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F64 && matchVal.second is C.F64 -> {
                    val (v19) = matchVal.first as C.F64
                    val (v29) = matchVal.second as C.F64
                    Ok(
                        C.f64(
                            when (val tryResult = this.f(v19, l1, v29, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F8e4m3 && matchVal.second is C.F8e4m3 -> {
                    val (v110) = matchVal.first as C.F8e4m3
                    val (v210) = matchVal.second as C.F8e4m3
                    Ok(
                        C.f8e4m3(
                            when (val tryResult = this.f(v110, l1, v210, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                else -> Err(Error.DTypeMismatchBinaryOp(lhs = v1.dtype(), rhs = v2.dtype(), op = Map2.OP).bt())
            }
        }
    }
}

public interface Map2InPlace {
    public val OP: String

    public fun <T : WithDType> f(v1: List<T>, l1: Layout, v2: List<T>, l2: Layout): Result<Unit>

    public fun map(v1: C, l1: Layout, v2: C, l2: Layout): Result<Unit> {
        run {
            val matchVal = Pair(v1, v2)
            when {
                matchVal.first is C.U8 && matchVal.second is C.U8 -> {
                    val (v11) = matchVal.first as C.U8
                    val (v21) = matchVal.second as C.U8
                    when (val tryResult = this.f(v11, l1, v21, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.U32 && matchVal.second is C.U32 -> {
                    val (v12) = matchVal.first as C.U32
                    val (v22) = matchVal.second as C.U32
                    when (val tryResult = this.f(v12, l1, v22, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.I16 && matchVal.second is C.I16 -> {
                    val (v13) = matchVal.first as C.I16
                    val (v23) = matchVal.second as C.I16
                    when (val tryResult = this.f(v13, l1, v23, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.I32 && matchVal.second is C.I32 -> {
                    val (v14) = matchVal.first as C.I32
                    val (v24) = matchVal.second as C.I32
                    when (val tryResult = this.f(v14, l1, v24, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.I64 && matchVal.second is C.I64 -> {
                    val (v15) = matchVal.first as C.I64
                    val (v25) = matchVal.second as C.I64
                    when (val tryResult = this.f(v15, l1, v25, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.Bf16 && matchVal.second is C.Bf16 -> {
                    val (v16) = matchVal.first as C.Bf16
                    val (v26) = matchVal.second as C.Bf16
                    when (val tryResult = this.f(v16, l1, v26, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.F16 && matchVal.second is C.F16 -> {
                    val (v17) = matchVal.first as C.F16
                    val (v27) = matchVal.second as C.F16
                    when (val tryResult = this.f(v17, l1, v27, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.F32 && matchVal.second is C.F32 -> {
                    val (v18) = matchVal.first as C.F32
                    val (v28) = matchVal.second as C.F32
                    when (val tryResult = this.f(v18, l1, v28, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.F64 && matchVal.second is C.F64 -> {
                    val (v19) = matchVal.first as C.F64
                    val (v29) = matchVal.second as C.F64
                    when (val tryResult = this.f(v19, l1, v29, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is C.F8e4m3 && matchVal.second is C.F8e4m3 -> {
                    val (v110) = matchVal.first as C.F8e4m3
                    val (v210) = matchVal.second as C.F8e4m3
                    when (val tryResult = this.f(v110, l1, v210, l2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                else -> {
                    val v111 = matchVal.first
                    val v211 = matchVal.second
                    when (val tryResult = Err(Error.DTypeMismatchBinaryOp(lhs = v111.dtype(), rhs = v211.dtype(), op = Map2InPlace.OP).bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        }
        return Ok(Unit)
    }
}

public interface Map2U8 {
    public val OP: String

    public fun <T : WithDType> f(v1: List<T>, l1: Layout, v2: List<T>, l2: Layout): Result<List<UByte>>

    public fun map(v1: C, l1: Layout, v2: C, l2: Layout): Result<C> {
        return run {
            val matchVal = Pair(v1, v2)
            when {
                matchVal.first is C.U8 && matchVal.second is C.U8 -> {
                    val (v11) = matchVal.first as C.U8
                    val (v21) = matchVal.second as C.U8
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v11, l1, v21, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.U32 && matchVal.second is C.U32 -> {
                    val (v12) = matchVal.first as C.U32
                    val (v22) = matchVal.second as C.U32
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v12, l1, v22, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.I16 && matchVal.second is C.I16 -> {
                    val (v13) = matchVal.first as C.I16
                    val (v23) = matchVal.second as C.I16
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v13, l1, v23, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.I32 && matchVal.second is C.I32 -> {
                    val (v14) = matchVal.first as C.I32
                    val (v24) = matchVal.second as C.I32
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v14, l1, v24, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.I64 && matchVal.second is C.I64 -> {
                    val (v15) = matchVal.first as C.I64
                    val (v25) = matchVal.second as C.I64
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v15, l1, v25, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.Bf16 && matchVal.second is C.Bf16 -> {
                    val (v16) = matchVal.first as C.Bf16
                    val (v26) = matchVal.second as C.Bf16
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v16, l1, v26, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F16 && matchVal.second is C.F16 -> {
                    val (v17) = matchVal.first as C.F16
                    val (v27) = matchVal.second as C.F16
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v17, l1, v27, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F32 && matchVal.second is C.F32 -> {
                    val (v18) = matchVal.first as C.F32
                    val (v28) = matchVal.second as C.F32
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v18, l1, v28, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F64 && matchVal.second is C.F64 -> {
                    val (v19) = matchVal.first as C.F64
                    val (v29) = matchVal.second as C.F64
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v19, l1, v29, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                matchVal.first is C.F8e4m3 && matchVal.second is C.F8e4m3 -> {
                    val (v110) = matchVal.first as C.F8e4m3
                    val (v210) = matchVal.second as C.F8e4m3
                    Ok(
                        C.u8(
                            when (val tryResult = this.f(v110, l1, v210, l2)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ),
                    )
                }
                else -> Err(Error.DTypeMismatchBinaryOp(lhs = v1.dtype(), rhs = v2.dtype(), op = Map2U8.OP).bt())
            }
        }
    }
}

public fun <T, U> binaryMap(lhsL: Layout, rhsL: Layout, lhs: List<T>, rhs: List<T>, f: (T, T) -> U): List<U> {
    val elCount = lhsL.shape().elemCount()
    val result: List<U> = ArrayList(elCount)
    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (innerLs, innerRs) = ndIter.innerStrides
    for ((lhsOff, rhsOff) in ndIter) {
        for (i in 0 until innerSize) {
            result.add(f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs]))
        }
    }
    return result
}

// Similar to binaryMap but with vectorized variants.
public fun <T> binaryMapVec(lhsL: Layout, rhsL: Layout, lhs: List<T>, rhs: List<T>, f: (T, T) -> T, fVec: (List<T>, List<T>, List<T>) -> Unit, fScalarVec: (T, List<T>, List<T>) -> Unit): List<T> {
    val elCount = lhsL.shape().elemCount()
    val ys: List<T> = ArrayList(elCount)
    val ysToSet =
        run {
            val s = ys.spareCapacityMut()
            transmute<List<T?>, List<T>>(s)
        }
    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (innerLs, innerRs) = ndIter.innerStrides
    var dstOff = 0
    for ((lhsOff, rhsOff) in ndIter) {
        run {
            val matchVal = Pair(innerLs, innerRs)
            when {
                matchVal.first == 1 && matchVal.second == 1 -> fVec(((lhs).slice(lhsOff until lhsOff + innerSize)), ((rhs).slice(rhsOff until rhsOff + innerSize)), ((ysToSet).slice(dstOff until dstOff + innerSize)))
                matchVal.first == 1 && matchVal.second == 0 -> {
                    val r = rhs[rhsOff]
                    fScalarVec(r, ((lhs).slice(lhsOff until lhsOff + innerSize)), ((ysToSet).slice(dstOff until dstOff + innerSize)))
                }
                matchVal.first == 0 && matchVal.second == 1 -> {
                    val l = lhs[lhsOff]
                    for (i in 0 until innerSize) {
                        ysToSet[dstOff + i] = f(l, rhs[rhsOff + i])
                    }
                }
                else -> {
                    for (i1 in 0 until innerSize) {
                        ysToSet[dstOff + i1] = f(lhs[lhsOff + i1 * innerLs], rhs[rhsOff + i1 * innerRs])
                    }
                }
            }
        }
        dstOff += innerSize
    }
    // SAFETY: all elCount elements have been written in the dispatch loop above.
    ys.setLen(elCount)
    return ys
}

public fun <T, U> unaryMap(vs: List<T>, layout: Layout, f: (T) -> U): List<U> =
    when (val matchVal = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val startOffset = matchVal.startOffset
            val len = matchVal.len
            ((vs).slice(startOffset until startOffset + len)).map({ v -> f(v) }).collect()
        }
        is StridedBlocks.UniformBlocks -> {
            val startOffset1 = matchVal.startOffset
            val blockLen = matchVal.blockLen
            val count = matchVal.count
            val srcStride = matchVal.srcStride
            var result = ArrayList<U>(count * blockLen)
            if (blockLen == 1) {
                for (i in 0 until count) {
                    val v = vs.getUnchecked(startOffset1 + i * srcStride)
                    result.add(f(v))
                }
            } else {
                for (i1 in 0 until count) {
                    val srcStart = startOffset1 + i1 * srcStride
                    for (offset in 0 until blockLen) {
                        val v1 = vs.getUnchecked(srcStart + offset)
                        result.add(f(v1))
                    }
                }
            }
            result
        }
        is StridedBlocks.MultipleBlocks -> {
            val blockStartIndex = matchVal.blockStartIndex
            val blockLen1 = matchVal.blockLen
            var result1 = ArrayList<U>(layout.shape().elemCount())
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen1 == 1) {
                for (index in blockStartIndex) {
                    val v2 = vs.getUnchecked(index)
                    result1.add(f(v2))
                }
            } else {
                for (index1 in blockStartIndex) {
                    for (offset1 in 0 until blockLen1) {
                        val v3 = vs.getUnchecked(index1 + offset1)
                        result1.add(f(v3))
                    }
                }
            }
            result1
        }
    }

public fun <T, U> unaryMapVec(vs: List<T>, layout: Layout, f: (T) -> U, fVec: (List<T>, List<U>) -> Unit): List<U> =
    when (val matchVal = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val startOffset = matchVal.startOffset
            val len = matchVal.len
            var ys: List<U> = ArrayList(len)
            val ysToSet = ys.spareCapacityMut()
            val ysToSet1 = transmute<List<U?>, List<U>>(ysToSet)
            fVec(((vs).slice(startOffset until startOffset + len)), ysToSet1)
            // SAFETY: values are all set by fVec.
            ys.setLen(len)
            ys
        }
        is StridedBlocks.UniformBlocks -> {
            val startOffset1 = matchVal.startOffset
            val blockLen = matchVal.blockLen
            val count = matchVal.count
            val srcStride = matchVal.srcStride
            val elCount = count * blockLen
            if (blockLen == 1) {
                val result = ArrayList<U>(count)
                for (i in 0 until count) {
                    val v = vs.getUnchecked(startOffset1 + i * srcStride)
                    result.add(f(v))
                }
                result
            } else {
                val ys1: List<U> = ArrayList(elCount)
                val ysToSet2 = ys1.spareCapacityMut()
                val ysToSet3 = transmute<List<U?>, List<U>>(ysToSet2)
                var dstIndex = 0
                for (i1 in 0 until count) {
                    val srcStart = startOffset1 + i1 * srcStride
                    fVec(((vs).slice(srcStart until srcStart + blockLen)), ((ysToSet3).slice(dstIndex until dstIndex + blockLen)))
                    dstIndex += blockLen
                }
                // SAFETY: values are all set by fVec.
                ys1.setLen(elCount)
                ys1
            }
        }
        is StridedBlocks.MultipleBlocks -> {
            val blockStartIndex = matchVal.blockStartIndex
            val blockLen1 = matchVal.blockLen
            val elCount1 = layout.shape().elemCount()
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen1 == 1) {
                val result1 = ArrayList<U>(elCount1)
                for (index in blockStartIndex) {
                    val v1 = vs.getUnchecked(index)
                    result1.add(f(v1))
                }
                result1
            } else {
                val ys2: List<U> = ArrayList(elCount1)
                val ysToSet4 = ys2.spareCapacityMut()
                val ysToSet5 = transmute<List<U?>, List<U>>(ysToSet4)
                var dstIndex1 = 0
                for (srcIndex in blockStartIndex) {
                    val vs1 = ((vs).slice(srcIndex until srcIndex + blockLen1))
                    val ys3 = ((ysToSet5).slice(dstIndex1 until dstIndex1 + blockLen1))
                    fVec(vs1, ys3)
                    dstIndex1 += blockLen1
                }
                // SAFETY: values are all set by fVec.
                ys2.setLen(elCount1)
                ys2
            }
        }
    }

// Below this many logical elements the barrier-pool split costs more than it saves.
public val PAR_ELEMWISE_MIN: Int = 64 * 1024

// Per-unit span: big enough to amortize dispatch, small enough to balance.
public val PAR_ELEMWISE_CHUNK: Int = 16 * 1024

public fun <T, U> unaryMapVecPar(vs: List<T>, layout: Layout, f: (T) -> U, fVec: (List<T>, List<U>) -> Unit): List<U> {
    when (val matchVal = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val startOffset = matchVal.startOffset
            val len = matchVal.len
            if (len >= PAR_ELEMWISE_MIN) {
                val src = ((vs).slice(startOffset until startOffset + len))
                val ys: List<U> = ArrayList(len)
                val ysPtr = ys.asMutPtr().toInt()
                val nUnits = len.divCeil(PAR_ELEMWISE_CHUNK)
                barrierPool().executeChunked(nUnits, { range ->
                    val ysPtr1 = MutPtr(ysPtr.toLong())
                    for (unit in range) {
                        val lo = unit * PAR_ELEMWISE_CHUNK
                        val hi = len.min(lo + PAR_ELEMWISE_CHUNK)
                        val dst = fromRawPartsMut(MutPtr((ysPtr1).rawAddress + ((lo).toLong() * (ObjectSlotLayout<U>()).strideBytes)), hi - lo)
                        fVec(((src).slice(lo until hi)), dst)
                    }
                })
                // SAFETY: every element is written by exactly one unit.
                ys.setLen(len)
                return ys
            }
        }
        else -> {}
    }
    return unaryMapVec(vs, layout, f, fVec)
}

public fun <T> binaryMapVecPar(lhsL: Layout, rhsL: Layout, lhs: List<T>, rhs: List<T>, f: (T, T) -> T, fVec: (List<T>, List<T>, List<T>) -> Unit, fScalarVec: (T, List<T>, List<T>) -> Unit): List<T> {
    val elCount = lhsL.shape().elemCount()
    if (elCount < PAR_ELEMWISE_MIN) {
        return binaryMapVec(lhsL, rhsL, lhs, rhs, f, fVec, fScalarVec)
    }
    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (lhsStride, rhsStride) = ndIter.innerStrides
    check(innerSize > 0)
    val ys: List<T> = ArrayList(elCount)
    val ysPtr = ys.asMutPtr().toInt()
    val nUnits = elCount.divCeil(PAR_ELEMWISE_CHUNK)
    barrierPool().executeChunked(nUnits, { range ->
        val ysPtr1 = MutPtr(ysPtr.toLong())
        for (unit in range) {
            var dstOffset = unit * PAR_ELEMWISE_CHUNK
            val hi = elCount.min(dstOffset + PAR_ELEMWISE_CHUNK)
            var innerOffset = dstOffset % innerSize
            val ndIter1 = NdIter.new(listOf(lhsL, rhsL))
            var offsets = ndIter1.nth(dstOffset / innerSize).expect("logical output offset must map to an NdIter block")
            while (dstOffset < hi) {
                val len = (innerSize - innerOffset).min(hi - dstOffset)
                val lhsOffset = offsets[0] + innerOffset * lhsStride
                val rhsOffset = offsets[1] + innerOffset * rhsStride
                // SAFETY: work units own disjoint output ranges, all written before setLen.
                val dst = fromRawPartsMut(MutPtr((ysPtr1).rawAddress + ((dstOffset).toLong() * (ObjectSlotLayout<T>()).strideBytes)), len)
                run {
                    val matchVal = Pair(lhsStride, rhsStride)
                    when {
                        matchVal.first == 1 && matchVal.second == 1 -> fVec(((lhs).slice(lhsOffset until lhsOffset + len)), ((rhs).slice(rhsOffset until rhsOffset + len)), dst)
                        matchVal.first == 1 && matchVal.second == 0 -> {
                            fScalarVec(rhs[rhsOffset], ((lhs).slice(lhsOffset until lhsOffset + len)), dst)
                        }
                        else -> {
                            for ((i, dst1) in dst.iterMut().enumerate()) {
                                dst1 = f(lhs[lhsOffset + i * lhsStride], rhs[rhsOffset + i * rhsStride])
                            }
                        }
                    }
                }
                dstOffset += len
                innerOffset = 0
                if (dstOffset < hi) {
                    offsets = ndIter1.next().expect("logical output range must stay within NdIter blocks")
                }
            }
        }
    })
    // SAFETY: every element is written by exactly one work unit.
    ys.setLen(elCount)
    return ys
}
