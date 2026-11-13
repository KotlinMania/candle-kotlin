package io.github.kotlinmania.candle.matmultests

import io.github.kotlinmania.DType.F16
import io.github.kotlinmania.DType.F32
import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Tensor
import io.github.kotlinmania.candlecore.testutils.assertTensorEq

public fun matmul(device: Device): Result<Unit> {
    val data = mutableListOf(1.0f, 2.0, 3.0, 4.0)
    val a =
        when (val tryResult = Tensor.fromSlice(data, Pair(2, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data1 = mutableListOf(1.0f, 2.0, 3.0, 4.0)
    val b =
        when (val tryResult = Tensor.fromSlice(data1, Pair(2, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val c =
        when (val tryResult = a.matmul(b)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = c.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7.0f, 10.0), listOf(15.0, 22.0)),
    )
    val data2 = mutableListOf(1.0f, 2.0)
    val a1 =
        when (val tryResult = Tensor.fromSlice(data2, Pair(2, 1), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data3 = mutableListOf(3.0f, 4.0)
    val b1 =
        when (val tryResult = Tensor.fromSlice(data3, Pair(1, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val c1 =
        when (val tryResult = a1.matmul(b1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = c1.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(3.0, 4.0), listOf(6.0, 8.0)),
    )
    val data4: Vec = (0 until 6).map({ i -> i.toFloat() }).collect()
    val a2 =
        when (val tryResult = Tensor.fromSlice(data4, Pair(2, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data5: Vec = (0 until 6).map({ i -> (i + 2).toFloat() }).collect()
    val b2 =
        when (val tryResult = Tensor.fromSlice(data5, Pair(3, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val c2 =
        when (val tryResult = a2.matmul(b2)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = c2.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(16.0, 19.0), listOf(52.0, 64.0)),
    )
    val data6: Vec = (0 until 12).map({ i -> i.toFloat() }).collect()
    val a3 =
        when (val tryResult = Tensor.fromSlice(data6, Triple(2, 2, 3), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data7: Vec = (0 until 12).map({ i -> (i + 2).toFloat() }).collect()
    val b3 =
        when (val tryResult = Tensor.fromSlice(data7, Triple(2, 3, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expected = listOf(listOf(listOf(16.0, 19.0), listOf(52.0, 64.0)), listOf(listOf(214.0, 235.0), listOf(304.0, 334.0)))
    val c3 =
        when (val tryResult = a3.matmul(b3)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = c3.toVec3<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expected,
    )
    // Also perform the matmul on contiguous transposed versions.
    val aTt =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = a3.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(!aTt.isContiguous())
    check(a3.dims() == aTt.dims())
    check(aTt.stride() == listOf(6, 1, 2))
    val bTt =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = b3.t()) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.contiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(!bTt.isContiguous())
    check(b3.dims() == bTt.dims())
    check(bTt.stride() == listOf(6, 1, 3))
    check(
        when (
            val tryResult =
                when (val tryResult = aTt.matmul(b3)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expected,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = a3.matmul(bTt)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expected,
    )
    check(
        when (
            val tryResult =
                when (val tryResult = aTt.matmul(bTt)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec3<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == expected,
    )
    return Result.Ok(Unit)
}

public fun matmulBf16(device: Device): Result<Unit> {
    if (!device.supportsBf16()) {
        return Result.Ok(Unit)
    }
    val data = mutableListOf(1.0f, 2.0, 3.0, 4.0)
    val a =
        when (
            val tryResult =
                when (val tryResult = Tensor.fromSlice(data, Pair(2, 2), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.BF16)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val data1 = mutableListOf(1.0f, 2.0, 3.0, 4.0)
    val b =
        when (
            val tryResult =
                when (val tryResult = Tensor.fromSlice(data1, Pair(2, 2), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.BF16)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val c =
        when (
            val tryResult =
                when (val tryResult = a.matmul(b)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toDtype(DType.F32)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = c.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(7.0f, 10.0), listOf(15.0, 22.0)),
    )
    return Result.Ok(Unit)
}

public fun broadcastMatmul(device: Device): Result<Unit> {
    val lhs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, listOf(3, 1, 4, 5), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Triple(6, 5, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out =
        when (val tryResult = lhs.broadcastMatmul(rhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out.dims() == listOf(3, 6, 4, 2))
    for (idx1 in 0 until 3) {
        for (idx2 in 0 until 6) {
            val out1 =
                when (val tryResult = out.i(Pair(idx1, idx2))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val lhs1 =
                when (val tryResult = lhs.i(Pair(idx1, 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val rhs1 =
                when (val tryResult = rhs.i(idx2)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            val out2 = lhs1.matmul(rhs1)
            val sumDiff2 =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (out1 - out2)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sqr()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            // With cuda, we see errors of up to ~1e-12.
            check(
                when (val tryResult = sumDiff2.toVec0<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } < 1e-6,
            )
        }
    }
    return Result.Ok(Unit)
}

// A stride-zero batch dim on the lhs, as produced by `broadcastAs`, must give the same
// result as materializing that batch. The batches of rhs cannot be folded into the columns
// of one matmul the way a batch-invariant rhs folds into the rows, so this exercises the
// per-batch loop.
public fun broadcastMatmulStrideZeroLhs(device: Device): Result<Unit> {
    for ((b, m, n, k) in listOf(listOf(32, 32, 32, 32), listOf(3, 1, 2, 5), listOf(4, 2, 3, 1), listOf(2, 5, 1, 3))) {
        val lhs =
            when (val tryResult = Tensor.randn(0.0f, 1.0f, Triple(1, m, k), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val rhs =
            when (val tryResult = Tensor.randn(0.0f, 1.0f, Triple(b, k, n), device)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val out =
            when (
                val tryResult =
                    when (val tryResult = lhs.broadcastAs(Triple(b, m, k))) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }.matmul(rhs)
            ) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        check(out.dims() == listOf(b, m, n))
        // Every batch is the same lhs against that batch of rhs.
        val lhs1 =
            when (val tryResult = lhs.i(0)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        for (idx in 0 until b) {
            val diff =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = out.i(idx)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } -
                                            when (
                                                val tryResult =
                                                    lhs1.matmul(
                                                        when (val tryResult = rhs.i(idx)) {
                                                            is Result.Ok<*> -> tryResult.value
                                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                                        },
                                                    )
                                            ) {
                                                is Result.Ok<*> -> tryResult.value
                                                is Result.Err<*> -> return Result.Err(tryResult.error)
                                            }
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sqr()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            check(
                when (val tryResult = diff.toVec0<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } < 1e-6,
            ) { "batch $idx differs" }
        }
    }
    return Result.Ok(Unit)
}

public fun zeroMatmul(device: Device): Result<Unit> {
    val lhs =
        when (val tryResult = Tensor.zeros(Pair(2, 0), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = Tensor.zeros(Pair(0, 3), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = lhs.matmul(rhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(output.dims() == listOf(2, 3))
    check(
        when (val tryResult = output.toVec2<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } == listOf(listOf(0.0, 0.0, 0.0), listOf(0.0, 0.0, 0.0)),
    )
    val lhs1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = Tensor.zeros(Triple(2, 3, 4), DType.F32, device)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.transpose(1, 2)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(1, 0, 0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = Tensor.zeros(Triple(2, 4, 3), DType.F32, device)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.transpose(1, 2)
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.narrow(2, 0, 0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(!lhs1.isContiguous())
    check(!rhs1.isContiguous())
    check(lhs1.dims() == listOf(2, 0, 3))
    check(rhs1.dims() == listOf(2, 3, 0))
    check(
        when (val tryResult = lhs1.matmul(rhs1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }.dims() == listOf(2, 0, 0),
    )
    return Result.Ok(Unit)
}

public fun assertMatmulError(device: Device, lhs: Pair<List<Int>, DType>, rhs: Pair<List<Int>, DType>, expected: String): Result<Unit> {
    val lhs1 =
        when (val tryResult = Tensor.zeros(lhs.first, lhs.second, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs1 =
        when (val tryResult = Tensor.zeros(rhs.first, rhs.second, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err = lhs1.matmul(rhs1).unwrapErr()
    check(err.toString().startsWith(expected)) { "unexpected error: $err" }
    return Result.Ok(Unit)
}

public fun zeroMatmulValidation(device: Device): Result<Unit> {
    val shapeError = "shape mismatch in matmul"
    when (val tryResult = assertMatmulError(device, Pair(listOf(0, 2), F32), Pair(listOf(3, 4), F16), shapeError)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = assertMatmulError(device, Pair(listOf(2, 3), F32), Pair(listOf(4, 0), F32), shapeError)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = assertMatmulError(device, Pair(listOf(0, 2, 3), F32), Pair(listOf(1, 3, 4), F32), shapeError)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    when (val tryResult = assertMatmulError(device, Pair(listOf(0, 2), F32), Pair(listOf(2, 3), F16), "dtype mismatch in matmul")) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

public fun zeroMatmulDeviceValidation(device: Device): Result<Unit> {
    if (device.isCpu()) {
        return Result.Ok(Unit)
    }
    val lhs =
        when (val tryResult = Tensor.zeros(Pair(0, 2), DType.F32, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = Tensor.zeros(Pair(2, 3), DType.F16, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val err = lhs.matmul(rhs).unwrapErr()
    check(err.toString().startsWith("device mismatch in matmul")) { "unexpected error: $err" }
    return Result.Ok(Unit)
}

// A rank-2 rhs is broadcast over the batch dims only, which `broadcastMatmul` folds into a
// single 2D matmul instead of copying the rhs. Check the folded path against the per-batch
// products it stands for, contiguous lhs (folded) and non-contiguous lhs (fallback) alike.
public fun broadcastMatmulRank2Rhs(device: Device): Result<Unit> {
    val rhs =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Pair(5, 2), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (lhs in listOf(
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Triple(3, 4, 5), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        },
        when (val tryResult = Tensor.randn(0.0f, 1.0f, listOf(3, 6, 4, 5), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        },
        when (val tryResult = Tensor.randn(0.0f, 1.0f, Triple(1, 1, 5), device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        },
        when (
            val tryResult =
                when (val tryResult = Tensor.randn(0.0f, 1.0f, Triple(3, 5, 4), device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(1, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        },
    )) {
        val out =
            when (val tryResult = lhs.broadcastMatmul(rhs)) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val dims = lhs.dims().toVec()
        val n = dims.len()
        dims[n - 1] = 2
        check(out.dims() == dims.asSlice())
        // Same product, computed the way the doc comment describes it.
        val batch: Int = ((lhs.dims()).slice(0 until n - 2)).iter().product()
        val (m, k) = Pair(lhs.dims()[n - 2], lhs.dims()[n - 1])
        val flat =
            when (val tryResult = lhs.reshape(Triple(batch, m, k))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        val out1 =
            when (val tryResult = out.reshape(Triple(batch, m, 2))) {
                is Result.Ok<*> -> tryResult.value
                is Result.Err<*> -> return Result.Err(tryResult.error)
            }
        for (b in 0 until batch) {
            val diff =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (
                                    val tryResult = (
                                        when (val tryResult = out1.i(b)) {
                                            is Result.Ok<*> -> tryResult.value
                                            is Result.Err<*> -> return Result.Err(tryResult.error)
                                        } -
                                            when (
                                                val tryResult =
                                                    when (val tryResult = flat.i(b)) {
                                                        is Result.Ok<*> -> tryResult.value
                                                        is Result.Err<*> -> return Result.Err(tryResult.error)
                                                    }.matmul(rhs)
                                            ) {
                                                is Result.Ok<*> -> tryResult.value
                                                is Result.Err<*> -> return Result.Err(tryResult.error)
                                            }
                                    )
                                ) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.sqr()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            check(
                when (val tryResult = diff.toVec0<Float>()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                } < 1e-6,
            )
        }
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun tensorDot(): Result<Unit> {
    val lhs =
        when (val tryResult = Tensor.new(listOf(1.0, 2.0, 3.0), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val rhs =
        when (val tryResult = Tensor.new(listOf(4.0, 5.0, 6.0), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expected =
        when (val tryResult = Tensor.new(32.0, Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val dotRet =
        when (val tryResult = lhs.dot(rhs)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(dotRet, expected)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

@kotlin.test.Test
public fun tensorMv(): Result<Unit> {
    val mat =
        when (val tryResult = Tensor.new(listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val vec =
        when (val tryResult = Tensor.new(listOf(1.0, 1.0, 1.0), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expected =
        when (val tryResult = Tensor.new(listOf(6.0, 15.0), Device.Cpu)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mvRet =
        when (val tryResult = mat.mv(vec)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    when (val tryResult = assertTensorEq(mvRet, expected)) {
        is Result.Ok<*> -> tryResult.value
        is Result.Err<*> -> return Result.Err(tryResult.error)
    }
    return Result.Ok(Unit)
}

// https://github.com/huggingface/candle/issues/1948
public fun squeezeMm(device: Device): Result<Unit> {
    val seqLen = 8
    val a =
        when (val tryResult = Tensor.zeros(Triple(1, seqLen, 16), DType.F32, device)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x =
        when (val tryResult = a.i(Triple(0 until 0, seqLen - 1, 0 until 0))) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val w =
        when (
            val tryResult =
                when (val tryResult = Tensor.zeros(Pair(32, 16), DType.F32, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.t()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val x1 =
        when (val tryResult = x.matmul(w)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(x1.dims() == listOf(1, 32))
    return Result.Ok(Unit)
}

// https://github.com/huggingface/candle/issues/1992
public fun mmLayout(device: Device): Result<Unit> {
    val a =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val b =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 8.0f, device)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 2))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm1 =
        when (val tryResult = a.matmul(b)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // Forces the layout to be:
    // shape: [1, 1, 4, 2], stride: [8, 2, 2, 1], startOffset: 0
    // This is still a contiguous matrix but matmul checks are only the two last dimensions have
    // non 1 sizes but matmul check may be reluctant to handle it.
    val b1 =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = b.transpose(1, 2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.forceContiguous()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.transpose(1, 2)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val mm2 =
        when (val tryResult = a.matmul(b1)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (mm1 - mm2)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.sumAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec0<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(diff == 0.0)
    return Result.Ok(Unit)
}
