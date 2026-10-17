// port-lint: source cpu_backend/utils.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.candle.CandleException as Error

// Helper functions to write CPU kernels.

public interface Map1 {
    public fun f(
        vs: List<Double>,
        layout: Layout,
    ): List<Double>

    public fun map(
        vs: CpuStorage,
        layout: Layout,
    ): CpuStorage =
        when (vs) {
            is CpuStorage.U8 -> Ok(CpuStorage.U8(self.f(vs, layout)))

            is CpuStorage.U32 -> Ok(CpuStorage.U32(self.f(vs, layout)))

            is CpuStorage.I16 -> Ok(CpuStorage.I16(self.f(vs, layout)))

            is CpuStorage.I32 -> Ok(CpuStorage.I32(self.f(vs, layout)))

            is CpuStorage.I64 -> Ok(CpuStorage.I64(self.f(vs, layout)))

            is CpuStorage.BF16 -> Ok(CpuStorage.BF16(self.f(vs, layout)))

            is CpuStorage.F16 -> Ok(CpuStorage.F16(self.f(vs, layout)))

            is CpuStorage.F32 -> Ok(CpuStorage.F32(self.f(vs, layout)))

            is CpuStorage.F64 -> Ok(CpuStorage.F64(self.f(vs, layout)))

            is CpuStorage.F8E4M3 -> Ok(CpuStorage.F8E4M3(self.f(vs, layout)))

            // Dummy types don't support Map1 operations
            is CpuStorage.F6E2M3 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())

            is CpuStorage.F6E3M2 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())

            is CpuStorage.F4 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())

            is CpuStorage.F8E8M0 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1").bt())
        }
}

public interface Map1Any {
    public fun f(
        vs: List<Double>,
        layout: Layout,
        wrap: (List<Double>) -> CpuStorage,
    ): CpuStorage

    public fun map(
        vs: CpuStorage,
        layout: Layout,
    ): CpuStorage =
        when (vs) {
            is CpuStorage.U8 -> Ok(self.f(vs, layout, CpuStorage::U8))

            is CpuStorage.U32 -> Ok(self.f(vs, layout, CpuStorage::U32))

            is CpuStorage.I16 -> Ok(self.f(vs, layout, CpuStorage::I16))

            is CpuStorage.I32 -> Ok(self.f(vs, layout, CpuStorage::I32))

            is CpuStorage.I64 -> Ok(self.f(vs, layout, CpuStorage::I64))

            is CpuStorage.BF16 -> Ok(self.f(vs, layout, CpuStorage::BF16))

            is CpuStorage.F16 -> Ok(self.f(vs, layout, CpuStorage::F16))

            is CpuStorage.F32 -> Ok(self.f(vs, layout, CpuStorage::F32))

            is CpuStorage.F64 -> Ok(self.f(vs, layout, CpuStorage::F64))

            is CpuStorage.F8E4M3 -> Ok(self.f(vs, layout, CpuStorage::F8E4M3))

            // Dummy types don't support Map1Any operations
            is CpuStorage.F6E2M3 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())

            is CpuStorage.F6E3M2 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())

            is CpuStorage.F4 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())

            is CpuStorage.F8E8M0 -> Err(Error.UnsupportedDTypeForOp(vs.dtype(), "map1any").bt())
        }
}

public interface Map2 {
    public val op: String

    public fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): List<Double>

    public fun map(
        v1: CpuStorage,
        l1: Layout,
        v2: CpuStorage,
        l2: Layout,
    ): CpuStorage =
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> {
                Ok(CpuStorage.U32(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> {
                Ok(CpuStorage.I16(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> {
                Ok(CpuStorage.I32(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> {
                Ok(CpuStorage.I64(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> {
                Ok(CpuStorage.BF16(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> {
                Ok(CpuStorage.F16(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> {
                Ok(CpuStorage.F32(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> {
                Ok(CpuStorage.F64(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> {
                Ok(CpuStorage.F8E4M3(self.f(v1, l1, v2, l2)))
            }

            else -> {
                Err(
                    Error
                        .DTypeMismatchBinaryOp(
                            v1.dtype(),
                            v2.dtype(),
                            self.OP,
                        ).bt(),
                )
            }
        }
}

public interface Map2InPlace {
    public val op: String

    public fun f(
        v1: MutableList<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    )

    public fun map(
        v1: CpuStorage,
        l1: Layout,
        v2: CpuStorage,
        l2: Layout,
    ) {
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> {
                self.f(v1, l1, v2, l2)
            }

            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> {
                self.f(v1, l1, v2, l2)
            }

            else -> {
                Err(
                    Error
                        .DTypeMismatchBinaryOp(
                            v1.dtype(),
                            v2.dtype(),
                            self.OP,
                        ).bt(),
                )
            }
        }
        Ok(Unit)
    }
}

public interface Map2U8 {
    public val op: String

    public fun f(
        v1: List<Double>,
        l1: Layout,
        v2: List<Double>,
        l2: Layout,
    ): UByteArray

    public fun map(
        v1: CpuStorage,
        l1: Layout,
        v2: CpuStorage,
        l2: Layout,
    ): CpuStorage =
        when {
            v1 is CpuStorage.U8 && v2 is CpuStorage.U8 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.U32 && v2 is CpuStorage.U32 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.I16 && v2 is CpuStorage.I16 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.I32 && v2 is CpuStorage.I32 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.I64 && v2 is CpuStorage.I64 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.BF16 && v2 is CpuStorage.BF16 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F16 && v2 is CpuStorage.F16 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F32 && v2 is CpuStorage.F32 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F64 && v2 is CpuStorage.F64 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            v1 is CpuStorage.F8E4M3 && v2 is CpuStorage.F8E4M3 -> {
                Ok(CpuStorage.U8(self.f(v1, l1, v2, l2)))
            }

            else -> {
                Err(
                    Error
                        .DTypeMismatchBinaryOp(
                            v1.dtype(),
                            v2.dtype(),
                            self.OP,
                        ).bt(),
                )
            }
        }
}

public fun <T, U> binaryMap(
    lhsL: Layout,
    rhsL: Layout,
    lhs: List<T>,
    rhs: List<T>,
    f: (T, T) -> U,
): List<U> {
    val elCount = lhsL.shape().elemCount()
    val result: MutableList<U> = Vec.withCapacity(elCount)

    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (innerLs, innerRs) = ndIter.innerStrides

    for ((lhsOff, rhsOff) in ndIter) {
        for (i in 0 until innerSize) {
            result.push(f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs]))
        }
    }

    return result
}

// Similar to binaryMap but with vectorized variants.
public fun <
    T,
> binaryMapVec(
    lhsL: Layout,
    rhsL: Layout,
    lhs: List<T>,
    rhs: List<T>,
    f: (T, T) -> T,
    fVec: (List<T>, List<T>, MutableList<T>) -> Unit,
    fScalarVec: (T, List<T>, MutableList<T>) -> Unit,
): List<T> {
    val elCount = lhsL.shape().elemCount()
    val ys: VecList<T> = Vec.withCapacity(elCount)
    val ysToSet =
        run {
            val s = ys.spareCapacityMut()

            s
        }

    val ndIter = NdIter.new(listOf(lhsL, rhsL))
    val innerSize = ndIter.innerSize
    val (innerLs, innerRs) = ndIter.innerStrides

    var dstOff = 0

    for ((lhsOff, rhsOff) in ndIter) {
        when (innerLs to innerRs) {
            1 to 1 -> {
                fVec(
                    lhs.subList(lhsOff, lhsOff + innerSize),
                    rhs.subList(rhsOff, rhsOff + innerSize),
                    ysToSet.subList(dstOff, dstOff + innerSize),
                )
            }

            1 to 0 -> {
                val r = rhs[rhsOff]
                fScalarVec(
                    r,
                    lhs.subList(lhsOff, lhsOff + innerSize),
                    ysToSet.subList(dstOff, dstOff + innerSize),
                )
            }

            0 to 1 -> {
                val l = lhs[lhsOff]
                for (i in 0 until innerSize) {
                    ysToSet[dstOff + i] = f(l, rhs[rhsOff + i])
                }
            }

            else -> {
                for (i in 0 until innerSize) {
                    ysToSet[dstOff + i] =
                        f(lhs[lhsOff + i * innerLs], rhs[rhsOff + i * innerRs])
                }
            }
        }
        dstOff += innerSize
    }

    // All elCount elements have been written in the dispatch loop above.
    ys.setLen(elCount)
    return ys
}

public fun <T, U> unaryMap(
    vs: List<T>,
    layout: Layout,
    f: (T) -> U,
): List<U> =
    when (val b = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            vs
                .subList(b.startOffset, b.startOffset + b.len)
                .iter()
                .map { v -> f(v) }
                .collect()
        }

        is StridedBlocks.UniformBlocks -> {
            val startOffset = b.startOffset
            val blockLen = b.blockLen
            val count = b.count
            val srcStride = b.srcStride
            val result: MutableList<U> = Vec.withCapacity(count * blockLen)
            if (blockLen == 1) {
                for (i in 0 until count) {
                    val v = vs.getUnchecked(startOffset + i * srcStride)
                    result.push(f(v))
                }
            } else {
                for (i in 0 until count) {
                    val srcStart = startOffset + i * srcStride
                    for (offset in 0 until blockLen) {
                        val v = vs.getUnchecked(srcStart + offset)
                        result.push(f(v))
                    }
                }
            }
            result
        }

        is StridedBlocks.MultipleBlocks -> {
            val (
                blockStartIndex,
                blockLen,
            ) = b
            val result: MutableList<U> = Vec.withCapacity(layout.shape().elemCount())
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen == 1) {
                for (index in blockStartIndex) {
                    val v = vs.getUnchecked(index)
                    result.push(f(v))
                }
            } else {
                for (index in blockStartIndex) {
                    for (offset in 0 until blockLen) {
                        val v = vs.getUnchecked(index + offset)
                        result.push(f(v))
                    }
                }
            }
            result
        }
    }

public fun <T, U> unaryMapVec(
    vs: List<T>,
    layout: Layout,
    f: (T) -> U,
    fVec: (List<T>, MutableList<U>) -> Unit,
): List<U> =
    when (val b = layout.stridedBlocks()) {
        is StridedBlocks.SingleBlock -> {
            val (startOffset, len) = b
            val ys: VecList<U> = Vec.withCapacity(len)
            val ysToSet =
                ys.spareCapacityMut()

            fVec(vs.subList(startOffset, startOffset + len), ysToSet)
            // Values are all set by fVec.
            ys.setLen(len)
            ys
        }

        is StridedBlocks.UniformBlocks -> {
            val startOffset = b.startOffset
            val blockLen = b.blockLen
            val count = b.count
            val srcStride = b.srcStride
            val elCount = count * blockLen
            if (blockLen == 1) {
                val result: MutableList<U> = Vec.withCapacity(count)
                for (i in 0 until count) {
                    val v = vs.getUnchecked(startOffset + i * srcStride)
                    result.push(f(v))
                }
                result
            } else {
                val ys: VecList<U> = Vec.withCapacity(elCount)
                val ysToSet =
                    ys.spareCapacityMut()

                var dstIndex = 0
                for (i in 0 until count) {
                    val srcStart = startOffset + i * srcStride
                    fVec(
                        vs.subList(srcStart, srcStart + blockLen),
                        ysToSet.subList(dstIndex, dstIndex + blockLen),
                    )
                    dstIndex += blockLen
                }
                // Values are all set by fVec.
                ys.setLen(elCount)
                ys
            }
        }

        is StridedBlocks.MultipleBlocks -> {
            val (
                blockStartIndex,
                blockLen,
            ) = b
            val elCount = layout.shape().elemCount()
            // Specialize the case where blockLen is one to avoid the second loop.
            if (blockLen == 1) {
                val result: MutableList<U> = Vec.withCapacity(elCount)
                for (index in blockStartIndex) {
                    val v = vs.getUnchecked(index)
                    result.push(f(v))
                }
                result
            } else {
                val ys: VecList<U> = Vec.withCapacity(elCount)
                val ysToSet =
                    ys.spareCapacityMut()

                var dstIndex = 0
                for (srcIndex in blockStartIndex) {
                    val vsSlice = vs.subList(srcIndex, srcIndex + blockLen)
                    val ysSlice = ysToSet.subList(dstIndex, dstIndex + blockLen)
                    fVec(vsSlice, ysSlice)
                    dstIndex += blockLen
                }
                // Values are all set by fVec.
                ys.setLen(elCount)
                ys
            }
        }
    }
