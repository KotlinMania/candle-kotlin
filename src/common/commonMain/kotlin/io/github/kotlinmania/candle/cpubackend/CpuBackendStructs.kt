package io.github.kotlinmania.candle.cpubackend

import io.github.kotlinmania.candle.accelerate.dgemm
import io.github.kotlinmania.candle.accelerate.sgemm
import io.github.kotlinmania.candle.backend.BackendDevice
import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.utils.Map1
import io.github.kotlinmania.candle.cpubackend.utils.Map1Any
import io.github.kotlinmania.candle.cpubackend.utils.Map2
import io.github.kotlinmania.candle.cpubackend.utils.Map2InPlace
import io.github.kotlinmania.candle.cpubackend.utils.Map2U8
import io.github.kotlinmania.candle.cpubackend.utils.binaryMap
import io.github.kotlinmania.candle.cpubackend.utils.binaryMapVec
import io.github.kotlinmania.candle.cpubackend.utils.unaryMap
import io.github.kotlinmania.candle.device.DeviceLocation
import io.github.kotlinmania.candle.dims3
import io.github.kotlinmania.candle.dims4
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.IntDType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.MatMulUnexpectedStriding
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.mkl.hgemm
import io.github.kotlinmania.candle.nditer.NdIter
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.utils.getNumThreads
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.gemm.Parallelism
import io.github.kotlinmania.gemm.gemm
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.numtraits.Float
import io.github.kotlinmania.numtraits.Num
import io.github.kotlinmania.rand.prelude.*
import io.github.kotlinmania.rand.rng
import io.github.kotlinmania.rayon.prelude.*
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.mem.mutPtrFromRustCast
import io.github.kotlinmania.rustlang.core.result.fromResidual

public typealias Storage = CpuStorage

public class CpuDevice : BackendDevice {
    public fun location(): DeviceLocation = DeviceLocation.Cpu

    public fun sameDevice(arg0: CpuDevice): Boolean = true

    public fun <T : WithDType> storageFromSlice(s: List<T>, toCpuStorageTFn: (Any?) -> Any?): Result<CpuStorage> = Ok(toCpuStorageTFn(s))

    public fun storageFromCpuStorage(s: CpuStorage): Result<CpuStorage> = Ok(s.clone())

    public fun storageFromCpuStorageOwned(s: CpuStorage): Result<CpuStorage> = Ok(s)

    public fun setSeed(seed: ULong): Result<Unit> = bail("cannot seed the CPU rng with set_seed")

    public fun getCurrentSeed(): Result<ULong> = bail("cannot get the CPU rng seed with get_current_seed")

    public fun randUniform(shape: Shape, dtype: DType, min: Double, max: Double): Result<CpuStorage> {
        val elemCount = shape.elemCount()
        val rng = rng()
        return when (dtype) {
            DType.u8, DType.u32, DType.i16, DType.i32, DType.i64, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> Err(Error.UnsupportedDTypeForOp(dtype, "rand_uniform").bt())
            DType.bf16 -> {
                val data = ArrayList(elemCount)
                val uniform =
                    when (val tryResult = Uniform.new(fromF64(min), fromF64(max)).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i in 0 until elemCount) {
                    data.add(rng.sample<Bf16>(uniform))
                }
                Ok(CpuStorage.bf16(data))
            }
            DType.f16 -> {
                val data1 = ArrayList(elemCount)
                val uniform1 =
                    when (val tryResult = Uniform.new(fromF64(min), fromF64(max)).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i1 in 0 until elemCount) {
                    data1.add(rng.sample<F16>(uniform1))
                }
                Ok(CpuStorage.f16(data1))
            }
            DType.f8e4m3 -> {
                val data2 = ArrayList(elemCount)
                val uniform2 =
                    when (val tryResult = Uniform.new(F8e4m3.fromF64(min), F8e4m3.fromF64(max)).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i2 in 0 until elemCount) {
                    data2.add(rng.sample<F8e4m3>(uniform2))
                }
                Ok(CpuStorage.f8e4m3(data2))
            }
            DType.f32 -> {
                val data3 = ArrayList(elemCount)
                val uniform3 =
                    when (val tryResult = Uniform.new(min.toFloat(), max.toFloat()).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i3 in 0 until elemCount) {
                    data3.add(rng.sample<Float>(uniform3))
                }
                Ok(CpuStorage.f32(data3))
            }
            DType.f64 -> {
                val data4 = ArrayList(elemCount)
                val uniform4 =
                    when (val tryResult = Uniform.new(min, max).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i4 in 0 until elemCount) {
                    data4.add(rng.sample<Double>(uniform4))
                }
                Ok(CpuStorage.f64(data4))
            }
        }
    }

    public fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): Result<CpuStorage> {
        val elemCount = shape.elemCount()
        val rng = rng()
        return when (dtype) {
            DType.u8, DType.u32, DType.i16, DType.i32, DType.i64, DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> Err(Error.UnsupportedDTypeForOp(dtype, "rand_normal").bt())
            DType.bf16 -> {
                val data = ArrayList(elemCount)
                val normal =
                    when (val tryResult = Normal.new(fromF64(mean), fromF64(std)).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i in 0 until elemCount) {
                    data.add(normal.sample(rng))
                }
                Ok(CpuStorage.bf16(data))
            }
            DType.f16 -> {
                val data1 = ArrayList(elemCount)
                val normal1 =
                    when (val tryResult = Normal.new(fromF64(mean), fromF64(std)).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i1 in 0 until elemCount) {
                    data1.add(normal1.sample(rng))
                }
                Ok(CpuStorage.f16(data1))
            }
            DType.f8e4m3 -> {
                val data2 = ArrayList(elemCount)
                val normal2 =
                    when (val tryResult = Normal.new(F8e4m3.fromF64(mean), F8e4m3.fromF64(std)).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i2 in 0 until elemCount) {
                    data2.add(normal2.sample(rng))
                }
                Ok(CpuStorage.f8e4m3(data2))
            }
            DType.f32 -> {
                val data3 = ArrayList(elemCount)
                val normal3 =
                    when (val tryResult = Normal.new(mean.toFloat(), std.toFloat()).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i3 in 0 until elemCount) {
                    data3.add(normal3.sample(rng))
                }
                Ok(CpuStorage.f32(data3))
            }
            DType.f64 -> {
                val data4 = ArrayList(elemCount)
                val normal4 =
                    when (val tryResult = Normal.new(mean, std).mapErr(Error::wrap)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for (i4 in 0 until elemCount) {
                    data4.add(normal4.sample(rng))
                }
                Ok(CpuStorage.f64(data4))
            }
        }
    }

    public fun allocUninit(shape: Shape, dtype: DType): Result<CpuStorage> {
        val elemCount = shape.elemCount()
        // The code below is highly unchecked but hopefully not directly unsound as we only consider
        // types that are Copy, not Drop, and for which all bit patterns are proper values.
        // It's still pretty risky, see the following for more details:
        // https://github.com/rust-lang/rust-clippy/issues/4483
        val storage =
            when (dtype) {
                DType.u8 -> {
                    val v = ArrayList(elemCount)
                    v.setLen(elemCount)
                    CpuStorage.u8(v)
                }
                DType.u32 -> {
                    val v1 = ArrayList(elemCount)
                    v1.setLen(elemCount)
                    CpuStorage.u32(v1)
                }
                DType.i16 -> {
                    val v2 = ArrayList(elemCount)
                    v2.setLen(elemCount)
                    CpuStorage.i16(v2)
                }
                DType.i32 -> {
                    val v3 = ArrayList(elemCount)
                    v3.setLen(elemCount)
                    CpuStorage.i32(v3)
                }
                DType.i64 -> {
                    val v4 = ArrayList(elemCount)
                    v4.setLen(elemCount)
                    CpuStorage.i64(v4)
                }
                DType.bf16 -> {
                    val v5 = ArrayList(elemCount)
                    v5.setLen(elemCount)
                    CpuStorage.bf16(v5)
                }
                DType.f16 -> {
                    val v6 = ArrayList(elemCount)
                    v6.setLen(elemCount)
                    CpuStorage.f16(v6)
                }
                DType.f32 -> {
                    val v7 = ArrayList(elemCount)
                    v7.setLen(elemCount)
                    CpuStorage.f32(v7)
                }
                DType.f64 -> {
                    val v8 = ArrayList(elemCount)
                    v8.setLen(elemCount)
                    CpuStorage.f64(v8)
                }
                DType.f8e4m3 -> {
                    val v9 = ArrayList(elemCount)
                    v9.setLen(elemCount)
                    CpuStorage.f8e4m3(v9)
                }
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Err(Error.UnsupportedDTypeForOp(dtype, "alloc_uninit").bt())
                }
            }
        return Ok(storage)
    }

    public fun zerosImpl(shape: Shape, dtype: DType): Result<CpuStorage> {
        val elemCount = shape.elemCount()
        val storage =
            when (dtype) {
                DType.u8 -> CpuStorage.u8(mutableListOf(0u.toUByte(), elemCount))
                DType.u32 -> CpuStorage.u32(mutableListOf(0u, elemCount))
                DType.i16 -> CpuStorage.i16(mutableListOf(0, elemCount))
                DType.i32 -> CpuStorage.i32(mutableListOf(0, elemCount))
                DType.i64 -> CpuStorage.i64(mutableListOf(0L, elemCount))
                DType.bf16 -> CpuStorage.bf16(mutableListOf(ZERO, elemCount))
                DType.f16 -> CpuStorage.f16(mutableListOf(ZERO, elemCount))
                DType.f32 -> CpuStorage.f32(mutableListOf(0.0f, elemCount))
                DType.f64 -> CpuStorage.f64(mutableListOf(0.0, elemCount))
                DType.f8e4m3 -> CpuStorage.f8e4m3(mutableListOf(F8e4m3.ZERO, elemCount))
                DType.f6e2m3, DType.f6e3m2, DType.f4, DType.f8e8m0 -> {
                    return Err(Error.UnsupportedDTypeForOp(dtype, "zeros").bt())
                }
            }
        return Ok(storage)
    }

    public fun synchronize(): Result<Unit> = Ok(Unit)

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CpuDevice> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CpuDevice> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CpuDevice =
                    CpuDevice()
                override fun writeNative(addr: Long, value: CpuDevice) {
                }
            }

        public fun new(arg0: ULong): Result<CpuDevice> = Ok(CpuDevice)
    }
}

public class Cmp(
    public var value: CmpOp = CmpOp.entries.first(),
) : Map2U8 {
    public var field0: CmpOp
        get() = value
        set(v) {
            value = v
        }

    public fun <T : WithDType> f(lhs: List<T>, lhsL: Layout, rhs: List<T>, rhsL: Layout): Result<List<UByte>> {
        val dst =
            when (this.value) {
                CmpOp.Eq -> binaryMap(lhsL, rhsL, lhs, rhs, { x, y -> UByte.from(x == y) })
                CmpOp.Ne -> binaryMap(lhsL, rhsL, lhs, rhs, { x, y -> UByte.from(x != y) })
                CmpOp.Lt -> binaryMap(lhsL, rhsL, lhs, rhs, { x, y -> UByte.from(x < y) })
                CmpOp.Le -> binaryMap(lhsL, rhsL, lhs, rhs, { x, y -> UByte.from(x <= y) })
                CmpOp.Gt -> binaryMap(lhsL, rhsL, lhs, rhs, { x, y -> UByte.from(x > y) })
                CmpOp.Ge -> binaryMap(lhsL, rhsL, lhs, rhs, { x, y -> UByte.from(x >= y) })
            }
        return Ok(dst)
    }

    public companion object {
        public val OP: String = "cmp"
    }
}

public class WCond<T : IntDType>(
    public var field0: List<T> = emptyList(),
    public var field1: Layout = Layout(),
)

public val OP: String = "where"

public fun <I : IntDType, T : WithDType> WCond<I>.f(t: List<T>, tL: Layout, f: List<T>, fL: Layout): Result<List<T>> {
    val vs =
        run {
            val matchVal = Triple(this.field1.contiguousOffsets(), tL.contiguousOffsets(), fL.contiguousOffsets())
            when {
                matchVal.first != null && matchVal.second != null && matchVal.third != null -> {
                    val o1 = matchVal.first.first
                    val o2 = matchVal.first.second
                    val oT1 = matchVal.second.first
                    val oT2 = matchVal.second.second
                    val oF1 = matchVal.third.first
                    val oF2 = matchVal.third.second
                    val pred = ((this.value).slice(o1 until o2))
                    val t1 = ((t).slice(oT1 until oT2))
                    val f1 = ((f).slice(oF1 until oF2))
                    (pred.zip((t1.iter().zip(f1.iter()))))
                        .map({ (p, tf) ->
                            val (t2, f2) = tf
                            if (p.isTrue()) {
                                t2
                            } else {
                                f2
                            }
                        })
                        .collect<Vec>()
                }
                else -> {
                    // Same allocation strategy as `binaryMapVec`
                    val elCount = this.field1.shape().elemCount()
                    val ys: List<T> = ArrayList(elCount)
                    val ysToSet =
                        run {
                            val s = ys.spareCapacityMut()
                            transmute<List<T?>, List<T>>(s)
                        }
                    val ndIter = NdIter.new(listOf(this.field1, tL, fL))
                    val (innerPs, innerTs, innerFs) = ndIter.innerStrides
                    val offset = { off, i -> off + i }
                    val boffset = { off -> off }
                    run {
                        val matchVal1 = Triple(innerPs, innerTs, innerFs)
                        when {
                            matchVal1.first == 1 && matchVal1.second == 1 && matchVal1.third == 1 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, offset, offset, offset)
                            }
                            matchVal1.first == 1 && matchVal1.second == 1 && matchVal1.third == 0 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, offset, offset, boffset)
                            }
                            matchVal1.first == 1 && matchVal1.second == 0 && matchVal1.third == 1 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, offset, boffset, offset)
                            }
                            matchVal1.first == 1 && matchVal1.second == 0 && matchVal1.third == 0 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, offset, boffset, boffset)
                            }
                            matchVal1.first == 0 && matchVal1.second == 1 && matchVal1.third == 1 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, boffset, offset, offset)
                            }
                            matchVal1.first == 0 && matchVal1.second == 1 && matchVal1.third == 0 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, boffset, offset, boffset)
                            }
                            matchVal1.first == 0 && matchVal1.second == 0 && matchVal1.third == 1 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, boffset, boffset, offset)
                            }
                            matchVal1.first == 0 && matchVal1.second == 0 && matchVal1.third == 0 -> {
                                whereImpl(ndIter, ysToSet, this.value, t, f, boffset, boffset, boffset)
                            }
                            else -> {
                                val offset1 = { off, i, inner -> off + i * inner }
                                whereImpl(ndIter, ysToSet, this.value, t, f, offset1, offset1, offset1)
                            }
                        }
                    }
                    // SAFETY: all elCount elements have been written in the dispatch loop above.
                    ys.setLen(elCount)
                    ys
                }
            }
        }
    return Ok(vs)
}

public class ReduceIndex(
    public var reduceDimIndex: Int = 0,
    public var useMin: Boolean = false,
    public var returnIndex: Boolean = false,
) : Map1Any {
    // The value gets replaced if f(s[currentAcc], s[i]) returns true.
    public fun <T, U> foldImpl(src: List<T>, srcL: Layout, f: (T, T) -> Boolean, g: (T, Int) -> U): Result<List<U>> {
        val reduceDimSize = srcL.dims()[this.reduceDimIndex]
        val reduceDimStride = srcL.stride()[this.reduceDimIndex]
        val dstLen = srcL.shape().elemCount() / reduceDimSize
        val dst: List<U> = ArrayList(dstLen)
        val dstToSet = dst.spareCapacityMut()
        val dstToSet1 = transmute<List<U?>, List<U>>(dstToSet)
        when (val matchVal = srcL.contiguousOffsets()) {
            null -> {
                val l =
                    when (val tryResult = srcL.narrow(this.reduceDimIndex, 0, 1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                for ((unstrIndex, srcIndex) in l.stridedIndex().enumerate()) {
                    val src1 = ((src).slice(srcIndex until src.size))
                    var acc = 0
                    var value = src1[0]
                    for (srcI in 0 until reduceDimSize) {
                        val s = src1[srcI * reduceDimStride]
                        if (f(value, s)) {
                            acc = srcI
                            value = s
                        }
                    }
                    dstToSet1[unstrIndex] = g(value, acc)
                }
            }
            else -> {
                val o1 = matchVal.first
                val o2 = matchVal.second
                val src2 = ((src).slice(o1 until o2))
                if (reduceDimStride == 1) {
                    for ((startSrcI, dstV) in dstToSet1.iterMut().enumerate()) {
                        val startSrcI1 = startSrcI * reduceDimSize
                        val src3 = ((src2).slice(startSrcI1 until startSrcI1 + reduceDimSize))
                        var acc1 = 0
                        var value1 = src3[0]
                        for ((srcI1, s1) in src3.iter().enumerate()) {
                            if (f(value1, s1)) {
                                acc1 = srcI1
                                value1 = s1
                            }
                        }
                        dstV = g(value1, acc1)
                    }
                } else {
                    for ((startSrcI2, dstV1) in dstToSet1.iterMut().enumerate()) {
                        val (p, q) = Pair(startSrcI2 / reduceDimStride, startSrcI2 % reduceDimStride)
                        // startSrcI = p * reduceDimStride + q
                        val startSrcI3 = p * reduceDimStride * reduceDimSize + q
                        val src4 = ((src2).slice(startSrcI3 until src2.size))
                        var acc2 = 0
                        var value2 = src4[0]
                        for (srcI2 in 0 until reduceDimSize) {
                            val s2 = src4[srcI2 * reduceDimStride]
                            if (f(value2, s2)) {
                                acc2 = srcI2
                                value2 = s2
                            }
                        }
                        dstV1 = g(value2, acc2)
                    }
                }
            }
        }
        dst.setLen(dstLen)
        return Ok(dst)
    }

    public fun <T : WithDType> f(src: List<T>, srcL: Layout, wrap: (List<T>) -> CpuStorage): Result<CpuStorage> {
        if (srcL.shape().elemCount() == 0) {
            when (val tryResult = Err(Error.EmptyTensor(op = "reduce").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val dst =
            run {
                val matchVal = Pair(this.returnIndex, this.useMin)
                when {
                    matchVal.first == false && matchVal.second == true ->
                        wrap(
                            when (val tryResult = this.foldImpl(src, srcL, { x, y -> x > y }, { v, i -> v })) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        )
                    matchVal.first == false && matchVal.second == false ->
                        wrap(
                            when (val tryResult = this.foldImpl(src, srcL, { x, y -> x < y }, { v, i -> v })) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        )
                    matchVal.first == true && matchVal.second == true -> {
                        CpuStorage.u32(
                            when (val tryResult = this.foldImpl(src, srcL, { x, y -> x > y }, { v, i -> i.toUInt() })) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        )
                    }
                    else -> {
                        CpuStorage.u32(
                            when (val tryResult = this.foldImpl(src, srcL, { x, y -> x < y }, { v, i -> i.toUInt() })) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        )
                    }
                }
            }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<ReduceIndex> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<ReduceIndex> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 16L
                override fun readNative(addr: Long): ReduceIndex =
                    ReduceIndex(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).readNative(addr + 9L),
                    )
                override fun writeNative(addr: Long, value: ReduceIndex) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.reduceDimIndex)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 8L, value.useMin)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.BOOL).writeNative(addr + 9L, value.returnIndex)
                }
            }
    }
}

public class ReduceSum(
    public var dstShape: Shape = Shape(),
    public var reduceDims: List<Int> = emptyList(),
    public var reduceDimsAndStride: List<Pair<Int, Int>> = emptyList(),
) : Map1 {
    public fun <T> foldImpl(src: List<T>, srcL: Layout, startElt: T, vecReduceSumTFn: (Any?) -> Any?): Result<List<T>> where T : WithDType {
        val dst = mutableListOf(startElt, this.dstShape.elemCount())
        when (val matchVal = srcL.contiguousOffsets()) {
            null -> {
                for ((unstrIndex, srcIndex) in srcL.stridedIndex().enumerate()) {
                    var dstIndex = unstrIndex
                    // Set the reduceDims indexes to 0.
                    for ((dim, stride) in ArrayDeque(this.reduceDimsAndStride)) {
                        // The compiler is able to optimize the following in a single divmod op.
                        val (pre, post) = Pair(dstIndex / stride, dstIndex % stride)
                        dstIndex = (pre / dim) * stride + post
                    }
                    dst[dstIndex] += src[srcIndex]
                }
            }
            else -> {
                val o1 = matchVal.first
                val o2 = matchVal.second
                val src1 = ((src).slice(o1 until o2))
                // Handle the case where we reduce over the last dimensions separately as it is
                // fairly common and easy to optimize. This rely on the layout being contiguous!
                // reduceDims is sorted, check if it is ranging from a to n-1.
                val reduceOverLastDims =
                    this.reduceDims
                        .iter()
                        .rev()
                        .enumerate()
                        .all({ (i, v) -> v == srcL.shape().rank() - 1 - i })
                if (reduceOverLastDims) {
                    val reduceSz = this.reduceDimsAndStride.map({ (u, _) -> u }).product<Int>()
                    for ((dstI, dstV) in ArrayDeque(dst).enumerate()) {
                        val srcI = dstI * reduceSz
                        vecReduceSumTFn(((src1).slice(srcI until srcI + reduceSz)).asPtr(), dstV, reduceSz)
                    }
                    return Ok(dst)
                }
                for ((unstrIndex1, src2) in src1.iter().enumerate()) {
                    var dstIndex1 = unstrIndex1
                    // Set the reduceDims indexes to 0.
                    for ((dim1, stride1) in ArrayDeque(this.reduceDimsAndStride)) {
                        // The compiler is able to optimize the following in a single divmod op.
                        val (pre1, post1) = Pair(dstIndex1 / stride1, dstIndex1 % stride1)
                        dstIndex1 = (pre1 / dim1) * stride1 + post1
                    }
                    dst[dstIndex1] += src2
                }
            }
        }
        return Ok(dst)
    }

    public fun <T : WithDType> f(src: List<T>, srcL: Layout, zeroTFn: (Any?) -> Any?): Result<List<T>> = this.foldImpl(src, srcL, zeroTFn())
}

public class Affine(
    public var field0: Double = 0.0,
    public var field1: Double = 0.0,
) : Map1 {
    public fun <T : WithDType> f(vs: List<T>, layout: Layout, fromF64TFn: (Any?) -> Any?): Result<List<T>> {
        val mul = fromF64TFn(this.value)
        val add = fromF64TFn(this.field1)
        return Ok(unaryMap(vs, layout, { v -> v * mul + add }))
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Affine> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Affine> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 16L
                override fun readNative(addr: Long): Affine =
                    Affine(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).readNative(addr + 8L),
                    )
                override fun writeNative(addr: Long, value: Affine) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr, value.field0)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.F64).writeNative(addr + 8L, value.field1)
                }
            }
    }
}

public class AvgPool2D(
    public var field0: Pair<Int, Int> = Pair<Int, Int>(0, 0),
    public var field1: Pair<Int, Int> = Pair<Int, Int>(0, 0),
) : Map1 {
    public fun <T : WithDType> f(src: List<T>, layout: Layout, fromF64TFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<List<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html
        val (kH, kW) = this.value
        val (sH, sW) = this.field1
        val (bSz, c, h, w) =
            when (val tryResult = layout.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val stride = layout.stride()
        val (strideH, strideW) = Pair(stride[2], stride[3])
        val hOut = (h - kH) / sH + 1
        val wOut = (w - kW) / sW + 1
        val srcIndex = layout.startOffset()
        val dst = mutableListOf(zeroTFn(), bSz * c * hOut * wOut)
        val scale = 1.0 / (kH * kW).toDouble()
        val scale1 = fromF64TFn(scale)
        for (bIdx in 0 until bSz) {
            val dst1 = ((dst).slice(bIdx * c * hOut * wOut until dst.size))
            val srcIndex1 = srcIndex + bIdx * stride[0]
            for (cIdx in 0 until c) {
                val dst2 = ((dst1).slice(cIdx * hOut * wOut until dst1.size))
                val srcIndex2 = srcIndex1 + cIdx * stride[1]
                for (hIdx in 0 until hOut) {
                    for (wIdx in 0 until wOut) {
                        var sum = zeroTFn()
                        for (m in 0 until kH) {
                            for (n in 0 until kW) {
                                val m1 = sH * hIdx + m
                                val n1 = sW * wIdx + n
                                sum += src[srcIndex2 + m1 * strideH + n1 * strideW]
                            }
                        }
                        dst2[hIdx * wOut + wIdx] = sum * scale1
                    }
                }
            }
        }
        return Ok(dst)
    }
}

public class MaxPool2D(
    public var field0: Pair<Int, Int> = Pair<Int, Int>(0, 0),
    public var field1: Pair<Int, Int> = Pair<Int, Int>(0, 0),
) : Map1 {
    public fun <T : WithDType> f(src: List<T>, layout: Layout): Result<List<T>> {
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        val (kH, kW) = this.value
        val (sH, sW) = this.field1
        val (bSz, c, h, w) =
            when (val tryResult = layout.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val stride = layout.stride()
        val (strideH, strideW) = Pair(stride[2], stride[3])
        val hOut = (h - kH) / sH + 1
        val wOut = (w - kW) / sW + 1
        val srcIndex = layout.startOffset()
        val dst = mutableListOf(T.zero(), bSz * c * hOut * wOut)
        for (bIdx in 0 until bSz) {
            val dst1 = ((dst).slice(bIdx * c * hOut * wOut until dst.size))
            val srcIndex1 = srcIndex + bIdx * stride[0]
            for (cIdx in 0 until c) {
                val dst2 = ((dst1).slice(cIdx * hOut * wOut until dst1.size))
                val srcIndex2 = srcIndex1 + cIdx * stride[1]
                for (hIdx in 0 until hOut) {
                    for (wIdx in 0 until wOut) {
                        var largest = src[srcIndex2 + sH * hIdx * strideH + sW * wIdx * strideW]
                        for (m in 0 until kH) {
                            for (n in 0 until kW) {
                                val m1 = sH * hIdx + m
                                val n1 = sW * wIdx + n
                                if (largest < src[srcIndex2 + m1 * strideH + n1 * strideW]) {
                                    largest = src[srcIndex2 + m1 * strideH + n1 * strideW]
                                }
                            }
                        }
                        dst2[hIdx * wOut + wIdx] = largest
                    }
                }
            }
        }
        return Ok(dst)
    }
}

public class UpsampleNearest1D(
    public var value: Int = 0,
) : Map1 {
    public var field0: Int
        get() = value
        set(v) {
            value = v
        }

    public fun <T : WithDType> f(src: List<T>, layout: Layout): Result<List<T>> {
        // Note: Specialized implementation for the case 2*sz?
        val dstSz = this.value
        val (bSz, c, srcSz) =
            when (val tryResult = layout.shape().dims3()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val stride = layout.stride()
        val strideSz = stride[2]
        val srcIndex = layout.startOffset()
        val scaleSz = srcSz.toDouble() / dstSz.toDouble()
        val dst = mutableListOf(T.zero(), bSz * c * dstSz)
        val srcIdxs = (0 until dstSz).map({ idx -> Int.min(srcSz - 1, (idx.toDouble() * scaleSz).toInt()) }).collect<Vec>()
        for (bIdx in 0 until bSz) {
            val dst1 = ((dst).slice(bIdx * c * dstSz until dst.size))
            val srcIndex1 = srcIndex + bIdx * stride[0]
            for (cIdx in 0 until c) {
                val dst2 = ((dst1).slice(cIdx * dstSz until dst1.size))
                val srcIndex2 = srcIndex1 + cIdx * stride[1]
                for ((idx, srcIdx) in srcIdxs.iter().enumerate()) {
                    dst2[idx] = src[srcIndex2 + srcIdx * strideSz]
                }
            }
        }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<UpsampleNearest1D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<UpsampleNearest1D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): UpsampleNearest1D =
                    UpsampleNearest1D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: UpsampleNearest1D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.value)
                }
            }
    }
}

public class UpsampleNearest2D(
    public var field0: Int = 0,
    public var field1: Int = 0,
) : Map1 {
    public fun <T : WithDType> f(src: List<T>, layout: Layout): Result<List<T>> {
        // Note: Specialized implementation for the case 2*h, 2*w?
        val (dstH, dstW) = Pair(this.value, this.field1)
        val (bSz, c, srcH, srcW) =
            when (val tryResult = layout.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val stride = layout.stride()
        val (strideH, strideW) = Pair(stride[2], stride[3])
        val srcIndex = layout.startOffset()
        val scaleH = srcH.toDouble() / dstH.toDouble()
        val scaleW = srcW.toDouble() / dstW.toDouble()
        val dst = mutableListOf(T.zero(), bSz * c * dstH * dstW)
        val srcHIdxs = (0 until dstH).map({ hIdx -> Int.min(srcH - 1, (hIdx.toDouble() * scaleH).toInt()) }).collect<Vec>()
        val srcWIdxs = (0 until dstW).map({ wIdx -> Int.min(srcW - 1, (wIdx.toDouble() * scaleW).toInt()) }).collect<Vec>()
        for (bIdx in 0 until bSz) {
            val dst1 = ((dst).slice(bIdx * c * dstH * dstW until dst.size))
            val srcIndex1 = srcIndex + bIdx * stride[0]
            for (cIdx in 0 until c) {
                val dst2 = ((dst1).slice(cIdx * dstH * dstW until dst1.size))
                val srcIndex2 = srcIndex1 + cIdx * stride[1]
                for ((hIdx, srcHIdx) in srcHIdxs.iter().enumerate()) {
                    for ((wIdx, srcWIdx) in srcWIdxs.iter().enumerate()) {
                        val srcIndex3 = srcIndex2 + srcHIdx * strideH + srcWIdx * strideW
                        dst2[hIdx * dstW + wIdx] = src[srcIndex3]
                    }
                }
            }
        }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<UpsampleNearest2D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<UpsampleNearest2D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 16L
                override fun readNative(addr: Long): UpsampleNearest2D =
                    UpsampleNearest2D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                    )
                override fun writeNative(addr: Long, value: UpsampleNearest2D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.field0)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.field1)
                }
            }
    }
}

public class UpsampleBilinear2D(
    public var targetH: Int = 0,
    public var targetW: Int = 0,
    public var alignCorners: Boolean = false,
    public var scaleHFactor: Double? = null,
    public var scaleWFactor: Double? = null,
) : Map1 {
    public fun <T : WithDType> f(src: List<T>, layout: Layout, fromF64TFn: (Any?) -> Any?): Result<List<T>> {
        val (batch, channels, heightIn, widthIn) =
            when (val tryResult = layout.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val heightOut = this.targetH
        val widthOut = this.targetW
        // Early return for identity case
        if (heightIn == heightOut && widthIn == widthOut) {
            return Ok(src.toVec())
        }
        val stride = layout.stride()
        val srcOffset = layout.startOffset()
        // Calculate scale factors following PyTorch's areaPixelComputeScale logic
        val scaleH =
            if (this.alignCorners) {
                if (heightOut > 1) {
                    (heightIn - 1).toDouble() / (heightOut - 1).toDouble()
                } else {
                    0.0
                }
            } else {
                // PyTorch's computeScalesValue logic:
                // If scaleFactor was provided, use 1.0 / scaleFactor
                // Otherwise, use inputSize / outputSize
                when (val matchVal = this.scaleHFactor) {
                    null -> {
                        heightIn.toDouble() / heightOut.toDouble()
                    }
                    else -> {
                        val scaleFactor = matchVal
                        1.0 / scaleFactor
                    }
                }
            }
        val scaleW =
            if (this.alignCorners) {
                if (widthOut > 1) {
                    (widthIn - 1).toDouble() / (widthOut - 1).toDouble()
                } else {
                    0.0
                }
            } else {
                when (val matchVal1 = this.scaleWFactor) {
                    null -> {
                        widthIn.toDouble() / widthOut.toDouble()
                    }
                    else -> {
                        val scaleFactor1 = matchVal1
                        1.0 / scaleFactor1
                    }
                }
            }
        // Precompute indices and weights for height
        val hIndices = ArrayList<T>(heightOut)
        for (hOut in 0 until heightOut) {
            val srcH =
                if (this.alignCorners) {
                    scaleH * hOut.toDouble()
                } else {
                    scaleH * (hOut.toDouble() + 0.5) - 0.5
                }
            val srcHClamped = srcH.max(0.0)
            val h0 = srcHClamped.floor().toInt()
            val h1 = (h0 + 1).min(heightIn - 1)
            val weightH = (srcHClamped - h0.toDouble()).clamp(0.0, 1.0)
            hIndices.add(Triple(h0, h1, weightH))
        }
        // Precompute indices and weights for width
        val wIndices = ArrayList<T>(widthOut)
        for (wOut in 0 until widthOut) {
            val srcW =
                if (this.alignCorners) {
                    scaleW * wOut.toDouble()
                } else {
                    scaleW * (wOut.toDouble() + 0.5) - 0.5
                }
            val srcWClamped = srcW.max(0.0)
            val w0 = srcWClamped.floor().toInt()
            val w1 = (w0 + 1).min(widthIn - 1)
            val weightW = (srcWClamped - w0.toDouble()).clamp(0.0, 1.0)
            wIndices.add(Triple(w0, w1, weightW))
        }
        // Allocate output
        val dst = mutableListOf(T.zero(), batch * channels * heightOut * widthOut)
        // Perform bilinear interpolation
        for (b in 0 until batch) {
            for (c in 0 until channels) {
                val baseIdx = srcOffset + b * stride[0] + c * stride[1]
                val dstBase = (b * channels + c) * heightOut * widthOut
                for ((hOut1, h01, h11, weightH1) in ArrayDeque(hIndices).enumerate()) {
                    for ((wOut1, w01, w11, weightW1) in ArrayDeque(wIndices).enumerate()) {
                        // Get four neighboring pixels
                        val idx00 = baseIdx + h01 * stride[2] + w01 * stride[3]
                        val idx10 = baseIdx + h01 * stride[2] + w11 * stride[3]
                        val idx01 = baseIdx + h11 * stride[2] + w01 * stride[3]
                        val idx11 = baseIdx + h11 * stride[2] + w11 * stride[3]
                        val v00 = src[idx00].toF64()
                        val v10 = src[idx10].toF64()
                        val v01 = src[idx01].toF64()
                        val v11 = src[idx11].toF64()
                        // Bilinear interpolation
                        val vTop = v00 * (1.0 - weightW1) + v10 * weightW1
                        val vBottom = v01 * (1.0 - weightW1) + v11 * weightW1
                        val value = vTop * (1.0 - weightH1) + vBottom * weightH1
                        dst[dstBase + hOut1 * widthOut + wOut1] = fromF64TFn(value)
                    }
                }
            }
        }
        return Ok(dst)
    }
}

public class Gather<I : IntDType>(
    public var ids: List<I> = emptyList(),
    public var idsL: Layout = Layout(),
    public var dim: Int = 0,
) : Map1 {
    public fun <T : WithDType> f(src: List<T>, srcL: Layout, maxValueIFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<List<T>> {
        val ids =
            when (val matchVal = this.idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "gather").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val a = matchVal.first
                    val b = matchVal.second
                    ((this.ids).slice(a until b))
                }
            }
        val src1 =
            when (val matchVal1 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "gather").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val a1 = matchVal1.first
                    val b1 = matchVal1.second
                    ((src).slice(a1 until b1))
                }
            }
        val dim = this.dim
        val idsDims = this.idsL.dims()
        val srcDims = srcL.dims()
        val dstLen: Int = idsDims.iter().product()
        val dstLeftLen: Int = ((idsDims).slice(0 until dim)).iter().product()
        val dstDimLen = idsDims[dim]
        val dstRightLen: Int = ((idsDims).slice(dim + 1 until idsDims.size)).iter().product()
        val srcDimLen = srcDims[dim]
        val srcRightLen: Int = ((srcDims).slice(dim + 1 until srcDims.size)).iter().product()
        val dst = mutableListOf(zeroTFn(), dstLen)
        for (leftI in 0 until dstLeftLen) {
            val startSrcIdx = leftI * srcRightLen * srcDimLen
            val startDstIdx = leftI * dstRightLen * dstDimLen
            for (i in 0 until dstDimLen) {
                val startDstIdx1 = startDstIdx + i * dstRightLen
                for (rightI in 0 until dstRightLen) {
                    val dstIdx = startDstIdx1 + rightI
                    val index = ids[dstIdx]
                    if (index == maxValueIFn()) {
                        dst[dstIdx] = zeroTFn()
                    } else {
                        val index1 = index.asUsize()
                        if (index1 >= srcDimLen) {
                            when (val tryResult = Err(Error.InvalidIndex(index = index1, size = srcDimLen, op = "gather").bt())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }
                        }
                        val srcIdx = startSrcIdx + index1 * srcRightLen + rightI
                        dst[dstIdx] = src1[srcIdx]
                    }
                }
            }
        }
        return Ok(dst)
    }
}

public class IndexSelect<T : IntDType>(
    public var ids: List<T> = emptyList(),
    public var idsL: Layout = Layout(),
    public var dim: Int = 0,
)

public fun <I : IntDType, T : WithDType> IndexSelect<I>.f(src: List<T>, layout: Layout, maxValueIFn: (Any?) -> Any?, zeroTFn: (Any?) -> Any?): Result<List<T>> {
    val src1 =
        when (val matchVal = layout.contiguousOffsets()) {
            null ->
                when (val tryResult = Err(Error.RequiresContiguous(op = "index-select").bt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            else -> {
                val a = matchVal.first
                val b = matchVal.second
                ((src).slice(a until b))
            }
        }
    val dim = this.dim
    val nIds1 =
        run {
            val matchVal1 = this.idsL.dims()
            when {
                matchVal1.size == 1 -> {
                    val nIds = matchVal1[0]
                    nIds
                }
                else -> {
                    val d = matchVal1
                    when (val tryResult = Err(Error.UnexpectedNumberOfDims(expected = 1, got = d.len(), shape = this.idsL.shape().clone()).bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        }
    val strideIds = this.idsL.stride()[0]
    val dstDims = layout.dims().toVec()
    val srcDim = dstDims[dim]
    dstDims[dim] = nIds1
    val dstLen: Int = dstDims.iter().product()
    val leftLen: Int = ((dstDims).slice(0 until dim)).iter().product()
    val rightLen: Int = ((dstDims).slice(dim + 1 until dstDims.size)).iter().product()
    val dst = mutableListOf(zeroTFn(), dstLen)
    for (leftI in 0 until leftLen) {
        val startSrcIdx = leftI * rightLen * srcDim
        val startDstIdx = leftI * rightLen * nIds1
        for (i in 0 until nIds1) {
            val startDstIdx1 = startDstIdx + i * rightLen
            val index = this.ids[this.idsL.startOffset() + strideIds * i]
            if (index == maxValueIFn()) {
                ((dst).slice(startDstIdx1 until startDstIdx1 + rightLen)).fill(zeroTFn())
            } else {
                val index1 = index.asUsize()
                if (index1 >= srcDim) {
                    when (val tryResult = Err(Error.InvalidIndex(index = index1, size = srcDim, op = "index-select").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                val startSrcIdx1 = startSrcIdx + index1 * rightLen
                ((dst).slice(startDstIdx1 until startDstIdx1 + rightLen)).copyFromSlice(((src1).slice(startSrcIdx1 until startSrcIdx1 + rightLen)))
            }
        }
    }
    return Ok(dst)
}

public class Set : ElemUpdate {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Set> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Set> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Set =
                    Set()
                override fun writeNative(addr: Long, value: Set) {
                }
            }

        public fun <T : WithDType> f(dst: T, src: T) {
            var dst1 = dst
            dst1 = src
        }
    }
}

public class Add : ElemUpdate {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Add> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Add> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): Add =
                    Add()
                override fun writeNative(addr: Long, value: Add) {
                }
            }

        public fun <T : WithDType> f(dst: T, src: T) {
            var dst1 = dst
            dst1 += src
        }
    }
}

public class Scatter<I : IntDType, M : ElemUpdate>(
    public var ids: List<I> = emptyList(),
    public var idsL: Layout = Layout(),
    public var dim: Int = 0,
    public var phantom: Unit = Unit,
) : Map2InPlace {
    public fun <T : WithDType> f(dst: List<T>, dstL: Layout, src: List<T>, srcL: Layout, maxValueIFn: (Any?) -> Any?, fMFn: (Any?) -> Any?): Result<Unit> {
        val dst1 =
            when (val matchVal = dstL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal.first
                    val o2 = matchVal.second
                    ((dst).slice(o1 until o2))
                }
            }
        val src1 =
            when (val matchVal1 = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o11 = matchVal1.first
                    val o21 = matchVal1.second
                    ((src).slice(o11 until o21))
                }
            }
        val dim = this.dim
        val idsDims = this.idsL.dims()
        val dstDims = dstL.dims()
        val dstDimLen = dstDims[dim]
        val dstRightLen: Int = ((dstDims).slice(dim + 1 until dstDims.size)).iter().product()
        val idsLeftLen: Int = ((idsDims).slice(0 until dim)).iter().product()
        val idsDimLen = idsDims[dim]
        val idsRightLen: Int = ((idsDims).slice(dim + 1 until idsDims.size)).iter().product()
        val ids =
            when (val matchVal2 = this.idsL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "scatter").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val a = matchVal2.first
                    val b = matchVal2.second
                    ((this.ids).slice(a until b))
                }
            }
        for (leftI in 0 until idsLeftLen) {
            val startIdsIdx = leftI * idsRightLen * idsDimLen
            val startDstIdx = leftI * dstRightLen * dstDimLen
            for (i in 0 until idsDimLen) {
                val startIdsIdx1 = startIdsIdx + i * idsRightLen
                for (rightI in 0 until dstRightLen) {
                    val idsIdx = startIdsIdx1 + rightI
                    val index = ids[idsIdx]
                    if (index == maxValueIFn()) {
                        continue
                    }
                    val index1 = index.asUsize()
                    if (index1 >= dstDimLen) {
                        when (val tryResult = Err(Error.InvalidIndex(index = index1, size = dstDimLen, op = "gather").bt())) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                    val dstIdx = startDstIdx + index1 * dstRightLen + rightI
                    fMFn(dst1[dstIdx], src1[idsIdx])
                }
            }
        }
        return Ok(Unit)
    }

    public companion object {
        public fun <I : IntDType, M : ElemUpdate> new(ids: List<I>, idsL: Layout, dim: Int): Scatter<I, M> = Scatter(ids = ids, idsL = idsL, dim = dim, phantom = Default.default())

        public val OP: String = "scatter"
    }
}

public class IndexAdd<I : IntDType>(
    public var ids: List<I> = emptyList(),
    public var dim: Int = 0,
) : Map2 {
    // https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_
    // v1, l1 -> self
    public fun <T : WithDType> f(v1: List<T>, l1: Layout, src: List<T>, srcL: Layout, maxValueIFn: (Any?) -> Any?): Result<List<T>> {
        val dstLen = l1.shape().elemCount()
        val dst = mutableListOf(T.zero(), dstLen)
        copyStridedSrc(v1, dst, 0, l1)
        val src1 =
            when (val matchVal = srcL.contiguousOffsets()) {
                null ->
                    when (val tryResult = Err(Error.RequiresContiguous(op = "index-add").bt())) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                else -> {
                    val o1 = matchVal.first
                    val o2 = matchVal.second
                    ((src).slice(o1 until o2))
                }
            }
        val dim = this.dim
        val maxIdx = l1.dims()[dim]
        val preDim = ((srcL.dims()).slice(0 until dim)).iter().product<Int>()
        val srcDimSz = srcL.dims()[dim]
        val postDim = ((srcL.dims()).slice(dim + 1 until srcL.dims().size)).iter().product<Int>()
        if (dim == 0) {
            for ((srcIdx, dstIdx) in this.ids.enumerate()) {
                if (dstIdx == maxValueIFn()) {
                    continue
                }
                val dstIdx1 = dstIdx.asUsize()
                if (dstIdx1 >= maxIdx) {
                    when (val tryResult = Err(Error.InvalidIndex(index = dstIdx1, op = "index-add", size = maxIdx))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                val srcIdx1 = srcIdx * postDim
                val dstIdx2 = dstIdx1 * postDim
                val src2 = ((src1).slice(srcIdx1 until srcIdx1 + postDim))
                val dst1 = ((dst).slice(dstIdx2 until dstIdx2 + postDim))
                for ((d, s) in (dst1.iterMut().zip(src2.iter()))) {
                    d += s
                }
            }
        } else {
            for ((srcIdx2, dstIdx3) in this.ids.enumerate()) {
                if (dstIdx3 == maxValueIFn()) {
                    continue
                }
                val dstIdx4 = dstIdx3.asUsize()
                if (dstIdx4 >= maxIdx) {
                    when (val tryResult = Err(Error.InvalidIndex(index = dstIdx4, op = "index-add", size = maxIdx))) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                for (preI in 0 until preDim) {
                    val preSrcI = (preI * srcDimSz + srcIdx2) * postDim
                    val preDstI = (preI * maxIdx + dstIdx4) * postDim
                    val src3 = ((src1).slice(preSrcI until preSrcI + postDim))
                    val dst2 = ((dst).slice(preDstI until preDstI + postDim))
                    for ((d1, s1) in (dst2.iterMut().zip(src3.iter()))) {
                        d1 += s1
                    }
                }
            }
        }
        return Ok(dst)
    }

    public companion object {
        public val OP: String = "index-add"
    }
}

public class Conv1D(
    public var value: ParamsConv1D = ParamsConv1D(),
) : Map2 {
    public var field0: ParamsConv1D
        get() = value
        set(v) {
            value = v
        }

    public fun <T : WithDType> f(inp: List<T>, inpL: Layout, k: List<T>, kL: Layout, zeroTFn: (Any?) -> Any?, vecDotTFn: (Any?) -> Any?): Result<List<T>> {
        val p = this.value
        val inp1 = ((inp).slice(inpL.startOffset() until inp.size))
        val k1 = ((k).slice(kL.startOffset() until k.size))
        val (inpS0, inpS1, inpS2) =
            when (val tryResult = dims3(inpL.stride())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val (kS0, kS1, kS2) =
            when (val tryResult = dims3(kL.stride())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lOut = p.lOut()
        val dstElems = p.cOut * lOut * p.bSize
        // The output shape is [bSize, cOut, lOut]
        val dst = mutableListOf(zeroTFn(), dstElems)
        // Note: Avoid making this copy if `inp` already has the appropriate layout.
        val inpCont = mutableListOf(zeroTFn(), p.bSize * p.cIn * p.lIn)
        for (bIdx in 0 until p.bSize) {
            for (srcL in 0 until p.lIn) {
                for (srcCIdx in 0 until p.cIn) {
                    val inpIdx = bIdx * inpS0 + srcCIdx * inpS1 + srcL * inpS2
                    inpCont[bIdx * p.lIn * p.cIn + srcL * p.cIn + srcCIdx] = inp1[inpIdx]
                }
            }
        }
        for (offset in 0 until p.kSize) {
            (0 until p.cOut).intoParIter().forEach({ dstCIdx ->
                val dstIdx = dstCIdx * lOut
                val kCont = (0 until p.cIn).map({ cInIdx -> k1[dstCIdx * kS0 + cInIdx * kS1 + offset * kS2] }).collect<Vec>()
                for (bIdx1 in 0 until p.bSize) {
                    val dstIdx1 = dstIdx + bIdx1 * p.cOut * lOut
                    for (dstL in 0 until lOut) {
                        val dstIdx2 = dstIdx1 + dstL
                        val srcL1 = p.stride * dstL + offset * p.dilation
                        if (srcL1 < p.padding || srcL1 >= p.padding + p.lIn) {
                            continue
                        }
                        val srcL2 = srcL1 - p.padding
                        val inpCont1 = ((inpCont).slice(bIdx1 * p.lIn * p.cIn + srcL2 * p.cIn until inpCont.size))
                        check(inpCont1.len() >= p.cIn)
                        check(kCont.len() >= p.cIn)
                        val d = zeroTFn()
                        vecDotTFn(inpCont1.asPtr(), kCont.asPtr(), d, p.cIn)
                        val dstP = dst.asPtr()
                        // Safety: dstIdx are uniques per dstCIdx which is used to parallelise
                        // the different tasks so no two threads can try to write at the same
                        // location.
                        run {
                            val ptr = mutPtrFromRustCast(dstP.add(dstIdx2))
                            (ptr).let { targetPtr ->
                                val targetLayout = ObjectSlotLayout<T>()
                                var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                                currentValue += (d)
                                targetLayout.writeNative(targetPtr.rawAddress, currentValue)
                            }
                        }
                    }
                }
            })
        }
        return Ok(dst)
    }

    public companion object {
        public val OP: String = "conv1d"
    }
}

public class Im2Col1D(
    public var lK: Int = 0,
    public var stride: Int = 0,
    public var dilation: Int = 0,
    public var padding: Int = 0,
) : Map1 {
    public fun lOut(l: Int): Int = (l + 2 * this.padding - this.dilation * (this.lK - 1) - 1) / this.stride + 1

    public fun <T : WithDType> f(vs: List<T>, layout: Layout): Result<List<T>> {
        val lK = this.lK
        val stride = this.stride
        val dilation = this.dilation
        val padding = this.padding
        val (b, c, l) =
            when (val tryResult = layout.shape().dims3()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lOut = this.lOut(l)
        val src = ((vs).slice(layout.startOffset() until vs.size))
        val dst = mutableListOf(T.zero(), b * lOut * c * lK)
        val (srcS0, srcS1, srcS2) =
            run {
                val s = layout.stride()
                Triple(s[0], s[1], s[2])
            }
        // Note: provide specialized kernels for the common use cases.
        // - lK = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for (bIdx in 0 until b) {
            val srcIdx = bIdx * srcS0
            val dstIdx = bIdx * lOut * c * lK
            for (lIdx in 0 until lOut) {
                val dstIdx1 = dstIdx + lIdx * c * lK
                for (cIdx in 0 until c) {
                    val dstIdx2 = dstIdx1 + cIdx * lK
                    val srcIdx1 = cIdx * srcS1 + srcIdx
                    for (lKIdx in 0 until lK) {
                        val srcL = lIdx * stride + lKIdx * dilation
                        if (padding != 0 && (srcL < padding || srcL >= l + padding)) {
                            continue
                        }
                        val srcL1 = srcL - padding
                        val srcIdx2 = srcIdx1 + srcL1 * srcS2
                        val dstIdx3 = dstIdx2 + lKIdx
                        dst[dstIdx3] = src[srcIdx2]
                    }
                }
            }
        }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col1D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col1D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 32L
                override fun readNative(addr: Long): Im2Col1D =
                    Im2Col1D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 24L),
                    )
                override fun writeNative(addr: Long, value: Im2Col1D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.lK)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.stride)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 16L, value.dilation)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 24L, value.padding)
                }
            }
    }
}

public class Im2Col(
    public var hK: Int = 0,
    public var wK: Int = 0,
    public var stride: Int = 0,
    public var dilation: Int = 0,
    public var padding: Int = 0,
) : Map1 {
    public fun hwOut(h: Int, w: Int): Pair<Int, Int> {
        val hOut = (h + 2 * this.padding - this.dilation * (this.hK - 1) - 1) / this.stride + 1
        val wOut = (w + 2 * this.padding - this.dilation * (this.wK - 1) - 1) / this.stride + 1
        return Pair(hOut, wOut)
    }

    public fun <T : WithDType> f(vs: List<T>, layout: Layout): Result<List<T>> {
        val hK = this.hK
        val wK = this.wK
        val stride = this.stride
        val dilation = this.dilation
        val padding = this.padding
        val (b, c, h, w) =
            when (val tryResult = layout.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val (hOut, wOut) = this.hwOut(h, w)
        val src = ((vs).slice(layout.startOffset() until vs.size))
        val dst = mutableListOf(T.zero(), b * hOut * wOut * c * hK * wK)
        val (srcS0, srcS1, srcS2, srcS3) =
            run {
                val s = layout.stride()
                listOf(s[0], s[1], s[2], s[3])
            }
        // Note: provide specialized kernels for the common use cases.
        // - hK = wK = 1
        // - padding = 0
        // - stride = 1
        // - dilation = 1
        for (bIdx in 0 until b) {
            val srcIdx = bIdx * srcS0
            val dstIdx = bIdx * hOut * wOut * c * hK * wK
            for (hIdx in 0 until hOut) {
                val dstIdx1 = dstIdx + hIdx * wOut * c * hK * wK
                for (wIdx in 0 until wOut) {
                    val dstIdx2 = dstIdx1 + wIdx * c * hK * wK
                    for (cIdx in 0 until c) {
                        val dstIdx3 = dstIdx2 + cIdx * hK * wK
                        val srcIdx1 = cIdx * srcS1 + srcIdx
                        for (hKIdx in 0 until hK) {
                            val srcH = hIdx * stride + hKIdx * dilation
                            if (padding != 0 && (srcH < padding || srcH >= h + padding)) {
                                continue
                            }
                            val srcH1 = srcH - padding
                            val srcIdx2 = srcIdx1 + srcH1 * srcS2
                            val dstIdx4 = dstIdx3 + hKIdx * wK
                            for (wKIdx in 0 until wK) {
                                val srcW = wIdx * stride + wKIdx * dilation
                                if (padding != 0 && (srcW < padding || srcW >= w + padding)) {
                                    continue
                                }
                                val srcW1 = srcW - padding
                                val srcIdx3 = srcIdx2 + srcW1 * srcS3
                                val dstIdx5 = dstIdx4 + wKIdx
                                dst[dstIdx5] = src[srcIdx3]
                            }
                        }
                    }
                }
            }
        }
        return Ok(dst)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Im2Col> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 40L
                override fun readNative(addr: Long): Im2Col =
                    Im2Col(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 8L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 16L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 24L),
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr + 32L),
                    )
                override fun writeNative(addr: Long, value: Im2Col) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.hK)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 8L, value.wK)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 16L, value.stride)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 24L, value.dilation)
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr + 32L, value.padding)
                }
            }
    }
}

public class Col2Im1D(
    public var stride: Int = 0,
) : Map1 {
    public fun <T : WithDType> f(col: List<T>, l: Layout): Result<List<T>> {
        val (bSize, lIn, cOut, kSize) =
            when (val tryResult = l.shape().dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val stride = this.stride
        val lOut = (lIn - 1) * stride + kSize
        val im = mutableListOf(T.zero(), bSize * cOut * lOut)
        val (dstS0, dstS1) = Pair(cOut * lOut, lOut)
        val (srcS0, srcS1, srcS2) = Triple(cOut * kSize * lIn, cOut * kSize, kSize)
        for (lInI in 0 until lIn) {
            for (kI in 0 until kSize) {
                val lOutI = lInI * stride + kI
                for (bI in 0 until bSize) {
                    for (cI in 0 until cOut) {
                        val dstIdx = bI * dstS0 + cI * dstS1 + lOutI
                        val srcIdx = bI * srcS0 + lInI * srcS1 + cI * srcS2 + kI
                        im[dstIdx] += col[srcIdx]
                    }
                }
            }
        }
        return Ok(im)
    }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<Col2Im1D> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<Col2Im1D> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): Col2Im1D =
                    Col2Im1D(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: Col2Im1D) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.stride)
                }
            }
    }
}

public class ConvTranspose1D(
    public var value: ParamsConvTranspose1D = ParamsConvTranspose1D(),
) : Map2 {
    public var field0: ParamsConvTranspose1D
        get() = value
        set(v) {
            value = v
        }

    public fun <T : WithDType> f(inp: List<T>, inpL: Layout, k: List<T>, kL: Layout, zeroTFn: (Any?) -> Any?, vecDotTFn: (Any?) -> Any?): Result<List<T>> {
        val p = this.value
        val inp1 = ((inp).slice(inpL.startOffset() until inp.size))
        val k1 = ((k).slice(kL.startOffset() until k.size))
        val (inpS0, inpS1, inpS2) =
            when (val tryResult = dims3(inpL.stride())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val (kS0, kS1, kS2) =
            when (val tryResult = dims3(kL.stride())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lOut = p.lOut()
        // Output shape: [bSize, cOut, lOut].
        val dstElems = p.cOut * lOut * p.bSize
        val dst = mutableListOf(zeroTFn(), dstElems)
        val dstS0 = p.cOut * lOut
        val dstS1 = lOut
        val dstS2 = 1
        // Note: Avoid making this copy if `inp` already has the appropriate layout.
        val inpCont = mutableListOf(zeroTFn(), p.bSize * p.cIn * p.lIn)
        val contS0 = p.lIn * p.cIn
        val contS1 = p.cIn
        for (bIdx in 0 until p.bSize) {
            for (lIdx in 0 until p.lIn) {
                for (cIdx in 0 until p.cIn) {
                    val srcIdx = bIdx * inpS0 + cIdx * inpS1 + lIdx * inpS2
                    val dstIdx = bIdx * contS0 + lIdx * contS1 + cIdx
                    inpCont[dstIdx] = inp1[srcIdx]
                }
            }
        }
        for (kIdx in 0 until p.kSize) {
            (0 until p.cOut).intoParIter().forEach({ dstCIdx ->
                val kCont = (0 until p.cIn).map({ cInIdx -> k1[cInIdx * kS0 + dstCIdx * kS1 + kIdx * kS2] }).collect<Vec>()
                for (bIdx1 in 0 until p.bSize) {
                    for (lIdx1 in 0 until p.lIn) {
                        val outIdx = lIdx1 * p.stride + kIdx * p.dilation
                        if (outIdx < p.padding) {
                            continue
                        }
                        val outIdx1 = outIdx - p.padding
                        if (outIdx1 < lOut) {
                            val inpCont1 = ((inpCont).slice(bIdx1 * contS0 + lIdx1 * contS1 until inpCont.size))
                            val dstIdx1 = bIdx1 * dstS0 + outIdx1 * dstS2 + dstCIdx * dstS1
                            val d = zeroTFn()
                            vecDotTFn(inpCont1.asPtr(), kCont.asPtr(), d, p.cIn)
                            val dstP = dst.asPtr()
                            // Safety: dstIdx are uniques per dstCIdx which is used to
                            // parallelise the different tasks so no two threads can try to
                            // write at the same location.
                            run {
                                val ptr = mutPtrFromRustCast(dstP.add(dstIdx1))
                                (ptr).let { targetPtr ->
                                    val targetLayout = ObjectSlotLayout<T>()
                                    var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                                    currentValue += (d)
                                    targetLayout.writeNative(targetPtr.rawAddress, currentValue)
                                }
                            }
                        }
                    }
                }
            })
        }
        return Ok(dst)
    }

    public companion object {
        public val OP: String = "conv_transpose1d"
    }
}

public class ConvTranspose2D(
    public var value: ParamsConvTranspose2D = ParamsConvTranspose2D(),
) : Map2 {
    public var field0: ParamsConvTranspose2D
        get() = value
        set(v) {
            value = v
        }

    public fun <T : WithDType> f(inp: List<T>, inpL: Layout, k: List<T>, kL: Layout, zeroTFn: (Any?) -> Any?, vecDotTFn: (Any?) -> Any?): Result<List<T>> {
        val p = this.value
        val inp1 = ((inp).slice(inpL.startOffset() until inp.size))
        val (inpS0, inpS1, inpS2, inpS3) =
            when (val tryResult = dims4(inpL.stride())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val k1 = ((k).slice(kL.startOffset() until k.size))
        val (kS0, kS1, kS2, kS3) =
            when (val tryResult = dims4(kL.stride())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val (outH, outW) = Pair(p.outH(), p.outW())
        // Output shape: [bSize, cOut, outH, outW].
        val dst = mutableListOf(zeroTFn(), p.bSize * p.cOut * outH * outW)
        val dstS0 = p.cOut * outH * outW
        val dstS1 = outH * outW
        val dstS2 = outW
        val dstS3 = 1
        // Note: Avoid making this copy if `inp` already has the appropriate layout.
        val inpCont = mutableListOf(zeroTFn(), p.bSize * p.cIn * p.iH * p.iW)
        val contS0 = p.iH * p.iW * p.cIn
        val contS1 = p.iW * p.cIn
        val contS2 = p.cIn
        for (bIdx in 0 until p.bSize) {
            for (hIdx in 0 until p.iH) {
                for (wIdx in 0 until p.iW) {
                    for (cIdx in 0 until p.cIn) {
                        val srcIdx = bIdx * inpS0 + cIdx * inpS1 + hIdx * inpS2 + wIdx * inpS3
                        val dstIdx = bIdx * contS0 + hIdx * contS1 + wIdx * contS2 + cIdx
                        inpCont[dstIdx] = inp1[srcIdx]
                    }
                }
            }
        }
        for (kY in 0 until p.kH) {
            for (kX in 0 until p.kW) {
                (0 until p.cOut).intoParIter().forEach({ dstCIdx ->
                    val kCont =
                        (0 until p.cIn)
                            .map({ cInIdx ->
                                k1[cInIdx * kS0 + dstCIdx * kS1 + kY * kS2 + kX * kS3]
                            })
                            .collect<Vec>()
                    for (bIdx1 in 0 until p.bSize) {
                        for (inpY in 0 until p.iH) {
                            for (inpX in 0 until p.iW) {
                                val outX = inpX * p.stride + kX * p.dilation
                                val outY = inpY * p.stride + kY * p.dilation
                                if (outX < p.padding || outY < p.padding) {
                                    continue
                                }
                                val outX1 = outX - p.padding
                                val outY1 = outY - p.padding
                                if (outX1 < outW && outY1 < outH) {
                                    val inpCont1 = ((inpCont).slice(bIdx1 * contS0 + inpY * contS1 + inpX * contS2 until inpCont.size))
                                    val dstIdx1 = bIdx1 * dstS0 + outY1 * dstS2 + outX1 * dstS3 + dstCIdx * dstS1
                                    val d = zeroTFn()
                                    vecDotTFn(inpCont1.asPtr(), kCont.asPtr(), d, p.cIn)
                                    val dstP = dst.asPtr()
                                    // Safety: dstIdx are uniques per dstCIdx which is used to
                                    // parallelise the different tasks so no two threads can try to
                                    // write at the same location.
                                    run {
                                        val ptr = mutPtrFromRustCast(dstP.add(dstIdx1))
                                        (ptr).let { targetPtr ->
                                            val targetLayout = ObjectSlotLayout<T>()
                                            var currentValue = targetLayout.readNative(targetPtr.rawAddress)
                                            currentValue += (d)
                                            targetLayout.writeNative(targetPtr.rawAddress, currentValue)
                                        }
                                    }
                                }
                            }
                        }
                    }
                })
            }
        }
        return Ok(dst)
    }

    public companion object {
        public val OP: String = "conv_transpose2d"
    }
}

public class MatMul(
    public var value: List<Any?> = emptyList(),
) : Map2 {
    public var field0: List<Any?>
        get() = value
        set(v) {
            value = v
        }

    public fun stridingError(lhsL: Layout, rhsL: Layout, msg: String): Error = Error.MatMulUnexpectedStriding(MatMulUnexpectedStriding(lhsL = lhsL.clone(), rhsL = rhsL.clone(), bmnk = this.value, msg = msg)).bt()

    public fun abSkip(lhsL: Layout, rhsL: Layout): Result<Pair<Int, Int>> {
        val lhsStride = lhsL.stride()
        val rhsStride = rhsL.stride()
        val rank = lhsStride.len()
        val (b, m, n, k) = this.value
        val aSkip: Int =
            run {
                val matchVal = ((lhsStride).slice(0 until rank - 2))
                when {
                    matchVal.size == 2 && matchVal[0] == matchVal[1] * lhsL.dims()[1] -> {
                        val s1 = matchVal[0]
                        val stride = matchVal[1]
                        stride
                    }
                    matchVal.size == 2 && lhsL.dims()[0] == 1 -> {
                        val stride1 = matchVal[1]
                        stride1
                    }
                    matchVal.size == 2 && lhsL.dims()[1] == 1 -> {
                        val stride2 = matchVal[0]
                        stride2
                    }
                    matchVal.size == 1 -> {
                        val stride3 = matchVal[0]
                        stride3
                    }
                    matchVal.isEmpty() -> m * k
                    else ->
                        when (val tryResult = Err(this.stridingError(lhsL, rhsL, "non-contiguous lhs"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        val bSkip: Int =
            run {
                val matchVal1 = ((rhsStride).slice(0 until rank - 2))
                when {
                    matchVal1.size == 2 && matchVal1[0] == matchVal1[1] * rhsL.dims()[1] -> {
                        val s11 = matchVal1[0]
                        val stride4 = matchVal1[1]
                        stride4
                    }
                    matchVal1.size == 2 && rhsL.dims()[0] == 1 -> {
                        val stride5 = matchVal1[1]
                        stride5
                    }
                    matchVal1.size == 2 && rhsL.dims()[1] == 1 -> {
                        val stride6 = matchVal1[0]
                        stride6
                    }
                    matchVal1.size == 1 -> {
                        val stride7 = matchVal1[0]
                        stride7
                    }
                    matchVal1.isEmpty() -> n * k
                    else ->
                        when (val tryResult = Err(this.stridingError(lhsL, rhsL, "non-contiguous rhs"))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        return Ok(Pair(aSkip, bSkip))
    }

    public fun <T> f(lhs: List<T>, lhsL: Layout, rhs: List<T>, rhsL: Layout, zeroTFn: (Any?) -> Any?, oneTFn: (Any?) -> Any?): Result<List<T>> where T : WithDType, T : Num {
        when (T.DTYPE) {
            DType.f16, DType.f32, DType.f64 -> {}
            else ->
                when (val tryResult = Err(Error.UnsupportedDTypeForOp(T.DTYPE, "matmul").bt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
        }
        val (b, m, n, k) = this.value
        val lhs1 = ((lhs).slice(lhsL.startOffset() until lhs.size))
        val rhs1 = ((rhs).slice(rhsL.startOffset() until rhs.size))
        val lhsStride = lhsL.stride()
        val rhsStride = rhsL.stride()
        val rank = lhsStride.len()
        val lhsCs = lhsStride[rank - 1]
        val lhsRs = lhsStride[rank - 2]
        val rhsCs = rhsStride[rank - 1]
        val rhsRs = rhsStride[rank - 2]
        val (aSkip, bSkip) =
            when (val tryResult = this.abSkip(lhsL, rhsL)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cSkip: Int = m * n
        val dstShape: Shape = Pair(m, n).into()
        val dstStrides = dstShape.strideContiguous()
        val dstRs = dstStrides[0]
        val dstCs = dstStrides[1]
        val dst = mutableListOf(zeroTFn(), b * m * n)
        val numThreads = getNumThreads()
        val parallelism =
            if (numThreads > 1) {
                Parallelism.Rayon(numThreads)
            } else {
                Parallelism.None
            }
        val (b1, m1, n1, k1) =
            if (bSkip == 0 && aSkip == m * k) {
                // A batch-invariant rhs lets the batches of lhs stack into the rows of a single
                // (b * m, k) matrix, which the destination already matches row for row.
                // aSkip and cSkip should be updated but step is always 0 so
                // it wouldn't matter.
                listOf(1, b * m, n, k)
            } else {
                // There is deliberately no mirrored fold for a batch-invariant lhs. Stacking the
                // batches of rhs into the columns of a (k, b * n) matrix would need those columns
                // to be adjacent, but rhs is batch-major, and the destination rows would still be
                // n apart while each merged row is b * n wide. Both make the merged gemm write
                // over itself, so a stride-zero batch on lhs goes through the loop below.
                listOf(b, m, n, k)
            }
        for (step in 0 until b1) {
            val lhsP = ((lhs1).slice(step * aSkip until lhs1.size))
            val rhsP = ((rhs1).slice(step * bSkip until rhs1.size))
            val dstP = ((dst).slice(step * cSkip until dst.size))
            gemm(m1, n1, k1, dstP.asMutPtr(), dstCs.toInt(), dstRs.toInt(), false, lhsP.asPtr(), lhsCs.toInt(), lhsRs.toInt(), rhsP.asPtr(), rhsCs.toInt(), rhsRs.toInt(), zeroTFn(), oneTFn(), false, false, false, parallelism)
        }
        return Ok(dst)
    }

    public fun <T> fMap2(lhs: List<T>, lhsL: Layout, rhs: List<T>, rhsL: Layout): Result<List<T>> where T : WithDType, T : Num {
        val (b, m, n, k) = this.value
        val lhs1 = ((lhs).slice(lhsL.startOffset() until lhs.size))
        val rhs1 = ((rhs).slice(rhsL.startOffset() until rhs.size))
        val lhsStride = lhsL.stride()
        val rhsStride = rhsL.stride()
        val (aSkip, bSkip) =
            when (val tryResult = this.abSkip(lhsL, rhsL)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cSkip: Int = m * n
        val rhsM1 = rhsStride[rhsStride.len() - 1]
        val rhsM2 = rhsStride[rhsStride.len() - 2]
        val lhsM1 = lhsStride[lhsStride.len() - 1]
        val lhsM2 = lhsStride[lhsStride.len() - 2]
        val (lda, transa) =
            if ((rhsM1 == 1 || n == 1) && (rhsM2 == n || k == 1)) {
                Pair(n.toInt(), 78u)
            } else if (rhsM1 == k && rhsM2 == 1) {
                Pair(k.toInt(), 84u)
            } else {
                when (val tryResult = Err(this.stridingError(lhsL, rhsL, "non-contiguous rhs"))) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        // The b tensor has dims batching, m, k (lhs)
        val (ldb, transb) =
            if ((lhsM1 == 1 || k == 1) && (lhsM2 == k || m == 1)) {
                Pair(k.toInt(), 78u)
            } else if (lhsM1 == m && lhsM2 == 1) {
                Pair(m.toInt(), 84u)
            } else {
                when (val tryResult = Err(this.stridingError(lhsL, rhsL, "non-contiguous lhs"))) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        val dst = mutableListOf(T.zero(), b * m * n)
        when (val matchVal = T.DTYPE) {
            DType.f16 -> {
                bail("the accelerate backend does not support f16 matmul")
            }
            DType.f32 -> {
                for (step in 0 until b) {
                    val lhsP = ((lhs1).slice(step * aSkip until lhs1.size))
                    val rhsP = ((rhs1).slice(step * bSkip until rhs1.size))
                    val dstP = ((dst).slice(step * cSkip until dst.size))
                    run {
                        val a = constPtrFromRustCast(rhsP.asPtr())
                        val b1 = constPtrFromRustCast(lhsP.asPtr())
                        val c = mutPtrFromRustCast(dstP.asMutPtr())
                        val a1 = fromRawParts(a, aSkip)
                        val b2 = fromRawParts(b1, bSkip)
                        val c1 = fromRawPartsMut(c, cSkip)
                        sgemm(transa, transb, n.toInt(), m.toInt(), k.toInt(), 1.0f, a1, lda, b2, ldb, 0.0f, c1, n.toInt())
                    }
                }
            }
            DType.f64 -> {
                for (step1 in 0 until b) {
                    val lhsP1 = ((lhs1).slice(step1 * aSkip until lhs1.size))
                    val rhsP1 = ((rhs1).slice(step1 * bSkip until rhs1.size))
                    val dstP1 = ((dst).slice(step1 * cSkip until dst.size))
                    run {
                        val a2 = constPtrFromRustCast(rhsP1.asPtr())
                        val b3 = constPtrFromRustCast(lhsP1.asPtr())
                        val c2 = mutPtrFromRustCast(dstP1.asMutPtr())
                        val a3 = fromRawParts(a2, aSkip)
                        val b4 = fromRawParts(b3, bSkip)
                        val c3 = fromRawPartsMut(c2, cSkip)
                        dgemm(transa, transb, n.toInt(), m.toInt(), k.toInt(), 1.0, a3, lda, b4, ldb, 0.0, c3, n.toInt())
                    }
                }
            }
            else -> {
                val dtype = matchVal
                when (val tryResult = Err(Error.UnsupportedDTypeForOp(dtype, "matmul").bt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        return Ok(dst)
    }

    public fun <T> fMap2Map2(lhs: List<T>, lhsL: Layout, rhs: List<T>, rhsL: Layout): Result<List<T>> where T : WithDType, T : Num {
        val (b, m, n, k) = this.value
        val lhs1 = ((lhs).slice(lhsL.startOffset() until lhs.size))
        val rhs1 = ((rhs).slice(rhsL.startOffset() until rhs.size))
        val lhsStride = lhsL.stride()
        val rhsStride = rhsL.stride()
        val (aSkip, bSkip) =
            when (val tryResult = this.abSkip(lhsL, rhsL)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cSkip: Int = m * n
        val rhsM1 = rhsStride[rhsStride.len() - 1]
        val rhsM2 = rhsStride[rhsStride.len() - 2]
        val lhsM1 = lhsStride[lhsStride.len() - 1]
        val lhsM2 = lhsStride[lhsStride.len() - 2]
        val (lda, transa) =
            if ((rhsM1 == 1 || n == 1) && (rhsM2 == n || k == 1)) {
                Pair(n.toInt(), 78u)
            } else if (rhsM1 == k && rhsM2 == 1) {
                Pair(k.toInt(), 84u)
            } else {
                when (val tryResult = Err(this.stridingError(lhsL, rhsL, "non-contiguous rhs"))) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        // The b tensor has dims batching, m, k (lhs)
        val (ldb, transb) =
            if ((lhsM1 == 1 || k == 1) && (lhsM2 == k || m == 1)) {
                Pair(k.toInt(), 78u)
            } else if (lhsM1 == m && lhsM2 == 1) {
                Pair(m.toInt(), 84u)
            } else {
                when (val tryResult = Err(this.stridingError(lhsL, rhsL, "non-contiguous lhs"))) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        val dst = mutableListOf(T.zero(), b * m * n)
        when (val matchVal = T.DTYPE) {
            DType.f16 -> {
                for (step in 0 until b) {
                    val lhsP = ((lhs1).slice(step * aSkip until lhs1.size))
                    val rhsP = ((rhs1).slice(step * bSkip until rhs1.size))
                    val dstP = ((dst).slice(step * cSkip until dst.size))
                    run {
                        val a = constPtrFromRustCast(rhsP.asPtr())
                        val b1 = constPtrFromRustCast(lhsP.asPtr())
                        val c = mutPtrFromRustCast(dstP.asMutPtr())
                        val a1 = fromRawParts(a, aSkip)
                        val b2 = fromRawParts(b1, bSkip)
                        val c1 = fromRawPartsMut(c, cSkip)
                        hgemm(transa, transb, n.toInt(), m.toInt(), k.toInt(), ONE, a1, lda, b2, ldb, ZERO, c1, n.toInt())
                    }
                }
            }
            DType.f32 -> {
                for (step1 in 0 until b) {
                    val lhsP1 = ((lhs1).slice(step1 * aSkip until lhs1.size))
                    val rhsP1 = ((rhs1).slice(step1 * bSkip until rhs1.size))
                    val dstP1 = ((dst).slice(step1 * cSkip until dst.size))
                    run {
                        val a2 = constPtrFromRustCast(rhsP1.asPtr())
                        val b3 = constPtrFromRustCast(lhsP1.asPtr())
                        val c2 = mutPtrFromRustCast(dstP1.asMutPtr())
                        val a3 = fromRawParts(a2, aSkip)
                        val b4 = fromRawParts(b3, bSkip)
                        val c3 = fromRawPartsMut(c2, cSkip)
                        io.github.kotlinmania.candle.mkl
                            .sgemm(transa, transb, n.toInt(), m.toInt(), k.toInt(), 1.0f, a3, lda, b4, ldb, 0.0f, c3, n.toInt())
                    }
                }
            }
            DType.f64 -> {
                for (step2 in 0 until b) {
                    val lhsP2 = ((lhs1).slice(step2 * aSkip until lhs1.size))
                    val rhsP2 = ((rhs1).slice(step2 * bSkip until rhs1.size))
                    val dstP2 = ((dst).slice(step2 * cSkip until dst.size))
                    run {
                        val a4 = constPtrFromRustCast(rhsP2.asPtr())
                        val b5 = constPtrFromRustCast(lhsP2.asPtr())
                        val c4 = mutPtrFromRustCast(dstP2.asMutPtr())
                        val a5 = fromRawParts(a4, aSkip)
                        val b6 = fromRawParts(b5, bSkip)
                        val c5 = fromRawPartsMut(c4, cSkip)
                        io.github.kotlinmania.candle.mkl
                            .dgemm(transa, transb, n.toInt(), m.toInt(), k.toInt(), 1.0, a5, lda, b6, ldb, 0.0, c5, n.toInt())
                    }
                }
            }
            else -> {
                val dtype = matchVal
                when (val tryResult = Err(Error.UnsupportedDTypeForOp(dtype, "matmul").bt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
        }
        return Ok(dst)
    }

    public companion object {
        public val OP: String = "mat_mul"
    }
}
