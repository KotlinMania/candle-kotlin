// port-lint: source storage.rs
package io.github.kotlinmania.candle.storage

import io.github.kotlinmania.candle.conv.ParamsConv1D
import io.github.kotlinmania.candle.conv.ParamsConv2D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose1D
import io.github.kotlinmania.candle.conv.ParamsConvTranspose2D
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.CudaStorage
import io.github.kotlinmania.candle.customop.CustomOp1
import io.github.kotlinmania.candle.customop.CustomOp2
import io.github.kotlinmania.candle.customop.CustomOp3
import io.github.kotlinmania.candle.customop.InplaceOpN
import io.github.kotlinmania.candle.customop.Src
import io.github.kotlinmania.candle.customop.allDistinct
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.op.BinaryOpT
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.UnaryOpT
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.parkinglot.RwLockReadGuard
import io.github.kotlinmania.parkinglot.RwLockWriteGuard

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead tryClone should be used.
public sealed class Storage {
    public data class Cpu(
        public var value: CpuStorage,
    ) : Storage()
    public data class Cuda(
        public var value: CudaStorage,
    ) : Storage()
    public data class Metal(
        public var value: MetalStorage,
    ) : Storage()

    public fun tryClone(layout: Layout): Result<Storage> {
        return when (this) {
            is Cpu -> {
                val (storage) = this
                Result.Ok(Storage.Cpu(storage.clone()))
            }
            is Cuda -> {
                val (storage1) = this
                val storage2 =
                    when (val tryResult = storage1.tryClone(layout)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage2))
            }
            is Metal -> {
                val (storage3) = this
                val storage4 =
                    when (val tryResult = storage3.tryClone(layout)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage4))
            }
        }
    }

    public fun device(): Device =
        when (this) {
            is Cpu -> Device.Cpu
            is Cuda -> {
                val (storage) = this
                Device.Cuda(storage.device().clone())
            }
            is Metal -> {
                val (storage1) = this
                Device.Metal(storage1.device().clone())
            }
        }

    public fun dtype(): DType =
        when (this) {
            is Cpu -> {
                val (storage) = this
                storage.dtype()
            }
            is Cuda -> {
                val (storage1) = this
                storage1.dtype()
            }
            is Metal -> {
                val (storage2) = this
                storage2.dtype()
            }
        }

    internal fun sameDevice(rhs: Storage, op: String): Result<Unit> {
        val lhsDevice = this.device()
        val rhsDevice = rhs.device()
        val lhs = lhsDevice.location()
        val rhs1 = rhsDevice.location()
        val sameDevice =
            if (this.device().isMetal()) {
                // On metal, we require the device to be exactly the same rather than
                // having the same location. In cuda this is not necessary as all CudaDevice on the
                // same GPU will use the same cuda stream.
                lhsDevice.sameDevice(rhsDevice)
            } else {
                lhs == rhs1
            }
        return if (!sameDevice) {
            Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs, rhs = rhs1, op = op).bt())
        } else {
            Result.Ok(Unit)
        }
    }

    internal fun sameDtype(rhs: Storage, op: String): Result<Unit> {
        val lhs = this.dtype()
        val rhs1 = rhs.dtype()
        return if (lhs != rhs1) {
            Result.Err(Error.DTypeMismatchBinaryOp(lhs = lhs, rhs = rhs1, op = op).bt())
        } else {
            Result.Ok(Unit)
        }
    }

    internal fun constSet(v: Scalar, l: Layout): Result<Unit> =
        when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                storage.constSet(v, l)
            }
            is Storage.Cuda -> {
                val (storage1) = this
                storage1.constSet(v, l)
            }
            is Storage.Metal -> {
                val (storage2) = this
                storage2.constSet(v, l)
            }
        }

    internal fun affine(layout: Layout, mul: Double, add: Double): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.affine(layout, mul, add)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.affine(layout, mul, add)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.affine(layout, mul, add)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun powf(layout: Layout, alpha: Double): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.powf(layout, alpha)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.powf(layout, alpha)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.powf(layout, alpha)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun elu(layout: Layout, alpha: Double): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.elu(layout, alpha)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.elu(layout, alpha)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.elu(layout, alpha)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun cmp(op: CmpOp, rhs: Storage, lhsLayout: Layout, rhsLayout: Layout): Result<Storage> {
        when (val tryResult = this.sameDevice(rhs, "cmp")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(rhs, "cmp")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (lhs) = matchVal.first as Storage.Cpu
                    val (rhs1) = matchVal.second as Storage.Cpu
                    val storage =
                        when (val tryResult = lhs.cmp(op, rhs1, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 =
                        when (val tryResult = lhs1.cmp(op, rhs2, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 =
                        when (val tryResult = lhs2.cmp(op, rhs3, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    // Should not happen because of the same device check above but we're defensive
                    // anyway.
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = "cmp").bt())
                }
            }
        }
    }

    internal fun reduceOp(op: ReduceOp, layout: Layout, s: List<Int>): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.reduceOp(op, layout, s)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.reduceOp(op, layout, s)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.reduceOp(op, layout, s)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun toDtype(layout: Layout, dtype: DType): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.toDtype(layout, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.toDtype(layout, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.toDtype(layout, dtype)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun applyOp1(l: Layout, c: CustomOp1): Result<Pair<Storage, Shape>> {
        return when (this) {
            is Cpu -> {
                val (storage) = this
                val (storage1, shape) =
                    when (val tryResult = c.cpuFwd(storage, l)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Pair(Storage.Cpu(storage1), shape))
            }
            is Cuda -> {
                val (storage2) = this
                val (storage3, shape1) =
                    when (val tryResult = c.cudaFwd(storage2, l)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Pair(Storage.Cuda(storage3), shape1))
            }
            is Metal -> {
                val (storage4) = this
                val (storage5, shape2) =
                    when (val tryResult = c.metalFwd(storage4, l)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Pair(Storage.Metal(storage5), shape2))
            }
        }
    }

    internal fun applyOp2(l1: Layout, t2: Storage, l2: Layout, c: CustomOp2): Result<Pair<Storage, Shape>> {
        when (val tryResult = this.sameDevice(t2, c.name())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, t2)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (s1) = matchVal.first as Cpu
                    val (s2) = matchVal.second as Cpu
                    val (s, shape) =
                        when (val tryResult = c.cpuFwd(s1, l1, s2, l2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Pair(Storage.Cpu(s), shape))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (s11) = matchVal.first as Cuda
                    val (s21) = matchVal.second as Cuda
                    val (s1, shape1) =
                        when (val tryResult = c.cudaFwd(s11, l1, s21, l2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Pair(Storage.Cuda(s1), shape1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (s12) = matchVal.first as Metal
                    val (s22) = matchVal.second as Metal
                    val (s2, shape2) =
                        when (val tryResult = c.metalFwd(s12, l1, s22, l2)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Pair(Storage.Metal(s2), shape2))
                }
                else -> error("unreachable")
            }
        }
    }

    internal fun applyOp3(l1: Layout, t2: Storage, l2: Layout, t3: Storage, l3: Layout, c: CustomOp3): Result<Pair<Storage, Shape>> {
        when (val tryResult = this.sameDevice(t2, c.name())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDevice(t3, c.name())) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Triple(this, t2, t3)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s1) = matchVal.first as Cpu
                    val (s2) = matchVal.second as Cpu
                    val (s3) = matchVal.third as Cpu
                    val (s, shape) =
                        when (val tryResult = c.cpuFwd(s1, l1, s2, l2, s3, l3)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Pair(Storage.Cpu(s), shape))
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s11) = matchVal.first as Cuda
                    val (s21) = matchVal.second as Cuda
                    val (s31) = matchVal.third as Cuda
                    val (s1, shape1) =
                        when (val tryResult = c.cudaFwd(s11, l1, s21, l2, s31, l3)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Pair(Storage.Cuda(s1), shape1))
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s12) = matchVal.first as Metal
                    val (s22) = matchVal.second as Metal
                    val (s32) = matchVal.third as Metal
                    val (s2, shape2) =
                        when (val tryResult = c.metalFwd(s12, l1, s22, l2, s32, l3)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Pair(Storage.Metal(s2), shape2))
                }
                else -> error("unreachable")
            }
        }
    }

    /**
     * Applies an custom in-place op for the `self` tensor.
     *
     * [Src.Aliased] use the same underlying storage as `self`, while [Src.Distinct] does not.
     * If there are aliases present the aliased forward function on `InplaceOpN` is called. Otherwise
     * the normal forward function can be used.
     */
    internal fun <C : InplaceOpN<N>> inplaceOp(dstL: Layout, srcs: List<Pair<Src<Storage>, Layout>>, c: C, n: Int): Result<Unit> {
        // Aliased sources are `self`. Only distinct needs checking.
        for ((s, _) in srcs.iter()) {
            if (s is Src.Distinct) {
                val (s1) = s
                when (val tryResult = this.sameDevice(s1, c.name())) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            }
        }
        return when (this) {
            is Storage.Cpu -> {
                val (dst) = this
                val operands: List<Pair<Src, Layout>> =
                    fromFn({ i ->
                        val (s2, l) = srcs[i]
                        val s4 =
                            when {
                                s2 is Src.Distinct && s2.value is Storage.Cpu -> {
                                    val (s3) = s2.value as Storage.Cpu
                                    Src.Distinct(s3)
                                }
                                s2 is Src.Aliased -> {
                                    val (rel) = s2
                                    Src.Aliased(rel)
                                }
                                else -> {
                                    error("same_device above rejects mismatched backends")
                                }
                            }
                        Pair(s4, l)
                    })
                when (val matchVal = allDistinct(operands)) {
                    null -> c.cpuFwdAliased(dst, dstL, operands)
                    else -> {
                        val distinct = matchVal
                        c.cpuFwd(dst, dstL, distinct)
                    }
                }
            }
            is Storage.Cuda -> {
                val (dst1) = this
                val operands1: List<Pair<Src, Layout>> =
                    fromFn({ i ->
                        val (s2, l) = srcs[i]
                        val s4 =
                            when {
                                s2 is Src.Distinct && s2.value is Storage.Cuda -> {
                                    val (s3) = s2.value as Storage.Cuda
                                    Src.Distinct(s3)
                                }
                                s2 is Src.Aliased -> {
                                    val (rel) = s2
                                    Src.Aliased(rel)
                                }
                                else -> {
                                    error("same_device above rejects mismatched backends")
                                }
                            }
                        Pair(s4, l)
                    })
                when (val matchVal1 = allDistinct(operands1)) {
                    null -> c.cudaFwdAliased(dst1, dstL, operands1)
                    else -> {
                        val distinct1 = matchVal1
                        c.cudaFwd(dst1, dstL, distinct1)
                    }
                }
            }
            is Storage.Metal -> {
                val (dst2) = this
                val operands2: List<Pair<Src, Layout>> =
                    fromFn({ i ->
                        val (s2, l) = srcs[i]
                        val s4 =
                            when {
                                s2 is Src.Distinct && s2.value is Storage.Metal -> {
                                    val (s3) = s2.value as Storage.Metal
                                    Src.Distinct(s3)
                                }
                                s2 is Src.Aliased -> {
                                    val (rel) = s2
                                    Src.Aliased(rel)
                                }
                                else -> {
                                    error("same_device above rejects mismatched backends")
                                }
                            }
                        Pair(s4, l)
                    })
                when (val matchVal2 = allDistinct(operands2)) {
                    null -> c.metalFwdAliased(dst2, dstL, operands2)
                    else -> {
                        val distinct2 = matchVal2
                        c.metalFwd(dst2, dstL, distinct2)
                    }
                }
            }
        }
    }

    internal fun <B : UnaryOpT> unaryImpl(layout: Layout): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.unaryImpl<B>(layout)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.unaryImpl<B>(layout)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.unaryImpl<B>(layout)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun <B : BinaryOpT> binaryImpl(rhs: Storage, lhsLayout: Layout, rhsLayout: Layout): Result<Storage> {
        when (val tryResult = this.sameDevice(rhs, B.NAME)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(rhs, B.NAME)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (lhs) = matchVal.first as Storage.Cpu
                    val (rhs1) = matchVal.second as Storage.Cpu
                    val storage =
                        when (val tryResult = lhs.binaryImpl<B>(rhs1, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 =
                        when (val tryResult = lhs1.binaryImpl<B>(rhs2, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 =
                        when (val tryResult = lhs2.binaryImpl<B>(rhs3, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    // Should not happen because of the same device check above but we're defensive
                    // anyway.
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = B.NAME).bt())
                }
            }
        }
    }

    internal fun conv1d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConv1D): Result<Storage> {
        when (val tryResult = this.sameDevice(kernel, "conv1d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(kernel, "conv1d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s =
                        when (val tryResult = inp.conv1d(l, kernel1, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 =
                        when (val tryResult = inp1.conv1d(l, kernel2, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 =
                        when (val tryResult = inp2.conv1d(l, kernel3, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv1d").bt())
                }
            }
        }
    }

    internal fun convTranspose1d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConvTranspose1D): Result<Storage> {
        when (val tryResult = this.sameDevice(kernel, "conv-transpose1d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(kernel, "conv-transpose1d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s =
                        when (val tryResult = inp.convTranspose1d(l, kernel1, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 =
                        when (val tryResult = inp1.convTranspose1d(l, kernel2, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 =
                        when (val tryResult = inp2.convTranspose1d(l, kernel3, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv-transpose1d").bt())
                }
            }
        }
    }

    internal fun conv2d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConv2D): Result<Storage> {
        when (val tryResult = this.sameDevice(kernel, "conv2d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(kernel, "conv2d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s =
                        when (val tryResult = inp.conv2d(l, kernel1, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 =
                        when (val tryResult = inp1.conv2d(l, kernel2, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 =
                        when (val tryResult = inp2.conv2d(l, kernel3, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv2d").bt())
                }
            }
        }
    }

    internal fun convTranspose2d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConvTranspose2D): Result<Storage> {
        when (val tryResult = this.sameDevice(kernel, "conv_transpose2d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(kernel, "conv_transpose2d")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s =
                        when (val tryResult = inp.convTranspose2d(l, kernel1, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 =
                        when (val tryResult = inp1.convTranspose2d(l, kernel2, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 =
                        when (val tryResult = inp2.convTranspose2d(l, kernel3, kernelL, params)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv_transpose2d").bt())
                }
            }
        }
    }

    internal fun avgPool2d(layout: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.avgPool2d(layout, kernelSize, stride)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.avgPool2d(layout, kernelSize, stride)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.avgPool2d(layout, kernelSize, stride)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun maxPool2d(layout: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.maxPool2d(layout, kernelSize, stride)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.maxPool2d(layout, kernelSize, stride)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.maxPool2d(layout, kernelSize, stride)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun upsampleNearest1d(layout: Layout, sz: Int): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.upsampleNearest1d(layout, sz)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.upsampleNearest1d(layout, sz)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.upsampleNearest1d(layout, sz)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun upsampleNearest2d(layout: Layout, h: Int, w: Int): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.upsampleNearest2d(layout, h, w)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.upsampleNearest2d(layout, h, w)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.upsampleNearest2d(layout, h, w)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun upsampleBilinear2d(layout: Layout, h: Int, w: Int, alignCorners: Boolean, scaleH: Double?, scaleW: Double?): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 =
                    when (val tryResult = storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 =
                    when (val tryResult = storage2.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 =
                    when (val tryResult = storage4.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                Result.Ok(Storage.Metal(storage5))
            }
        }
    }

    internal fun whereCond(layout: Layout, t: Storage, layoutT: Layout, f: Storage, layoutF: Layout): Result<Storage> {
        when (val tryResult = this.sameDevice(t, "where")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDevice(f, "where")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = t.sameDtype(f, "where")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Triple(this, t, f)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu && matchVal.third is Storage.Cpu -> {
                    val (cond) = matchVal.first as Storage.Cpu
                    val (t1) = matchVal.second as Storage.Cpu
                    val (f1) = matchVal.third as Storage.Cpu
                    val storage =
                        when (val tryResult = cond.whereCond(layout, t1, layoutT, f1, layoutF)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (cond1) = matchVal.first as Cuda
                    val (t2) = matchVal.second as Cuda
                    val (f2) = matchVal.third as Cuda
                    val storage1 =
                        when (val tryResult = cond1.whereCond(layout, t2, layoutT, f2, layoutF)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (cond2) = matchVal.first as Metal
                    val (t3) = matchVal.second as Metal
                    val (f3) = matchVal.third as Metal
                    val storage2 =
                        when (val tryResult = cond2.whereCond(layout, t3, layoutT, f3, layoutF)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> {
                    val lhs = matchVal.second
                    val rhs = matchVal.third
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "where").bt())
                }
            }
        }
    }

    internal fun gather(l: Layout, indexes: Storage, indexesL: Layout, d: Int): Result<Storage> {
        when (val tryResult = this.sameDevice(indexes, "index-add")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, indexes)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val storage =
                        when (val tryResult = s.gather(l, indexes1, indexesL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val storage1 =
                        when (val tryResult = s1.gather(l, indexes2, indexesL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val storage2 =
                        when (val tryResult = s2.gather(l, indexes3, indexesL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> error("unreachable")
            }
        }
    }

    internal fun scatterSet(l: Layout, indexes: Storage, indexesL: Layout, source: Storage, sourceL: Layout, d: Int): Result<Unit> {
        when (val tryResult = this.sameDevice(indexes, "scatter-set")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDevice(source, "scatter-set")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        run {
            val matchVal = Triple(this, indexes, source)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val (source1) = matchVal.third as Cpu
                    when (val tryResult = s.scatterSet(l, indexes1, indexesL, source1, sourceL, d)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val (source2) = matchVal.third as Cuda
                    when (val tryResult = s1.scatterSet(l, indexes2, indexesL, source2, sourceL, d)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val (source3) = matchVal.third as Metal
                    when (val tryResult = s2.scatterSet(l, indexes3, indexesL, source3, sourceL, d)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                else -> error("unreachable")
            }
        }
        return Result.Ok(Unit)
    }

    internal fun scatterAdd(l: Layout, indexes: Storage, indexesL: Layout, source: Storage, sourceL: Layout, d: Int): Result<Unit> {
        when (val tryResult = this.sameDevice(indexes, "scatter-add")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDevice(source, "scatter-add")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        run {
            val matchVal = Triple(this, indexes, source)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val (source1) = matchVal.third as Cpu
                    when (val tryResult = s.scatterAddSet(l, indexes1, indexesL, source1, sourceL, d)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val (source2) = matchVal.third as Cuda
                    when (val tryResult = s1.scatterAddSet(l, indexes2, indexesL, source2, sourceL, d)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val (source3) = matchVal.third as Metal
                    when (val tryResult = s2.scatterAddSet(l, indexes3, indexesL, source3, sourceL, d)) {
                        is Result.Ok<*> -> tryResult.value
                        is Result.Err<*> -> return Result.Err(tryResult.error)
                    }
                }
                else -> error("unreachable")
            }
        }
        return Result.Ok(Unit)
    }

    internal fun indexAdd(l: Layout, indexes: Storage, indexesL: Layout, source: Storage, sourceL: Layout, d: Int): Result<Storage> {
        when (val tryResult = this.sameDevice(indexes, "index-add")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDevice(source, "index-add")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Triple(this, indexes, source)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val (source1) = matchVal.third as Cpu
                    val storage =
                        when (val tryResult = s.indexAdd(l, indexes1, indexesL, source1, sourceL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val (source2) = matchVal.third as Cuda
                    val storage1 =
                        when (val tryResult = s1.indexAdd(l, indexes2, indexesL, source2, sourceL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val (source3) = matchVal.third as Metal
                    val storage2 =
                        when (val tryResult = s2.indexAdd(l, indexes3, indexesL, source3, sourceL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> error("unreachable")
            }
        }
    }

    internal fun indexSelect(rhs: Storage, lhsL: Layout, rhsL: Layout, d: Int): Result<Storage> {
        when (val tryResult = this.sameDevice(rhs, "index-select")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (lhs) = matchVal.first as Cpu
                    val (rhs1) = matchVal.second as Cpu
                    val storage =
                        when (val tryResult = lhs.indexSelect(rhs1, lhsL, rhsL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 =
                        when (val tryResult = lhs1.indexSelect(rhs2, lhsL, rhsL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 =
                        when (val tryResult = lhs2.indexSelect(rhs3, lhsL, rhsL, d)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = "index-select").bt())
                }
            }
        }
    }

    internal fun matmul(rhs: Storage, bmnk: List<Any?>, lhsLayout: Layout, rhsLayout: Layout): Result<Storage> {
        when (val tryResult = this.sameDevice(rhs, "matmul")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        when (val tryResult = this.sameDtype(rhs, "matmul")) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (lhs) = matchVal.first as Cpu
                    val (rhs1) = matchVal.second as Cpu
                    val storage =
                        when (val tryResult = lhs.matmul(rhs1, bmnk, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 =
                        when (val tryResult = lhs1.matmul(rhs2, bmnk, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 =
                        when (val tryResult = lhs2.matmul(rhs3, bmnk, lhsLayout, rhsLayout)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }
                    Result.Ok(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = "matmul").bt())
                }
            }
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    internal fun copyStridedSrc(dst: Storage, dstOffset: Int, srcL: Layout): Result<Unit> {
        return run {
            val matchVal = Pair(this, dst)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (src) = matchVal.first as Cpu
                    val (dst1) = matchVal.second as Cpu
                    src.copyStridedSrc(dst1, dstOffset, srcL)
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (src1) = matchVal.first as Cuda
                    val (dst2) = matchVal.second as Cuda
                    Result.Ok(
                        when (val tryResult = src1.copyStridedSrc(dst2, dstOffset, srcL)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (src2) = matchVal.first as Metal
                    val (dst3) = matchVal.second as Metal
                    Result.Ok(
                        when (val tryResult = src2.copyStridedSrc(dst3, dstOffset, srcL)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "copy").bt())
                }
            }
        }
    }

    internal fun copy2d(dst: Storage, d1: Int, d2: Int, srcS: Int, dstS: Int, srcO: Int, dstO: Int): Result<Unit> {
        return run {
            val matchVal = Pair(this, dst)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (src) = matchVal.first as Cpu
                    val (dst1) = matchVal.second as Cpu
                    src.copy2d(dst1, d1, d2, srcS, dstS, srcO, dstO)
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (src1) = matchVal.first as Cuda
                    val (dst2) = matchVal.second as Cuda
                    Result.Ok(
                        when (val tryResult = src1.copy2d(dst2, d1, d2, srcS, dstS, srcO, dstO)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (src2) = matchVal.first as Metal
                    val (dst3) = matchVal.second as Metal
                    Result.Ok(
                        when (val tryResult = src2.copy2d(dst3, d1, d2, srcS, dstS, srcO, dstO)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        },
                    )
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.Err(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "copy2d").bt())
                }
            }
        }
    }
}

public typealias StorageRef = RwLockReadGuard<Storage>

public typealias StorageMutRef = RwLockWriteGuard<Storage>
