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
    public data class Cpu(public var value: CpuStorage) : Storage()
    public data class Cuda(public var value: CudaStorage) : Storage()
    public data class Metal(public var value: MetalStorage) : Storage()

    public open fun tryClone(layout: Layout): Result<Storage> {
        return when (this) {
            is Cpu -> {
                val (storage) = this
                Result.success(Storage.Cpu(storage.clone()))
            }
            is Cuda -> {
                val (storage1) = this
                val storage2 = storage1.tryClone(layout).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage2))
            }
            is Metal -> {
                val (storage3) = this
                val storage4 = storage3.tryClone(layout).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage4))
            }
        }
    }

    public open fun device(): Device =
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

    public open fun dtype(): DType =
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

    internal open fun sameDevice(rhs: Storage, op: String): Result<Unit> {
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
            Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs, rhs = rhs1, op = op).bt())
        } else {
            Result.success(Unit)
        }
    }

    internal open fun sameDtype(rhs: Storage, op: String): Result<Unit> {
        val lhs = this.dtype()
        val rhs1 = rhs.dtype()
        return if (lhs != rhs1) {
            Result.failure(Error.DTypeMismatchBinaryOp(lhs = lhs, rhs = rhs1, op = op).bt())
        } else {
            Result.success(Unit)
        }
    }

    internal open fun constSet(v: Scalar, l: Layout): Result<Unit> =
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

    internal open fun affine(layout: Layout, mul: Double, add: Double): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.affine(layout, mul, add).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.affine(layout, mul, add).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.affine(layout, mul, add).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun powf(layout: Layout, alpha: Double): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.powf(layout, alpha).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.powf(layout, alpha).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.powf(layout, alpha).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun elu(layout: Layout, alpha: Double): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.elu(layout, alpha).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.elu(layout, alpha).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.elu(layout, alpha).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun cmp(op: CmpOp, rhs: Storage, lhsLayout: Layout, rhsLayout: Layout): Result<Storage> {
        this.sameDevice(rhs, "cmp").getOrElse { return Result.failure(it) }
        this.sameDtype(rhs, "cmp").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (lhs) = matchVal.first as Storage.Cpu
                    val (rhs1) = matchVal.second as Storage.Cpu
                    val storage = lhs.cmp(op, rhs1, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 = lhs1.cmp(op, rhs2, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 = lhs2.cmp(op, rhs3, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    // Should not happen because of the same device check above but we're defensive
                    // anyway.
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = "cmp").bt())
                }
            }
        }
    }

    internal open fun reduceOp(op: ReduceOp, layout: Layout, s: List<Int>): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.reduceOp(op, layout, s).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.reduceOp(op, layout, s).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.reduceOp(op, layout, s).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun toDtype(layout: Layout, dtype: DType): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.toDtype(layout, dtype).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.toDtype(layout, dtype).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.toDtype(layout, dtype).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun applyOp1(l: Layout, c: CustomOp1): Result<Pair<Storage, Shape>> {
        return when (this) {
            is Cpu -> {
                val (storage) = this
                val (storage1, shape) = c.cpuFwd(storage, l).getOrElse { return Result.failure(it) }
                Result.success(Pair(Storage.Cpu(storage1), shape))
            }
            is Cuda -> {
                val (storage2) = this
                val (storage3, shape1) = c.cudaFwd(storage2, l).getOrElse { return Result.failure(it) }
                Result.success(Pair(Storage.Cuda(storage3), shape1))
            }
            is Metal -> {
                val (storage4) = this
                val (storage5, shape2) = c.metalFwd(storage4, l).getOrElse { return Result.failure(it) }
                Result.success(Pair(Storage.Metal(storage5), shape2))
            }
        }
    }

    internal open fun applyOp2(l1: Layout, t2: Storage, l2: Layout, c: CustomOp2): Result<Pair<Storage, Shape>> {
        this.sameDevice(t2, c.name()).getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, t2)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (s1) = matchVal.first as Cpu
                    val (s2) = matchVal.second as Cpu
                    val (s, shape) = c.cpuFwd(s1, l1, s2, l2).getOrElse { return Result.failure(it) }
                    Result.success(Pair(Storage.Cpu(s), shape))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (s11) = matchVal.first as Cuda
                    val (s21) = matchVal.second as Cuda
                    val (s1, shape1) = c.cudaFwd(s11, l1, s21, l2).getOrElse { return Result.failure(it) }
                    Result.success(Pair(Storage.Cuda(s1), shape1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (s12) = matchVal.first as Metal
                    val (s22) = matchVal.second as Metal
                    val (s2, shape2) = c.metalFwd(s12, l1, s22, l2).getOrElse { return Result.failure(it) }
                    Result.success(Pair(Storage.Metal(s2), shape2))
                }
                else -> error("unreachable")
            }
        }
    }

    internal open fun applyOp3(l1: Layout, t2: Storage, l2: Layout, t3: Storage, l3: Layout, c: CustomOp3): Result<Pair<Storage, Shape>> {
        this.sameDevice(t2, c.name()).getOrElse { return Result.failure(it) }
        this.sameDevice(t3, c.name()).getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Triple(this, t2, t3)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s1) = matchVal.first as Cpu
                    val (s2) = matchVal.second as Cpu
                    val (s3) = matchVal.third as Cpu
                    val (s, shape) = c.cpuFwd(s1, l1, s2, l2, s3, l3).getOrElse { return Result.failure(it) }
                    Result.success(Pair(Storage.Cpu(s), shape))
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s11) = matchVal.first as Cuda
                    val (s21) = matchVal.second as Cuda
                    val (s31) = matchVal.third as Cuda
                    val (s1, shape1) = c.cudaFwd(s11, l1, s21, l2, s31, l3).getOrElse { return Result.failure(it) }
                    Result.success(Pair(Storage.Cuda(s1), shape1))
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s12) = matchVal.first as Metal
                    val (s22) = matchVal.second as Metal
                    val (s32) = matchVal.third as Metal
                    val (s2, shape2) = c.metalFwd(s12, l1, s22, l2, s32, l3).getOrElse { return Result.failure(it) }
                    Result.success(Pair(Storage.Metal(s2), shape2))
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
    internal open fun <C : InplaceOpN<N>> inplaceOp(dstL: Layout, srcs: List<Pair<Src<Storage>, Layout>>, c: C, n: Int): Result<Unit> {
        // Aliased sources are `self`. Only distinct needs checking.
        for ((s, _) in srcs.iter()) {
            if (s is Src.Distinct) {
                val (s1) = s
                this.sameDevice(s1, c.name()).getOrElse { return Result.failure(it) }
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

    internal open fun <B : UnaryOpT> unaryImpl(layout: Layout): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.unaryImpl<B>(layout).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.unaryImpl<B>(layout).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.unaryImpl<B>(layout).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun <B : BinaryOpT> binaryImpl(rhs: Storage, lhsLayout: Layout, rhsLayout: Layout): Result<Storage> {
        this.sameDevice(rhs, B.NAME).getOrElse { return Result.failure(it) }
        this.sameDtype(rhs, B.NAME).getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (lhs) = matchVal.first as Storage.Cpu
                    val (rhs1) = matchVal.second as Storage.Cpu
                    val storage = lhs.binaryImpl<B>(rhs1, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 = lhs1.binaryImpl<B>(rhs2, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 = lhs2.binaryImpl<B>(rhs3, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    // Should not happen because of the same device check above but we're defensive
                    // anyway.
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = B.NAME).bt())
                }
            }
        }
    }

    internal open fun conv1d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConv1D): Result<Storage> {
        this.sameDevice(kernel, "conv1d").getOrElse { return Result.failure(it) }
        this.sameDtype(kernel, "conv1d").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s = inp.conv1d(l, kernel1, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 = inp1.conv1d(l, kernel2, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 = inp2.conv1d(l, kernel3, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv1d").bt())
                }
            }
        }
    }

    internal open fun convTranspose1d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConvTranspose1D): Result<Storage> {
        this.sameDevice(kernel, "conv-transpose1d").getOrElse { return Result.failure(it) }
        this.sameDtype(kernel, "conv-transpose1d").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s = inp.convTranspose1d(l, kernel1, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 = inp1.convTranspose1d(l, kernel2, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 = inp2.convTranspose1d(l, kernel3, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv-transpose1d").bt())
                }
            }
        }
    }

    internal open fun conv2d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConv2D): Result<Storage> {
        this.sameDevice(kernel, "conv2d").getOrElse { return Result.failure(it) }
        this.sameDtype(kernel, "conv2d").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s = inp.conv2d(l, kernel1, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 = inp1.conv2d(l, kernel2, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 = inp2.conv2d(l, kernel3, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv2d").bt())
                }
            }
        }
    }

    internal open fun convTranspose2d(l: Layout, kernel: Storage, kernelL: Layout, params: ParamsConvTranspose2D): Result<Storage> {
        this.sameDevice(kernel, "conv_transpose2d").getOrElse { return Result.failure(it) }
        this.sameDtype(kernel, "conv_transpose2d").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, kernel)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (inp) = matchVal.first as Storage.Cpu
                    val (kernel1) = matchVal.second as Storage.Cpu
                    val s = inp.convTranspose2d(l, kernel1, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(s))
                }
                matchVal.first is Storage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (inp1) = matchVal.first as Storage.Cuda
                    val (kernel2) = matchVal.second as Storage.Cuda
                    val s1 = inp1.convTranspose2d(l, kernel2, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(s1))
                }
                matchVal.first is Storage.Metal && matchVal.second is Storage.Metal -> {
                    val (inp2) = matchVal.first as Storage.Metal
                    val (kernel3) = matchVal.second as Storage.Metal
                    val s2 = inp2.convTranspose2d(l, kernel3, kernelL, params).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(s2))
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "conv_transpose2d").bt())
                }
            }
        }
    }

    internal open fun avgPool2d(layout: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.avgPool2d(layout, kernelSize, stride).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.avgPool2d(layout, kernelSize, stride).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.avgPool2d(layout, kernelSize, stride).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun maxPool2d(layout: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.maxPool2d(layout, kernelSize, stride).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.maxPool2d(layout, kernelSize, stride).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.maxPool2d(layout, kernelSize, stride).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun upsampleNearest1d(layout: Layout, sz: Int): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.upsampleNearest1d(layout, sz).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.upsampleNearest1d(layout, sz).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.upsampleNearest1d(layout, sz).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun upsampleNearest2d(layout: Layout, h: Int, w: Int): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.upsampleNearest2d(layout, h, w).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.upsampleNearest2d(layout, h, w).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.upsampleNearest2d(layout, h, w).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun upsampleBilinear2d(layout: Layout, h: Int, w: Int, alignCorners: Boolean, scaleH: Double?, scaleW: Double?): Result<Storage> {
        return when (this) {
            is Storage.Cpu -> {
                val (storage) = this
                val storage1 = storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cpu(storage1))
            }
            is Cuda -> {
                val (storage2) = this
                val storage3 = storage2.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW).getOrElse { return Result.failure(it) }
                Result.success(Storage.Cuda(storage3))
            }
            is Metal -> {
                val (storage4) = this
                val storage5 = storage4.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW).getOrElse { return Result.failure(it) }
                Result.success(Storage.Metal(storage5))
            }
        }
    }

    internal open fun whereCond(layout: Layout, t: Storage, layoutT: Layout, f: Storage, layoutF: Layout): Result<Storage> {
        this.sameDevice(t, "where").getOrElse { return Result.failure(it) }
        this.sameDevice(f, "where").getOrElse { return Result.failure(it) }
        t.sameDtype(f, "where").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Triple(this, t, f)
            when {
                matchVal.first is Storage.Cpu && matchVal.second is Storage.Cpu && matchVal.third is Storage.Cpu -> {
                    val (cond) = matchVal.first as Storage.Cpu
                    val (t1) = matchVal.second as Storage.Cpu
                    val (f1) = matchVal.third as Storage.Cpu
                    val storage = cond.whereCond(layout, t1, layoutT, f1, layoutF).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (cond1) = matchVal.first as Cuda
                    val (t2) = matchVal.second as Cuda
                    val (f2) = matchVal.third as Cuda
                    val storage1 = cond1.whereCond(layout, t2, layoutT, f2, layoutF).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (cond2) = matchVal.first as Metal
                    val (t3) = matchVal.second as Metal
                    val (f3) = matchVal.third as Metal
                    val storage2 = cond2.whereCond(layout, t3, layoutT, f3, layoutF).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> {
                    val lhs = matchVal.second
                    val rhs = matchVal.third
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "where").bt())
                }
            }
        }
    }

    internal open fun gather(l: Layout, indexes: Storage, indexesL: Layout, d: Int): Result<Storage> {
        this.sameDevice(indexes, "index-add").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, indexes)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val storage = s.gather(l, indexes1, indexesL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val storage1 = s1.gather(l, indexes2, indexesL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val storage2 = s2.gather(l, indexes3, indexesL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> error("unreachable")
            }
        }
    }

    internal open fun scatterSet(l: Layout, indexes: Storage, indexesL: Layout, source: Storage, sourceL: Layout, d: Int): Result<Unit> {
        this.sameDevice(indexes, "scatter-set").getOrElse { return Result.failure(it) }
        this.sameDevice(source, "scatter-set").getOrElse { return Result.failure(it) }
        run {
            val matchVal = Triple(this, indexes, source)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val (source1) = matchVal.third as Cpu
                    s.scatterSet(l, indexes1, indexesL, source1, sourceL, d).getOrElse { return Result.failure(it) }
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val (source2) = matchVal.third as Cuda
                    s1.scatterSet(l, indexes2, indexesL, source2, sourceL, d).getOrElse { return Result.failure(it) }
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val (source3) = matchVal.third as Metal
                    s2.scatterSet(l, indexes3, indexesL, source3, sourceL, d).getOrElse { return Result.failure(it) }
                }
                else -> error("unreachable")
            }
        }
        return Result.success(Unit)
    }

    internal open fun scatterAdd(l: Layout, indexes: Storage, indexesL: Layout, source: Storage, sourceL: Layout, d: Int): Result<Unit> {
        this.sameDevice(indexes, "scatter-add").getOrElse { return Result.failure(it) }
        this.sameDevice(source, "scatter-add").getOrElse { return Result.failure(it) }
        run {
            val matchVal = Triple(this, indexes, source)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val (source1) = matchVal.third as Cpu
                    s.scatterAddSet(l, indexes1, indexesL, source1, sourceL, d).getOrElse { return Result.failure(it) }
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val (source2) = matchVal.third as Cuda
                    s1.scatterAddSet(l, indexes2, indexesL, source2, sourceL, d).getOrElse { return Result.failure(it) }
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val (source3) = matchVal.third as Metal
                    s2.scatterAddSet(l, indexes3, indexesL, source3, sourceL, d).getOrElse { return Result.failure(it) }
                }
                else -> error("unreachable")
            }
        }
        return Result.success(Unit)
    }

    internal open fun indexAdd(l: Layout, indexes: Storage, indexesL: Layout, source: Storage, sourceL: Layout, d: Int): Result<Storage> {
        this.sameDevice(indexes, "index-add").getOrElse { return Result.failure(it) }
        this.sameDevice(source, "index-add").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Triple(this, indexes, source)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu && matchVal.third is Cpu -> {
                    val (s) = matchVal.first as Cpu
                    val (indexes1) = matchVal.second as Cpu
                    val (source1) = matchVal.third as Cpu
                    val storage = s.indexAdd(l, indexes1, indexesL, source1, sourceL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda && matchVal.third is Cuda -> {
                    val (s1) = matchVal.first as Cuda
                    val (indexes2) = matchVal.second as Cuda
                    val (source2) = matchVal.third as Cuda
                    val storage1 = s1.indexAdd(l, indexes2, indexesL, source2, sourceL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal && matchVal.third is Metal -> {
                    val (s2) = matchVal.first as Metal
                    val (indexes3) = matchVal.second as Metal
                    val (source3) = matchVal.third as Metal
                    val storage2 = s2.indexAdd(l, indexes3, indexesL, source3, sourceL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> error("unreachable")
            }
        }
    }

    internal open fun indexSelect(rhs: Storage, lhsL: Layout, rhsL: Layout, d: Int): Result<Storage> {
        this.sameDevice(rhs, "index-select").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (lhs) = matchVal.first as Cpu
                    val (rhs1) = matchVal.second as Cpu
                    val storage = lhs.indexSelect(rhs1, lhsL, rhsL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 = lhs1.indexSelect(rhs2, lhsL, rhsL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 = lhs2.indexSelect(rhs3, lhsL, rhsL, d).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = "index-select").bt())
                }
            }
        }
    }

    internal open fun matmul(rhs: Storage, bmnk: List<Any?>, lhsLayout: Layout, rhsLayout: Layout): Result<Storage> {
        this.sameDevice(rhs, "matmul").getOrElse { return Result.failure(it) }
        this.sameDtype(rhs, "matmul").getOrElse { return Result.failure(it) }
        return run {
            val matchVal = Pair(this, rhs)
            when {
                matchVal.first is Cpu && matchVal.second is Cpu -> {
                    val (lhs) = matchVal.first as Cpu
                    val (rhs1) = matchVal.second as Cpu
                    val storage = lhs.matmul(rhs1, bmnk, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cpu(storage))
                }
                matchVal.first is Cuda && matchVal.second is Cuda -> {
                    val (lhs1) = matchVal.first as Cuda
                    val (rhs2) = matchVal.second as Cuda
                    val storage1 = lhs1.matmul(rhs2, bmnk, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Cuda(storage1))
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (lhs2) = matchVal.first as Metal
                    val (rhs3) = matchVal.second as Metal
                    val storage2 = lhs2.matmul(rhs3, bmnk, lhsLayout, rhsLayout).getOrElse { return Result.failure(it) }
                    Result.success(Storage.Metal(storage2))
                }
                else -> {
                    val lhs3 = matchVal.first
                    val rhs4 = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs3.device().location(), rhs = rhs4.device().location(), op = "matmul").bt())
                }
            }
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    internal open fun copyStridedSrc(dst: Storage, dstOffset: Int, srcL: Layout): Result<Unit> {
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
                    Result.success(src1.copyStridedSrc(dst2, dstOffset, srcL).getOrElse { return Result.failure(it) })
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (src2) = matchVal.first as Metal
                    val (dst3) = matchVal.second as Metal
                    Result.success(src2.copyStridedSrc(dst3, dstOffset, srcL).getOrElse { return Result.failure(it) })
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "copy").bt())
                }
            }
        }
    }

    internal open fun copy2d(dst: Storage, d1: Int, d2: Int, srcS: Int, dstS: Int, srcO: Int, dstO: Int): Result<Unit> {
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
                    Result.success(src1.copy2d(dst2, d1, d2, srcS, dstS, srcO, dstO).getOrElse { return Result.failure(it) })
                }
                matchVal.first is Metal && matchVal.second is Metal -> {
                    val (src2) = matchVal.first as Metal
                    val (dst3) = matchVal.second as Metal
                    Result.success(src2.copy2d(dst3, d1, d2, srcS, dstS, srcO, dstO).getOrElse { return Result.failure(it) })
                }
                else -> {
                    val lhs = matchVal.first
                    val rhs = matchVal.second
                    Result.failure(Error.DeviceMismatchBinaryOp(lhs = lhs.device().location(), rhs = rhs.device().location(), op = "copy2d").bt())
                }
            }
        }
    }
}

public typealias StorageRef = RwLockReadGuard<Storage>

public typealias StorageMutRef = RwLockWriteGuard<Storage>
