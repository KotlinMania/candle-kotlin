// port-lint: source tensor.rs
package io.github.kotlinmania.candle.tensor

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.ModuleT
import io.github.kotlinmania.candle.ToUsize2
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.cudabackend.utils.S
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.device.NdArray
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dtype.FloatDType
import io.github.kotlinmania.candle.dtype.WithDType
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.one
import io.github.kotlinmania.candle.op.Abs
import io.github.kotlinmania.candle.op.Add
import io.github.kotlinmania.candle.op.BackpropOp
import io.github.kotlinmania.candle.op.BinaryOp
import io.github.kotlinmania.candle.op.Ceil
import io.github.kotlinmania.candle.op.CmpOp
import io.github.kotlinmania.candle.op.Cos
import io.github.kotlinmania.candle.op.Div
import io.github.kotlinmania.candle.op.Erf
import io.github.kotlinmania.candle.op.Exp
import io.github.kotlinmania.candle.op.Floor
import io.github.kotlinmania.candle.op.Gelu
import io.github.kotlinmania.candle.op.GeluErf
import io.github.kotlinmania.candle.op.Log
import io.github.kotlinmania.candle.op.Maximum
import io.github.kotlinmania.candle.op.Minimum
import io.github.kotlinmania.candle.op.Mul
import io.github.kotlinmania.candle.op.Neg
import io.github.kotlinmania.candle.op.Op
import io.github.kotlinmania.candle.op.Recip
import io.github.kotlinmania.candle.op.ReduceOp
import io.github.kotlinmania.candle.op.Relu
import io.github.kotlinmania.candle.op.Round
import io.github.kotlinmania.candle.op.Sign
import io.github.kotlinmania.candle.op.Silu
import io.github.kotlinmania.candle.op.Sin
import io.github.kotlinmania.candle.op.Sqr
import io.github.kotlinmania.candle.op.Sqrt
import io.github.kotlinmania.candle.op.Sub
import io.github.kotlinmania.candle.op.Tanh
import io.github.kotlinmania.candle.op.UnaryOp
import io.github.kotlinmania.candle.scalar.Scalar
import io.github.kotlinmania.candle.scalar.TensorOrScalar
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.shape.Dim
import io.github.kotlinmania.candle.shape.Dims
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.shape.ShapeWithOneHole
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candle.storage.StorageMutRef
import io.github.kotlinmania.candle.storage.StorageRef
import io.github.kotlinmania.candle.stridedindex.StridedBlocks
import io.github.kotlinmania.candle.stridedindex.StridedIndex
import io.github.kotlinmania.candle.zero
import io.github.kotlinmania.parkinglot.RwLock
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.sync.atomic

// Tensors are N-dimensional matrixes of elements using a single data type.

public typealias Target = Tensor

/**
 * Unique identifier for tensors.
 */
public data class TensorId(
    public var value: Int = 0,
) {
    public var field0: Int
        get() = value
        set(v) {
            value = v
        }

    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<TensorId> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<TensorId> {
                override val alignBytes: Long = 8L
                override val sizeBytes: Long = 8L
                override fun readNative(addr: Long): TensorId =
                    TensorId(
                        (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).readNative(addr),
                    )
                override fun writeNative(addr: Long, value: TensorId) {
                    (io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts.USIZE64).writeNative(addr, value.value)
                }
            }

        public fun new(): TensorId {
            val COUNTER: atomic.AtomicUsize = atomic.AtomicUsize.new(1)
            return TensorId(COUNTER.fetchAdd(1, atomic.Ordering.Relaxed))
        }
    }
}

public class Tensor internal constructor(
    public var id: TensorId,
    storageCtorArg: RwLock<Storage>,
    public var layout: Layout,
    public var op: BackpropOp,
    public var isVariable: Boolean,
    public var dtype: DType,
    public var device: Device,
) : AsRef<Tensor>,
    Add<B>,
    Sub<B>,
    Mul<B>,
    Div<B>,
    From<Pair<Storage, S>> {
    // As we provide inner mutability on the tensor content, the alternatives are:
    // - Using a mutex, this would have the highest cost when retrieving the storage but would
    // prevent errors when concurrent access takes place. Mutex would also be subject to
    // deadlocks for example using the current code if the same tensor is used twice by a single
    // binary op.
    // - Using a refcell unchecked cell would have some intermediary cost, borrow checking would be
    // verified dynamically, but the resulting tensors would not be send or sync.
    // - Using an unchecked cell would have the lowest cost but undefined behavior on concurrent
    // accesses.
    // Ideally, we would use Arc<Storage> for tensors on which we don't plan on modifying the data
    // and Arc<Mutex<Storage>> for tensors where the data could be modified, e.g. variables but
    // that's tricky to encode in the current setup.
    public var storage: RwLock<Storage> = storageCtorArg
        internal set

    public constructor() : this(
        id = TensorId(),
        storageCtorArg = RwLock<Storage>(),
        layout = Layout(),
        op = BackpropOp(),
        isVariable = false,
        dtype = DType.entries.first(),
        device = Device(),
    )

    public fun asRef(): Tensor = this

    public fun deref(): Tensor = this.value.asRef()

    public fun constSet(value: Scalar): Result<Unit> = this.storageMut().constSet(value, this.layout())

    public fun zeroSet(): Result<Unit> = this.constSet(zero(this.dtype()))

    public fun oneSet(): Result<Unit> = this.constSet(one(this.dtype()))

    /**
     * Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
     *
     * ```kotlin
     * import candlecore.DType
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
     * val b = a.onesLike().getOrElse { return Result.failure(it) }
     * // b == a + 1
     * ```
     */
    public fun onesLike(): Result<Tensor> = Tensor.ones(this.shape(), this.dtype(), this.device())

    /**
     * Creates a new tensor filled with zeros with same shape, dtype, and device as the other
     * tensor.
     *
     * ```kotlin
     * import candlecore.DType
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
     * val b = a.zerosLike().getOrElse { return Result.failure(it) }
     * // b is on CPU f32.
     * ```
     */
    public fun zerosLike(): Result<Tensor> = Tensor.zeros(this.shape(), this.dtype(), this.device())

    /**
     * Creates a new tensor filled with uninitialized memory of the same shape, dtype, and device as the other
     * tensor.
     *
     * # Safety
     * This returns uninitialized memory.
     *
     * ```kotlin
     * import candlecore.DType
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
     * val b = a.emptyLike().getOrElse { return Result.failure(it) }
     * ```
     */
    public fun emptyLike(): Result<Tensor> = Tensor.empty(this.shape(), this.dtype(), this.device())

    public fun randLike(lo: Double, up: Double): Result<Tensor> = Tensor.randF64Impl(lo, up, this.shape(), this.dtype(), this.device(), false)

    public fun randnLike(mean: Double, stdev: Double): Result<Tensor> = Tensor.randnF64Impl(mean, stdev, this.shape(), this.dtype(), this.device(), false)

    internal fun sameShapeBinaryOp(rhs: Tensor, op: String): Result<Shape> {
        val lhs = this.shape()
        val rhs1 = rhs.shape()
        return if (lhs != rhs1) {
            Err(Error.ShapeMismatchBinaryOp(lhs = lhs.clone(), rhs = rhs1.clone(), op = op).bt())
        } else {
            Ok(lhs)
        }
    }

    /**
     * Returns true if the computation graph should track this op, that is if it is
     * a variable or if it has some variable as dependencies.
     */
    public fun trackOp(): Boolean = this.isVariable || this.op.isSome()

    public fun add(rhs: Tensor): Result<Tensor> {
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs, "add")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().binaryImpl<Add>(rhs.storage(), this.layout(), rhs.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, rhs, { t1, t2 -> Op.Binary(t1, t2, BinaryOp.Add) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun mul(rhs: Tensor): Result<Tensor> {
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs, "mul")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().binaryImpl<Mul>(rhs.storage(), this.layout(), rhs.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, rhs, { t1, t2 -> Op.Binary(t1, t2, BinaryOp.Mul) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun sub(rhs: Tensor): Result<Tensor> {
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs, "sub")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().binaryImpl<Sub>(rhs.storage(), this.layout(), rhs.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, rhs, { t1, t2 -> Op.Binary(t1, t2, BinaryOp.Sub) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun div(rhs: Tensor): Result<Tensor> {
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs, "div")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().binaryImpl<Div>(rhs.storage(), this.layout(), rhs.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, rhs, { t1, t2 -> Op.Binary(t1, t2, BinaryOp.Div) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun <T : TensorOrScalar> maximum(rhs: T): Result<Tensor> {
        val rhs3 =
            when (
                val matchVal =
                    when (val tryResult = rhs.toTensorScalar()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            ) {
                is Tensor -> {
                    val (rhs1) = matchVal
                    rhs1
                }
                is Scalar -> {
                    val (rhs2) = matchVal
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = rhs2.toDtype(this.dtype())) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDevice(this.device())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.broadcastAs(this.shape())
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs3, "maximum")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (this.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().binaryImpl<Maximum>(rhs3.storage(), this.layout(), rhs3.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, rhs3, { t1, t2 -> Op.Binary(t1, t2, BinaryOp.Maximum) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun <T : TensorOrScalar> minimum(rhs: T): Result<Tensor> {
        val rhs3 =
            when (
                val matchVal =
                    when (val tryResult = rhs.toTensorScalar()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            ) {
                is Tensor -> {
                    val (rhs1) = matchVal
                    rhs1
                }
                is Scalar -> {
                    val (rhs2) = matchVal
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = rhs2.toDtype(this.dtype())) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDevice(this.device())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.broadcastAs(this.shape())
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs3, "minimum")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (this.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().binaryImpl<Minimum>(rhs3.storage(), this.layout(), rhs3.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, rhs3, { t1, t2 -> Op.Binary(t1, t2, BinaryOp.Minimum) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun broadcastAdd(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_add")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.add(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.add(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.add(rhs)
                else -> lhs.add(rhs)
            }
        }
    }

    public fun broadcastMul(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_mul")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.mul(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.mul(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.mul(rhs)
                else -> lhs.mul(rhs)
            }
        }
    }

    public fun broadcastSub(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_sub")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.sub(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.sub(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.sub(rhs)
                else -> lhs.sub(rhs)
            }
        }
    }

    public fun broadcastDiv(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_div")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.div(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.div(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.div(rhs)
                else -> lhs.div(rhs)
            }
        }
    }

    public fun broadcastMaximum(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_maximum")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.maximum(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.maximum(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.maximum(rhs)
                else -> lhs.maximum(rhs)
            }
        }
    }

    public fun broadcastMinimum(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_minimum")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.minimum(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.minimum(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.minimum(rhs)
                else -> lhs.minimum(rhs)
            }
        }
    }

    public fun broadcastEq(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_eq")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.eq(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.eq(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.eq(rhs)
                else -> lhs.eq(rhs)
            }
        }
    }

    public fun broadcastNe(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_ne")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.ne(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.ne(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.ne(rhs)
                else -> lhs.ne(rhs)
            }
        }
    }

    public fun broadcastLt(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_lt")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.lt(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.lt(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.lt(rhs)
                else -> lhs.lt(rhs)
            }
        }
    }

    public fun broadcastLe(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_le")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.le(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.le(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.le(rhs)
                else -> lhs.le(rhs)
            }
        }
    }

    public fun broadcastGt(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_gt")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.gt(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.gt(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.gt(rhs)
                else -> lhs.gt(rhs)
            }
        }
    }

    public fun broadcastGe(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val shape =
            when (val tryResult = lhs.shape().broadcastShapeBinaryOp(rhs.shape(), "broadcast_ge")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = shape != lhs.shape()
        val rBroadcast = shape != rhs.shape()
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.ge(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == false && matchVal.second == true ->
                    lhs.ge(
                        when (val tryResult = rhs.broadcastAs(shape)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (val tryResult = lhs.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.ge(rhs)
                else -> lhs.ge(rhs)
            }
        }
    }

    public fun recip(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Recip>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Recip) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun neg(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Neg>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Neg) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun exp(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Exp>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Exp) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun log(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Log>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Log) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun sin(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Sin>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Sin) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun cos(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Cos>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Cos) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun tanh(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Tanh>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Tanh) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun abs(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Abs>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Abs) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun sqr(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Sqr>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Sqr) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun sqrt(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Sqrt>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Sqrt) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun gelu(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Gelu>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Gelu) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun geluErf(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<GeluErf>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.GeluErf) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun erf(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Erf>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Erf) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun relu(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Relu>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Relu) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun silu(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Silu>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Silu) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun ceil(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Ceil>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Ceil) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun floor(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Floor>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Floor) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun round(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Round>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Round) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    public fun sign(): Result<Tensor> {
        val shape = this.shape()
        if (shape.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().unaryImpl<Sign>(this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { s -> Op.Unary(s, UnaryOp.Sign) })
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    /**
     * Round element of the input tensor to the nearest integer.
     *
     * If the number of decimals is negative, it specifies the number of positions to the left of
     * the decimal point.
     */
    public fun roundTo(decimals: Int): Result<Tensor> {
        val mult = 10.0.powi(decimals)
        return when (
            val tryResult =
                when (val tryResult = (this * mult)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.round()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        } * (1.0 / mult)
    }

    /**
     * Retrieves the single scalar value hold in the tensor. If the tensor contains multiple
     * dimensions, an error is returned instead.
     */
    public fun <S : WithDType> toScalar(cpuStorageAsSliceSFn: (Any?) -> Any?): Result<S> {
        if (this.rank() != 0) {
            when (val tryResult = Err(Error.UnexpectedNumberOfDims(expected = 0, got = this.rank(), shape = this.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val fromCpuStorage =
            closure@{ cpuStorage: CpuStorage ->
                val data =
                    when (val tryResult = cpuStorageAsSliceSFn(cpuStorage)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return@closure Result.fromResidual(Result.Err(tryResult.value))
                    }
                Ok(data[this.layout().startOffset()])
            }
        return when (val matchVal = this.storage()) {
            is Storage.Cpu -> {
                val (cpuStorage) = matchVal
                fromCpuStorage(cpuStorage)
            }
            is Storage.Cuda -> {
                val (storage) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is Storage.Metal -> {
                val (storage1) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage1.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
        }
    }

    /**
     * An alias for `toScalar`.
     */
    public fun <S : WithDType> toVec0(): Result<S> = this.toScalar<S>()

    /**
     * Repeat this tensor along the specified dimensions.
     */
    public fun repeat(shape: Shape): Result<Tensor> {
        // Similar to PyTorch, we extend the number of dimensions of self if needed.
        val repeats = shape
        val repeats1 = repeats.dims()
        var inp =
            if (this.rank() < repeats1.len()) {
                val shape1 = listOf(mutableListOf(1, repeats1.len() - this.rank()), this.dims().toVec()).concat()
                when (val tryResult = this.reshape(shape1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            } else {
                this.clone()
            }
        for ((idx, repeat) in repeats1.iter().enumerate()) {
            inp =
                when (repeat) {
                    0 ->
                        when (val tryResult = inp.narrow(idx, 0, 0)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    1 -> inp
                    else -> {
                        val repeat1 = repeat
                        when (val tryResult = Tensor.cat(mutableListOf(inp, repeat1), idx)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
        }
        return Ok(inp)
    }

    /**
     * This operation multiplies the input tensor by `mul` then adds `add` and return the result.
     * The input values `mul` and `add` are casted to the appropriate type so some rounding might
     * be performed.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val a1 = a.affine(4.0, -2.0).getOrElse { return Result.failure(it) }
     * check(a1.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(-2.0, 2.0), listOf(6.0, 10.0)))
     * ```
     */
    public fun affine(mul: Double, add: Double): Result<Tensor> {
        if (this.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().affine(this.layout(), mul, add)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { arg -> Op.Affine(arg = arg, mul = mul, add = add) })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    /**
     * Applies the Exponential Linear Unit (ELU) function on each element of the input tensor.
     */
    public fun elu(alpha: Double): Result<Tensor> {
        if (this.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().elu(this.layout(), alpha)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { t -> Op.Elu(t, alpha) })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    /**
     * Raise the tensor to some float exponent `e`.
     */
    public fun powf(e: Double): Result<Tensor> {
        if (this.elemCount() == 0) {
            return Ok(this.clone())
        }
        val storage =
            when (val tryResult = this.storage().powf(this.layout(), e)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { t -> Op.Powf(t, e) })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    internal fun checkDim(dim: Int, op: String): Result<Unit> {
        return if (dim >= this.dims().size) {
            when (val tryResult = Err(Error.DimOutOfRange(shape = this.shape().clone(), dim = dim, op = op).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        } else {
            Ok(Unit)
        }
    }

    /**
     * Split a tensor into the specified number of chunks, this may return less chunks than
     * specified.
     */
    public fun <D : Dim> chunk(chunks: Int, dim: D): Result<List<Tensor>> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "chunk")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val size =
            when (val tryResult = this.dim(dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return if (size < chunks) {
            (0 until size).map({ i -> this.narrow(dim1, i, 1) }).collect()
        } else {
            val chunkSize = size / chunks
            val cntAdditional = size % chunks
            val tensors = mutableListOf<Tensor>()
            var sumChunkSize = 0
            for (i in 0 until chunks) {
                val chunkSize1 =
                    if (i < cntAdditional) {
                        chunkSize + 1
                    } else {
                        chunkSize
                    }
                val tensor =
                    when (val tryResult = this.narrow(dim1, sumChunkSize, chunkSize1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                tensors.add(tensor)
                sumChunkSize += chunkSize1
            }
            Ok(tensors)
        }
    }

    /**
     * Returns a new tensor that is a narrowed version of the input, the dimension `dim`
     * ranges from `start` to `start + len`.
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.new(listOf(listOf(0.0f, 1.0, 2.0), listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val b = a.narrow(0, 1, 2).getOrElse { return Result.failure(it) }
     * check(b.shape().dims() == listOf(2, 3))
     * check(b.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(3.0, 4.0, 5.0), listOf(6.0, 7.0, 8.0)))
     * val c = a.narrow(1, 1, 1).getOrElse { return Result.failure(it) }
     * check(c.shape().dims() == listOf(3, 1))
     * check(c.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(1.0), listOf(4.0), listOf(7.0)))
     * ```
     */
    public fun <D : Dim> narrow(dim: D, start: Int, len: Int): Result<Tensor> {
        val dims = this.dims()
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "narrow")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val err =
            { msg ->
                Err(Error.NarrowInvalidArgs(shape = this.shape().clone(), dim = dim1, start = start, len = len, msg = msg).bt())
            }
        if (start > dims[dim1]) {
            when (val tryResult = err("start > dim_len")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        if (start.saturatingAdd(len) > dims[dim1]) {
            when (val tryResult = err("start + len > dim_len")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        return if (start == 0 && dims[dim1] == len) {
            Ok(this.clone())
        } else {
            val op = BackpropOp.new1(this, { t -> Op.Narrow(t, dim1, start, len) })
            val layout =
                when (val tryResult = this.layout().narrow(dim1, start, len)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val tensor = Tensor(id = TensorId.new(), storage = this.storage.clone(), layout = layout, op = op, isVariable = false, dtype = this.dtype, device = this.device.clone())
            Ok(Tensor(tensor))
        }
    }

    public fun squeezeDims(dims: List<Int>): Result<Tensor> =
        when {
            dims.isEmpty() -> Ok(this)
            dims.size == 1 -> {
                val i = dims[0]
                this.squeeze(i)
            }
            else -> {
                val dims1 = dims
                val dims2 =
                    this
                        .dims()
                        .enumerate()
                        .filterMap({ (dimIdx, v) ->
                            if (dims1.contains(dimIdx)) {
                                null
                            } else {
                                v
                            }
                        })
                        .collect<Vec>()
                this.reshape(dims2)
            }
        }

    public fun <D : Dim> reduceImpl(dim: D, keepdim: Boolean, op: ReduceOp): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), op.name())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val storage =
            when (val tryResult = this.storage().reduceOp(op, this.layout(), listOf(dim1))) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dims = this.dims().toVec()
        dims[dim1] = 1
        val op1 =
            when (op) {
                ReduceOp.Sum, ReduceOp.Min, ReduceOp.Max -> {
                    BackpropOp.new1(this, { arg -> Op.Reduce(arg, op, dims.toVec()) })
                }
                ReduceOp.ArgMin, ReduceOp.ArgMax -> BackpropOp.none()
            }
        val res = fromStorage(storage, dims, op1, false)
        return if (keepdim) {
            Ok(res)
        } else {
            res.squeezeDims(listOf(dim1))
        }
    }

    public fun <D : Dims> sumImpl(sumDims: D, keepdim: Boolean): Result<Tensor> {
        val sumDims1 =
            when (val tryResult = sumDims.toIndexes(this.shape(), "sum")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val storage =
            when (val tryResult = this.storage().reduceOp(ReduceOp.Sum, this.layout(), sumDims1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dims = this.dims().toVec()
        for (sumDim in sumDims1.iter()) {
            dims[sumDim] = 1
        }
        val op = BackpropOp.new1(this, { a -> Op.Reduce(a, ReduceOp.Sum, dims.toVec()) })
        val sum = fromStorage(storage, dims, op, false)
        return if (keepdim) {
            Ok(sum)
        } else {
            sum.squeezeDims(sumDims1)
        }
    }

    /**
     * Roll the tensor input along the given dimension.
     * Elements that are shifted beyond the last position are re-introduced at the first position.
     *
     * ```kotlin
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor1 = tensor.roll(1, 0).getOrElse { return Result.failure(it) }
     * check(tensor1.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(4.0, 5.0), listOf(0.0, 1.0), listOf(2.0, 3.0)))
     * val tensor2 = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor3 = tensor2.roll(-1, 0).getOrElse { return Result.failure(it) }
     * check(tensor3.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(2.0, 3.0), listOf(4.0, 5.0), listOf(0.0, 1.0)))
     * ```
     */
    public fun <D> roll(shift: Int, dim: D): Result<Tensor> where D : Dim {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "roll")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dimSize =
            when (val tryResult = this.dim(dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val shift1 = shift.remEuclid(dimSize.toInt()).toInt()
        return if (shift1 == 0) {
            Ok(this.clone())
        } else {
            val a =
                when (val tryResult = this.narrow(dim1, 0, dimSize - shift1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val b =
                when (val tryResult = this.narrow(dim1, dimSize - shift1, shift1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            Tensor.cat(listOf(b, a), dim1)
        }
    }

    /**
     * Returns the sum of all elements in the input tensor. The sum is performed over all the
     * input dimensions.
     *
     * The resulting tensor has a shape that is similar to the shape of the input tensor, except
     * that the number of elements for each dimension index in `sumDims` is 1.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val s = a.sumKeepdim(0).getOrElse { return Result.failure(it) }
     * check(s.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(2.0, 4.0)))
     * val s1 = a.sumKeepdim(1).getOrElse { return Result.failure(it) }
     * check(s1.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(1.0), listOf(5.0)))
     * val s2 = a.sumKeepdim(Pair(0, 1)).getOrElse { return Result.failure(it) }
     * check(s2.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(6.0)))
     * ```
     */
    public fun <D : Dims> sumKeepdim(sumDims: D): Result<Tensor> = this.sumImpl(sumDims, true)

    /**
     * Returns the sum of all elements in the input tensor. The sum is performed over all the
     * input dimensions and compared to `sumKeepdim` these dimensions are squeezed rather than
     * kept.
     */
    public fun <D : Dims> sum(sumDims: D): Result<Tensor> = this.sumImpl(sumDims, false)

    /**
     * Returns the mean of all elements in the input tensor. The mean is performed over all the
     * input dimensions.
     *
     * The resulting tensor has a shape that is similar to the shape of the input tensor, except
     * that the number of elements for each dimension index in `meanDims` is 1.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val a = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val s = a.meanKeepdim(0).getOrElse { return Result.failure(it) }
     * check(s.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(1.0, 2.0)))
     * val s1 = a.meanKeepdim(1).getOrElse { return Result.failure(it) }
     * check(s1.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(0.5), listOf(2.5)))
     * val s2 = a.meanKeepdim(Pair(0, 1)).getOrElse { return Result.failure(it) }
     * check(s2.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(1.5)))
     * ```
     */
    public fun <D : Dims> meanKeepdim(meanDims: D): Result<Tensor> {
        val meanDims1 =
            when (val tryResult = meanDims.toIndexes(this.shape(), "mean-keepdim")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val reducedDim: Int = meanDims1.iter().map({ i -> this.dims()[i] }).product()
        val scale = 1.0 / (reducedDim.toDouble())
        return when (val tryResult = this.sumImpl(meanDims1, true)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        } * scale
    }

    /**
     * Returns the mean of all elements in the input tensor. The mean is performed over all the
     * input dimensions and compared to `meanKeepdim` these dimensions are squeezed rather than
     * kept.
     */
    public fun <D : Dims> mean(meanDims: D): Result<Tensor> {
        val meanDims1 =
            when (val tryResult = meanDims.toIndexes(this.shape(), "mean")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val reducedDim: Int = meanDims1.iter().map({ i -> this.dims()[i] }).product()
        val scale = 1.0 / (reducedDim.toDouble())
        return when (val tryResult = this.sumImpl(meanDims1, false)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        } * scale
    }

    /**
     * Returns the unbiased variance over the selected dimension.
     */
    public fun <D : Dim> varKeepdim(dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "var")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val mean =
            when (val tryResult = this.meanKeepdim(dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val squares =
            when (
                val tryResult =
                    when (val tryResult = this.broadcastSub(mean)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.sqr()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return when (val tryResult = squares.sumImpl(dim1, true)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        } /
            (
                when (val tryResult = this.dim(dim1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                } - 1
            ).toDouble()
    }

    /**
     * Returns the unbiased variance over the selected dimension.
     */
    public fun <D : Dim> variable(dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "var")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return when (val tryResult = this.varKeepdim(dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.squeeze(dim1)
    }

    /**
     * Gathers the maximum value across the selected dimension. The resulting shape has the same
     * number of dimensions as the original tensor and the select dimension has a single element.
     */
    public fun <D : Dim> maxKeepdim(dim: D): Result<Tensor> = this.reduceImpl(dim, true, ReduceOp.Max)

    /**
     * Similar to `maxKeepdim` but the target dimension is squeezed.
     */
    public fun <D : Dim> max(dim: D): Result<Tensor> = this.reduceImpl(dim, false, ReduceOp.Max)

    /**
     * Gathers the minimum value across the selected dimension. The resulting shape has the same
     * number of dimensions as the original tensor and the select dimension has a single element.
     */
    public fun <D : Dim> minKeepdim(dim: D): Result<Tensor> = this.reduceImpl(dim, true, ReduceOp.Min)

    /**
     * Similar to `minKeepdim` but the target dimension is squeezed.
     */
    public fun <D : Dim> min(dim: D): Result<Tensor> = this.reduceImpl(dim, false, ReduceOp.Min)

    public fun <D : Dim> argmaxKeepdim(dim: D): Result<Tensor> = this.reduceImpl(dim, true, ReduceOp.ArgMax)

    /**
     * Similar to `argmaxKeepdim` but the target dimension is squeezed.
     */
    public fun <D : Dim> argmax(dim: D): Result<Tensor> = this.reduceImpl(dim, false, ReduceOp.ArgMax)

    public fun <D : Dim> argminKeepdim(dim: D): Result<Tensor> = this.reduceImpl(dim, true, ReduceOp.ArgMin)

    /**
     * Similar to `argminKeepdim` but the target dimension is squeezed.
     */
    public fun <D : Dim> argmin(dim: D): Result<Tensor> = this.reduceImpl(dim, false, ReduceOp.ArgMin)

    /**
     * Element-wise comparison between two tensors, e.g. equality, greater than, ... The actual
     * comparison operation is specified by the `op` argument.
     *
     * The returned tensor has the same shape as the original tensors and uses `UByte` elements.
     */
    public fun <T : TensorOrScalar> cmp(rhs: T, op: CmpOp): Result<Tensor> {
        val rhs3 =
            when (
                val matchVal =
                    when (val tryResult = rhs.toTensorScalar()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
            ) {
                is Tensor -> {
                    val (rhs1) = matchVal
                    rhs1
                }
                is Scalar -> {
                    val (rhs2) = matchVal
                    when (
                        val tryResult =
                            when (
                                val tryResult =
                                    when (val tryResult = rhs2.toDtype(this.dtype())) {
                                        is Result.Ok -> tryResult.value
                                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                    }.toDevice(this.device())
                            ) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.broadcastAs(this.shape())
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
            }
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(rhs3, "cmp")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val storage =
            when (val tryResult = this.storage().cmp(op, rhs3.storage(), this.layout(), rhs3.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op1 = BackpropOp.new1(this, { a -> Op.Cmp(a, op) })
        return Ok(fromStorage(storage, shape.dims(), op1, false))
    }

    /**
     * Element-wise equality.
     */
    public fun <T : TensorOrScalar> eq(rhs: T): Result<Tensor> = this.cmp(rhs, CmpOp.Eq)

    /**
     * Element-wise non-equality.
     */
    public fun <T : TensorOrScalar> ne(rhs: T): Result<Tensor> = this.cmp(rhs, CmpOp.Ne)

    /**
     * Element-wise comparison with lower-than, the returned tensor uses value 1 where `self <
     * rhs` and 0 otherwise.
     */
    public fun <T : TensorOrScalar> lt(rhs: T): Result<Tensor> = this.cmp(rhs, CmpOp.Lt)

    /**
     * Element-wise comparison with greater-than, the returned tensor uses value 1 where `self >
     * rhs` and 0 otherwise.
     */
    public fun <T : TensorOrScalar> gt(rhs: T): Result<Tensor> = this.cmp(rhs, CmpOp.Gt)

    /**
     * Element-wise comparison with greater-equal, the returned tensor uses value 1 where `self >=
     * rhs` and 0 otherwise.
     */
    public fun <T : TensorOrScalar> ge(rhs: T): Result<Tensor> = this.cmp(rhs, CmpOp.Ge)

    /**
     * Element-wise comparison with lower-equal, the returned tensor uses value 1 where `self <=
     * rhs` and 0 otherwise.
     */
    public fun <T : TensorOrScalar> le(rhs: T): Result<Tensor> = this.cmp(rhs, CmpOp.Le)

    /**
     * Clamp the tensor values to be between `min` and `max`.
     */
    public fun <T1 : TensorOrScalar, T2 : TensorOrScalar> clamp(min: T1, max: T2): Result<Tensor> {
        return when (val tryResult = this.maximum(min)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.minimum(max)
    }

    /**
     * Interpolate the input tensor to the `targetSize` size, taking the value of the nearest element.
     *
     * The input tensor should have three dimensions, `(batch, channels, l)`, the returned
     * tensor also has three dimensions, `(batch, channels, targetSize)`.
     */
    public fun interpolate1d(targetSize: Int): Result<Tensor> {
        val (n, c, l) =
            when (val tryResult = this.dims3()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { arg -> Op.UpsampleNearest1D(arg = arg, targetSize = targetSize) })
        val storage =
            when (val tryResult = this.storage().upsampleNearest1d(this.layout(), targetSize)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(fromStorage(storage, Triple(n, c, targetSize), op, false))
    }

    /**
     * Alias for `interpolate1d`.
     */
    public fun upsampleNearest1d(targetSize: Int): Result<Tensor> = this.interpolate1d(targetSize)

    /**
     * Interpolate the input tensor to the `(targetH, targetW)` size, taking the value of the
     * nearest element.
     *
     * The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
     * tensor also has four dimensions, `(batch, channels, targetH, targetW)`.
     */
    public fun interpolate2d(targetH: Int, targetW: Int): Result<Tensor> {
        val (n, c, h, w) =
            when (val tryResult = this.dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { arg -> Op.UpsampleNearest2D(arg = arg, targetH = targetH, targetW = targetW) })
        val storage =
            when (val tryResult = this.storage().upsampleNearest2d(this.layout(), targetH, targetW)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(fromStorage(storage, listOf(n, c, targetH, targetW), op, false))
    }

    /**
     * Alias for `interpolate2d`.
     */
    public fun upsampleNearest2d(targetH: Int, targetW: Int): Result<Tensor> = this.interpolate2d(targetH, targetW)

    /**
     * Bilinear interpolation to resize the input tensor to the specified size.
     *
     * The input tensor should have four dimensions: `(batch, channels, h, w)`.
     * The returned tensor also has four dimensions: `(batch, channels, targetH, targetW)`.
     *
     * # Arguments
     *
     * * `targetH` - Target height
     * * `targetW` - Target width
     * * `alignCorners` - If true, corner pixels are aligned. If false (default),
     *   pixels are treated as areas (matches PyTorch default behavior).
     *
     * # Example
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val t = Tensor.arange(0.0f, 16.0f, Device.Cpu).getOrElse { return Result.failure(it) }.reshape(listOf(1, 1, 4, 4)).getOrElse { return Result.failure(it) }
     * val upsampled = t.upsampleBilinear2d(8, 8, false).getOrElse { return Result.failure(it) }
     * check(upsampled.dims() == listOf(1, 1, 8, 8))
     * ```
     */
    public fun upsampleBilinear2d(targetH: Int, targetW: Int, alignCorners: Boolean): Result<Tensor> {
        val (n, c, h, w) =
            when (val tryResult = this.dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new1(this, { arg -> Op.UpsampleBilinear2D(arg = arg, targetH = targetH, targetW = targetW, alignCorners = alignCorners) })
        // Pass None for scale factors (size mode)
        val storage =
            when (val tryResult = this.storage().upsampleBilinear2d(this.layout(), targetH, targetW, alignCorners, null, null)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(fromStorage(storage, listOf(n, c, targetH, targetW), op, false))
    }

    /**
     * Bilinear interpolation using scale factors.
     *
     * Similar to `upsampleBilinear2d` but uses scale factors instead of absolute sizes.
     * This matches PyTorch's `interpolate(scaleFactor = ...)` behavior.
     *
     * # Arguments
     *
     * * `scaleH` - Height scaling factor
     * * `scaleW` - Width scaling factor
     * * `alignCorners` - If true, corner pixels are aligned
     *
     * # Example
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val t = Tensor.arange(0.0f, 16.0f, Device.Cpu).getOrElse { return Result.failure(it) }.reshape(listOf(1, 1, 4, 4)).getOrElse { return Result.failure(it) }
     * // Scale by 2x in both dimensions
     * val upsampled = t.upsampleBilinear2dWithScale(2.0, 2.0, false).getOrElse { return Result.failure(it) }
     * check(upsampled.dims() == listOf(1, 1, 8, 8))
     * ```
     */
    public fun upsampleBilinear2dWithScale(scaleH: Double, scaleW: Double, alignCorners: Boolean): Result<Tensor> {
        val (n, c, heightIn, widthIn) =
            when (val tryResult = this.dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // Calculate output size (floor, matching PyTorch)
        val heightOut = (heightIn.toDouble() * scaleH).floor().toInt()
        val widthOut = (widthIn.toDouble() * scaleW).floor().toInt()
        // Early return if size unchanged
        if (heightIn == heightOut && widthIn == widthOut) {
            return Ok(this.clone())
        }
        val op = BackpropOp.new1(this, { arg -> Op.UpsampleBilinear2D(arg = arg, targetH = heightOut, targetW = widthOut, alignCorners = alignCorners) })
        // Pass original scale factors (scaleFactor mode)
        // This ensures PyTorch-compatible scale calculation
        val storage =
            when (val tryResult = this.storage().upsampleBilinear2d(this.layout(), heightOut, widthOut, alignCorners, scaleH, scaleW)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(fromStorage(storage, listOf(n, c, heightOut, widthOut), op, false))
    }

    /**
     * 2D average pooling over an input tensor with multiple channels.
     *
     * The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
     * tensor also has four dimensions, `(batch, channels, h', w')`. The pooling is performed on
     * the two last dimensions using a kernel of size `sz`. The returned element is the average
     * value over the kernel window.
     */
    public fun <T : ToUsize2> avgPool2d(sz: T): Result<Tensor> {
        val sz1 = sz.toUsize2()
        return this.avgPool2dWithStride(sz1, sz1)
    }

    /**
     * Same as `avgPool2d` but with a `stride` that can be set to a value different from the
     * kernel size.
     */
    public fun <T : ToUsize2> avgPool2dWithStride(kernelSize: T, stride: T): Result<Tensor> {
        val kernelSize1 = kernelSize.toUsize2()
        val stride1 = stride.toUsize2()
        val (n, c, h, w) =
            when (val tryResult = this.dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (h < kernelSize1.value || w < kernelSize1.field1) {
            bail("kernel-size {kernel_size:?} is larger than the input size {h},{w}")
        }
        // https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html#torch.nn.AvgPool2d
        val hOut = (h - kernelSize1.value) / stride1.value + 1
        val wOut = (w - kernelSize1.field1) / stride1.field1 + 1
        val op = BackpropOp.new1(this, { arg -> Op.AvgPool2D(arg = arg, kernelSize = kernelSize1, stride = stride1) })
        val storage =
            when (val tryResult = this.storage().avgPool2d(this.layout(), kernelSize1, stride1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(fromStorage(storage, listOf(n, c, hOut, wOut), op, false))
    }

    /**
     * 2D max pooling over an input tensor with multiple channels.
     *
     * The input tensor should have four dimensions, `(batch, channels, h, w)`, the returned
     * tensor also has four dimensions, `(batch, channels, h', w')`. The pooling is performed on
     * the two last dimensions using a kernel of size `sz`, the returned element is the maximum
     * value over the kernel window.
     */
    public fun <T : ToUsize2> maxPool2d(sz: T): Result<Tensor> {
        val sz1 = sz.toUsize2()
        return this.maxPool2dWithStride(sz1, sz1)
    }

    /**
     * Same as `maxPool2d` but with a `stride` that can be set to a value different from the
     * kernel size.
     */
    public fun <T : ToUsize2> maxPool2dWithStride(kernelSize: T, stride: T): Result<Tensor> {
        val kernelSize1 = kernelSize.toUsize2()
        val stride1 = stride.toUsize2()
        val (n, c, h, w) =
            when (val tryResult = this.dims4()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (h < kernelSize1.value || w < kernelSize1.field1) {
            bail("kernel-size {kernel_size:?} is larger than the input size {h},{w}")
        }
        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        val hOut = (h - kernelSize1.value) / stride1.value + 1
        val wOut = (w - kernelSize1.field1) / stride1.field1 + 1
        val op = BackpropOp.new1(this, { arg -> Op.MaxPool2D(arg = arg, kernelSize = kernelSize1, stride = stride1) })
        val storage =
            when (val tryResult = this.storage().maxPool2d(this.layout(), kernelSize1, stride1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(fromStorage(storage, listOf(n, c, hOut, wOut), op, false))
    }

    /**
     * Computes the dot product of two 1D tensors.
     *
     * - If inputs are 1D vectors (`[n]`), returns their scalar dot product.
     * - Panics if shapes are not compatible
     * - Not supported for integer dtypes
     *
     * # Example (vectors)
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val t1 = Tensor.new(listOf(1.0, 2.0, 3.0), Device.Cpu).getOrElse { return Result.failure(it) }
     * val t2 = Tensor.new(listOf(4.0, 5.0, 6.0), Device.Cpu).getOrElse { return Result.failure(it) }
     * val res = t1.dot(t2).getOrElse { return Result.failure(it) }
     * check(res.toScalar<Double>().getOrElse { return Result.failure(it) } == 32.0)
     * ```
     */
    public fun dot(rhs: Tensor): Result<Tensor> {
        if (this.dims().size != 1 || rhs.dims().size != 1) {
            return Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = rhs.shape().clone(), op = "dot"))
        }
        return (this * rhs).andThen({ ret -> ret.sumAll() })
    }

    /**
     * Computes the **Frobenius norm** (L2 norm of all elements) of the tensor.
     * - Output is `sqrt(sum(x ^ 2))`.
     * - Always returns a scalar (`[]` shape).
     *
     * # Example
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val t = Tensor.new(listOf(listOf(3.0, 4.0), listOf(0.0, 0.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val norm = t.norm().getOrElse { return Result.failure(it) }
     * check(norm.toScalar<Double>().getOrElse { return Result.failure(it) } == 5.0)
     * ```
     */
    public fun norm(): Result<Tensor> {
        if (this.dtype().isInt()) {
            bail("norm not supported for integer dtypes")
        }
        return this.sqr().andThen({ x -> x.sumAll() }).andThen({ x -> x.sqrt() })
    }

    /**
     * Performs strict matrix-vector multiplication (`[m, n] * [n] = [m]`).
     *
     * - If `self` is a matrix (`[m, n]`) and `rhs` is a vector (`[n]`), returns a vector (`[m]`).
     * - **No broadcasting**: Panics if `self` is not 2D or if `rhs` is not 1D with matching size.
     *
     * # Example
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val mat = Tensor.new(listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val vec = Tensor.new(listOf(1.0, 1.0, 1.0), Device.Cpu).getOrElse { return Result.failure(it) }
     * val res = mat.mv(vec).getOrElse { return Result.failure(it) }
     * check(res.toVec1<Double>().getOrElse { return Result.failure(it) } == listOf(6.0, 15.0))
     * ```
     */
    public fun mv(rhs: Tensor): Result<Tensor> {
        // Strict shape checks
        val lhsDims = this.dims()
        val rhsDims = rhs.dims()
        if (lhsDims.size != 2 || rhsDims.size != 1 || lhsDims[1] != rhsDims[0]) {
            return Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = rhs.shape().clone(), op = "mv"))
        }
        // Direct matmul after ensuring rhs is column vector
        return when (
            val tryResult =
                this.matmul(
                    when (val tryResult = rhs.unsqueeze(1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.squeeze(1)
    }

    /**
     * Returns the matrix-multiplication of the input tensor with the other provided tensor.
     *
     * # Arguments
     *
     * * `self` - A tensor with dimensions `b1, b2, ..., bi, m, k`.
     * * `rhs` - A tensor with dimensions `b1, b2, ..., bi, k, n`.
     *
     * The resulting tensor has dimensions `b1, b2, ..., bi, m, n`.
     */
    public fun matmul(rhs: Tensor): Result<Tensor> {
        val aDims = this.shape().dims()
        val bDims = rhs.shape().dims()
        val dim = aDims.len()
        if (dim < 2 || bDims.len() != dim) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = rhs.shape().clone(), op = "matmul").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val m = aDims[dim - 2]
        val k = aDims[dim - 1]
        val k2 = bDims[dim - 2]
        val n = bDims[dim - 1]
        val cShape = Shape.from(((aDims).slice(0 until dim - 2))).extend(listOf(m, n))
        val batching: Int = ((aDims).slice(0 until dim - 2)).iter().product()
        val batchingB: Int = ((bDims).slice(0 until dim - 2)).iter().product()
        if (k != k2 || batching != batchingB) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = rhs.shape().clone(), op = "matmul").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        if (cShape.elemCount() == 0 || k == 0) {
            run {
                val lhsStorage = this.storage()
                val rhsStorage = rhs.storage()
                when (val tryResult = lhsStorage.sameDevice(rhsStorage, "matmul")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
                when (val tryResult = lhsStorage.sameDtype(rhsStorage, "matmul")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
            val storage =
                when (val tryResult = this.device().zeros(cShape, this.dtype())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val op = BackpropOp.new2(this, rhs, Op.Matmul)
            return Ok(fromStorage(storage, cShape, op, false))
        }
        val storage1 =
            when (val tryResult = this.storage().matmul(rhs.storage(), listOf(batching, m, n, k), this.layout(), rhs.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op1 = BackpropOp.new2(this, rhs, Op.Matmul)
        return Ok(fromStorage(storage1, cShape, op1, false))
    }

    /**
     * Matrix-multiplication with broadcasting support.
     *
     * Compared to `matmul` the two matrixes are allowed to have different dimensions as long as
     * they are compatible for broadcast. E.g. if `self` has shape `(j, 1, n, k)` and `rhs` has
     * shape `(l, k, m)`, the output will have shape `(j, l, n, m)`.
     */
    public fun broadcastMatmul(rhs: Tensor): Result<Tensor> {
        val lhs = this
        val (lShape, rShape) =
            when (val tryResult = lhs.shape().broadcastShapeMatmul(rhs.shape())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val lBroadcast = lShape != lhs.shape()
        val rBroadcast = rShape != rhs.shape()
        // Note: Avoid concretising the broadcasted matrixes via contiguous.
        return run {
            val matchVal = Pair(lBroadcast, rBroadcast)
            when {
                matchVal.first == true && matchVal.second == true ->
                    when (
                        val tryResult =
                            when (val tryResult = lhs.broadcastAs(lShape)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.contiguous()
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.matmul(
                        when (
                            val tryResult =
                                when (val tryResult = rhs.broadcastAs(rShape)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }.contiguous()
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                // A rank-2 rhs is only broadcast over the batch dimensions, so the whole product is
                // a single 2D matmul once the leading dims of lhs are folded into its row dimension.
                // Broadcasting the rhs instead would copy it `batch` times -- for an lmHead that is
                // the entire vocabulary matrix, per call. Same trick, and same contiguity guard, as
                // `candleNn.Linear.forward`.
                matchVal.first == false && matchVal.second == true && rhs.rank() == 2 && lhs.isContiguous() -> {
                    val (lhsDims, rhsDims) = Pair(lhs.dims(), rhs.dims())
                    val (m, k) = Pair(lhsDims[lhs.rank() - 2], lhsDims[lhs.rank() - 1])
                    val n = rhsDims[1]
                    val batch: Int = ((lhsDims).slice(0 until lhs.rank() - 2)).iter().product()
                    val outDims = lhsDims.toVec()
                    outDims.pop()
                    outDims.push(n)
                    when (
                        val tryResult =
                            when (val tryResult = lhs.reshape(Pair(batch * m, k))) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.matmul(rhs)
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.reshape(outDims)
                }
                matchVal.first == false && matchVal.second == true ->
                    lhs.matmul(
                        when (
                            val tryResult =
                                when (val tryResult = rhs.broadcastAs(rShape)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }.contiguous()
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                matchVal.first == true && matchVal.second == false ->
                    when (
                        val tryResult =
                            when (val tryResult = lhs.broadcastAs(lShape)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.contiguous()
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.matmul(rhs)
                else -> lhs.matmul(rhs)
            }
        }
    }

    /**
     * Returns a tensor with the same shape as the input tensor, the values are taken from
     * `onTrue` if the input tensor value is not zero, and `onFalse` at the positions where the
     * input tensor is equal to zero.
     */
    public fun whereCond(onTrue: Tensor, onFalse: Tensor): Result<Tensor> {
        val shape =
            when (val tryResult = this.sameShapeBinaryOp(onFalse, "where_cond")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val storage =
            when (val tryResult = this.storage().whereCond(this.layout(), onTrue.storage(), onTrue.layout(), onFalse.storage(), onFalse.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new3(this, onTrue, onFalse, Op.WhereCond)
        return Ok(fromStorage(storage, shape, op, false))
    }

    /**
     * Returns a tensor with the values from the `self` tensor at the index corresponding to the
     * values hold in the `ids` tensor.
     *
     * # Arguments
     *
     * * `self` - A tensor with dimensions `v, h`.
     * * `ids` - A tensor with dimensions `s` and with integer values between 0 and v (exclusive).
     *
     * The resulting tensor has dimensions `s, h`. `s` is called the sequence length, `v` the
     * vocabulary size, and `h` the hidden size.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val values = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val ids = Tensor.new(listOf(2u, 1u, 2u), Device.Cpu).getOrElse { return Result.failure(it) }
     * val emb = values.embedding(ids).getOrElse { return Result.failure(it) }
     * check(emb.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(4.0, 5.0), listOf(2.0, 3.0), listOf(4.0, 5.0)))
     * ```
     */
    public fun embedding(ids: Tensor): Result<Tensor> {
        if (this.rank() != 2 || ids.rank() != 1) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = ids.shape().clone(), op = "embedding").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        return this.indexSelect(ids, 0)
    }

    public fun scatterChecks(indexes: Tensor, source: Tensor, dim: Int): Result<Unit> {
        val sourceDims = source.dims()
        val selfDims = this.dims()
        val mismatch1 =
            if (sourceDims.size != selfDims.size) {
                true
            } else {
                var mismatch = false
                for ((i, d1, d2) in (selfDims.zip(sourceDims)).enumerate()) {
                    if (i != dim && d1 != d2) {
                        mismatch = true
                        break
                    }
                }
                mismatch
            }
        if (mismatch1) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(op = "scatter (self, src)", lhs = this.shape().clone(), rhs = source.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        if (indexes.dims() != source.dims()) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(op = "scatter (indexes, src)", lhs = indexes.shape().clone(), rhs = source.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        return Ok(Unit)
    }

    public fun <D : Dim> scatter(indexes: Tensor, source: Tensor, dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "scatter")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.scatterChecks(indexes, source, dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val shape = this.shape()
        val storage =
            when (val tryResult = this.device().allocUninit(shape, this.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val layout = Layout.contiguous(shape)
        when (val tryResult = storage.scatterSet(layout, indexes.storage(), indexes.layout(), source.storage(), source.layout(), dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val op =
            BackpropOp.new3(this, indexes, source, { t1, t2, t3 ->
                Op.Scatter(t1, t2, t3, dim1)
            })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    public fun <D : Dim> scatterSet(indexes: Tensor, source: Tensor, dim: D): Result<Unit> {
        if (this.sameStorage(source)) {
            bail("cannot use slice_set when self and src share their storage")
        }
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "scatter-set")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.scatterChecks(indexes, source, dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        when (val tryResult = this.storageMut().scatterSet(this.layout(), indexes.storage(), indexes.layout(), source.storage(), source.layout(), dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }

    public fun <D : Dim> scatterAdd(indexes: Tensor, source: Tensor, dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "scatter-add")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.scatterChecks(indexes, source, dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val shape = this.shape()
        val storage =
            when (val tryResult = this.device().allocUninit(shape, this.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val layout = Layout.contiguous(shape)
        when (val tryResult = storage.scatterAdd(layout, indexes.storage(), indexes.layout(), source.storage(), source.layout(), dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val op =
            BackpropOp.new3(this, indexes, source, { t1, t2, t3 ->
                Op.ScatterAdd(t1, t2, t3, dim1)
            })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    public fun <D : Dim> scatterAddSet(indexes: Tensor, source: Tensor, dim: D): Result<Unit> {
        if (this.sameStorage(source)) {
            bail("cannot use slice_set when self and src share their storage")
        }
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "scatter-add-set")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.scatterChecks(indexes, source, dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        when (val tryResult = this.storageMut().scatterAdd(this.layout(), indexes.storage(), indexes.layout(), source.storage(), source.layout(), dim1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(Unit)
    }

    /**
     * Embeds the values of the `src` tensor into the `self` tensor on the specified dimension.
     */
    public fun <D : Dim> sliceScatter(src: Tensor, dim: D, start: Int): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "slice-scatter")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return if (dim1 == 0) {
            this.sliceScatter0(src, start)
        } else {
            // Note: Maybe we want to add a more efficient implementation at some point.
            when (
                val tryResult =
                    when (val tryResult = this.transpose(0, dim1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.sliceScatter0(
                        when (val tryResult = src.transpose(0, dim1)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                        start,
                    )
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.transpose(0, dim1)
        }
    }

    /**
     * Embeds the values of the `src` tensor into the `self` tensor on the first dimension.
     */
    public fun sliceScatter0(src: Tensor, start: Int): Result<Tensor> {
        if (this.dtype() != src.dtype()) {
            when (val tryResult = Err(Error.DTypeMismatchBinaryOp(lhs = this.dtype(), rhs = src.dtype(), op = "slice-scatter").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        if (this.device().location() != src.device.location()) {
            when (val tryResult = Err(Error.DeviceMismatchBinaryOp(lhs = this.device().location(), rhs = src.device().location(), op = "slice-scatter").bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        if (this.rank() != src.rank()) {
            when (val tryResult = Err(Error.UnexpectedNumberOfDims(expected = this.rank(), got = src.rank(), shape = src.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val shapeOk =
            (this.dims().zip(src.dims())).enumerate().all({ (dimIdx, pair) ->
                val (d1, d2) = pair
                if (0 == dimIdx) {
                    d2 + start <= d1
                } else {
                    d1 == d2
                }
            })
        if (!shapeOk) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(op = "slice-scatter (self, src)", lhs = this.shape().clone(), rhs = src.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val storage =
            when (val tryResult = this.device().allocUninit(this.shape(), this.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val offset = start * ((src.dims()).slice(1 until src.dims().size)).iter().product<Int>()
        when (val tryResult = src.storage().copyStridedSrc(storage, offset, src.layout())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val op = BackpropOp.new2(this, src, { t1, t2 -> Op.SliceScatter0(t1, t2, start) })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    /**
     * Accumulate element from `source` at indexes `indexes` and add them to `self`.
     */
    public fun <D : Dim> indexAdd(indexes: Tensor, source: Tensor, dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "index-add")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val sourceDims = source.dims()
        val selfDims = this.dims()
        val mismatch1 =
            if (sourceDims.size != selfDims.size) {
                true
            } else {
                var mismatch = false
                for ((i, d1, d2) in (selfDims.zip(sourceDims)).enumerate()) {
                    if (i != dim1 && d1 != d2) {
                        mismatch = true
                        break
                    }
                }
                mismatch
            }
        if (mismatch1) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(op = "index-add (self, source)", lhs = this.shape().clone(), rhs = source.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        // The number of element in indexes must match the dimension on which the add is
        // performed on the source tensor (and the index values from `indexes` are taken from
        // the target tensor self)
        val indexesLen =
            when (val tryResult = indexes.dims1()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (sourceDims[dim1] != indexesLen) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(op = "index-add (ids, source))", lhs = indexes.shape().clone(), rhs = source.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val storage =
            when (val tryResult = this.storage().indexAdd(this.layout(), indexes.storage(), indexes.layout(), source.storage(), source.layout(), dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op =
            BackpropOp.new3(this, indexes, source, { t1, t2, t3 ->
                Op.IndexAdd(t1, t2, t3, dim1)
            })
        return Ok(fromStorage(storage, this.shape(), op, false))
    }

    /**
     * Gather values across the target dimension.
     *
     * # Arguments
     *
     * * `self` - The input tensor.
     * * `indexes` - The indices of elements to gather, this should have same number of dimensions as `self`
     *   and indexes.dims()[d] <= self.dims()[d] for all dimensions d != dim
     * * `dim` - the target dimension.
     *
     * The resulting tensor has the same shape as `indexes` and use values from `self` indexed on
     * dimension `dim` by the values in `indexes`.
     */
    public fun <D : Dim> gather(indexes: Tensor, dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "gather")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val selfDims = this.dims()
        val indexesDims = indexes.dims()
        val mismatch1 =
            if (indexesDims.size != selfDims.size) {
                true
            } else {
                var mismatch = false
                for ((i, d1, d2) in (selfDims.zip(indexesDims)).enumerate()) {
                    if (i != dim1 && d1 < d2) {
                        mismatch = true
                        break
                    }
                }
                mismatch
            }
        if (mismatch1) {
            when (val tryResult = Err(Error.ShapeMismatchBinaryOp(op = "gather", lhs = this.shape().clone(), rhs = indexes.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val storage =
            when (val tryResult = this.storage().gather(this.layout(), indexes.storage(), indexes.layout(), dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val op = BackpropOp.new2(this, indexes, { t1, t2 -> Op.Gather(t1, t2, dim1) })
        return Ok(fromStorage(storage, indexes.shape(), op, false))
    }

    /**
     * Select values for the input tensor at the target indexes across the specified dimension.
     *
     * The `indexes` is argument is an int tensor with a single dimension.
     * The output has the same number of dimension as the `self` input. The target dimension of
     * the output has length the length of `indexes` and the values are taken from `self` using
     * the index from `indexes`. Other dimensions have the same number of elements as the input
     * tensor.
     */
    public fun <D : Dim> indexSelect(indexes: Tensor, dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "index-select")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val indexesLen =
            run {
                val matchVal = indexes.dims()
                when {
                    matchVal.size == 1 -> {
                        val l = matchVal[0]
                        l
                    }
                    else ->
                        when (val tryResult = Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = indexes.shape().clone(), op = "index-select").bt())) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        val storage =
            when (val tryResult = this.storage().indexSelect(indexes.storage(), this.layout(), indexes.layout(), dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dims = this.dims().toVec()
        dims[dim1] = indexesLen
        val op = BackpropOp.new2(this, indexes, { t1, t2 -> Op.IndexSelect(t1, t2, dim1) })
        return Ok(fromStorage(storage, dims, op, false))
    }

    /**
     * Returns an iterator over position of the elements in the storage when ranging over the
     * index tuples in lexicographic order.
     */
    public fun stridedIndex(): StridedIndex = this.layout.stridedIndex()

    /**
     * Similar to `stridedIndex` but returns the position of the start of each contiguous block
     * as well as the length of the contiguous blocks. For a contiguous tensor, the index iterator
     * will only return the start offset and the size would be the number of elements in the
     * tensor.
     */
    public fun stridedBlocks(): StridedBlocks = this.layout.stridedBlocks()

    /**
     * Returns the data contained in a 1D tensor as a vector of scalar values.
     */
    public fun <S : WithDType> toVec1(cpuStorageAsSliceSFn: (Any?) -> Any?): Result<List<S>> {
        if (this.rank() != 1) {
            when (val tryResult = Err(Error.UnexpectedNumberOfDims(expected = 1, got = this.rank(), shape = this.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val fromCpuStorage =
            closure1@{ cpuStorage: CpuStorage ->
                val data =
                    when (val tryResult = cpuStorageAsSliceSFn(cpuStorage)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return@closure1 Result.fromResidual(Result.Err(tryResult.value))
                    }
                val data1 =
                    when (val matchVal = this.layout.contiguousOffsets()) {
                        null -> this.stridedIndex().map({ i -> data[i] }).collect()
                        else -> {
                            val o1 = matchVal.first
                            val o2 = matchVal.second
                            ((data).slice(o1 until o2)).toVec()
                        }
                    }
                Ok(data1)
            }
        return when (val matchVal = this.storage()) {
            is Storage.Cpu -> {
                val (storage) = matchVal
                fromCpuStorage(storage)
            }
            is Storage.Cuda -> {
                val (storage1) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage1.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is Storage.Metal -> {
                val (storage2) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage2.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
        }
    }

    /**
     * Returns the data contained in a 2D tensor as a vector of vector of scalar values.
     */
    public fun <S : WithDType> toVec2(cpuStorageAsSliceSFn: (Any?) -> Any?): Result<List<List<S>>> {
        val (dim1, dim2) =
            when (val tryResult = this.dims2()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val fromCpuStorage =
            closure2@{ cpuStorage: CpuStorage ->
                val data =
                    when (val tryResult = cpuStorageAsSliceSFn(cpuStorage)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return@closure2 Result.fromResidual(Result.Err(tryResult.value))
                    }
                var rows = mutableListOf()
                when (val matchVal = this.layout.contiguousOffsets()) {
                    null -> {
                        val srcIndex = this.stridedIndex()
                        for (idxRow in 0 until dim1) {
                            val row = (0 until dim2).map({ data[srcIndex.next()!!] }).collect()
                            rows.add(row)
                        }
                        check(srcIndex.next() == null)
                    }
                    else -> {
                        val o1 = matchVal.first
                        val o2 = matchVal.second
                        val data1 = ((data).slice(o1 until o2))
                        for (idxRow in 0 until dim1) {
                            rows.add(((data1).slice(idxRow * dim2 until (idxRow + 1) * dim2)).toVec())
                        }
                    }
                }
                Ok(rows)
            }
        return when (val matchVal = this.storage()) {
            is Storage.Cpu -> {
                val (storage) = matchVal
                fromCpuStorage(storage)
            }
            is Storage.Cuda -> {
                val (storage1) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage1.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is Storage.Metal -> {
                val (storage2) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage2.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
        }
    }

    /**
     * Returns the data contained in a 3D tensor.
     */
    public fun <S : WithDType> toVec3(cpuStorageAsSliceSFn: (Any?) -> Any?): Result<List<List<List<S>>>> {
        val (dim1, dim2, dim3) =
            when (val tryResult = this.dims3()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val fromCpuStorage =
            closure3@{ cpuStorage: CpuStorage ->
                val data =
                    when (val tryResult = cpuStorageAsSliceSFn(cpuStorage)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return@closure3 Result.fromResidual(Result.Err(tryResult.value))
                    }
                var topRows = mutableListOf()
                when (val matchVal = this.layout.contiguousOffsets()) {
                    null -> {
                        val srcIndex = this.stridedIndex()
                        for (idx in 0 until dim1) {
                            val rows = mutableListOf()
                            for (jdx in 0 until dim2) {
                                val row = (0 until dim3).map({ data[srcIndex.next()!!] }).collect()
                                rows.add(row)
                            }
                            topRows.add(rows)
                        }
                        check(srcIndex.next() == null)
                    }
                    else -> {
                        val o1 = matchVal.first
                        val o2 = matchVal.second
                        val data1 = ((data).slice(o1 until o2))
                        val dim23 = dim2 * dim3
                        for (idx1 in 0 until dim1) {
                            val data2 = ((data1).slice(idx1 * dim23 until (idx1 + 1) * dim23))
                            val rows1 = mutableListOf()
                            for (idx2 in 0 until dim2) {
                                rows1.add(((data2).slice(idx2 * dim3 until (idx2 + 1) * dim3)).toVec())
                            }
                            topRows.add(rows1)
                        }
                    }
                }
                Ok(topRows)
            }
        return when (val matchVal = this.storage()) {
            is Storage.Cpu -> {
                val (storage) = matchVal
                fromCpuStorage(storage)
            }
            is Storage.Cuda -> {
                val (storage1) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage1.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            is Storage.Metal -> {
                val (storage2) = matchVal
                fromCpuStorage(
                    when (val tryResult = storage2.toCpuStorage()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
        }
    }

    /**
     * The tensor shape, i.e. dimension sizes on each axis.
     */
    public fun shape(): Shape = this.layout().shape()

    /**
     * The dimension size for this tensor on each axis.
     */
    public fun dims(): List<Int> = this.shape().dims()

    /**
     * The dimension size for a specified dimension index.
     */
    public fun <D : Dim> dim(dim: D): Result<Int> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "dim")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(this.dims()[dim1])
    }

    public fun stride(): List<Int> = this.layout.stride()

    /**
     * The number of dimensions for this tensor, 0 for a scalar tensor, 1 for a 1D tensor, etc.
     */
    public fun rank(): Int = this.shape().rank()

    /**
     * The number of elements stored in this tensor.
     */
    public fun elemCount(): Int = this.shape().elemCount()

    /**
     * Computes the max of all the elements in this tensor and returns a tensor holding this
     * scalar with zero dimensions.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor1 = tensor.maxAll().getOrElse { return Result.failure(it) }
     * check(tensor1.toScalar<Float>().getOrElse { return Result.failure(it) } == 5.0)
     * ```
     */
    public fun maxAll(): Result<Tensor> {
        return if (this.rank() == 0) {
            Ok(this.clone())
        } else {
            when (val tryResult = this.flattenAll()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.max(0)
        }
    }

    /**
     * Computes the min of all the elements in this tensor and returns a tensor holding this
     * scalar with zero dimensions.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor1 = tensor.minAll().getOrElse { return Result.failure(it) }
     * check(tensor1.toScalar<Float>().getOrElse { return Result.failure(it) } == 0.0)
     * ```
     */
    public fun minAll(): Result<Tensor> {
        return if (this.rank() == 0) {
            Ok(this.clone())
        } else {
            when (val tryResult = this.flattenAll()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.min(0)
        }
    }

    /**
     * Computes the sum of all the elements in this tensor and returns a tensor holding this
     * scalar with zero dimensions.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor1 = tensor.sumAll().getOrElse { return Result.failure(it) }
     * check(tensor1.toScalar<Float>().getOrElse { return Result.failure(it) } == 15.0)
     * ```
     */
    public fun sumAll(): Result<Tensor> {
        val dims: Vec = (0 until this.rank()).collect()
        return this.sum(dims)
    }

    public fun meanAll(): Result<Tensor> {
        return when (val tryResult = this.sumAll()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        } / this.elemCount().toDouble()
    }

    public fun <D1 : Dim, D2 : Dim> flatten(startDim: D1?, endDim: D2?): Result<Tensor> {
        return if (this.rank() == 0) {
            this.reshape(1)
        } else {
            val startDim1 =
                when (startDim) {
                    null -> 0
                    else -> {
                        val dim = startDim
                        when (val tryResult = dim.toIndex(this.shape(), "flatten")) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
            val endDim1 =
                when (endDim) {
                    null -> this.rank() - 1
                    else -> {
                        val dim1 = endDim
                        when (val tryResult = dim1.toIndex(this.shape(), "flatten")) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                }
            if (startDim1 < endDim1) {
                val dims = this.dims()
                val dstDims = ((dims).slice(0 until startDim1)).toVec()
                dstDims.push(((dims).slice(startDim1 until endDim1 + 1)).iter().product<Int>())
                if (endDim1 + 1 < dims.size) {
                    dstDims.extend(((dims).slice(endDim1 + 1 until dims.size)))
                }
                this.reshape(dstDims)
            } else {
                Ok(this.clone())
            }
        }
    }

    /**
     * Flattens the input tensor on the dimension indexes from `0` to `endDim` (inclusive).
     */
    public fun <D : Dim> flattenTo(endDim: D): Result<Tensor> = this.flatten(None, endDim)

    /**
     * Flattens the input tensor on the dimension indexes from `startDim` (inclusive) to the last
     * dimension.
     */
    public fun <D : Dim> flattenFrom(startDim: D): Result<Tensor> = this.flatten(startDim, None)

    /**
     * Flattens the input tensor by reshaping it into a one dimension tensor.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor1 = tensor.flattenAll().getOrElse { return Result.failure(it) }
     * check(tensor1.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
     * ```
     */
    public fun flattenAll(): Result<Tensor> = this.flatten(None, None)

    /**
     * Returns the sub-tensor fixing the index at `i` on the first dimension.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val t = tensor.get(0).getOrElse { return Result.failure(it) }
     * check(t.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(0.0, 1.0))
     * val t1 = tensor.get(1).getOrElse { return Result.failure(it) }
     * check(t1.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(2.0, 3.0))
     * ```
     */
    public fun get(i: Int): Result<Tensor> {
        val dims = this.dims()
        return if (dims.isEmpty()) {
            Ok(this.clone())
        } else {
            when (val tryResult = this.narrow(0, i, 1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.reshape(((dims).slice(1 until dims.size)))
        }
    }

    /**
     * Returns the sub-tensor fixing the index at `index` on the dimension `dim`.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val t = tensor.getOnDim(1, 0).getOrElse { return Result.failure(it) }
     * check(t.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(0.0, 2.0, 4.0))
     * val t1 = tensor.getOnDim(1, 1).getOrElse { return Result.failure(it) }
     * check(t1.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(1.0, 3.0, 5.0))
     * val t2 = tensor.getOnDim(0, 1).getOrElse { return Result.failure(it) }
     * check(t2.toVec1<Float>().getOrElse { return Result.failure(it) } == listOf(2.0, 3.0))
     * ```
     */
    public fun <D : Dim> getOnDim(dim: D, index: Int): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "get_on_dim")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return when (val tryResult = this.narrow(dim1, index, 1)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.squeeze(dim1)
    }

    /**
     * Returns a tensor that is a transposed version of the input, the two last dimensions of the
     * input are swapped.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(listOf(listOf(0.0f, 1.0), listOf(2.0, 3.0), listOf(4.0, 5.0)), Device.Cpu).getOrElse { return Result.failure(it) }
     * val tensor1 = tensor.t().getOrElse { return Result.failure(it) }
     * check(tensor1.toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(0.0, 2.0, 4.0), listOf(1.0, 3.0, 5.0)))
     * ```
     */
    public fun t(): Result<Tensor> {
        val rank = this.rank()
        if (rank < 2) {
            when (val tryResult = Err(Error.UnexpectedNumberOfDims(expected = 2, got = rank, shape = this.shape().clone()).bt())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        return this.transpose(rank - 2, rank - 1)
    }

    /**
     * Returns a tensor that is a transposed version of the input, the given dimensions are
     * swapped.
     */
    public fun <D1 : Dim, D2 : Dim> transpose(dim1: D1, dim2: D2): Result<Tensor> {
        val dim11 =
            when (val tryResult = dim1.toIndex(this.shape(), "transpose")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dim21 =
            when (val tryResult = dim2.toIndex(this.shape(), "transpose")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (dim11 == dim21) {
            return Ok(this.clone())
        }
        val op = BackpropOp.new1(this, { t -> Op.Transpose(t, dim11, dim21) })
        val tensor =
            Tensor(
                id = TensorId.new(),
                storage = this.storage.clone(),
                layout =
                    when (val tryResult = this.layout.transpose(dim11, dim21)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                op = op,
                isVariable = false,
                dtype = this.dtype,
                device = this.device.clone(),
            )
        return Ok(Tensor(tensor))
    }

    /**
     * Returns a tensor with the same data as the input where the dimensions have been permuted.
     * dims must be a permutation, i.e. include each dimension index exactly once.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.arange(0u, 120u, Device.Cpu).getOrElse { return Result.failure(it) }.reshape(listOf(2, 3, 4, 5)).getOrElse { return Result.failure(it) }
     * check(tensor.dims() == listOf(2, 3, 4, 5))
     * val tensor1 = tensor.permute(listOf(2, 3, 1, 0)).getOrElse { return Result.failure(it) }
     * check(tensor1.dims() == listOf(4, 5, 3, 2))
     * ```
     */
    public fun <D : Dims> permute(dims: D): Result<Tensor> {
        val dims1 =
            when (val tryResult = dims.toIndexes(this.shape(), "permute")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // O(n^2) permutation check but these arrays are small.
        val isPermutation = dims1.len() == this.rank() && (0 until dims1.len()).all({ i -> dims1.contains(i) })
        if (!isPermutation) {
            bail("dimension mismatch in permute, tensor {:?}, dims: {:?}", this.dims(), dims1)
        }
        val op = BackpropOp.new1(this, { t -> Op.Permute(t, dims1.clone()) })
        val tensor =
            Tensor(
                id = TensorId.new(),
                storage = this.storage.clone(),
                layout =
                    when (val tryResult = this.layout.permute(dims1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                op = op,
                isVariable = false,
                dtype = this.dtype,
                device = this.device.clone(),
            )
        return Ok(Tensor(tensor))
    }

    /**
     * Returns true if the data is stored in a C contiguous (aka row major) way.
     */
    public fun isContiguous(): Boolean = this.layout.isContiguous()

    /**
     * Returns true if the data is stored in a Fortran contiguous (aka column major) way.
     */
    public fun isFortranContiguous(): Boolean = this.layout.isFortranContiguous()

    /**
     * Compared to clone, this copies the actual storage but may fail because of running out of
     * memory.
     */
    public fun copy(): Result<Tensor> {
        val op = BackpropOp.new1(this, Op.Copy)
        val tensor =
            Tensor(
                id = TensorId.new(),
                storage =
                    RwLock.new(
                        when (val tryResult = this.storage().tryClone(this.layout())) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                layout = this.layout.clone(),
                op = op,
                isVariable = false,
                dtype = this.dtype,
                device = this.device.clone(),
            )
        return Ok(Tensor(tensor))
    }

    /**
     * Returns a new tensor detached from the current graph, gradient are not propagated through
     * this new node. The storage of this tensor is shared with the initial tensor.
     *
     * If the tensor is already detached from the computation graph, the same tensor is returned.
     */
    public fun detach(): Tensor =
        if (this.op.isNone() && !this.isVariable) {
            this.clone()
        } else {
            val tensor = Tensor(id = TensorId.new(), storage = this.storage.clone(), layout = this.layout.clone(), op = BackpropOp.none(), isVariable = false, dtype = this.dtype, device = this.device.clone())
            Tensor(tensor)
        }

    /**
     * If the target device is the same as the tensor device, only a shallow copy is performed.
     */
    public fun toDevice(device: Device): Result<Tensor> {
        return if (this.device().sameDevice(device)) {
            Ok(this.clone())
        } else {
            val storage6 =
                run {
                    val matchVal = Pair(this.storage(), device)
                    when {
                        matchVal.first is Storage.Cpu && matchVal.second is Device.Cuda -> {
                            val (storage) = matchVal.first as Storage.Cpu
                            val (cuda) = matchVal.second as Device.Cuda
                            Storage.Cuda(
                                when (val tryResult = cuda.storageFromCpuStorage(storage)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                            )
                        }
                        matchVal.first is Storage.Cpu && matchVal.second is Device.Metal -> {
                            val (storage1) = matchVal.first as Storage.Cpu
                            val (metal) = matchVal.second as Device.Metal
                            Storage.Metal(
                                when (val tryResult = metal.storageFromCpuStorage(storage1)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                            )
                        }
                        matchVal.first is Storage.Cuda && matchVal.second == Device.Cpu -> {
                            val (storage2) = matchVal.first as Storage.Cuda
                            Storage.Cpu(
                                when (val tryResult = storage2.toCpuStorage()) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                            )
                        }
                        matchVal.first is Storage.Metal && matchVal.second == Device.Cpu -> {
                            val (storage3) = matchVal.first as Storage.Metal
                            Storage.Cpu(
                                when (val tryResult = storage3.toCpuStorage()) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                },
                            )
                        }
                        matchVal.first is Storage.Cuda && matchVal.second is Device.Cuda -> {
                            val (storage4) = matchVal.first as Storage.Cuda
                            val (cuda1) = matchVal.second as Device.Cuda
                            // can't clone storage if it's the same device because of the underlying device ptr
                            val dstStorage =
                                when (val tryResult = storage4.transferToDevice(cuda1)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            Storage.Cuda(dstStorage)
                        }
                        matchVal.first is Storage.Cpu && matchVal.second == Device.Cpu -> {
                            val (storage5) = matchVal.first as Storage.Cpu
                            Storage.Cpu(storage5.clone())
                        }
                        else -> {
                            bail("not implemented yet, self.device: {:?}, device: {:?}", this.device(), device)
                        }
                    }
                }
            val op = BackpropOp.new1(this, Op.ToDevice)
            val tensor = Tensor(id = TensorId.new(), storage = RwLock.new(storage6), layout = this.layout.clone(), op = op, isVariable = false, dtype = this.dtype, device = device.clone())
            Ok(Tensor(tensor))
        }
    }

    /**
     * Returns a new tensor duplicating data from the original tensor. New dimensions are inserted
     * on the left.
     */
    public fun broadcastLeft(leftShape: Shape): Result<Tensor> {
        val leftShape1 = leftShape
        val dims = leftShape1.intoDims()
        dims.extend(this.dims())
        return this.broadcastAs(dims)
    }

    /**
     * Broadcast the input tensor to the target shape. This returns an error if the input shape is
     * not compatible with the target shape.
     *
     * If the input shape is `i1, i2, ... iK`, the target shape has to have `k` dimensions or
     * more and shape `j1, ..., jL, t1, t2, ..., tK`. The dimensions `j1` to `jL` can have
     * any value, the dimension `tA` must be equal to `iA` if `iA` is different from 1. If
     * `iA` is equal to 1, any value can be used.
     */
    public fun broadcastAs(shape: Shape): Result<Tensor> {
        val tensor =
            Tensor(
                id = TensorId.new(),
                storage = this.storage.clone(),
                layout =
                    when (val tryResult = this.layout.broadcastAs(shape)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                op = BackpropOp.new1(this, Op.Broadcast),
                isVariable = false,
                dtype = this.dtype,
                device = this.device.clone(),
            )
        return Ok(Tensor(tensor))
    }

    /**
     * An alias for broadcastAs.
     */
    public fun expand(shape: Shape): Result<Tensor> = this.broadcastAs(shape)

    /**
     * Casts the input tensor to the target `dtype`.
     *
     * ```kotlin
     * import candlecore.Device
     * import candlecore.Tensor
     *
     * val tensor = Tensor.new(3.14159265358979, Device.Cpu).getOrElse { return Result.failure(it) }
     * check(tensor.toScalar<Double>().getOrElse { return Result.failure(it) } == 3.14159265358979)
     * val tensor1 = tensor.toDtype(DType.F32).getOrElse { return Result.failure(it) }
     * check(tensor1.toScalar<Float>().getOrElse { return Result.failure(it) } == 3.1415927)
     * ```
     */
    public fun toDtype(dtype: DType): Result<Tensor> {
        return if (this.dtype() == dtype) {
            Ok(this.clone())
        } else {
            val shape = this.shape()
            val storage =
                when (val tryResult = this.storage().toDtype(this.layout(), dtype)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val op = BackpropOp.new1(this, Op.ToDType)
            Ok(fromStorage(storage, shape.clone(), op, false))
        }
    }

    /**
     * Returns a tensor that is in row major order. This is the same as the original tensor if it
     * was already contiguous, otherwise a copy is triggered.
     */
    public fun contiguous(): Result<Tensor> {
        return if (this.isContiguous()) {
            Ok(this.clone())
        } else {
            val shape = this.shape()
            val storage =
                when (val tryResult = this.device().allocUninit(shape, this.dtype())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            val op = BackpropOp.new1(this, Op.Copy)
            Ok(fromStorage(storage, shape.clone(), op, false))
        }
    }

    /**
     * Returns a tensor that is in row major order. This always makes a copy.
     */
    public fun forceContiguous(): Result<Tensor> {
        val shape = this.shape()
        val storage =
            when (val tryResult = this.device().allocUninit(shape, this.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val op = BackpropOp.new1(this, Op.Copy)
        return Ok(fromStorage(storage, shape.clone(), op, false))
    }

    /**
     * Create a variable based on the values currently stored in a tensor. The storage is always
     * copied.
     */
    internal fun makeVar(): Result<Tensor> {
        val shape = this.shape().clone()
        val storage =
            when (val tryResult = this.device().allocUninit(shape, this.dtype())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(fromStorage(storage, shape, BackpropOp.none(), true))
    }

    /**
     * Reshape returns a tensor with the target shape provided that the number of elements of the
     * original tensor is the same.
     * If the input tensor is contiguous, this is a view on the original data. Otherwise this uses
     * a new storage and copies the data over, the returned tensor is always contiguous.
     *
     * The shape can be specified using a tuple of `Int` and at most one `()` in which case
     * the behavior is the same as when using `- 1` in PyTorch: this dimension size is adjusted so
     * as to match the number of elements in the tensor.
     *
     * ```kotlin
     * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
     * val c = a.reshape(Pair(1, 6)).getOrElse { return Result.failure(it) }
     * check(c.shape().dims() == listOf(1, 6))
     * val c1 = a.reshape(Pair(3, 2)).getOrElse { return Result.failure(it) }
     * check(c1.shape().dims() == listOf(3, 2))
     * val c2 = a.reshape(Triple(2, Unit, 1)).getOrElse { return Result.failure(it) }
     * check(c2.shape().dims() == listOf(2, 3, 1))
     * ```
     */
    public fun <S : ShapeWithOneHole> reshape(s: S): Result<Tensor> {
        val shape =
            when (val tryResult = s.intoShape(this.elemCount())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (shape.elemCount() != this.elemCount()) {
            return Err(Error.ShapeMismatchBinaryOp(lhs = this.shape().clone(), rhs = shape, op = "reshape").bt())
        }
        val op = BackpropOp.new1(this, Op.Reshape)
        return if (this.isContiguous()) {
            val tensor = Tensor(id = TensorId.new(), storage = this.storage.clone(), layout = Layout.contiguousWithOffset(shape, this.layout.startOffset()), op = op, isVariable = false, dtype = this.dtype, device = this.device.clone())
            Ok(Tensor(tensor))
        } else {
            val storage =
                when (val tryResult = this.device().allocUninit(shape, this.dtype())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            when (val tryResult = this.storage().copyStridedSrc(storage, 0, this.layout())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            Ok(fromStorage(storage, shape, op, false))
        }
    }

    /**
     * Creates a new tensor with the specified dimension removed if its size was one.
     *
     * ```kotlin
     * val a = Tensor.zeros(Triple(2, 3, 1), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
     * val c = a.squeeze(2).getOrElse { return Result.failure(it) }
     * check(c.shape().dims() == listOf(2, 3))
     * val c1 = a.squeeze(D.Minus1).getOrElse { return Result.failure(it) }
     * check(c1.shape().dims() == listOf(2, 3))
     * ```
     */
    public fun <D : Dim> squeeze(dim: D): Result<Tensor> {
        // The PyTorch semantics are to return the same tensor if the target dimension
        // does not have a size of 1.
        val dims = this.dims()
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "squeeze")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return if (dims[dim1] == 1) {
            val dims1 = dims.toVec()
            val strides = this.stride().toVec()
            dims1.remove(dim1)
            strides.remove(dim1)
            val tensor = Tensor(id = TensorId.new(), storage = this.storage.clone(), layout = Layout.new(dims1.into(), strides, this.layout.startOffset()), op = BackpropOp.new1(this, Op.Reshape), isVariable = false, dtype = this.dtype, device = this.device.clone())
            Ok(Tensor(tensor))
        } else {
            Ok(this.clone())
        }
    }

    /**
     * Creates a new tensor with a dimension of size one inserted at the specified position.
     *
     * ```kotlin
     * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
     * val c = a.unsqueeze(0).getOrElse { return Result.failure(it) }
     * check(c.shape().dims() == listOf(1, 2, 3))
     * val c1 = a.unsqueeze(D.Minus1).getOrElse { return Result.failure(it) }
     * check(c1.shape().dims() == listOf(2, 3, 1))
     * ```
     */
    public fun <D : Dim> unsqueeze(dim: D): Result<Tensor> {
        val dims = this.dims().toVec()
        val strides = this.stride().toVec()
        val dim1 =
            when (val tryResult = dim.toIndexPlusOne(this.shape(), "unsqueeze")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // Cannot panic because toIndexPlusOne already checks dimensions
        dims.insert(dim1, 1)
        // Any stride would work here, but we pick one so as to maximize the probability to remain
        // C contiguous.
        val stride =
            if (dim1 < strides.len()) {
                strides[dim1]
            } else {
                1
            }
        strides.insert(dim1, stride)
        val tensor = Tensor(id = TensorId.new(), storage = this.storage.clone(), layout = Layout.new(dims.into(), strides, this.layout.startOffset()), op = BackpropOp.new1(this, Op.Reshape), isVariable = false, dtype = this.dtype, device = this.device.clone())
        return Ok(Tensor(tensor))
    }

    /**
     * Pad the input tensor using 0s along dimension `dim`. This adds `left` elements before the
     * input tensor values and `right` elements after.
     */
    public fun <D : Dim> padWithZeros(dim: D, left: Int, right: Int): Result<Tensor> {
        return if (left == 0 && right == 0) {
            Ok(this.clone())
        } else if (left == 0) {
            val dim1 =
                when (val tryResult = dim.toIndex(this.shape(), "pad_with_zeros")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val dims = this.dims().toVec()
            dims[dim1] = right
            val right1 =
                when (val tryResult = Tensor.zeros(dims.asSlice(), this.dtype, this.device())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            Tensor.cat(listOf(this, right1), dim1)
        } else if (right == 0) {
            val dim2 =
                when (val tryResult = dim.toIndex(this.shape(), "pad_with_zeros")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val dims1 = this.dims().toVec()
            dims1[dim2] = left
            val left1 =
                when (val tryResult = Tensor.zeros(dims1.asSlice(), this.dtype, this.device())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            Tensor.cat(listOf(left1, this), dim2)
        } else {
            val dim3 =
                when (val tryResult = dim.toIndex(this.shape(), "pad_with_zeros")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val dims2 = this.dims().toVec()
            dims2[dim3] = left
            val left2 =
                when (val tryResult = Tensor.zeros(dims2.asSlice(), this.dtype, this.device())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            dims2[dim3] = right
            val right2 =
                when (val tryResult = Tensor.zeros(dims2.asSlice(), this.dtype, this.device())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            Tensor.cat(listOf(left2, this, right2), dim3)
        }
    }

    /**
     * Pad the input tensor using same values along dimension `dim`. This adds `left` elements before the
     * input tensor values and `right` elements after.
     */
    public fun <D : Dim> padWithSame(dim: D, left: Int, right: Int): Result<Tensor> {
        return if (left == 0 && right == 0) {
            Ok(this.clone())
        } else if (this.elemCount() == 0) {
            bail("cannot use pad_with_same on an empty tensor")
        } else if (left == 0) {
            val dim1 =
                when (val tryResult = dim.toIndex(this.shape(), "pad_with_same")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val r =
                when (
                    val tryResult =
                        this.narrow(
                            dim1,
                            when (val tryResult = this.dim(dim1)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            } - 1,
                            1,
                        )
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val v = mutableListOf(this)
            for (unused in 0 until right) {
                v.add(r)
            }
            Tensor.cat(v, dim1)
        } else if (right == 0) {
            val dim2 =
                when (val tryResult = dim.toIndex(this.shape(), "pad_with_same")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val l =
                when (val tryResult = this.narrow(dim2, 0, 1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val v1 = mutableListOf()
            for (unused1 in 0 until left) {
                v1.add(l)
            }
            v1.add(this)
            Tensor.cat(v1, dim2)
        } else {
            val dim3 =
                when (val tryResult = dim.toIndex(this.shape(), "pad_with_same")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val l1 =
                when (val tryResult = this.narrow(dim3, 0, 1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val r1 =
                when (
                    val tryResult =
                        this.narrow(
                            dim3,
                            when (val tryResult = this.dim(dim3)) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            } - 1,
                            1,
                        )
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val v2 = mutableListOf()
            for (unused2 in 0 until left) {
                v2.add(l1)
            }
            v2.add(this)
            for (unused3 in 0 until right) {
                v2.add(r1)
            }
            Tensor.cat(v2, dim3)
        }
    }

    /**
     * Run the `forward` method of `m` on `self`.
     */
    public fun <M : Module> apply(m: M): Result<Tensor> = m.forward(this)

    /**
     * Run the `forward` method of `m` on `self`.
     */
    public fun <M : ModuleT> applyT(m: M, train: Boolean): Result<Tensor> = m.forwardT(this, train)

    /**
     * Acquire write lock on storage and returns guard.
     */
    internal fun storageMut(): StorageMutRef = this.storage.write()

    // If we extend the visibility of this function to be usable outside of this crate, we should
    // make it unchecked.
    internal fun storageMutAndLayout(): Pair<StorageMutRef, Layout> {
        val storage = this.storage.write()
        return Pair(storage, this.layout)
    }

    /**
     * The storage used by this tensor, together with the layout to use to access it safely.
     */
    public fun storageAndLayout(): Pair<StorageRef, Layout> {
        val storage = this.storage.read()
        return Pair(storage, this.layout)
    }

    /**
     * Unique key for this tensor's storage. Equal keys mean the tensors share the same allocation.
     */
    internal fun storageKey(): Int {
        val lock: RwLock<Storage> = this.storage.asRef()
        return lock.addr()
    }

    /**
     * Check if two tensors share the same underlying allocation.
     */
    internal fun sameStorage(rhs: Tensor): Boolean = this.storageKey() == rhs.storageKey()

    /**
     * Normalize a 'relative' axis value: positive values are kept, negative
     * values means counting the dimensions from the back.
     */
    public fun normalizeAxis(axis: Long): Result<Int> {
        val rank = this.rank().toLong()
        return if (rank <= axis) {
            bail("axis {axis} is too large, tensor rank {rank}")
        } else if (0L <= axis) {
            Ok(axis.toInt())
        } else {
            val naxis = rank + axis
            if (naxis < 0L) {
                bail("axis {axis} is too small, tensor rank {rank}")
            }
            Ok(naxis.toInt())
        }
    }

    /**
     * Returns the cumulative sum of elements of the input tensor summed over the specified
     * dimension.
     *
     * This operation is most efficient when dim is the last dimension of the tensor.
     */
    public fun <D : Dim> cumsum(dim: D): Result<Tensor> {
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "cumsum")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val rank = this.rank()
        if (rank == 0) {
            return Ok(this.clone())
        }
        val nAxis =
            when (val tryResult = this.dim(dim1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val triu =
            when (val tryResult = Tensor.triu2(nAxis, this.dtype(), this.device())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return if (rank == 1) {
            when (
                val tryResult =
                    when (val tryResult = this.unsqueeze(0)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.matmul(triu)
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.squeeze(0)
        } else {
            val last = rank - 1
            val t =
                when (val tryResult = this.transpose(dim1, last)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t1 =
                when (val tryResult = t.broadcastMatmul(triu)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            t1.transpose(dim1, last)
        }
    }

    /**
     * Returns a copy of `self` where the values within `ranges` have been replaced with the
     * content of `src`.
     */
    public fun <D : RangeBounds<Int>> sliceAssign(ranges: List<D>, src: Tensor): Result<Tensor> {
        val src1 = src
        val srcDims = src1.dims()
        val selfDims = this.dims()
        if (selfDims.size != srcDims.size) {
            bail("slice-assign requires input with the same rank {} <> {}", selfDims.size, srcDims.size)
        }
        if (selfDims.size != ranges.size) {
            bail("slice-assign requires input with the same rank as there are ranges {} <> {}", selfDims.size, ranges.size)
        }
        var src2 = src1.clone()
        var mask =
            when (val tryResult = Tensor.ones(src2.shape(), DType.U8, src2.device())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        for ((i, range) in ranges.enumerate()) {
            val startIncluded =
                when (val matchVal = range.startBound()) {
                    Bound.Unbounded -> 0
                    is Bound.Included -> {
                        val (v) = matchVal
                        v
                    }
                    is Bound.Excluded -> {
                        val (v1) = matchVal
                        v1 + 1
                    }
                }
            val endExcluded =
                when (val matchVal1 = range.endBound()) {
                    Bound.Unbounded -> selfDims[i]
                    is Bound.Included -> {
                        val (v2) = matchVal1
                        v2 + 1
                    }
                    is Bound.Excluded -> {
                        val (v3) = matchVal1
                        v3
                    }
                }
            if (endExcluded <= startIncluded) {
                bail("slice-assign: empty range for dim {i}, {start_included} {end_excluded}")
            }
            if (selfDims[i] < endExcluded) {
                bail("slice-assign: upper bound is out of range for dim {i}, {end_excluded} {}", selfDims[i])
            }
            if (endExcluded - startIncluded != srcDims[i]) {
                bail("slice-assign: the range for dim {i} ({start_included}..{end_excluded}) does not match the size of src {}", srcDims[i])
            }
            src2 =
                when (val tryResult = src2.padWithZeros(i, startIncluded, selfDims[i] - endExcluded)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            mask =
                when (val tryResult = mask.padWithZeros(i, startIncluded, selfDims[i] - endExcluded)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
        }
        return mask.whereCond(src2, this)
    }

    /**
     * Returns log(sum(exp(tensor), dim)).
     */
    public fun <D : Dims> logSumExp(sumDims: D): Result<Tensor> {
        val sumDims1 =
            when (val tryResult = sumDims.toIndexes(this.shape(), "log-sum-exp")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (sumDims1.isEmpty()) {
            return Ok(this.clone())
        }
        val max =
            when (
                val tryResult =
                    ((sumDims1).slice(1 until sumDims1.size)).iter().tryFold(
                        when (val tryResult = this.maxKeepdim(sumDims1[0])) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                        { max, dim ->
                            max.maxKeepdim(dim)
                        },
                    )
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val exp =
            when (
                val tryResult =
                    when (val tryResult = this.broadcastSub(max)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }.exp()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val sum =
            when (val tryResult = exp.sum(sumDims1.clone())) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return when (val tryResult = sum.log()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        } + max.squeezeDims(sumDims1)
    }

    /**
     * Pointwise pow operation.
     */
    public fun pow(rhs: Tensor): Result<Tensor> {
        return when (
            val tryResult =
                rhs.mul(
                    when (val tryResult = this.log()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.exp()
    }

    /**
     * Broadcasting version of `pow`.
     */
    public fun broadcastPow(rhs: Tensor): Result<Tensor> {
        return when (
            val tryResult =
                rhs.broadcastMul(
                    when (val tryResult = this.log()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.exp()
    }

    /**
     * Returns a new tensor with the order of elements reversed along the specified dimensions.
     * This function makes a copy of the tensor’s data.
     *
     * ```kotlin
     * val t = Tensor.arange(0.0, 6.0, Device.Cpu).getOrElse { return Result.failure(it) }.reshape(Pair(2, 3)).getOrElse { return Result.failure(it) }
     * check(t.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(0.0, 1.0, 2.0), listOf(3.0, 4.0, 5.0)))
     * val tFlipped = t.flip(listOf(0)).getOrElse { return Result.failure(it) }
     * check(tFlipped.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(3.0, 4.0, 5.0), listOf(0.0, 1.0, 2.0)))
     * ```
     */
    public fun flip(dims: List<Int>): Result<Tensor> {
        var result = this.clone()
        for (dim in dims) {
            val size =
                when (val tryResult = result.dim(dim)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val indices: List<Long> = (0 until size).rev().map({ x -> x.toLong() }).collect()
            val indicesTensor =
                when (val tryResult = Tensor.fromVec(indices, size, result.device())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            result =
                when (val tryResult = result.indexSelect(indicesTensor, dim)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
        }
        return Ok(result)
    }

    /**
     * Returns a view of which contains all slices of size `size` from self tensor in the dimension
     * `dim` and stepped by `step`.
     */
    public fun <D : Dim> unfold(dim: D, size: Int, step: Int): Result<Tensor> {
        // https://github.com/pytorch/pytorch/blob/75b0720a97ac5d82e8a7a1a6ae7c5f7a87d7183d/aten/src/ATen/native/TensorShape.cpp#L3785-L3804
        val sizes = this.dims().toVec()
        val strides = this.stride().toVec()
        val dim1 =
            when (val tryResult = dim.toIndex(this.shape(), "unfold")) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val maxLen =
            if (this.dims().isEmpty()) {
                1
            } else {
                sizes[dim1]
            }
        if (size > maxLen) {
            bail("unsqueeze: maximum size for tensor at dimension {dim} is {max_len} but size is {size}")
        }
        sizes.push(size)
        strides.push(
            if (this.dims().isEmpty()) {
                1
            } else {
                strides[dim1]
            },
        )
        if (!this.dims().isEmpty()) {
            sizes[dim1] = ((sizes[dim1].toFloat() - size.toFloat()) / step.toFloat() + 1.0f).toInt()
            strides[dim1] *= step
        }
        val tensor = Tensor(id = TensorId.new(), storage = this.storage.clone(), layout = Layout.new(sizes.into(), strides, this.layout.startOffset()), op = BackpropOp.new1(this, Op.Reshape), isVariable = false, dtype = this.dtype, device = this.device.clone())
        return Ok(Tensor(tensor))
    }

    public fun add(rhs: B): Result<Tensor> = Tensor.add(this, rhs.borrow())

    public fun add(rhs: Result<B>): Result<Tensor> {
        return Tensor.add(
            this,
            when (val tryResult = rhs) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.borrow(),
        )
    }

    public fun add(rhs: Double): Result<Tensor> = this.affine({ 1.0(rhs) }, { v -> v(rhs) })

    public fun sub(rhs: B): Result<Tensor> = Tensor.sub(this, rhs.borrow())

    public fun sub(rhs: Result<B>): Result<Tensor> {
        return Tensor.sub(
            this,
            when (val tryResult = rhs) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.borrow(),
        )
    }

    public fun sub(rhs: Double): Result<Tensor> = this.affine({ 1.0(rhs) }, { v: Double -> -v(rhs) })

    public fun mul(rhs: B): Result<Tensor> = Tensor.mul(this, rhs.borrow())

    public fun mul(rhs: Result<B>): Result<Tensor> {
        return Tensor.mul(
            this,
            when (val tryResult = rhs) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.borrow(),
        )
    }

    public fun mul(rhs: Double): Result<Tensor> = this.affine({ v -> v(rhs) }, { 0.0(rhs) })

    public fun div(rhs: B): Result<Tensor> = Tensor.div(this, rhs.borrow())

    public fun div(rhs: Result<B>): Result<Tensor> {
        return Tensor.div(
            this,
            when (val tryResult = rhs) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.borrow(),
        )
    }

    public fun div(rhs: Double): Result<Tensor> = this.affine({ v -> 1.0 / v(rhs) }, { 0.0(rhs) })

    public companion object {
        public operator fun invoke(
            id: TensorId = TensorId(),
            storage: RwLock<Storage> = RwLock<Storage>(),
            layout: Layout = Layout(),
            op: BackpropOp = BackpropOp(),
            isVariable: Boolean = false,
            dtype: DType = DType.entries.first(),
            device: Device = Device(),
        ): Tensor =
            Tensor(
                id = id,
                storageCtorArg = storage,
                layout = layout,
                op = op,
                isVariable = isVariable,
                dtype = dtype,
                device = device,
            )

        internal fun onesImpl(shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Result<Tensor> {
            val none = BackpropOp.none()
            val shape1 = shape
            val storage =
                when (val tryResult = device.allocUninit(shape1, dtype)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val layout = Layout.contiguous(shape1.clone())
            when (val tryResult = storage.constSet(one(dtype), layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            return Ok(fromStorage(storage, shape1, none, isVariable))
        }

        /**
         * Creates a new tensor filled with ones.
         *
         * ```kotlin
         * import candlecore.DType
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.ones(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
         * val b = Tensor.fromSlice(listOf(1.0f, 1.0, 1.0, 1.0, 1.0, 1.0), Pair(2, 3), Device.Cpu).getOrElse { return Result.failure(it) }
         * // a == b
         * ```
         */
        public fun ones(shape: Shape, dtype: DType, device: Device): Result<Tensor> = Tensor.onesImpl(shape, dtype, device, false)

        // Do not expose outside of the crate, the `isVariable = true` case should only be accessed from
        // the variable module.
        internal fun zerosImpl(shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Result<Tensor> {
            val none = BackpropOp.none()
            val shape1 = shape
            val storage =
                when (val tryResult = device.zeros(shape1, dtype)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(fromStorage(storage, shape1, none, isVariable))
        }

        /**
         * Creates a new tensor filled with zeros.
         *
         * ```kotlin
         * import candlecore.DType
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
         * val b = Tensor.fromSlice(listOf(0.0f, 0.0, 0.0, 0.0, 0.0, 0.0), Pair(2, 3), Device.Cpu).getOrElse { return Result.failure(it) }
         * // a == b
         * ```
         */
        public fun zeros(shape: Shape, dtype: DType, device: Device): Result<Tensor> = Tensor.zerosImpl(shape, dtype, device, false)

        // Do not expose outside of the crate, the `isVariable = true` case should only be accessed from
        // the variable module.
        internal fun emptyImpl(shape: Shape, dtype: DType, device: Device, isVariable: Boolean): Result<Tensor> {
            val none = BackpropOp.none()
            val shape1 = shape
            val storage =
                when (val tryResult = device.allocUninit(shape1, dtype)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(fromStorage(storage, shape1, none, isVariable))
        }

        /**
         * Creates a new tensor filled with uninitialized memory.
         *
         * # Safety
         * This returns uninitialized memory.
         *
         * ```kotlin
         * import candlecore.DType
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.empty(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
         * // a == b
         * ```
         */
        public fun empty(shape: Shape, dtype: DType, device: Device): Result<Tensor> = Tensor.emptyImpl(shape, dtype, device, false)

        internal fun <T : FloatDType> randImpl(lo: T, up: T, s: Shape, device: Device, isVariable: Boolean): Result<Tensor> {
            val s1 = s
            val storage =
                when (val tryResult = device.randUniform(lo, up, s1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, s1, none, isVariable))
        }

        internal fun randF64Impl(lo: Double, up: Double, s: Shape, dtype: DType, device: Device, isVariable: Boolean): Result<Tensor> {
            val s1 = s
            val storage =
                when (val tryResult = device.randUniformF64(lo, up, s1, dtype)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, s1, none, isVariable))
        }

        /**
         * Creates a new tensor initialized with values sampled uniformly between `lo` and `up`.
         */
        public fun <T : FloatDType> rand(lo: T, up: T, s: Shape, device: Device): Result<Tensor> = Tensor.randImpl(lo, up, s, device, false)

        internal fun <T : FloatDType> randnImpl(mean: T, std: T, s: Shape, device: Device, isVariable: Boolean): Result<Tensor> {
            val s1 = s
            val storage =
                when (val tryResult = device.randNormal(mean, std, s1)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, s1, none, isVariable))
        }

        internal fun randnF64Impl(mean: Double, std: Double, s: Shape, dtype: DType, device: Device, isVariable: Boolean): Result<Tensor> {
            val s1 = s
            val storage =
                when (val tryResult = device.randNormalF64(mean, std, s1, dtype)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, s1, none, isVariable))
        }

        /**
         * Creates a new tensor initialized with values sampled from a normal distribution with the
         * specified `mean` and standard deviation `std`.
         */
        public fun <T : FloatDType> randn(mean: T, std: T, s: Shape, device: Device): Result<Tensor> = Tensor.randnImpl(mean, std, s, device, false)

        internal fun <A : NdArray> newImpl(array: A, shape: Shape, device: Device, isVariable: Boolean): Result<Tensor> {
            val n: Int = shape.elemCount()
            val bufferSize: Int =
                when (val tryResult = array.shape()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.elemCount()
            if (bufferSize != n) {
                return Err(Error.ShapeMismatch(bufferSize = bufferSize, shape = shape).bt())
            }
            val storage =
                when (val tryResult = device.storage(array)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, shape, none, isVariable))
        }

        /**
         * Creates a new tensor on the specified device using the content and shape of the input.
         */
        public fun <A : NdArray> new(array: A, device: Device): Result<Tensor> {
            val shape =
                when (val tryResult = array.shape()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Tensor.newImpl(array, shape, device, false)
        }

        /**
         * Returns a new tensor with all the elements having the same specified value.
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.full(3.5, Pair(2, 4), Device.Cpu).getOrElse { return Result.failure(it) }
         * check(a.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(3.5, 3.5, 3.5, 3.5), listOf(3.5, 3.5, 3.5, 3.5)))
         * ```
         */
        public fun <D : WithDType> full(value: D, shape: Shape, device: Device): Result<Tensor> {
            val none = BackpropOp.none()
            val shape1 = shape
            val storage =
                when (val tryResult = device.allocUninit(shape1, D.DTYPE)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val layout = Layout.contiguous(shape1.clone())
            when (val tryResult = storage.constSet(value.toScalar(), layout)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
            return Ok(fromStorage(storage, shape1, none, false))
        }

        /**
         * Creates a new 1D tensor from an iterator.
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.fromIter(listOf(1.0, 2.0, 3.0, 4.0).intoIter(), Device.Cpu).getOrElse { return Result.failure(it) }
         * check(a.toVec1<Double>().getOrElse { return Result.failure(it) } == listOf(1.0, 2.0, 3.0, 4.0))
         * ```
         */
        public fun <D : WithDType> fromIter(iter: Any, device: Device): Result<Tensor> {
            val data = iter.intoIter().collect<Vec>()
            val len = data.len()
            return Tensor.fromVecImpl(data, len, device, false)
        }

        /**
         * Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
         * difference `1` from `start`.
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.arange(2.0, 5.0, Device.Cpu).getOrElse { return Result.failure(it) }
         * check(a.toVec1<Double>().getOrElse { return Result.failure(it) } == listOf(2.0, 3.0, 4.0))
         * ```
         */
        public fun <D : WithDType> arange(start: D, end: D, device: Device, oneDFn: (Any?) -> Any?): Result<Tensor> = Tensor.arangeStep(start, end, oneDFn(), device)

        /**
         * Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
         * difference `step` from `start`.
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.arangeStep(2.0, 4.0, 0.5, Device.Cpu).getOrElse { return Result.failure(it) }
         * check(a.toVec1<Double>().getOrElse { return Result.failure(it) } == listOf(2.0, 2.5, 3.0, 3.5))
         * ```
         */
        public fun <D : WithDType> arangeStep(start: D, end: D, step: D, device: Device, isZeroDFn: (Any?) -> Any?, zeroDFn: (Any?) -> Any?): Result<Tensor> {
            if (isZeroDFn(step)) {
                bail("step cannot be zero")
            }
            val data = mutableListOf()
            var current = start
            if (step >= zeroDFn()) {
                while (current < end) {
                    data.add(current)
                    current += step
                }
            } else {
                while (current > end) {
                    data.add(current)
                    current += step
                }
            }
            val len = data.size
            return Tensor.fromVecImpl(data, len, device, false)
        }

        internal fun <S : ShapeWithOneHole, D : WithDType> fromVecImpl(data: List<D>, shape: S, device: Device, isVariable: Boolean): Result<Tensor> {
            val shape1 =
                when (val tryResult = shape.intoShape(data.size)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val storage =
                when (val tryResult = device.storageOwned(data)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, shape1, none, isVariable))
        }

        /**
         * Creates a new tensor initialized with values from the input vector. The number of elements
         * in this vector must be the same as the number of elements defined by the shape.
         * If the device is cpu, no data copy is made.
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val a = Tensor.fromVec(mutableListOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), Pair(2, 3), Device.Cpu).getOrElse { return Result.failure(it) }
         * check(a.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(1.0, 2.0, 3.0), listOf(4.0, 5.0, 6.0)))
         * ```
         */
        public fun <S : ShapeWithOneHole, D : WithDType> fromVec(data: List<D>, shape: S, device: Device): Result<Tensor> = Tensor.fromVecImpl(data, shape, device, false)

        /**
         * Creates a new tensor initialized with values from the input slice. The number of elements
         * in this vector must be the same as the number of elements defined by the shape.
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Tensor
         *
         * val values = mutableListOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
         * val a = Tensor.fromSlice(values[1 until 7], Pair(2, 3), Device.Cpu).getOrElse { return Result.failure(it) }
         * check(a.toVec2<Double>().getOrElse { return Result.failure(it) } == listOf(listOf(2.0, 3.0, 4.0), listOf(5.0, 6.0, 7.0)))
         * ```
         */
        public fun <S : ShapeWithOneHole, D : WithDType> fromSlice(array: List<D>, shape: S, device: Device): Result<Tensor> {
            val shape1 =
                when (val tryResult = shape.intoShape(array.size)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val storage =
                when (val tryResult = device.storageFromSlice(array)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val none = BackpropOp.none()
            return Ok(fromStorage(storage, shape1, none, false))
        }

        /**
         * Creates a fresh tensor structure based on a storage and a shape.
         *
         * # Note
         * - This uses contiguous strides
         * - Ensure the shape is compatible with the shape of the storage.
         */
        public fun fromStorage(storage: Storage, shape: Shape, op: BackpropOp, isVariable: Boolean): Tensor = fromStorage(storage, shape, op, isVariable)

        /**
         * Creates grids of coordinates specified by the 1D inputs.
         *
         * # Arguments
         *
         * * `args` - A slice of 1D tensors.
         * * `xyIndexing` - Whether to use xy indexing or ij indexing. If xy is selected, the
         *   first dimension corresponds to the cardinality of the second input and the second
         *   dimension corresponds to the cardinality of the first input. If ij is selected, the
         *   dimensions are in the same order as the cardinality of the inputs.
         *
         * # Examples
         *
         * ```kotlin
         * import candlecore.Device
         * import candlecore.Shape
         * import candlecore.Tensor
         *
         * val x = Tensor.new(listOf(1.0f, 2.0, 3.0), Device.Cpu).getOrElse { return Result.failure(it) }
         * val y = Tensor.new(listOf(4.0f, 5.0, 6.0), Device.Cpu).getOrElse { return Result.failure(it) }
         * val gridsXy = Tensor.meshgrid(listOf(x, y), true).getOrElse { return Result.failure(it) }
         * check(gridsXy.len() == 2)
         * check(gridsXy[0].dims() == listOf(3, 3))
         * check(gridsXy[0].toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(1.0, 2.0, 3.0), listOf(1.0, 2.0, 3.0), listOf(1.0, 2.0, 3.0)))
         * check(gridsXy[1].toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(4.0, 4.0, 4.0), listOf(5.0, 5.0, 5.0), listOf(6.0, 6.0, 6.0)))
         * val gridsIj = Tensor.meshgrid(listOf(x, y), false).getOrElse { return Result.failure(it) }
         * check(gridsIj[0].toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(1.0, 1.0, 1.0), listOf(2.0, 2.0, 2.0), listOf(3.0, 3.0, 3.0)))
         * check(gridsIj[1].toVec2<Float>().getOrElse { return Result.failure(it) } == listOf(listOf(4.0, 5.0, 6.0), listOf(4.0, 5.0, 6.0), listOf(4.0, 5.0, 6.0)))
         * ```
         *
         * # Errors
         *
         * * Will return `Err` if `args` contains less than 2 tensors.
         */
        public fun <A : AsRef<Tensor>> meshgrid(args: List<A>, xyIndexing: Boolean): Result<List<Tensor>> {
            if (args.size <= 1) {
                when (val tryResult = Err(Error.OpRequiresAtLeastTwoTensors(op = "meshgrid").bt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
            val args1: Vec =
                if (xyIndexing) {
                    args.iter().rev().collect()
                } else {
                    args.iter().collect()
                }
            val shape = ArrayList<Tensor>(args1.len())
            for (arg in args1.iter()) {
                shape.add(
                    when (val tryResult = arg.asRef().dims1()) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
            }
            val grids = ArrayList<Tensor>(args1.len())
            for (idx in 0 until args1.len()) {
                val ones = mutableListOf(1, args1.len())
                ones[idx] = shape[idx]
                val arg1 =
                    when (val tryResult = args1[idx].asRef().reshape(ones)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                val repeats = shape.toMutableList()
                repeats[idx] = 1
                val repeatedTensor =
                    when (val tryResult = arg1.repeat(repeats)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                grids.add(repeatedTensor)
            }
            if (xyIndexing) {
                grids.reverse()
            }
            return Ok(grids)
        }

        /**
         * Stacks two or more tensors along a particular dimension.
         *
         * All tensors must have the same rank, and the output has one additional rank
         *
         * ```kotlin
         * val a = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
         * val b = Tensor.zeros(Pair(2, 3), DType.F32, Device.Cpu).getOrElse { return Result.failure(it) }
         * val c = Tensor.stack(listOf(a, b), 0).getOrElse { return Result.failure(it) }
         * check(c.shape().dims() == listOf(2, 2, 3))
         * val c1 = Tensor.stack(listOf(a, b), 2).getOrElse { return Result.failure(it) }
         * check(c1.shape().dims() == listOf(2, 3, 2))
         * ```
         */
        public fun <A : AsRef<Tensor>, D : Dim> stack(args: List<A>, dim: D): Result<Tensor> {
            if (args.isEmpty()) {
                when (val tryResult = Err(Error.OpRequiresAtLeastOneTensor(op = "stack").bt())) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            }
            val dim1 =
                when (val tryResult = dim.toIndexPlusOne(args[0].asRef().shape(), "stack")) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val args1 =
                when (val tryResult = args.map({ t -> t.asRef().unsqueeze(dim1) }).collect<Result<Vec>>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Tensor.cat(args1, dim1)
        }

        /**
         * Returns a lower triangular matrix of ones of size n by n.
         */
        public fun tril2(n: Int, dtype: DType, device: Device): Result<Tensor> {
            val t =
                when (val tryResult = Tensor.arange(0u, n.toUInt(), device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t1 =
                when (
                    val tryResult =
                        when (val tryResult = t.reshape(Pair(1, n))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.broadcastAs(Pair(n, n))
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t2 =
                when (
                    val tryResult =
                        when (val tryResult = t.reshape(Pair(n, 1))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.broadcastAs(Pair(n, n))
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return when (val tryResult = t1.le(t2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.toDtype(dtype)
        }

        /**
         * Returns an upper triangular matrix of ones of size n by n.
         */
        public fun triu2(n: Int, dtype: DType, device: Device): Result<Tensor> {
            val t =
                when (val tryResult = Tensor.arange(0u, n.toUInt(), device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t1 =
                when (
                    val tryResult =
                        when (val tryResult = t.reshape(Pair(1, n))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.broadcastAs(Pair(n, n))
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t2 =
                when (
                    val tryResult =
                        when (val tryResult = t.reshape(Pair(n, 1))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.broadcastAs(Pair(n, n))
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return when (val tryResult = t1.ge(t2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.toDtype(dtype)
        }

        /**
         * Returns a matrix with a diagonal of ones of size n by n.
         */
        public fun eye(n: Int, dtype: DType, device: Device): Result<Tensor> {
            val t =
                when (val tryResult = Tensor.arange(0u, n.toUInt(), device)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t1 =
                when (
                    val tryResult =
                        when (val tryResult = t.reshape(Pair(1, n))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.broadcastAs(Pair(n, n))
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            val t2 =
                when (
                    val tryResult =
                        when (val tryResult = t.reshape(Pair(n, 1))) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.broadcastAs(Pair(n, n))
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return when (val tryResult = t1.eq(t2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.toDtype(dtype)
        }

        public fun from(tupleArg0: Pair<Storage, Shape>): Tensor = fromStorage(storage, shape, BackpropOp.none(), false)
    }
}

/**
 * Creates a fresh tensor structure based on a storage and a shape, this uses contiguous strides.
 */
internal fun fromStorage(storage: Storage, shape: Shape, op: BackpropOp, isVariable: Boolean): Tensor {
    val dtype = storage.dtype()
    val device = storage.device()
    val tensor = Tensor(id = TensorId.new(), storage = RwLock.new(storage), layout = Layout.contiguous(shape), op = op, isVariable = isVariable, dtype = dtype, device = device)
    return Tensor(tensor)
}

public fun <B : Borrow<Tensor>> Result<B>.add(rhs: Tensor): Result<Tensor> {
    return Tensor.add(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.addTensor(rhs: Tensor): Result<Tensor> {
    return Tensor.add(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.sub(rhs: Tensor): Result<Tensor> {
    return Tensor.sub(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.subTensor(rhs: Tensor): Result<Tensor> {
    return Tensor.sub(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.mul(rhs: Tensor): Result<Tensor> {
    return Tensor.mul(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.mulTensor(rhs: Tensor): Result<Tensor> {
    return Tensor.mul(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.div(rhs: Tensor): Result<Tensor> {
    return Tensor.div(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun <B : Borrow<Tensor>> Result<B>.divTensor(rhs: Tensor): Result<Tensor> {
    return Tensor.div(
        when (val tryResult = this) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.borrow(),
        rhs,
    )
}

public fun Double.add(rhs: Tensor): Result<Tensor> = rhs + this

public fun Double.addTensor(rhs: Tensor): Result<Tensor> = rhs + this

public fun Double.mul(rhs: Tensor): Result<Tensor> = rhs * this

public fun Double.mulTensor(rhs: Tensor): Result<Tensor> = rhs * this

public fun Double.sub(rhs: Tensor): Result<Tensor> = rhs.affine(-1.0, this)

public fun Double.subTensor(rhs: Tensor): Result<Tensor> = rhs.affine(-1.0, this)

public fun Double.div(rhs: Tensor): Result<Tensor> {
    return when (val tryResult = rhs.recip()) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    } * this
}

public fun Double.divTensor(rhs: Tensor): Result<Tensor> {
    return when (val tryResult = rhs.recip()) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    } * this
}
