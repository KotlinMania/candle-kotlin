// port-lint: source candle-core/src/storage.rs
package io.github.kotlinmania.candle

public sealed class Storage {
    public data class Cpu(
        val storage: CpuStorage,
    ) : Storage()

    public data class Cuda(
        val storage: CudaStorage,
    ) : Storage()

    public data class Metal(
        val storage: MetalStorage,
    ) : Storage()

    public fun tryClone(layout: Layout): Storage =
        when (this) {
            is Cpu -> Cpu(storage.tryClone(layout))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun device(): Device =
        when (this) {
            is Cpu -> Device.Cpu
            is Cuda -> Device.Cuda(CudaDevice())
            is Metal -> Device.Metal(MetalDevice())
        }

    public fun dtype(): DType =
        when (this) {
            is Cpu -> storage.dtype()
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun sameDevice(rhs: Storage, op: String) {
        val lhsDevice = device().location()
        val rhsDevice = rhs.device().location()
        if (lhsDevice != rhsDevice) {
            throw CandleException.DeviceMismatchBinaryOp(lhsDevice, rhsDevice, op)
        }
    }

    public fun sameDtype(rhs: Storage, op: String) {
        val lhsDType = dtype()
        val rhsDType = rhs.dtype()
        if (lhsDType != rhsDType) {
            throw CandleException.DTypeMismatchBinaryOp(lhsDType, rhsDType, op)
        }
    }

    public fun constSet(v: Scalar, l: Layout) {
        when (this) {
            is Cpu -> storage.constSet(v, l)
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }
    }

    public fun affine(layout: Layout, mul: Double, add: Double): Storage =
        when (this) {
            is Cpu -> Cpu(storage.affine(layout, mul, add))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun powf(layout: Layout, e: Double): Storage =
        when (this) {
            is Cpu -> Cpu(storage.powf(layout, e))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun elu(layout: Layout, alpha: Double): Storage =
        when (this) {
            is Cpu -> Cpu(storage.elu(layout, alpha))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun unaryImpl(layout: Layout, op: UnaryOp): Storage =
        when (this) {
            is Cpu -> Cpu(storage.unaryImpl(layout, op))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun binaryImpl(
        rhs: Storage,
        lhsLayout: Layout,
        rhsLayout: Layout,
        op: BinaryOp,
    ): Storage {
        sameDevice(rhs, op.name)
        sameDtype(rhs, op.name)
        return when (this) {
            is Cpu -> Cpu(storage.binaryImpl((rhs as Cpu).storage, lhsLayout, rhsLayout, op))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }
    }

    public fun cmp(
        op: CmpOp,
        rhs: Storage,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        sameDevice(rhs, op.name)
        sameDtype(rhs, op.name)
        return when (this) {
            is Cpu -> Cpu(storage.cmp(op, (rhs as Cpu).storage, lhsLayout, rhsLayout))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }
    }

    public fun whereCond(
        layout: Layout,
        onTrue: Storage,
        trueLayout: Layout,
        onFalse: Storage,
        falseLayout: Layout,
    ): Storage =
        when (this) {
            is Cpu ->
                Cpu(
                    storage.whereCond(
                        layout,
                        (onTrue as Cpu).storage,
                        trueLayout,
                        (onFalse as Cpu).storage,
                        falseLayout,
                    ),
                )
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun toDtype(layout: Layout, dtype: DType): Storage =
        when (this) {
            is Cpu -> Cpu(storage.toDType(layout, dtype))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun toCpuStorage(): CpuStorage =
        when (this) {
            is Cpu -> storage
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun matmul(
        rhs: Storage,
        bmnk: List<Int>,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        sameDevice(rhs, "matmul")
        sameDtype(rhs, "matmul")
        return when (this) {
            is Cpu -> Cpu(storage.matmul((rhs as Cpu).storage, bmnk, lhsLayout, rhsLayout))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }
    }

    public fun reduceOp(
        op: ReduceOp,
        layout: Layout,
        reduceDims: List<Int>,
    ): Storage =
        when (this) {
            is Cpu -> Cpu(storage.reduceOp(op, layout, reduceDims))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }

    public fun indexSelect(
        ids: Storage,
        layout: Layout,
        idsLayout: Layout,
        dim: Int,
    ): Storage =
        when (this) {
            is Cpu -> Cpu(storage.indexSelect((ids as Cpu).storage, layout, idsLayout, dim))
            is Cuda -> throw CandleException.NotCompiledWithCudaSupport
            is Metal -> throw CandleException.NotCompiledWithMetalSupport
        }
}
