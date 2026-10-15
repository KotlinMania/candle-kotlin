// port-lint: source dummy_metal_backend.rs
package io.github.kotlinmania.candle

public sealed class MetalError : Exception() {
    public data class Message(
        override val message: String,
    ) : MetalError() {
        override fun toString(): String = message
    }

    public companion object {
        public fun from(e: String): MetalError = Message(e)
    }
}

public data class MetalDevice(
    val gpuId: Int = 0,
) : BackendDevice {
    public interface Device
    public interface Storage

    public fun id(): DeviceId = DeviceId(gpuId)

    override fun location(): DeviceLocation =
        throw CandleException.NotCompiledWithMetalSupport

    override fun sameDevice(other: BackendDevice): Boolean =
        throw CandleException.NotCompiledWithMetalSupport

    override fun setSeed(seed: Long): Unit =
        throw CandleException.NotCompiledWithMetalSupport

    override fun getCurrentSeed(): Long =
        throw CandleException.NotCompiledWithMetalSupport

    override fun zerosImpl(shape: Shape, dtype: DType): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun allocUninit(shape: Shape, dtype: DType): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    public fun <T> storageFromSlice(slice: List<T>): MetalStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun storageFromCpuStorage(storage: CpuStorage): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun storageFromCpuStorageOwned(storage: CpuStorage): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun randUniform(shape: Shape, dtype: DType, min: Double, max: Double): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun synchronize() {}

    public companion object {
        public fun new(ordinal: Int): MetalDevice =
            throw CandleException.NotCompiledWithMetalSupport
    }
}

public class MetalStorage : BackendStorage {
    public fun transferToDevice(dst: MetalDevice): MetalStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun tryClone(layout: Layout): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun dtype(): DType =
        throw CandleException.NotCompiledWithMetalSupport

    override fun device(): BackendDevice =
        throw CandleException.NotCompiledWithMetalSupport

    override fun constSet(v: Scalar, l: Layout): Unit =
        throw CandleException.NotCompiledWithMetalSupport

    override fun toCpuStorage(): CpuStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun affine(layout: Layout, mul: Double, add: Double): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun powf(layout: Layout, e: Double): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun elu(layout: Layout, alpha: Double): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun reduceOp(op: ReduceOp, layout: Layout, dims: List<Int>): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun cmp(op: CmpOp, rhs: BackendStorage, lhsL: Layout, rhsL: Layout): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun toDtype(layout: Layout, targetDType: DType): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    public fun <B : UnaryOpT> unaryImpl(layout: Layout): MetalStorage =
        throw CandleException.NotCompiledWithMetalSupport

    public fun <B : BinaryOpT> binaryImpl(
        rhs: MetalStorage,
        lhsL: Layout,
        rhsL: Layout,
    ): MetalStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun whereCond(
        layout: Layout,
        onTrue: BackendStorage,
        trueL: Layout,
        onFalse: BackendStorage,
        falseL: Layout,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun indexSelect(rhs: BackendStorage, lhsL: Layout, rhsL: Layout, dim: Int): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun gather(l: Layout, indexes: BackendStorage, indexesL: Layout, dim: Int): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        throw CandleException.NotCompiledWithMetalSupport

    override fun scatterAddSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): Unit =
        throw CandleException.NotCompiledWithMetalSupport

    override fun indexAdd(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun copyStridedSrc(dst: BackendStorage, dstOffset: Int, srcLayout: Layout): Unit =
        throw CandleException.NotCompiledWithMetalSupport

    override fun copy2d(
        dst: BackendStorage,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    ): Unit =
        throw CandleException.NotCompiledWithMetalSupport

    override fun avgPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun maxPool2d(
        l: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun upsampleNearest1d(l: Layout, targetSize: Int): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun upsampleNearest2d(l: Layout, targetH: Int, targetW: Int): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport

    override fun upsampleBilinear2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean,
        scaleH: Double?,
        scaleW: Double?,
    ): BackendStorage =
        throw CandleException.NotCompiledWithMetalSupport
}
