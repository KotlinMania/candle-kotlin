// port-lint: source backend.rs
package io.github.kotlinmania.candle

/**
 * Traits to Define Backend Behavior
 */
public interface BackendStorage {
    public fun tryClone(layout: Layout): BackendStorage

    public fun dtype(): DType

    public fun device(): BackendDevice

    public fun toCpuStorage(): CpuStorage

    public fun affine(layout: Layout, mul: Double, add: Double): BackendStorage

    public fun powf(layout: Layout, e: Double): BackendStorage

    public fun elu(layout: Layout, alpha: Double): BackendStorage

    public fun reduceOp(op: ReduceOp, layout: Layout, dims: List<Int>): BackendStorage

    public fun cmp(op: CmpOp, rhs: BackendStorage, lhsL: Layout, rhsL: Layout): BackendStorage

    public fun toDtype(layout: Layout, targetDType: DType): BackendStorage

    public fun whereCond(
        layout: Layout,
        onTrue: BackendStorage,
        trueL: Layout,
        onFalse: BackendStorage,
        falseL: Layout,
    ): BackendStorage

    public fun conv1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): BackendStorage

    public fun convTranspose1d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): BackendStorage

    public fun conv2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): BackendStorage

    public fun convTranspose2d(
        l: Layout,
        kernel: BackendStorage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): BackendStorage

    public fun avgPool2d(l: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): BackendStorage

    public fun maxPool2d(l: Layout, kernelSize: Pair<Int, Int>, stride: Pair<Int, Int>): BackendStorage

    public fun upsampleNearest1d(l: Layout, targetSize: Int): BackendStorage

    public fun upsampleNearest2d(l: Layout, targetH: Int, targetW: Int): BackendStorage

    public fun upsampleBilinear2d(
        l: Layout,
        targetH: Int,
        targetW: Int,
        alignCorners: Boolean,
        scaleH: Double? = null,
        scaleW: Double? = null,
    ): BackendStorage

    public fun gather(l: Layout, indexes: BackendStorage, indexesL: Layout, dim: Int): BackendStorage

    public fun scatterSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    )

    public fun scatterAddSet(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    )

    public fun indexSelect(rhs: BackendStorage, lhsL: Layout, rhsL: Layout, dim: Int): BackendStorage

    public fun indexAdd(
        l: Layout,
        indexes: BackendStorage,
        indexesL: Layout,
        src: BackendStorage,
        srcL: Layout,
        dim: Int,
    ): BackendStorage

    public fun matmul(
        rhs: BackendStorage,
        bmnk: List<Int>,
        lhsL: Layout,
        rhsL: Layout,
    ): BackendStorage

    public fun copyStridedSrc(dst: BackendStorage, dstOffset: Int, srcLayout: Layout)

    public fun copy2d(
        dst: BackendStorage,
        d1: Int,
        d2: Int,
        srcStride1: Int,
        dstStride1: Int,
        srcOffset: Int,
        dstOffset: Int,
    )

    public fun constSet(v: Scalar, l: Layout)
}

public interface BackendDevice {
    public fun location(): DeviceLocation

    public fun sameDevice(other: BackendDevice): Boolean

    public fun zerosImpl(shape: Shape, dtype: DType): BackendStorage

    public fun allocUninit(shape: Shape, dtype: DType): BackendStorage

    public fun storageFromCpuStorage(storage: CpuStorage): BackendStorage

    public fun storageFromCpuStorageOwned(storage: CpuStorage): BackendStorage

    public fun randUniform(shape: Shape, dtype: DType, min: Double, max: Double): BackendStorage

    public fun randNormal(shape: Shape, dtype: DType, mean: Double, std: Double): BackendStorage

    public fun setSeed(seed: Long)

    public fun getCurrentSeed(): Long

    /**
     * Synchronize should block until all the operations on the device are completed.
     */
    public fun synchronize()
}
