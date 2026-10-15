// port-lint: source storage.rs
package io.github.kotlinmania.candle

// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead tryClone should be used.
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
            is Cuda -> {
                val cloned = storage.tryClone(layout) as CudaStorage
                Cuda(cloned)
            }
            is Metal -> {
                val cloned = storage.tryClone(layout) as MetalStorage
                Metal(cloned)
            }
        }

    public fun device(): Device =
        when (this) {
            is Cpu -> Device.Cpu
            is Cuda -> Device.Cuda(storage.device() as CudaDevice)
            is Metal -> Device.Metal(storage.device() as MetalDevice)
        }

    public fun dtype(): DType =
        when (this) {
            is Cpu -> storage.dtype()
            is Cuda -> storage.dtype()
            is Metal -> storage.dtype()
        }

    public fun sameDevice(rhs: Storage, op: String) {
        val lhsDevice = device()
        val rhsDevice = rhs.device()
        val lhs = lhsDevice.location()
        val rhsLoc = rhsDevice.location()
        val sameDevice = if (device().isMetal()) {
            // On metal, we require the device to be exactly the same rather than
            // having the same location. In cuda this is not necessary as all CudaDevice on the
            // same GPU will use the same cuda stream.
            lhsDevice.sameDevice(rhsDevice)
        } else {
            lhs == rhsLoc
        }
        if (!sameDevice) {
            throw CandleException.DeviceMismatchBinaryOp(lhs, rhsLoc, op)
        }
    }

    public fun sameDtype(rhs: Storage, op: String) {
        val lhs = dtype()
        val rhsDtype = rhs.dtype()
        if (lhs != rhsDtype) {
            throw CandleException.DTypeMismatchBinaryOp(lhs, rhsDtype, op)
        }
    }

    public fun constSet(v: Scalar, l: Layout) {
        when (this) {
            is Cpu -> storage.constSet(v, l)
            is Cuda -> storage.constSet(v, l)
            is Metal -> storage.constSet(v, l)
        }
    }

    public fun affine(layout: Layout, mul: Double, add: Double): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.affine(layout, mul, add)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.affine(layout, mul, add) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.affine(layout, mul, add) as MetalStorage
                Metal(s)
            }
        }

    public fun powf(layout: Layout, alpha: Double): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.powf(layout, alpha)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.powf(layout, alpha) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.powf(layout, alpha) as MetalStorage
                Metal(s)
            }
        }

    public fun elu(layout: Layout, alpha: Double): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.elu(layout, alpha)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.elu(layout, alpha) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.elu(layout, alpha) as MetalStorage
                Metal(s)
            }
        }

    public fun cmp(
        op: CmpOp,
        rhs: Storage,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        sameDevice(rhs, "cmp")
        sameDtype(rhs, "cmp")
        return when {
            this is Cpu && rhs is Cpu -> {
                val s = storage.cmp(op, rhs.storage, lhsLayout, rhsLayout)
                Cpu(s)
            }
            this is Cuda && rhs is Cuda -> {
                val s = storage.cmp(op, rhs.storage, lhsLayout, rhsLayout) as CudaStorage
                Cuda(s)
            }
            this is Metal && rhs is Metal -> {
                val s = storage.cmp(op, rhs.storage, lhsLayout, rhsLayout) as MetalStorage
                Metal(s)
            }
            else -> {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                throw CandleException.DeviceMismatchBinaryOp(
                    device().location(),
                    rhs.device().location(),
                    "cmp",
                )
            }
        }
    }

    public fun reduceOp(
        op: ReduceOp,
        layout: Layout,
        s: List<Int>,
    ): Storage =
        when (this) {
            is Cpu -> {
                val out = storage.reduceOp(op, layout, s)
                Cpu(out)
            }
            is Cuda -> {
                val out = storage.reduceOp(op, layout, s) as CudaStorage
                Cuda(out)
            }
            is Metal -> {
                val out = storage.reduceOp(op, layout, s) as MetalStorage
                Metal(out)
            }
        }

    public fun toDtype(layout: Layout, dtype: DType): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.toDType(layout, dtype)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.toDtype(layout, dtype) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.toDtype(layout, dtype) as MetalStorage
                Metal(s)
            }
        }

    public fun applyOp1(l: Layout, c: CustomOp1): Pair<Storage, Shape> =
        when (this) {
            is Cpu -> {
                val (s, shape) = c.cpuFwd(storage, l)
                Pair(Cpu(s), shape)
            }
            is Cuda -> {
                val (s, shape) = c.cudaFwd(storage, l)
                Pair(Cuda(s), shape)
            }
            is Metal -> {
                val (s, shape) = c.metalFwd(storage, l)
                Pair(Metal(s), shape)
            }
        }

    public fun applyOp2(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        c: CustomOp2,
    ): Pair<Storage, Shape> {
        sameDevice(t2, c.name())
        return when {
            this is Cpu && t2 is Cpu -> {
                val (s, shape) = c.cpuFwd(storage, l1, t2.storage, l2)
                Pair(Cpu(s), shape)
            }
            this is Cuda && t2 is Cuda -> {
                val (s, shape) = c.cudaFwd(storage, l1, t2.storage, l2)
                Pair(Cuda(s), shape)
            }
            this is Metal && t2 is Metal -> {
                val (s, shape) = c.metalFwd(storage, l1, t2.storage, l2)
                Pair(Metal(s), shape)
            }
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun applyOp3(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        t3: Storage,
        l3: Layout,
        c: CustomOp3,
    ): Pair<Storage, Shape> {
        sameDevice(t2, c.name())
        sameDevice(t3, c.name())
        return when {
            this is Cpu && t2 is Cpu && t3 is Cpu -> {
                val (s, shape) = c.cpuFwd(storage, l1, t2.storage, l2, t3.storage, l3)
                Pair(Cpu(s), shape)
            }
            this is Cuda && t2 is Cuda && t3 is Cuda -> {
                val (s, shape) = c.cudaFwd(storage, l1, t2.storage, l2, t3.storage, l3)
                Pair(Cuda(s), shape)
            }
            this is Metal && t2 is Metal && t3 is Metal -> {
                val (s, shape) = c.metalFwd(storage, l1, t2.storage, l2, t3.storage, l3)
                Pair(Metal(s), shape)
            }
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun inplaceOp1(l: Layout, c: InplaceOp1) {
        when (this) {
            is Cpu -> c.cpuFwd(storage, l)
            is Cuda -> c.cudaFwd(storage, l)
            is Metal -> c.metalFwd(storage, l)
        }
    }

    public fun inplaceOp2(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        c: InplaceOp2,
    ) {
        sameDevice(t2, c.name())
        when {
            this is Cpu && t2 is Cpu -> c.cpuFwd(storage, l1, t2.storage, l2)
            this is Cuda && t2 is Cuda -> c.cudaFwd(storage, l1, t2.storage, l2)
            this is Metal && t2 is Metal -> c.metalFwd(storage, l1, t2.storage, l2)
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun inplaceOp3(
        l1: Layout,
        t2: Storage,
        l2: Layout,
        t3: Storage,
        l3: Layout,
        c: InplaceOp3,
    ) {
        sameDevice(t2, c.name())
        sameDevice(t3, c.name())
        when {
            this is Cpu && t2 is Cpu && t3 is Cpu -> c.cpuFwd(storage, l1, t2.storage, l2, t3.storage, l3)
            this is Cuda && t2 is Cuda && t3 is Cuda -> c.cudaFwd(storage, l1, t2.storage, l2, t3.storage, l3)
            this is Metal && t2 is Metal && t3 is Metal -> c.metalFwd(storage, l1, t2.storage, l2, t3.storage, l3)
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun unaryImpl(layout: Layout, op: UnaryOp): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.unaryImpl(layout, op)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.unaryImpl<UnaryOpT>(layout)
                Cuda(s)
            }
            is Metal -> {
                val s = storage.unaryImpl<UnaryOpT>(layout)
                Metal(s)
            }
        }

    public fun binaryImpl(
        rhs: Storage,
        lhsLayout: Layout,
        rhsLayout: Layout,
        op: BinaryOp,
    ): Storage {
        sameDevice(rhs, op.name)
        sameDtype(rhs, op.name)
        return when {
            this is Cpu && rhs is Cpu -> {
                val s = storage.binaryImpl(rhs.storage, lhsLayout, rhsLayout, op)
                Cpu(s)
            }
            this is Cuda && rhs is Cuda -> {
                val s = storage.binaryImpl<BinaryOpT>(rhs.storage, lhsLayout, rhsLayout)
                Cuda(s)
            }
            this is Metal && rhs is Metal -> {
                val s = storage.binaryImpl<BinaryOpT>(rhs.storage, lhsLayout, rhsLayout)
                Metal(s)
            }
            else -> {
                // Should not happen because of the same device check above but we're defensive
                // anyway.
                throw CandleException.DeviceMismatchBinaryOp(
                    device().location(),
                    rhs.device().location(),
                    op.name,
                )
            }
        }
    }

    public fun conv1d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConv1D,
    ): Storage {
        sameDevice(kernel, "conv1d")
        sameDtype(kernel, "conv1d")
        return when {
            this is Cpu && kernel is Cpu -> {
                val s = storage.conv1d(l, kernel.storage, kernelL, params)
                Cpu(s)
            }
            this is Cuda && kernel is Cuda -> {
                val s = storage.conv1d(l, kernel.storage, kernelL, params) as CudaStorage
                Cuda(s)
            }
            this is Metal && kernel is Metal -> {
                val s = storage.conv1d(l, kernel.storage, kernelL, params) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                kernel.device().location(),
                "conv1d",
            )
        }
    }

    public fun convTranspose1d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConvTranspose1D,
    ): Storage {
        sameDevice(kernel, "conv-transpose1d")
        sameDtype(kernel, "conv-transpose1d")
        return when {
            this is Cpu && kernel is Cpu -> {
                val s = storage.convTranspose1d(l, kernel.storage, kernelL, params)
                Cpu(s)
            }
            this is Cuda && kernel is Cuda -> {
                val s = storage.convTranspose1d(l, kernel.storage, kernelL, params) as CudaStorage
                Cuda(s)
            }
            this is Metal && kernel is Metal -> {
                val s = storage.convTranspose1d(l, kernel.storage, kernelL, params) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                kernel.device().location(),
                "conv-transpose1d",
            )
        }
    }

    public fun conv2d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConv2D,
    ): Storage {
        sameDevice(kernel, "conv2d")
        sameDtype(kernel, "conv2d")
        return when {
            this is Cpu && kernel is Cpu -> {
                val s = storage.conv2d(l, kernel.storage, kernelL, params)
                Cpu(s)
            }
            this is Cuda && kernel is Cuda -> {
                val s = storage.conv2d(l, kernel.storage, kernelL, params) as CudaStorage
                Cuda(s)
            }
            this is Metal && kernel is Metal -> {
                val s = storage.conv2d(l, kernel.storage, kernelL, params) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                kernel.device().location(),
                "conv2d",
            )
        }
    }

    public fun convTranspose2d(
        l: Layout,
        kernel: Storage,
        kernelL: Layout,
        params: ParamsConvTranspose2D,
    ): Storage {
        sameDevice(kernel, "conv_transpose2d")
        sameDtype(kernel, "conv_transpose2d")
        return when {
            this is Cpu && kernel is Cpu -> {
                val s = storage.convTranspose2d(l, kernel.storage, kernelL, params)
                Cpu(s)
            }
            this is Cuda && kernel is Cuda -> {
                val s = storage.convTranspose2d(l, kernel.storage, kernelL, params) as CudaStorage
                Cuda(s)
            }
            this is Metal && kernel is Metal -> {
                val s = storage.convTranspose2d(l, kernel.storage, kernelL, params) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                kernel.device().location(),
                "conv_transpose2d",
            )
        }
    }

    public fun avgPool2d(
        layout: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.avgPool2d(layout, kernelSize, stride)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.avgPool2d(layout, kernelSize, stride) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.avgPool2d(layout, kernelSize, stride) as MetalStorage
                Metal(s)
            }
        }

    public fun maxPool2d(
        layout: Layout,
        kernelSize: Pair<Int, Int>,
        stride: Pair<Int, Int>,
    ): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.maxPool2d(layout, kernelSize, stride)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.maxPool2d(layout, kernelSize, stride) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.maxPool2d(layout, kernelSize, stride) as MetalStorage
                Metal(s)
            }
        }

    public fun upsampleNearest1d(layout: Layout, sz: Int): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.upsampleNearest1d(layout, sz)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.upsampleNearest1d(layout, sz) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.upsampleNearest1d(layout, sz) as MetalStorage
                Metal(s)
            }
        }

    public fun upsampleNearest2d(layout: Layout, h: Int, w: Int): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.upsampleNearest2d(layout, h, w)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.upsampleNearest2d(layout, h, w) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.upsampleNearest2d(layout, h, w) as MetalStorage
                Metal(s)
            }
        }

    public fun upsampleBilinear2d(
        layout: Layout,
        h: Int,
        w: Int,
        alignCorners: Boolean,
        scaleH: Double? = null,
        scaleW: Double? = null,
    ): Storage =
        when (this) {
            is Cpu -> {
                val s = storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW)
                Cpu(s)
            }
            is Cuda -> {
                val s = storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW) as CudaStorage
                Cuda(s)
            }
            is Metal -> {
                val s = storage.upsampleBilinear2d(layout, h, w, alignCorners, scaleH, scaleW) as MetalStorage
                Metal(s)
            }
        }

    public fun whereCond(
        layout: Layout,
        t: Storage,
        layoutT: Layout,
        f: Storage,
        layoutF: Layout,
    ): Storage {
        sameDevice(t, "where")
        sameDevice(f, "where")
        t.sameDtype(f, "where")
        return when {
            this is Cpu && t is Cpu && f is Cpu -> {
                val s = storage.whereCond(layout, t.storage, layoutT, f.storage, layoutF)
                Cpu(s)
            }
            this is Cuda && t is Cuda && f is Cuda -> {
                val s = storage.whereCond(layout, t.storage, layoutT, f.storage, layoutF) as CudaStorage
                Cuda(s)
            }
            this is Metal && t is Metal && f is Metal -> {
                val s = storage.whereCond(layout, t.storage, layoutT, f.storage, layoutF) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                t.device().location(),
                f.device().location(),
                "where",
            )
        }
    }

    public fun gather(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        d: Int,
    ): Storage {
        sameDevice(indexes, "index-add")
        return when {
            this is Cpu && indexes is Cpu -> {
                val s = storage.gather(indexes.storage, l, indexesL, d)
                Cpu(s)
            }
            this is Cuda && indexes is Cuda -> {
                val s = storage.gather(l, indexes.storage, indexesL, d) as CudaStorage
                Cuda(s)
            }
            this is Metal && indexes is Metal -> {
                val s = storage.gather(l, indexes.storage, indexesL, d) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun scatterSet(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        source: Storage,
        sourceL: Layout,
        d: Int,
    ) {
        sameDevice(indexes, "scatter-set")
        sameDevice(source, "scatter-set")
        when {
            this is Cpu && indexes is Cpu && source is Cpu -> {
                storage.scatterSet(l, indexes.storage, indexesL, source.storage, sourceL, d)
            }
            this is Cuda && indexes is Cuda && source is Cuda -> {
                storage.scatterSet(l, indexes.storage, indexesL, source.storage, sourceL, d)
            }
            this is Metal && indexes is Metal && source is Metal -> {
                storage.scatterSet(l, indexes.storage, indexesL, source.storage, sourceL, d)
            }
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun scatterAdd(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        source: Storage,
        sourceL: Layout,
        d: Int,
    ) {
        sameDevice(indexes, "scatter-add")
        sameDevice(source, "scatter-add")
        when {
            this is Cpu && indexes is Cpu && source is Cpu -> {
                storage.scatterAddSet(l, indexes.storage, indexesL, source.storage, sourceL, d)
            }
            this is Cuda && indexes is Cuda && source is Cuda -> {
                storage.scatterAddSet(l, indexes.storage, indexesL, source.storage, sourceL, d)
            }
            this is Metal && indexes is Metal && source is Metal -> {
                storage.scatterAddSet(l, indexes.storage, indexesL, source.storage, sourceL, d)
            }
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun indexAdd(
        l: Layout,
        indexes: Storage,
        indexesL: Layout,
        source: Storage,
        sourceL: Layout,
        d: Int,
    ): Storage {
        sameDevice(indexes, "index-add")
        sameDevice(source, "index-add")
        return when {
            this is Cpu && indexes is Cpu && source is Cpu -> {
                val s = storage.indexAdd(l, indexes.storage, indexesL, source.storage, sourceL, d)
                Cpu(s)
            }
            this is Cuda && indexes is Cuda && source is Cuda -> {
                val s = storage.indexAdd(l, indexes.storage, indexesL, source.storage, sourceL, d) as CudaStorage
                Cuda(s)
            }
            this is Metal && indexes is Metal && source is Metal -> {
                val s = storage.indexAdd(l, indexes.storage, indexesL, source.storage, sourceL, d) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.Msg("unreachable")
        }
    }

    public fun indexSelect(
        rhs: Storage,
        lhsL: Layout,
        rhsL: Layout,
        d: Int,
    ): Storage {
        sameDevice(rhs, "index-select")
        return when {
            this is Cpu && rhs is Cpu -> {
                val s = storage.indexSelect(rhs.storage, lhsL, rhsL, d)
                Cpu(s)
            }
            this is Cuda && rhs is Cuda -> {
                val s = storage.indexSelect(rhs.storage, lhsL, rhsL, d) as CudaStorage
                Cuda(s)
            }
            this is Metal && rhs is Metal -> {
                val s = storage.indexSelect(rhs.storage, lhsL, rhsL, d) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                rhs.device().location(),
                "index-select",
            )
        }
    }

    public fun matmul(
        rhs: Storage,
        bmnk: List<Int>,
        lhsLayout: Layout,
        rhsLayout: Layout,
    ): Storage {
        sameDevice(rhs, "matmul")
        sameDtype(rhs, "matmul")
        return when {
            this is Cpu && rhs is Cpu -> {
                val s = storage.matmul(rhs.storage, bmnk, lhsLayout, rhsLayout)
                Cpu(s)
            }
            this is Cuda && rhs is Cuda -> {
                val s = storage.matmul(rhs.storage, bmnk, lhsLayout, rhsLayout) as CudaStorage
                Cuda(s)
            }
            this is Metal && rhs is Metal -> {
                val s = storage.matmul(rhs.storage, bmnk, lhsLayout, rhsLayout) as MetalStorage
                Metal(s)
            }
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                rhs.device().location(),
                "matmul",
            )
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    public fun copyStridedSrc(
        dst: Storage,
        dstOffset: Int,
        srcL: Layout,
    ) {
        when {
            this is Cpu && dst is Cpu -> storage.copyStridedSrc(dst.storage, dstOffset, srcL)
            this is Cuda && dst is Cuda -> storage.copyStridedSrc(dst.storage, dstOffset, srcL)
            this is Metal && dst is Metal -> storage.copyStridedSrc(dst.storage, dstOffset, srcL)
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                dst.device().location(),
                "copy",
            )
        }
    }

    public fun copy2d(
        dst: Storage,
        d1: Int,
        d2: Int,
        srcS: Int,
        dstS: Int,
        srcO: Int,
        dstO: Int,
    ) {
        when {
            this is Cpu && dst is Cpu -> storage.copy2d(dst.storage, d1, d2, srcS, dstS, srcO, dstO)
            this is Cuda && dst is Cuda -> storage.copy2d(dst.storage, d1, d2, srcS, dstS, srcO, dstO)
            this is Metal && dst is Metal -> storage.copy2d(dst.storage, d1, d2, srcS, dstS, srcO, dstO)
            else -> throw CandleException.DeviceMismatchBinaryOp(
                device().location(),
                dst.device().location(),
                "copy2d",
            )
        }
    }

    public fun toCpuStorage(): CpuStorage =
        when (this) {
            is Cpu -> storage
            is Cuda -> storage.toCpuStorage()
            is Metal -> storage.toCpuStorage()
        }
}
