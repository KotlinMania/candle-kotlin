// port-lint: source candle-core/src/variable.rs
package io.github.kotlinmania.candle

public class Var(
    public val tensor: Tensor,
) {
    public fun shape(): Shape = tensor.shape()

    public fun dtype(): DType = tensor.dtype()

    public fun device(): Device = tensor.device()

    public fun asTensor(): Tensor = tensor

    public fun set(src: Tensor) {
        if (src.shape() != tensor.shape()) {
            throw CandleException.ShapeMismatch(src.elemCount(), tensor.shape())
        }
        if (src.dtype() != tensor.dtype()) {
            throw CandleException.DTypeMismatchBinaryOp(tensor.dtype(), src.dtype(), "set")
        }
        val cont = src.contiguous()
        val srcCpu = (cont.storage() as Storage.Cpu).storage
        val dstCpu = (tensor.storage() as Storage.Cpu).storage
        val dstStrided = tensor.layout().stridedIndex()
        var srcIdx = 0
        while (dstStrided.hasNext()) {
            val dIdx = dstStrided.next()
            setStorageVal(dstCpu, dIdx, srcCpu, srcIdx++)
        }
    }

    private fun setStorageVal(dst: CpuStorage, dstIdx: Int, src: CpuStorage, srcIdx: Int) {
        when (dst) {
            is CpuStorage.F32 -> dst.data[dstIdx] = src.getAsFloat(srcIdx)
            is CpuStorage.F64 -> dst.data[dstIdx] = src.getAsDouble(srcIdx)
            is CpuStorage.I32 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt()
            is CpuStorage.I64 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toLong()
            is CpuStorage.U32 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toLong().toUInt()
            is CpuStorage.U8 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt().toUByte()
            is CpuStorage.I16 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt().toShort()
            is CpuStorage.F16 -> dst.data[dstIdx] = Float16Utils.floatToF16Bits(src.getAsFloat(srcIdx))
            is CpuStorage.BF16 -> dst.data[dstIdx] = Float16Utils.floatToBF16Bits(src.getAsFloat(srcIdx))
            is CpuStorage.F8E4M3 -> dst.data[dstIdx] = src.getAsDouble(srcIdx).toInt().toByte()
        }
    }

    public companion object {
        public fun zeros(shape: Shape, dtype: DType, device: Device): Var =
            Var(Tensor.zeros(shape, dtype, device))

        public fun ones(shape: Shape, dtype: DType, device: Device): Var =
            Var(Tensor.ones(shape, dtype, device))

        public fun fromTensor(t: Tensor): Var = Var(t)

        public fun rand(min: Float, max: Float, shape: Shape, device: Device): Var =
            Var(Tensor.rand(min, max, shape, device))

        public fun randn(mean: Float, std: Float, shape: Shape, device: Device): Var =
            Var(Tensor.randn(mean, std, shape, device))
    }
}
