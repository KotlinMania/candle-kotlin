package io.github.kotlinmania.candle

private fun QTensor.asRef(): QTensor = this

sealed class QMatMul : Module {
    private val self: QMatMul get() = this

    data class QTensorVariant(val qtensor: QTensor) : QMatMul() {
        val t: QTensor get() = qtensor
    }
    data class TensorVariant(val tensor: Tensor) : QMatMul() {
        val t: Tensor get() = tensor
        val w: Tensor get() = tensor
    }
    data class TensorF16Variant(val tensor: Tensor) : QMatMul() {
        val t: Tensor get() = tensor
        val w: Tensor get() = tensor
    }

    fun dequantizeF16(): CandleResult<Tensor> {
        return when (self) {
            is QTensorVariant -> {
                val qt = (this as QTensorVariant).t
                qt.dequantizeF16(qt.device())
            }
            is TensorVariant -> CandleResult.Ok((this as TensorVariant).t.toDtype(DType.F16))
            is TensorF16Variant -> CandleResult.Ok((this as TensorF16Variant).t.clone())
        }
    }

    fun forwardViaF16(xs: Tensor): CandleResult<Tensor> {
        val w = self.dequantizeF16().unwrapOrElse { return CandleResult.Err(it) }
        val inDtype = xs.dtype()
        val wT = when (xs.dims().size) {
            4 -> QuantizedHelper.broadcastLeftT2(w, xs.dims()[0], xs.dims()[1]).unwrapOrElse { return CandleResult.Err(it) }
            3 -> QuantizedHelper.broadcastLeftT1(w, xs.dims()[0]).unwrapOrElse { return CandleResult.Err(it) }
            else -> w.t()
        }
        return QuantizedHelper.matmulViaF16(xs, wT, inDtype)
    }

    fun indexedMoeForward(x: Tensor, ids: Tensor): CandleResult<Tensor> {
        return when (self) {
            is QTensorVariant -> (this as QTensorVariant).t.indexedMoeForward(x, ids)
            else -> {
                throw IllegalStateException("Not implemented!")
            }
        }
    }

    override fun forward(xs: Tensor): Tensor {
        return when (self) {
            is QTensorVariant -> xs.applyOp1NoBwd((this as QTensorVariant).t.asRef())
            is TensorVariant -> {
                val w = when (xs.dims().size) {
                    4 -> QuantizedHelper.broadcastLeftT2((this as TensorVariant).w, xs.dims()[0], xs.dims()[1]).unwrapOrElse { throw it }
                    3 -> QuantizedHelper.broadcastLeftT1((this as TensorVariant).w, xs.dims()[0]).unwrapOrElse { throw it }
                    else -> (this as TensorVariant).w.t()
                }
                xs.matmul(w)
            }
            is TensorF16Variant -> {
                val inDtype = xs.dtype()
                val w = when (xs.dims().size) {
                    4 -> QuantizedHelper.broadcastLeftT2((this as TensorF16Variant).w, xs.dims()[0], xs.dims()[1]).unwrapOrElse { throw it }
                    3 -> QuantizedHelper.broadcastLeftT1((this as TensorF16Variant).w, xs.dims()[0]).unwrapOrElse { throw it }
                    else -> (this as TensorF16Variant).w.t()
                }
                QuantizedHelper.matmulViaF16(xs, w, inDtype).unwrapOrElse { throw it }
            }
        }
    }

    companion object {
        fun fromArc(qtensor: QTensor): CandleResult<QMatMul> {
            val dequantize = when (qtensor.dtype()) {
                GgmlDType.F32, GgmlDType.F16, GgmlDType.BF16 -> true
                else -> QuantizedHelper.dequantizeAllFlag()
            }
            val t = if (dequantize) {
                val tensor = qtensor.dequantize(qtensor.device()).unwrapOrElse { return CandleResult.Err(it) }
                TensorVariant(tensor)
            } else if (QuantizedHelper.dequantizeAllF16Flag()) {
                val tensor = qtensor.dequantizeF16(qtensor.device()).unwrapOrElse { return CandleResult.Err(it) }
                TensorF16Variant(tensor)
            } else {
                QTensorVariant(qtensor)
            }
            return CandleResult.Ok(t)
        }

        fun fromQtensor(qtensor: QTensor): CandleResult<QMatMul> {
            return fromArc(qtensor)
        }
    }
}
