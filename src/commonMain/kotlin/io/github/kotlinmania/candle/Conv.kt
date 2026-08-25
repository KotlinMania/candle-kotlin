// port-lint: source candle-core/src/conv.rs
package io.github.kotlinmania.candle

public data class ParamsConv1D(
    val bSize: Int,
    val lIn: Int,
    val cOut: Int,
    val cIn: Int,
    val kSize: Int,
    val padding: Int,
    val stride: Int,
    val dilation: Int,
) {
    public fun lOut(): Int =
        (lIn + 2 * padding - dilation * (kSize - 1) - 1) / stride + 1

    public fun outDims(): List<Int> = listOf(bSize, cOut, lOut())
}

public data class ParamsConvTranspose1D(
    val bSize: Int,
    val lIn: Int,
    val cOut: Int,
    val cIn: Int,
    val kSize: Int,
    val padding: Int,
    val outputPadding: Int,
    val stride: Int,
    val dilation: Int,
) {
    public fun lOut(): Int =
        (lIn - 1) * stride - 2 * padding + dilation * (kSize - 1) + outputPadding + 1

    public fun outDims(): List<Int> = listOf(bSize, cOut, lOut())
}

public data class ParamsConv2D(
    val bSize: Int,
    val iH: Int,
    val iW: Int,
    val kH: Int,
    val kW: Int,
    val cOut: Int,
    val cIn: Int,
    val padding: Int,
    val stride: Int,
    val dilation: Int,
) {
    public fun outH(): Int =
        (iH + 2 * padding - dilation * (kH - 1) - 1) / stride + 1

    public fun outW(): Int =
        (iW + 2 * padding - dilation * (kW - 1) - 1) / stride + 1

    public fun outDims(): List<Int> = listOf(bSize, cOut, outH(), outW())
}

public data class ParamsConvTranspose2D(
    val bSize: Int,
    val iH: Int,
    val iW: Int,
    val kH: Int,
    val kW: Int,
    val cOut: Int,
    val cIn: Int,
    val padding: Int,
    val outputPadding: Int,
    val stride: Int,
    val dilation: Int,
) {
    public fun outH(): Int =
        (iH - 1) * stride - 2 * padding + dilation * (kH - 1) + outputPadding + 1

    public fun outW(): Int =
        (iW - 1) * stride - 2 * padding + dilation * (kW - 1) + outputPadding + 1

    public fun outDims(): List<Int> = listOf(bSize, cOut, outH(), outW())
}
