package io.github.kotlinmania.candle

import io.github.kotlinmania.serdejson.JsonParser
import io.github.kotlinmania.serdejson.Value

internal class SafetensorHeader(
    val dataStart: Int,
    val tensors: List<SafetensorHeaderEntry>,
)

internal class SafetensorHeaderEntry(
    val name: String,
    val dtype: String,
    val shape: IntArray,
    val start: Int,
    val end: Int,
) {
    fun validateSize(bitsPerElement: Int) {
        var elements = 1L
        for (dimension in shape) {
            require(dimension == 0 || elements <= Long.MAX_VALUE / dimension) {
                "safetensors shape overflows for tensor $name"
            }
            elements *= dimension
        }
        require(elements <= Long.MAX_VALUE / bitsPerElement) {
            "safetensors tensor size overflows for tensor $name"
        }
        val bits = elements * bitsPerElement
        require(bits % 8 == 0L) { "safetensors tensor $name is not byte aligned" }
        require(bits / 8 == (end - start).toLong()) {
            "safetensors shape and data length disagree for tensor $name"
        }
    }
}

internal fun readSafetensorHeader(data: ByteArray): SafetensorHeader {
    require(data.size >= 8) { "safetensors buffer too small" }
    var headerLength = 0uL
    for (index in 0 until 8) {
        headerLength = headerLength or ((data[index].toULong() and 0xffuL) shl (index * 8))
    }
    require(headerLength <= 100_000_000uL && headerLength <= (data.size - 8).toULong()) {
        "invalid safetensors header length"
    }
    val dataStart = 8 + headerLength.toInt()
    val parser = JsonParser(data.decodeToString(8, dataStart, throwOnInvalidSequence = true))
    val parsed =
        parser.parseValue().getOrElse { error ->
            throw IllegalArgumentException("invalid safetensors JSON header: $error")
        }
    require(!parser.hasMore()) { "trailing data in safetensors JSON header" }
    val objectValues = requireNotNull(parsed.asObject()) { "safetensors header must be an object" }
    val dataLength = data.size - dataStart
    val tensors = ArrayList<SafetensorHeaderEntry>()
    for ((name, value) in objectValues.entries()) {
        if (name == "__metadata__") {
            if (value != Value.Null) {
                val metadata = requireNotNull(value.asObject()) { "safetensors metadata must be an object" }
                require(metadata.entries().all { it.second.asStr() != null }) {
                    "safetensors metadata values must be strings"
                }
            }
            continue
        }
        val fields = requireNotNull(value.asObject()) { "safetensors tensor $name must be an object" }
        val dtype = requireNotNull(fields["dtype"]?.asStr()) { "missing or invalid dtype for tensor $name" }
        val dimensions = requireNotNull(fields["shape"]?.asArray()) { "missing or invalid shape for tensor $name" }
        val shape = IntArray(dimensions.size) { dimensions[it].nonnegativeInt("shape for tensor $name", Int.MAX_VALUE) }
        val offsets =
            requireNotNull(fields["data_offsets"]?.asArray()) { "missing or invalid offsets for tensor $name" }
        require(offsets.size == 2) { "safetensors tensor $name must have two offsets" }
        val start = offsets[0].nonnegativeInt("start offset for tensor $name", dataLength)
        val end = offsets[1].nonnegativeInt("end offset for tensor $name", dataLength)
        require(end >= start) { "reversed safetensors offsets for tensor $name" }
        tensors.add(SafetensorHeaderEntry(name, dtype, shape, start, end))
    }
    tensors.sortWith(compareBy<SafetensorHeaderEntry> { it.start }.thenBy { it.end })
    var nextOffset = 0
    for (tensor in tensors) {
        require(tensor.start == nextOffset) { "non-contiguous safetensors offsets for tensor ${tensor.name}" }
        nextOffset = tensor.end
    }
    require(nextOffset == dataLength) { "safetensors metadata does not cover the complete data buffer" }
    return SafetensorHeader(dataStart, tensors)
}

private fun Value.nonnegativeInt(
    field: String,
    maximum: Int,
): Int {
    val number = requireNotNull(asU64()) { "invalid safetensors $field" }
    require(number <= maximum.toULong()) { "safetensors $field exceeds the supported range" }
    return number.toInt()
}
