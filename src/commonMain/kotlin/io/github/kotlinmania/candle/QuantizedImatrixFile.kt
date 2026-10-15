// port-lint: source quantized/imatrix_file.rs
package io.github.kotlinmania.candle

private val inMemoryImatrixFiles: MutableMap<String, ByteArray> = LinkedHashMap()

fun loadImatrix(fname: String): CandleResult<Map<String, FloatArray>> {
    val allData = HashMap<String, FloatArray>()

    val buffer = inMemoryImatrixFiles[fname]
        ?: return CandleResult.Err(CandleError.Msg("Failed to open $fname: file not found"))

    val cursor = NpyByteReader(buffer)
    val readI32Le = { label: String ->
        val b = ByteArray(4)
        when (val res = cursor.readExact(b)) {
            is CandleResult.Ok -> {
                val v = (b[0].toInt() and 0xFF) or
                    ((b[1].toInt() and 0xFF) shl 8) or
                    ((b[2].toInt() and 0xFF) shl 16) or
                    ((b[3].toInt() and 0xFF) shl 24)
                CandleResult.Ok(v)
            }
            is CandleResult.Err -> CandleResult.Err(CandleError.Msg("$label: ${res.error}"))
        }
    }

    val nEntries = when (val res = readI32Le("Failed to read number of entries")) {
        is CandleResult.Ok -> res.value
        is CandleResult.Err -> return CandleResult.Err(res.error)
    }

    if (nEntries < 1) {
        return CandleResult.Err(CandleError.Msg("No data in file $fname").bt())
    }

    for (i in 0 until nEntries) {
        // Read length of the name
        val len = when (val res = readI32Le("Failed to read name length for entry ${i + 1}")) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }

        // Read the name
        val nameBuf = ByteArray(len)
        when (val res = cursor.readExact(nameBuf)) {
            is CandleResult.Ok -> {}
            is CandleResult.Err -> return CandleResult.Err(
                CandleError.Msg("Failed to read name for entry ${i + 1}: ${res.error}")
            )
        }
        val name = nameBuf.decodeToString()

        // Read ncall and nval
        val ncall = when (val res = readI32Le("Failed to read ncall for entry ${i + 1}")) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }

        val nval = when (val res = readI32Le("Failed to read nval for entry ${i + 1}")) {
            is CandleResult.Ok -> res.value
            is CandleResult.Err -> return CandleResult.Err(res.error)
        }

        if (nval < 1) {
            return CandleResult.Err(CandleError.Msg("Invalid nval for entry ${i + 1}: $nval").bt())
        }

        val data = FloatArray(nval)
        for (j in 0 until nval) {
            val bits = readI32Le("Failed to read f32 for entry ${i + 1}").unwrap()
            val v = Float.fromBits(bits)
            data[j] = if (ncall == 0) {
                v
            } else {
                v / ncall.toFloat()
            }
        }
        allData[name] = data
    }

    return CandleResult.Ok(allData)
}
