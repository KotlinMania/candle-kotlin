// port-lint: source quantized/imatrix_file.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result

private val inMemoryImatrixFiles: MutableMap<String, ByteArray> = LinkedHashMap()

public fun loadImatrix(fname: String): Result<HashMap<String, MutableList<Float>>> {
    val allData = HashMap<String, MutableList<Float>>()

    val file = inMemoryImatrixFiles[fname]
        ?: throw CandleError.msg("Failed to open $fname: file not found")
    val buffer = file.copyOf()

    val cursor = NpyByteReader(buffer)

    val nEntries = cursor
        .readI32()
        .unwrap()

    if (nEntries < 1) {
        bail("No data in file $fname")
    }

    for (i in 0 until nEntries) {
        // Read length of the name
        val len = cursor.readI32().unwrap()

        // Read the name
        val nameBuf = ByteArray(len)
        cursor.readExact(nameBuf).unwrap()
        val name = nameBuf.decodeToString()

        // Read ncall and nval
        val ncall = cursor.readI32().unwrap()

        val nval = cursor.readI32().unwrap()

        if (nval < 1) {
            bail("Invalid nval for entry ${i + 1}: $nval")
        }

        val data = ArrayList<Float>(nval)
        for (j in 0 until nval) {
            val v = cursor.readF32().unwrap()
            if (ncall == 0) {
                data.add(v)
            } else {
                data.add(v / ncall.toFloat())
            }
        }
        allData[name] = data
    }

    return Ok(allData)
}
