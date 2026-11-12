// port-lint: source quantized/imatrix_file.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.CandleResult.Ok

private fun Any.asRef(): String = toString()

public fun <P : Any> loadImatrix(fname: P): CandleResult<kotlin.collections.HashMap<String, MutableList<Float>>> {
    val allData = HashMap.new<String, MutableList<Float>>()

    val file =
        File
            .open(fname)
            .mapErr { e ->
                CandleException.msg(
                    format(
                        "Failed to open {}: {}",
                        fname.asRef().display(),
                        e,
                    ),
                )
            }.unwrap()
    val buffer = Vec.new<Byte>()
    file
        .readToEnd(buffer)
        .mapErr { e ->
            CandleException.msg(
                format(
                    "Failed to read file {}: {}",
                    fname.asRef().display(),
                    e,
                ),
            )
        }.unwrap()

    val cursor = Cursor.new(buffer)

    val nEntries =
        cursor
            .readI32<LittleEndian>()
            .mapErr { e -> CandleException.msg(format("Failed to read number of entries: $e")) }
            .unwrap()

    if (nEntries < 1) {
        bail(format("No data in file {}", fname.asRef().display()))
    }

    for (i in 0 until nEntries) {
        // Read length of the name
        val len =
            cursor
                .readI32<LittleEndian>()
                .mapErr { e ->
                    CandleException.msg(
                        format(
                            "Failed to read name length for entry {}: {}",
                            i + 1,
                            e,
                        ),
                    )
                }.unwrap()

        // Read the name
        val nameBuf = ByteArray(len)
        cursor
            .readExact(nameBuf)
            .mapErr { e ->
                CandleException.msg(format("Failed to read name for entry {}: {}", i + 1, e))
            }.unwrap()
        val name =
            String
                .fromUtf8(nameBuf)
                .mapErr { e ->
                    CandleException.msg(format("Invalid UTF-8 name for entry {}: {}", i + 1, e))
                }.unwrap()

        // Read ncall and nval
        val ncall =
            cursor
                .readI32<LittleEndian>()
                .mapErr { e ->
                    CandleException.msg(format("Failed to read ncall for entry {}: {}", i + 1, e))
                }.unwrap()

        val nval =
            cursor
                .readI32<LittleEndian>()
                .mapErr { e ->
                    CandleException.msg(format("Failed to read nval for entry {}: {}", i + 1, e))
                }.unwrap()

        if (nval < 1) {
            bail(format("Invalid nval for entry {}: {}", i + 1, nval))
        }

        val data = Vec.withCapacity<Float>(nval)
        repeat(nval) {
            val v = cursor.readF32<LittleEndian>().unwrap()
            if (ncall == 0) {
                data.push(v)
            } else {
                data.push(v / ncall.toFloat())
            }
        }
        allData.insert(name, data)
    }

    return Ok(kotlin.collections.HashMap(allData))
}
