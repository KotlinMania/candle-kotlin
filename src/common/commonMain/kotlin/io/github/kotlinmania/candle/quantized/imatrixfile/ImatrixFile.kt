// port-lint: source quantized/imatrix_file.rs
package io.github.kotlinmania.candle.quantized.imatrixfile

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail

public fun <P : AsRef<Path>> loadImatrix(fname: P): Result<Map<String, List<Float>>> {
    val allData = HashMap.new()
    var file =
        File.open(fname).mapErr({ e ->
            Error.msg("Failed to open ${fname.asRef().display()}: $e")
        }).getOrElse { return Result.failure(it) }
    val buffer = mutableListOf()
    file.readToEnd(buffer).mapErr({ e ->
        Error.msg("Failed to read file ${fname.asRef().display()}: $e")
    }).getOrElse { return Result.failure(it) }
    val cursor = Cursor.new(buffer)
    val nEntries = (cursor.readI32<LittleEndian>().mapErr({ e -> Error.msg("Failed to read number of entries: $e") }).getOrElse { return Result.failure(it) }).toInt()
    if (nEntries < 1) {
        bail("No data in file {}", fname.asRef().display())
    }
    for (i in 0 until nEntries) {
        // Read length of the name
        val len =
            (cursor.readI32<LittleEndian>().mapErr({ e ->
                Error.msg("Failed to read name length for entry ${i + 1}: $e")
            }).getOrElse { return Result.failure(it) }).toInt()
        // Read the name
        val nameBuf = mutableListOf(0u.toUByte(), len)
        cursor.readExact(nameBuf).mapErr({ e ->
            Error.msg("Failed to read name for entry ${i + 1}: $e")
        }).getOrElse { return Result.failure(it) }
        val name =
            String.fromUtf8(nameBuf).mapErr({ e ->
                Error.msg("Invalid UTF-8 name for entry ${i + 1}: $e")
            }).getOrElse { return Result.failure(it) }
        // Read ncall and nval
        val ncall =
            (cursor.readI32<LittleEndian>().mapErr({ e ->
                Error.msg("Failed to read ncall for entry ${i + 1}: $e")
            }).getOrElse { return Result.failure(it) }).toInt()
        val nval =
            (cursor.readI32<LittleEndian>().mapErr({ e ->
                Error.msg("Failed to read nval for entry ${i + 1}: $e")
            }).getOrElse { return Result.failure(it) }).toInt()
        if (nval < 1) {
            bail("Invalid nval for entry {}: {}", i + 1, nval)
        }
        val data = ArrayList(nval)
        for (unused in 0 until nval) {
            val v = cursor.readF32<LittleEndian>().unwrap()
            if (ncall == 0) {
                data.add(v)
            } else {
                data.add(v / ncall.toFloat())
            }
        }
        allData.insert(name, data)
    }
    return Result.success(allData)
}
