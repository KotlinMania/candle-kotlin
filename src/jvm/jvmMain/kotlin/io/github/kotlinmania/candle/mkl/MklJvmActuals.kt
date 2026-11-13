@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

private object MklJniBridge {
    init {
        loadNativeLibrary()
    }

    private fun loadNativeLibrary() {
        val osRaw = System.getProperty("os.name", "").lowercase()
        val archRaw = System.getProperty("os.arch", "").lowercase()
        val hostPlatform =
            when {
                osRaw.contains("win") -> "windows"
                osRaw.contains("mac") || osRaw.contains("darwin") -> "macos"
                else -> "linux"
            }
        val hostArch =
            when {
                archRaw == "aarch64" || archRaw == "arm64" -> "aarch64"
                archRaw == "amd64" || archRaw == "x86_64" -> "x86_64"
                else -> archRaw
            }
        val libFileName =
            when (hostPlatform) {
                "windows" -> "candle_jni.dll"
                "macos" -> "libcandle_jni.dylib"
                else -> "libcandle_jni.so"
            }
        val pathSep = System.getProperty("path.separator", ":")
        val libPathProp = System.getProperty("java.library.path", "")
        for (dirEntry in libPathProp.split(pathSep)) {
            if (dirEntry.isEmpty()) continue
            val candidate = java.io.File(dirEntry, libFileName)
            if (candidate.isFile) {
                System.load(candidate.absolutePath)
                return
            }
        }
        val resourcePath = "io/github/kotlinmania/candle/native/$hostPlatform-$hostArch/$libFileName"
        val stream = MklJniBridge::class.java.classLoader?.getResourceAsStream(resourcePath)
        if (stream != null) {
            val suffix = "." + libFileName.substringAfterLast('.')
            val extracted =
                java.nio.file.Files
                    .createTempFile("candle_jni-", suffix)
                    .toFile()
            extracted.deleteOnExit()
            stream.use { input ->
                extracted.outputStream().use { output ->
                    input.copyTo(output)
                }
            }
            System.load(extracted.absolutePath)
            return
        }
        System.loadLibrary("candle_jni")
    }

    @JvmStatic
    external fun candleMklVsTanh(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdTanh(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsExp(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdExp(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsLn(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdLn(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsSin(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdSin(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsCos(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdCos(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsSqrt(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVdSqrt(n: Int, a: Long, y: Long)

    @JvmStatic
    external fun candleMklVsAdd(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdAdd(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsSub(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdSub(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsMul(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdMul(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsDiv(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdDiv(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsFmax(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdFmax(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVsFmin(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklVdFmin(n: Int, a: Long, b: Long, y: Long)

    @JvmStatic
    external fun candleMklSgemm(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleMklDgemm(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleMklHgemm(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)
}

public actual fun vsTanh(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsTanh(n, a.rawAddress, y.rawAddress)
}

public actual fun vdTanh(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdTanh(n, a.rawAddress, y.rawAddress)
}

public actual fun vsExp(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsExp(n, a.rawAddress, y.rawAddress)
}

public actual fun vdExp(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdExp(n, a.rawAddress, y.rawAddress)
}

public actual fun vsLn(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsLn(n, a.rawAddress, y.rawAddress)
}

public actual fun vdLn(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdLn(n, a.rawAddress, y.rawAddress)
}

public actual fun vsSin(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsSin(n, a.rawAddress, y.rawAddress)
}

public actual fun vdSin(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdSin(n, a.rawAddress, y.rawAddress)
}

public actual fun vsCos(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsCos(n, a.rawAddress, y.rawAddress)
}

public actual fun vdCos(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdCos(n, a.rawAddress, y.rawAddress)
}

public actual fun vsSqrt(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsSqrt(n, a.rawAddress, y.rawAddress)
}

public actual fun vdSqrt(n: CInt, a: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdSqrt(n, a.rawAddress, y.rawAddress)
}

public actual fun vsAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsAdd(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdAdd(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsSub(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdSub(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsMul(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdMul(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsDiv(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdDiv(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsFmax(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdFmax(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vsFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVsFmin(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun vdFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr) {
    MklJniBridge.candleMklVdFmin(n, a.rawAddress, b.rawAddress, y.rawAddress)
}

public actual fun sgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    MklJniBridge.candleMklSgemm(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun dgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    MklJniBridge.candleMklDgemm(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun hgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr) {
    MklJniBridge.candleMklHgemm(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}
