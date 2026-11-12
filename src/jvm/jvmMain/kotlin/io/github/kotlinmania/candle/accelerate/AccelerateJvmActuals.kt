@file:OptIn(kotlin.ExperimentalUnsignedTypes::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.libc.CChar
import io.github.kotlinmania.libc.CDouble
import io.github.kotlinmania.libc.CFloat
import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

private object AccelerateJniBridge {
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
        val stream = AccelerateJniBridge::class.java.classLoader?.getResourceAsStream(resourcePath)
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
    external fun candleAccelerateSgemmFfi(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleAccelerateDgemmFfi(transa: Long, transb: Long, m: Long, n: Long, k: Long, alpha: Long, a: Long, lda: Long, b: Long, ldb: Long, beta: Long, c: Long, ldc: Long)

    @JvmStatic
    external fun candleAccelerateVvexpf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvexp(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsqrtf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsqrt(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsinf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvsin(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvcosf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvcos(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvlogf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvlog(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvtanhf(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVvtanh(dst: Long, src: Long, len: Long)

    @JvmStatic
    external fun candleAccelerateVDspVaddD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVadd(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVsubD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVsub(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmulD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmul(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVdivD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVdiv(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVminD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmin(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmaxD(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)

    @JvmStatic
    external fun candleAccelerateVDspVmax(arg0: Long, arg1: Long, arg2: Long, arg3: Long, arg4: Long, arg5: Long, arg6: Long)
}

public actual fun sgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CFloat>, a: ConstPtr<CFloat>, lda: ConstPtr<CInt>, b: ConstPtr<CFloat>, ldb: ConstPtr<CInt>, beta: ConstPtr<CFloat>, c: MutPtr<CFloat>, ldc: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateSgemmFfi(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun dgemmFfi(transa: ConstPtr<CChar>, transb: ConstPtr<CChar>, m: ConstPtr<CInt>, n: ConstPtr<CInt>, k: ConstPtr<CInt>, alpha: ConstPtr<CDouble>, a: ConstPtr<CDouble>, lda: ConstPtr<CInt>, b: ConstPtr<CDouble>, ldb: ConstPtr<CInt>, beta: ConstPtr<CDouble>, c: MutPtr<CDouble>, ldc: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateDgemmFfi(transa.rawAddress, transb.rawAddress, m.rawAddress, n.rawAddress, k.rawAddress, alpha.rawAddress, a.rawAddress, lda.rawAddress, b.rawAddress, ldb.rawAddress, beta.rawAddress, c.rawAddress, ldc.rawAddress)
}

public actual fun vvexpf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvexpf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvexp(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvexp(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsqrtf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvsqrtf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsqrt(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvsqrt(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsinf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvsinf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvsin(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvsin(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvcosf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvcosf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvcos(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvcos(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvlogf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvlogf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvlog(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvlog(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvtanhf(dst: MutPtr<CFloat>, src: ConstPtr<CFloat>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvtanhf(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vvtanh(dst: MutPtr<CDouble>, src: ConstPtr<CDouble>, len: ConstPtr<CInt>) {
    AccelerateJniBridge.candleAccelerateVvtanh(dst.rawAddress, src.rawAddress, len.rawAddress)
}

public actual fun vDspVaddD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVaddD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVadd(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVadd(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVsubD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVsubD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVsub(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVsub(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmulD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmulD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmul(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmul(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVdivD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVdivD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVdiv(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVdiv(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVminD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVminD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmin(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmin(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmaxD(arg0: ConstPtr<CDouble>, arg1: CLong, arg2: ConstPtr<CDouble>, arg3: CLong, arg4: MutPtr<CDouble>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmaxD(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}

public actual fun vDspVmax(arg0: ConstPtr<CFloat>, arg1: CLong, arg2: ConstPtr<CFloat>, arg3: CLong, arg4: MutPtr<CFloat>, arg5: CLong, arg6: CUlong) {
    AccelerateJniBridge.candleAccelerateVDspVmax(arg0.rawAddress, arg1, arg2.rawAddress, arg3, arg4.rawAddress, arg5, arg6.toLong())
}
