// port-lint: tests npy.rs
package io.github.kotlinmania.candle.npy

import io.github.kotlinmania.candle.dtype.DType

// Numpy support for tensors.
//
// The spec for the npy format can be found in
// [npy-format](https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html).
// The functions from this module can be used to read tensors from npy/npz files
// or write tensors to these files. A npy file contains a single tensor (unnamed)
// whereas a npz file can contain multiple named tensors. npz files are also compressed.
//
// These two formats are easy to use in Python using the numpy library.
//
// ```kotlin
// import numpy as np x = np.arange(10) np.save("test.npy", x) x = np.load("test.npy") values = { "x": x, "xPlusOne": x + 1 } np.savez("test.npz", * * values) values = np.loadz("test.npz")
// ```

@kotlin.test.Test
public fun parse() {
    val h = "{'descr': '<f8', 'fortran_order': False, 'shape': (128,), }"
    check(Header.parse(h).unwrap() == Header(descr = DType.F64, fortranOrder = false, shape = mutableListOf(128)))
    val h1 = "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128), }"
    val h2 = Header.parse(h1).unwrap()
    check(h2 == Header(descr = DType.F32, fortranOrder = true, shape = mutableListOf(256, 1, 128)))
    check(h2.toString().unwrap() == "{'descr': '<f4', 'fortran_order': True, 'shape': (256,1,128,), }")
    val h3 = Header(descr = DType.U32, fortranOrder = false, shape = mutableListOf())
    check(h3.toString().unwrap() == "{'descr': '<u4', 'fortran_order': False, 'shape': (), }")
}
