#ifndef CANDLE_MKL_MKL_WRAPPER_H
#define CANDLE_MKL_MKL_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void candle_mkl_vsTanh(int32_t n, void* a, void* y);
void candle_mkl_vdTanh(int32_t n, void* a, void* y);
void candle_mkl_vsExp(int32_t n, void* a, void* y);
void candle_mkl_vdExp(int32_t n, void* a, void* y);
void candle_mkl_vsLn(int32_t n, void* a, void* y);
void candle_mkl_vdLn(int32_t n, void* a, void* y);
void candle_mkl_vsSin(int32_t n, void* a, void* y);
void candle_mkl_vdSin(int32_t n, void* a, void* y);
void candle_mkl_vsCos(int32_t n, void* a, void* y);
void candle_mkl_vdCos(int32_t n, void* a, void* y);
void candle_mkl_vsSqrt(int32_t n, void* a, void* y);
void candle_mkl_vdSqrt(int32_t n, void* a, void* y);
void candle_mkl_vsAdd(int32_t n, void* a, void* b, void* y);
void candle_mkl_vdAdd(int32_t n, void* a, void* b, void* y);
void candle_mkl_vsSub(int32_t n, void* a, void* b, void* y);
void candle_mkl_vdSub(int32_t n, void* a, void* b, void* y);
void candle_mkl_vsMul(int32_t n, void* a, void* b, void* y);
void candle_mkl_vdMul(int32_t n, void* a, void* b, void* y);
void candle_mkl_vsDiv(int32_t n, void* a, void* b, void* y);
void candle_mkl_vdDiv(int32_t n, void* a, void* b, void* y);
void candle_mkl_vsFmax(int32_t n, void* a, void* b, void* y);
void candle_mkl_vdFmax(int32_t n, void* a, void* b, void* y);
void candle_mkl_vsFmin(int32_t n, void* a, void* b, void* y);
void candle_mkl_vdFmin(int32_t n, void* a, void* b, void* y);
void candle_mkl_sgemm(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candle_mkl_dgemm(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candle_mkl_hgemm(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);

#ifdef __cplusplus
}
#endif

#endif /* CANDLE_MKL_MKL_WRAPPER_H */
