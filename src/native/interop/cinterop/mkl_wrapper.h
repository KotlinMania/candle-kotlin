#ifndef CANDLE_MKL_MKL_WRAPPER_H
#define CANDLE_MKL_MKL_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void candleMklVsTanh(int32_t n, void* a, void* y);
void candleMklVdTanh(int32_t n, void* a, void* y);
void candleMklVsExp(int32_t n, void* a, void* y);
void candleMklVdExp(int32_t n, void* a, void* y);
void candleMklVsLn(int32_t n, void* a, void* y);
void candleMklVdLn(int32_t n, void* a, void* y);
void candleMklVsSin(int32_t n, void* a, void* y);
void candleMklVdSin(int32_t n, void* a, void* y);
void candleMklVsCos(int32_t n, void* a, void* y);
void candleMklVdCos(int32_t n, void* a, void* y);
void candleMklVsSqrt(int32_t n, void* a, void* y);
void candleMklVdSqrt(int32_t n, void* a, void* y);
void candleMklVsAdd(int32_t n, void* a, void* b, void* y);
void candleMklVdAdd(int32_t n, void* a, void* b, void* y);
void candleMklVsSub(int32_t n, void* a, void* b, void* y);
void candleMklVdSub(int32_t n, void* a, void* b, void* y);
void candleMklVsMul(int32_t n, void* a, void* b, void* y);
void candleMklVdMul(int32_t n, void* a, void* b, void* y);
void candleMklVsDiv(int32_t n, void* a, void* b, void* y);
void candleMklVdDiv(int32_t n, void* a, void* b, void* y);
void candleMklVsFmax(int32_t n, void* a, void* b, void* y);
void candleMklVdFmax(int32_t n, void* a, void* b, void* y);
void candleMklVsFmin(int32_t n, void* a, void* b, void* y);
void candleMklVdFmin(int32_t n, void* a, void* b, void* y);
void candleMklSgemm(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candleMklDgemm(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candleMklHgemm(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);

#ifdef __cplusplus
}
#endif

#endif /* CANDLE_MKL_MKL_WRAPPER_H */
