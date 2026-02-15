#ifndef MMCV_NVCC_GLIBC_COMPAT_H_
#define MMCV_NVCC_GLIBC_COMPAT_H_

#ifdef __CUDACC__
// Avoid glibc C23 IEC60559 rsqrt/rsqrtf declarations that conflict with
// CUDA's math declarations under NVCC + C++17.
#include <features.h>
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT
#define __GLIBC_USE_IEC_60559_FUNCS_EXT 0
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#endif

#endif  // MMCV_NVCC_GLIBC_COMPAT_H_
