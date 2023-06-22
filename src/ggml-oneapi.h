#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

void   ggml_init_oneapi(void);

void   ggml_oneapi_mul(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
bool   ggml_oneapi_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
size_t ggml_oneapi_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst);
void   ggml_oneapi_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, void * wdata, size_t wsize);

// TODO: export these with GGML_API
void * ggml_oneapi_host_malloc(size_t size);
void   ggml_oneapi_host_free(void * ptr);

void ggml_oneapi_transform_tensor(struct ggml_tensor * tensor);
void ggml_oneapi_load_data(const char * fname, struct ggml_tensor * tensors, size_t offset);

#ifdef  __cplusplus
}
#endif
