#include <migraph/gpu/rocblas.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {

rocblas_handle_ptr create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    return rocblas_handle_ptr{handle};
}

rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s)
{
    rocblas_handle_ptr rb = create_rocblas_handle_ptr();
    rocblas_set_stream(rb.get(), s);
    return rb;
}

} // namespace gpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph
