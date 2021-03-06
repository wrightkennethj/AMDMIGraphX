#include <migraphx/gpu/device/sub.hpp>
#include <migraphx/gpu/device/nary.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void sub(hipStream_t stream, const argument& result, const argument& arg1, const argument& arg2)
{
    nary(stream, result, arg1, arg2)([](auto x, auto y) { return y - x; });
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
