#ifndef MIGRAPHX_GUARD_RTGLIB_TANH_HPP
#define MIGRAPHX_GUARD_RTGLIB_TANH_HPP

#include <migraphx/shape.hpp>
#include <migraphx/gpu/miopen.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct miopen_tanh
{
    shared<activation_descriptor> ad;
    std::string name() const { return "gpu::tanh"; }
    shape compute_shape(const std::vector<shape>& inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
