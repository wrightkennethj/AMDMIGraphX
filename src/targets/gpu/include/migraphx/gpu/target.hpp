#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_MIOPEN_TARGET_HPP

#include <migraphx/program.hpp>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct target
{
    std::string name() const;
    std::vector<pass> get_passes(migraphx::context& gctx) const;
    migraphx::context get_context() const;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
