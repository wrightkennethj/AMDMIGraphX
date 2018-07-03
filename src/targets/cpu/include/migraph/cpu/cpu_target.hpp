#ifndef MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_CPU_TARGET_HPP

#include <migraph/program.hpp>

namespace migraph {
namespace cpu {

struct cpu_target
{
    std::string name() const;
    void apply(program& p) const;
    context get_context() const { return {}; }
};

} // namespace cpu

} // namespace migraph

#endif