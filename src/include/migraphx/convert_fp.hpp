#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_CONVERT_FP_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_CONVERT_FP_HPP

#include <list>
#include <unordered_map>
#include <migraphx/operation.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/target.hpp>
#include <migraphx/tracer.hpp>
#include <migraphx/env.hpp>
#include <migraphx/config.hpp>
#include <algorithm>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct program;

// temp implementation to write it as a pass for testing
struct convert_fp
{
    std::string name() const { return "convert_fp"; }
    void apply(program& prog) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
