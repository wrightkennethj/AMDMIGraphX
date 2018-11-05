#include <migraph/eliminate_allocation.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/ranges.hpp>
#include <migraph/stringutils.hpp>
#include <migraph/pass_config.hpp>

namespace migraph {
inline namespace MIGRAPH_INLINE_NS {

void eliminate_allocation::apply(program& p) const
{
    assert(alignment > 0);
    if(!enabled(MIGRAPH_DISABLE_MEMORY_COLORING{}))
        return;

    std::size_t n = 0;
    std::vector<std::pair<instruction_ref, std::size_t>> allocs;
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != allocation_op)
            continue;
        allocs.emplace_back(ins, n);
        std::size_t size    = ins->get_shape().bytes();
        std::size_t padding = (alignment - (size % alignment)) % alignment;
        n += size + padding;
    }
    auto mem = p.add_parameter("memory", shape{shape::int8_type, {n}});
    for(auto&& pp : allocs)
    {
        auto ins    = pp.first;
        auto s      = ins->get_shape();
        auto offset = pp.second;
        p.replace_instruction(ins, op::load{s, offset}, mem);
    }
}

} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph
