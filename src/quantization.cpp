#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void quantize(program& prog)
{
    bool reduced_precision = false;
    for(auto ins : iterator_for(prog))
    {
        // literal is 32-bit, convert to 16-bit
        if(ins->name() == "@literal")
        {
            shape s = ins->get_shape();
            // convert float_type to half_type
            if(s.type() == shape::float_type)
            {
                std::vector<float> values;
                auto l_fp32 = ins->get_literal();
                l_fp32.visit([&](auto val) { values.assign(val.begin(), val.end()); });
                auto l_fp16 = prog.add_literal(literal({shape::half_type, s.lens()}, values));
                prog.replace_instruction(ins, l_fp16);
                reduced_precision = true;
            }
        }
        // parameters is 32-bit add an operator to convert the
        // parameter to 16-bit
        else if(ins->name() == "@param")
        {
            shape s = ins->get_shape();
            // for float_type parameter, add an instruction to
            // convert float_type to half_type
            if(s.type() == shape::float_type)
            {
                instruction_ref ins_16{};
                if(ins == std::prev(prog.end()))
                {
                    ins_16 = prog.add_instruction(op::fp_conversion{}, ins);
                }
                else
                {
                    ins_16 = prog.insert_instruction(std::next(ins), op::fp_conversion{}, ins);
                }

                prog.replace_instruction(ins, ins_16);
                reduced_precision = true;
            }
        }
        else
        {
            ins->recompute_shape();
        }
    }

    // add another instruction at last to convert fp16 to fp32
    if(reduced_precision)
    {
        auto ins = std::prev(prog.end());
        if(ins->get_shape().type() == shape::half_type)
        {
            prog.add_instruction(op::fp_conversion{false}, ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
