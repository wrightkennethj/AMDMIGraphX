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

instruction_ref convert_fp32_fp16(program& prog, instruction_ref& ins)
{
    assert(ins->get_shape().type() == shape::float_type);
    assert(ins->name() == "@literal" || ins->name() == "@param");
    instruction_ref ins_fp16{};
    if(ins->name() == "@literal")
    {
        std::vector<float> values;
        auto l_fp32 = ins->get_literal();
        shape s     = ins->get_shape();
        l_fp32.visit([&](auto val) { values.assign(val.begin(), val.end()); });
        ins_fp16 = prog.add_literal(literal({shape::half_type, s.lens()}, values));
    }
    else if(ins->name() == "@param")
    {
        if(ins == std::prev(prog.end()))
        {
            ins_fp16 = prog.add_instruction(op::fp_conversion{}, ins);
        }
        else
        {
            ins_fp16 = prog.insert_instruction(std::next(ins), op::fp_conversion{}, ins);
        }
    }

    return ins_fp16;
}

// void replace_arguments(program& prog, instruction_ref ins, std::unordered_map<instruction_ref,
// instruction_ref>& convrted_ins)
// {
//    if (ins->get_shape().type() != shape::float_type)
//    {
//        convrted_ins[ins] = ins;
//        return;
//    }

//    auto inputs = ins->inputs();
//    for (auto input : inputs)
//    {
//        if (convrted_ins.count(input) > 0)
//        {
//            if (convrted_ins[input] != input)
//            {
//                instruction::replace_argument(ins, input, convrted_ins[input], false);
//            }
//        }
//        else
//        {
//            if (input->get_shape().type() == shape::float_type)
//            {
//                if (input->name() == "@literal" || input->name() == "@param")
//                {
//                    auto ins_fp16 = convert_fp32_fp16(prog, input);
//                    convrted_ins[input] = ins_fp16;
//                    instruction::replace_argument(ins, input, ins_fp16, false);
//                }
//                else
//                {
//                    replace_arguments(prog, input, convrted_ins);
//                }
//            }
//            else
//            {
//                convrted_ins[input] = input;
//            }
//        }
//    }

//    std::cout << "temp_prog = \n" << prog << std::endl;
//    std::cout << "ins_shape = " << ins->get_shape() << std::endl;
//    for (auto input : ins->inputs())
//    {
//        std::cout << "input_shape = " << input->get_shape() << std::endl;
//    }

//    ins->recompute_shape();

//    return;
// }

// void replace_instruction(program& prog,
//                    instruction_ref ins,
//                    std::unordered_map<instruction_ref, instruction_ref>& convrted_ins)
// {
//    std::vector<std::string> vec_op_names = {
//        "dot", "convolution", "add", "sub", "mul", "div", "max", "min"};

//    if(convrted_ins.count(ins) > 0)
//    {
//        return;
//    }

//    if(ins->get_shape().type() != shape::float_type)
//    {
//        return;
//    }

//    if(ins->name() == "@literal" || ins->name() == "@param")
//    {
//        convrted_ins[ins] = convert_fp32_fp16(prog, ins);
//        auto outputs = ins->outputs();
//        if (!std::any_of(outputs.begin(), outputs.end(), [&](auto i) {
//            return std::find(vec_op_names.begin(), vec_op_names.end(), i->name()) !=
//            vec_op_names.end();
//        }))
//        {
//            prog.replace_instruction(ins, convrted_ins[ins]);
//        }
//        else
//        {
//            for(auto output : outputs)
//            {
//                if (std::find(vec_op_names.begin(), vec_op_names.end(), output->name()) !=
//                vec_op_names.end())
//                {
//                    replace_arguments(prog, output, convrted_ins);
//                }
//            }
//        }
//    }
// }

// instruction_ref
// replace_instruction(program& prog,
//                     instruction_ref ins,
//                     std::unordered_map<instruction_ref, instruction_ref> convrted_ins)
// {
//     std::vector<std::string> vec_op_names = {
//         "dot", "convolution", "add", "sub", "mul", "div", "max", "min"};
//     if(convrted_ins.count(ins) > 0)
//     {
//         return convrted_ins[ins];
//     }

//     if(ins->get_shape().type() != shape::float_type)
//     {
//         return ins;
//     }

//     if(ins->name() == "@literal" || ins->name() == "@param")
//     {
//         instruction_ref ins_fp16{};
//         if(ins->name() == "@literal")
//         {
//             std::vector<float> values;
//             auto l_fp32 = ins->get_literal();
//             shape s     = ins->get_shape();
//             l_fp32.visit([&](auto val) { values.assign(val.begin(), val.end()); });
//             ins_fp16 = prog.add_literal(literal({shape::half_type, s.lens()}, values));
//         }
//         else if(ins->name() == "@param")
//         {
//             if(ins == std::prev(prog.end()))
//             {
//                 ins_fp16 = prog.add_instruction(op::fp_conversion{}, ins);
//             }
//             else
//             {
//                 ins_fp16 = prog.insert_instruction(std::next(ins), op::fp_conversion{}, ins);
//             }
//         }

//         convrted_ins[ins] = ins_fp16;

//         auto outputs = ins->outputs();
//         for(auto output : outputs)
//         {
//             if(std::find(vec_op_names.begin(), vec_op_names.end(), output->name()) !=
//                vec_op_names.end())
//             {
//                 auto inputs = output->inputs();
//                 for(auto input : inputs)
//                 {
//                     if(input != ins)
//                     {
//                         convrted_ins[input] = replace_instruction(prog, input, convrted_ins);
//                     }

//                     if(convrted_ins[input] != input)
//                     {
//                         instruction::replace_argument(output, input, convrted_ins[input], false);
//                     }
//                 }
//                 output->recompute_shape();
//             }
//         }
//         // prog.replace_instruction(ins, ins_fp16);

//         return ins_fp16;
//     }
//     else
//     {
//         auto inputs = ins->inputs();
//         for(auto input : inputs)
//         {
//             convrted_ins[input] = replace_instruction(prog, input, convrted_ins);
//             if(convrted_ins[input] != input)
//             {
//                 instruction::replace_argument(ins, input, convrted_ins[input], false);
//             }
//         }

//         ins->recompute_shape();

//         return ins;
//     }
// }

// void quantize(program& prog)
// {
//    std::unordered_map<instruction_ref, instruction_ref> instruction_map;

//    bool reduced_precision = false;
//    for(auto ins : iterator_for(prog))
//    {
//        if(instruction_map.count(ins) > 0)
//        {
//            continue;
//        }

//        // convert float_type to half_type
//        if((ins->name() == "@literal" || ins->name() == "@param") &&
//           ins->get_shape().type() == shape::float_type)
//        {
//            replace_instruction(prog, ins, instruction_map);
//            reduced_precision = true;
//        }
//        else
//        {
//            ins->recompute_shape();
//        }
//    }

//    // add another instruction at last to convert fp16 to fp32
//    if(reduced_precision)
//    {
//        auto ins = std::prev(prog.end());
//        if(ins->get_shape().type() == shape::half_type)
//        {
//            prog.add_instruction(op::fp_conversion{false}, ins);
//        }
//    }

//    std::cout << "prog = " << prog << std::endl;
// }

void quantize(program& prog)
{
    bool reduced_precision = false;
    for(auto ins : iterator_for(prog))
    {
        // convert float_type to half_type
        if((ins->name() == "@literal" || ins->name() == "@param") &&
           ins->get_shape().type() == shape::float_type)
        {
            auto ins_fp16 = convert_fp32_fp16(prog, ins);
            // prog.replace_instruction(ins, ins_fp16, false);
            // std::cout << "prog0 = \n" << prog << std::endl;
            auto outputs = ins->outputs();
            for(auto output : outputs)
            {
                if(output != ins_fp16)
                {
                    instruction::replace_argument(output, ins, ins_fp16, false);
                }
            }
            // std::cout << "prog1 = \n" << prog << std::endl;

            reduced_precision = true;
        }
    }

    // add another instruction at last to convert fp16 to fp32
    if(reduced_precision)
    {
        for(auto ins : iterator_for(prog))
        {
            // std::cout << "prog2 = \n" << prog << std::endl;
            if(ins->name() != "@literal" && ins->name() != "@param")
            {
                // std::cout << "name = " << ins->name() << std::endl;
                // auto inputs = ins->inputs();
                // for (auto input : inputs)
                //{
                //    std::cout << "input's shape = " << input->get_shape() << " " << input->name()
                //    << std::endl;
                //}
                ins->recompute_ins_shape();
            }
        }

        auto ins = std::prev(prog.end());
        if(ins->get_shape().type() == shape::half_type)
        {
            prog.add_instruction(op::fp_conversion{false}, ins);
        }
    }

    // std::cout << "prog = " << prog << std::endl;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
