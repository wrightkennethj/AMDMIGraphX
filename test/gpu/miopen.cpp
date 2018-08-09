
#include <migraph/program.hpp>
#include <migraph/operators.hpp>
#include <migraph/generate.hpp>
#include <migraph/cpu/cpu_target.hpp>
#include <migraph/gpu/target.hpp>
#include <migraph/gpu/miopen.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/manage_ptr.hpp>

#include <miopen/miopen.h>

#include "test.hpp"
#include "verify.hpp"

template <class V>
migraph::argument run_cpu()
{
    V v;
    auto p = v.create_program();
    p.compile(migraph::cpu::cpu_target{});
    return p.eval(v.create_params());
}

template <class V>
migraph::argument run_gpu()
{
    V v;
    auto p = v.create_program();
    p.compile(migraph::gpu::target{});

    auto m = v.create_params();
    for(auto&& e : m)
    {
        e.second = migraph::gpu::to_gpu(e.second);
    }

    m["output"] = migraph::gpu::to_gpu(migraph::generate_argument(p.get_parameter_shape("output")));

    return migraph::gpu::from_gpu(p.eval(m));
}

template <class V>
void verify_program()
{
    auto cpu_arg = run_cpu<V>();
    auto gpu_arg = run_gpu<V>();
    visit_all(cpu_arg, gpu_arg)([](auto cpu, auto gpu) { EXPECT(test::verify_range(cpu, gpu)); });
}

struct test_literals
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input = p.add_literal(
            generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}}));
        auto weights = p.add_literal(
            generate_literal(migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}}));
        auto conv = p.add_instruction(migraph::convolution{}, input, weights);
        p.add_instruction(migraph::activation{"relu"}, conv);
        return p;
    }

    migraph::program::parameter_map create_params() const { return {}; }
};

struct test_add
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x = p.add_parameter("x", s);
        auto y = p.add_parameter("y", s);
        p.add_instruction(migraph::add{}, x, y);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {3}});
        m["y"] = migraph::generate_argument({migraph::shape::float_type, {3}});
        return m;
    }
};

struct test_add_broadcast
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {3}};
        auto x  = p.add_parameter("x", {migraph::shape::float_type, {2, 2, 3}});
        auto y  = p.add_parameter("y", {migraph::shape::float_type, {2, 2}});
        auto by = p.add_instruction(migraph::broadcast{0}, x, y);
        p.add_instruction(migraph::add{}, x, by);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {2, 2, 3}});
        m["y"] = migraph::generate_argument({migraph::shape::float_type, {2, 2}});
        return m;
    }
};

struct test_conv_relu
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input = p.add_parameter("x", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto conv = p.add_instruction(migraph::convolution{}, input, weights);
        p.add_instruction(migraph::activation{"relu"}, conv);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 3, 3}});
        m["w"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 3, 3}});
        return m;
    }
};

struct test_conv_pooling
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto input =
            p.add_parameter("x", migraph::shape{migraph::shape::float_type, {4, 3, 32, 32}});
        auto weights =
            p.add_parameter("w", migraph::shape{migraph::shape::float_type, {4, 3, 3, 3}});
        auto conv    = p.add_instruction(migraph::convolution{}, input, weights);
        auto pooling = p.add_instruction(migraph::pooling{"max"}, conv);
        p.add_instruction(migraph::activation{"relu"}, pooling);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 32, 32}});
        m["w"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 3, 3}});
        return m;
    }
};

struct test_gemm_nn
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {4, 5}});
        auto b = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {5, 3}});
        p.add_instruction(migraph::gemm{}, a, b);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["a"] = migraph::generate_argument({migraph::shape::float_type, {4, 5}});
        m["b"] = migraph::generate_argument({migraph::shape::float_type, {5, 3}});
        return m;
    }
};

struct test_gemm_nt
{
    migraph::program create_program() const
    {
        migraph::program p;
        auto a = p.add_parameter("a", migraph::shape{migraph::shape::float_type, {4, 5}});
        auto b = p.add_parameter("b", migraph::shape{migraph::shape::float_type, {3, 5}});
        std::vector<int64_t> perm = {1,0};
        auto bt = p.add_instruction(migraph::transpose{perm}, b);
        p.add_instruction(migraph::gemm{}, a, bt);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["a"] = migraph::generate_argument({migraph::shape::float_type, {4, 5}});
        m["b"] = migraph::generate_argument({migraph::shape::float_type, {3, 5}});
        return m;
    }
};

struct test_contiguous
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {4, 4, 4, 3}, {48, 4, 1, 16}};
        auto x = p.add_parameter("x", s);
        p.add_instruction(migraph::contiguous{}, x);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] =
            migraph::generate_argument({migraph::shape::float_type, {4, 4, 4, 3}, {48, 4, 1, 16}});
        return m;
    }
};

struct test_transpose
{
    migraph::program create_program() const
    {
        migraph::program p;
        migraph::shape s{migraph::shape::float_type, {4, 3, 4, 4}};
        auto x                    = p.add_parameter("x", s);
        std::vector<int64_t> perm = {0, 2, 3, 1};
        auto l                    = p.add_instruction(migraph::transpose{perm}, x);
        p.add_instruction(migraph::contiguous{}, l);
        return p;
    }

    migraph::program::parameter_map create_params() const
    {
        migraph::program::parameter_map m;
        m["x"] = migraph::generate_argument({migraph::shape::float_type, {4, 3, 4, 4}});
        return m;
    }
};

void wst_gemm_test()
{
    std::vector<float> a = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                            1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                            -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                            -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    // NN
    {
        std::vector<float> c = {-1.56327541e+00,
                                -7.09570140e-01,
                                -5.37424982e-01,
                                -2.22994831e-01,
                                -2.15586437e+00,
                                2.09177941e-03,
                                -1.47279677e+00,
                                2.02627040e-01,
                                -6.04527691e-01,
                                -1.29885596e+00,
                                2.16294914e+00,
                                -1.48101497e-01};
        migraph::program p;
        migraph::shape a_shape{migraph::shape::float_type, {4, 5}};
        migraph::shape b_shape{migraph::shape::float_type, {5, 3}};
        auto al = p.add_literal(migraph::literal{a_shape, a});
        auto bl = p.add_literal(migraph::literal{b_shape, b});
        p.add_instruction(migraph::gemm{}, al, bl);
        p.compile(migraph::gpu::target{});
        auto result = p.eval({});
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        float tol = 1e-6;
        for(int i = 0; i < results_vector.size(); i++)
        {
            EXPECT(std::abs(results_vector[i] - c[i]) < tol);
        }
    }
    // NT
    {
        std::vector<float> c = {-0.28500289,
                                -0.60820148,
                                1.00851644,
                                0.85093479,
                                -0.54693915,
                                -1.56869325,
                                -0.97044707,
                                0.12430357,
                                0.6781955,
                                -2.37512276,
                                1.2701225,
                                -0.54703633};
        migraph::program p;
        migraph::shape a_shape{migraph::shape::float_type, {4, 5}};
        migraph::shape b_shape{migraph::shape::float_type, {3, 5}};
        auto al                   = p.add_literal(migraph::literal{a_shape, a});
        auto bl                   = p.add_literal(migraph::literal{b_shape, b});
        std::vector<int64_t> perm = {1, 0};
        auto bt                   = p.add_instruction(migraph::transpose{perm}, bl);
        p.add_instruction(migraph::gemm{1.0f, 0.0f}, al, bt);
        p.compile(migraph::cpu::cpu_target{});
        auto result = p.eval({});
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        float tol = 1e-6;
        for(int i = 0; i < results_vector.size(); i++)
        {
            EXPECT(std::abs(results_vector[i] - c[i]) < tol);
        }
    }
    // TN
    {
        std::vector<float> c = {0.08103308,
                                -1.01355749,
                                -0.49229794,
                                -1.78781014,
                                -0.1151139,
                                -0.0256696,
                                1.01611274,
                                0.32658809,
                                0.76131256,
                                -0.07680248,
                                0.66096994,
                                1.23196651};
        migraph::program p;
        migraph::shape a_shape{migraph::shape::float_type, {5, 4}};
        migraph::shape b_shape{migraph::shape::float_type, {5, 3}};
        auto al                   = p.add_literal(migraph::literal{a_shape, a});
        auto bl                   = p.add_literal(migraph::literal{b_shape, b});
        std::vector<int64_t> perm = {1, 0};
        auto at                   = p.add_instruction(migraph::transpose{perm}, al);
        p.add_instruction(migraph::gemm{1.0f, 0.0f}, at, bl);
        p.compile(migraph::cpu::cpu_target{});
        auto result = p.eval({});
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        float tol = 1e-6;
        for(int i = 0; i < results_vector.size(); i++)
        {
            EXPECT(std::abs(results_vector[i] - c[i]) < tol);
        }
    }
    // TT
    {
        std::vector<float> c = {1.26490865,
                                -0.8707044,
                                2.43410874,
                                -1.1972181,
                                -0.32065833,
                                -0.93707533,
                                0.05059333,
                                0.48048166,
                                -1.94676043,
                                0.15601288,
                                0.6720962,
                                0.33086335};
        migraph::program p;
        migraph::shape a_shape{migraph::shape::float_type, {5, 4}};
        migraph::shape b_shape{migraph::shape::float_type, {3, 5}};
        auto al                   = p.add_literal(migraph::literal{a_shape, a});
        auto bl                   = p.add_literal(migraph::literal{b_shape, b});
        std::vector<int64_t> perm = {1, 0};
        auto at                   = p.add_instruction(migraph::transpose{perm}, al);
        auto bt                   = p.add_instruction(migraph::transpose{perm}, bl);
        p.add_instruction(migraph::gemm{1.0f, 0.0f}, at, bt);
        p.compile(migraph::cpu::cpu_target{});
        auto result = p.eval({});
        std::vector<float> results_vector(12);
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        float tol = 1e-6;
        for(int i = 0; i < results_vector.size(); i++)
        {
            EXPECT(std::abs(results_vector[i] - c[i]) < tol);
        }
    }
}

int main()
{
    // verify_program<test_add>();
    // verify_program<test_add_broadcast>();
    // verify_program<test_conv_relu>();
    // verify_program<test_conv_pooling>();
    // verify_program<test_gemm_nn>();
    // verify_program<test_gemm_nt>();
    // verify_program<test_contiguous>();
    // verify_program<test_transpose>();
    wst_gemm_test();
}
