
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)

add_library(migraphx 
    auto_contiguous.cpp
    common_subexpression_elimination.cpp
    propagate_constant.cpp
    dead_code_elimination.cpp
    eliminate_allocation.cpp
    eliminate_contiguous.cpp
    eliminate_concat.cpp
    eliminate_identity.cpp
    eliminate_pad.cpp
    fwd_conv_batchnorm_rewrite.cpp
    rewrite_rnn.cpp
    env.cpp
    generate.cpp
    instruction.cpp
    program.cpp
    shape.cpp
    schedule.cpp
    pass_manager.cpp
    simplify_algebra.cpp
    simplify_reshapes.cpp
    opt/memory_coloring.cpp
    opt/memory_coloring_impl.cpp
)
rocm_clang_tidy_check(migraphx)
rocm_install_targets(
  TARGETS migraphx
  INCLUDE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_path(HALF_INCLUDE_DIR half.hpp)
# TODO: Fix the incorrect path
target_include_directories(migraphx SYSTEM PUBLIC $<BUILD_INTERFACE:${HALF_INCLUDE_DIR}>)

set(PACKAGE_DEPENDS)

add_subdirectory(onnx)
add_subdirectory(tf)

add_subdirectory(py)
add_subdirectory(targets/cpu)
if(MIGRAPHX_ENABLE_GPU)
list(APPEND PACKAGE_DEPENDS MIOpen rocblas)
add_subdirectory(targets/gpu)
endif()

rocm_export_targets(
  TARGETS migraphx::migraphx
  NAMESPACE migraphx::
  DEPENDS
    ${PACKAGE_DEPENDS}
)
 

