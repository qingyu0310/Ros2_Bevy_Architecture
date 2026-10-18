# CUDA / TensorRT 的发现逻辑。
# 构建期由 framework/CMakeLists.txt include；同时作为 CONFIG_EXTRAS 装进
# share/framework/cmake/，下游 find_package(framework) 时先跑这一段，把
# CUDA::cudart 与 TensorRT::nvinfer 建出来——framework_vision 的导出接口里引用它们，
# 没有这两个目标，targets export 会直接报 "target was not found"。
# 和旧工程一样从项目本地 .venv 找；新工程没有自己的一份，默认指向旧工程那份
# 下面的 CUDAToolkit_ROOT 要作为 <PackageName>_ROOT 生效；本工程的
# cmake_minimum_required 是 3.8，这个策略默认未设置会告警并忽略该变量
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

set(DUST_DEPENDENCY_ROOT "/home/qingyudd01/projects/vscode/opencv/.venv" CACHE PATH
    "Root of the project-local CUDA and TensorRT environment")
if(NOT DEFINED CUDAToolkit_ROOT)
    set(CUDAToolkit_ROOT "${DUST_DEPENDENCY_ROOT}" CACHE PATH "CUDA Toolkit root")
endif()
find_package(CUDAToolkit 12 REQUIRED)

set(TensorRT_ROOT "${DUST_DEPENDENCY_ROOT}" CACHE PATH "TensorRT installation root")
if(NOT TARGET TensorRT::nvinfer)
    find_path(TensorRT_INCLUDE_DIR NAMES NvInfer.h
        HINTS "${TensorRT_ROOT}/include" "$ENV{TensorRT_ROOT}/include" REQUIRED)
    find_library(TensorRT_LIBRARY NAMES nvinfer
        HINTS "${TensorRT_ROOT}/lib" "${TensorRT_ROOT}/lib64" REQUIRED)
    add_library(TensorRT::nvinfer SHARED IMPORTED GLOBAL)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()
