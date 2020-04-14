cmake_minimum_required(VERSION 2.8)
project(SuperVoxelCuda)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11 -O3 -march=native -D_MWAITXINTRIN_H_INCLUDED")


FIND_PACKAGE(CUDA REQUIRED)

#set(CUDA_SEPARABLE_COMPILATION ON)
set(NVCC_COMPILER_BINDIR "" CACHE PATH  "Directory where nvcc should look for C++ compiler. This is passed to nvcc through the --compiler-bindir option.")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC;-arch=sm_50;-O3;-D_FORCE_INLINES)
if (NVCC_COMPILER_BINDIR)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--compiler-bindir=${NVCC_COMPILER_BINDIR}")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)


find_package(Glog REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/Thirdparty/flann/src/cpp
${CUDA_INCLUDE_DIRS}
${GLOG_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# cuda_add_library(normal SHARED
# src/normalEstimation.cu
# )

# target_link_libraries(normal #PROPERTIES LINKER_LANGUAGE C
# ${CUDA_LIBRARIES}
# )

cuda_add_library(supervoxel SHARED
src/normalEstimation.cu
src/supervoxel.cpp
)
target_link_libraries(supervoxel
${GLOG_LIBRARIES}
${CUDA_LIBRARIES}
# ${PROJECT_SOURCE_DIR}/lib/libnormal.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cpp.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cuda.so
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# cuda_add_executable(test
# src/test_flann_cuda.cu
# )
# target_link_libraries(test ${GLOG_LIBRARIES} ${CUDA_LIBRARIES}
# ${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann.so
# ${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cpp.so
# ${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cuda.so)

add_executable(svcluster
        src/main.cpp)
target_link_libraries(svcluster 
${CUDA_LIBRARIES}
${GLOG_LIBRARIES}
# ${PROJECT_SOURCE_DIR}/lib/libnormal.so
${PROJECT_SOURCE_DIR}/lib/libsupervoxel.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cpp.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cuda.so
)

