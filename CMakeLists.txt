cmake_minimum_required(VERSION 2.8)
project(SuperVoxelCuda)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native -pthread -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native -std=c++11")


FIND_PACKAGE(CUDA REQUIRED)

OPTION(WITH_CUDA "Build with CUDA support?" ${CUDA_FOUND})

IF(WITH_CUDA)
  # Auto-detect the CUDA compute capability.
#  SET(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
#  IF(NOT DEFINED CUDA_COMPUTE_CAPABILITY)
#    INCLUDE("${CMAKE_MODULE_PATH}/CUDACheckCompute.cmake")
#  ENDIF()

  # Set the compute capability flags.
#  FOREACH(compute_capability ${CUDA_COMPUTE_CAPABILITY})
#    LIST(APPEND CUDA_NVCC_FLAGS --generate-code arch=compute_${compute_capability},code=sm_${compute_capability})
#  ENDFOREACH()

  # Enable fast math.
  SET(CUDA_NVCC_FLAGS --use_fast_math ; ${CUDA_NVCC_FLAGS})

  # If on Windows, make it possible to enable GPU debug information.
  IF(MSVC_IDE)
    OPTION(ENABLE_CUDA_DEBUGGING "Enable CUDA debugging?" OFF)
    IF(ENABLE_CUDA_DEBUGGING)
      SET(CUDA_NVCC_FLAGS -G; ${CUDA_NVCC_FLAGS})
    ENDIF()
  ENDIF()

  # If on Mac OS X 10.9 (Mavericks), make sure everything compiles and links using the correct C++ Standard Library.
  IF(${CMAKE_SYSTEM} MATCHES "Darwin-13.")
    SET(CUDA_HOST_COMPILER /usr/bin/clang)
    SET(CUDA_NVCC_FLAGS -Xcompiler -stdlib=libstdc++; -Xlinker -stdlib=libstdc++; ${CUDA_NVCC_FLAGS})
  ENDIF()

  # If on Linux:
  IF(${CMAKE_SYSTEM} MATCHES "Linux")
    # Make sure that C++11 support is enabled when compiling with nvcc. From CMake 3.5 onwards,
    # the host flag -std=c++11 is automatically propagated to nvcc. Manually setting it prevents
    # the project from building.
    IF(${CMAKE_VERSION} VERSION_LESS 3.5)
      SET(CUDA_NVCC_FLAGS -std=c++11; ${CUDA_NVCC_FLAGS})
    ENDIF()

    # Work around an Ubuntu 16.04 compilation error.
    IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER 5.0)
      ADD_DEFINITIONS(-D_FORCE_INLINES)
    ENDIF()
  ENDIF()

  # If not on Windows, disable some annoying nvcc warnings.
  IF(NOT MSVC_IDE)
    SET(CUDA_NVCC_FLAGS -Xcudafe "--diag_suppress=cc_clobber_ignored" ; -Xcudafe "--diag_suppress=set_but_not_used" ; ${CUDA_NVCC_FLAGS})
  ENDIF()
ELSE()
  ADD_DEFINITIONS(-DCOMPILE_WITHOUT_CUDA)
ENDIF()

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

cuda_add_library(normal SHARED
src/normalEstimation.cu
)

target_link_libraries(normal #PROPERTIES LINKER_LANGUAGE C
${CUDA_LIBRARIES}
)

add_library(supervoxel SHARED
src/supervoxel.cpp
)
target_link_libraries(supervoxel
${GLOG_LIBRARIES}
${CUDA_LIBRARIES}
${PROJECT_SOURCE_DIR}/lib/libnormal.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cpp.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cuda.so
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

cuda_add_executable(test
src/test_flann_cuda.cu
)
target_link_libraries(test ${GLOG_LIBRARIES} ${CUDA_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cpp.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cuda.so)

add_executable(svcluster
        src/main.cpp)
target_link_libraries(svcluster 
${CUDA_LIBRARIES}
${GLOG_LIBRARIES}
${PROJECT_SOURCE_DIR}/lib/libnormal.so
${PROJECT_SOURCE_DIR}/lib/libsupervoxel.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cpp.so
${PROJECT_SOURCE_DIR}/Thirdparty/flann/lib/libflann_cuda.so
)

