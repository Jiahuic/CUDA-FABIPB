cmake_minimum_required(VERSION 3.16)

project(fmm_PB VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(FMM_PB_BLA_VENDOR "" CACHE STRING
  "Optional BLAS/LAPACK vendor hint passed to BLA_VENDOR (e.g. OpenBLAS, Apple)")
if(FMM_PB_BLA_VENDOR)
  set(BLA_VENDOR "${FMM_PB_BLA_VENDOR}")
endif()

option(FMM_PB_ALLOW_UNSTABLE_OPENBLAS_0320
  "Allow configure with OpenBLAS 0.3.20, which is known to trigger unstable preconditioner solves on some machines"
  OFF)

set(FMM_PB_SOURCES
  src/fabipb.c
  src/gkSetup.c
  src/input.c
  src/gkGlobal.c
  src/expan.c
  src/moments.c
  src/numQuad.c
  src/fmm.c
  src/kernel.c
  src/precond_fmm.c
  src/gmres.c
  src/treecode.c
  src/direct_backend.c
)

option(FMM_PB_ENABLE_CUDA "Build CUDA GPU backend hooks when a CUDA toolkit is available" ON)
include(CheckLanguage)
check_language(CUDA)
set(FMM_PB_HAS_CUDA OFF)
if(FMM_PB_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(FMM_PB_HAS_CUDA ON)
  list(APPEND FMM_PB_SOURCES src/gpu_backend_cuda.cu)
else()
  list(APPEND FMM_PB_SOURCES src/gpu_backend_stub.c)
endif()

add_executable(fabipb ${FMM_PB_SOURCES})

target_include_directories(fabipb
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nanoshaper-master/build/NanoShaper")
  target_compile_definitions(fabipb PRIVATE
    FMM_PB_DEFAULT_NANOSHAPER_BIN="${CMAKE_CURRENT_SOURCE_DIR}/nanoshaper-master/build/NanoShaper")
endif()

if(FMM_PB_HAS_CUDA)
  target_compile_definitions(fabipb PRIVATE FMM_PB_HAS_CUDA_BACKEND=1)
else()
  target_compile_definitions(fabipb PRIVATE FMM_PB_HAS_CUDA_BACKEND=0)
endif()

find_package(BLAS QUIET)
if(NOT BLAS_FOUND)
  message(FATAL_ERROR
    "BLAS was not found. This project requires an installed BLAS library. "
    "libf2c is not a replacement for BLAS because src/gmres.c calls BLAS "
    "routines directly."
  )
endif()

find_package(LAPACK QUIET)
if(NOT LAPACK_FOUND)
  message(FATAL_ERROR
    "LAPACK was not found. Please install LAPACK and rerun CMake."
  )
endif()

set(FMM_PB_OPENBLAS_REALPATHS "")
foreach(_blas_lib IN LISTS BLAS_LIBRARIES LAPACK_LIBRARIES)
  if(IS_ABSOLUTE "${_blas_lib}" AND EXISTS "${_blas_lib}")
    get_filename_component(_blas_realpath "${_blas_lib}" REALPATH)
    if(_blas_realpath MATCHES "openblas")
      list(APPEND FMM_PB_OPENBLAS_REALPATHS "${_blas_realpath}")
    endif()
  endif()
endforeach()
list(REMOVE_DUPLICATES FMM_PB_OPENBLAS_REALPATHS)

set(FMM_PB_OPENBLAS_0320_DETECTED OFF)
foreach(_openblas_path IN LISTS FMM_PB_OPENBLAS_REALPATHS)
  if(_openblas_path MATCHES "0\.3\.20")
    set(FMM_PB_OPENBLAS_0320_DETECTED ON)
  endif()
endforeach()

if(FMM_PB_OPENBLAS_0320_DETECTED)
  if(FMM_PB_ALLOW_UNSTABLE_OPENBLAS_0320)
    message(WARNING
      "OpenBLAS 0.3.20 was detected in the resolved BLAS/LAPACK libraries: ${FMM_PB_OPENBLAS_REALPATHS}. "
      "This version has triggered unstable preconditioner solves on tested machines. Continue only for explicit comparison work."
    )
  else()
    message(FATAL_ERROR
      "OpenBLAS 0.3.20 was detected in the resolved BLAS/LAPACK libraries: ${FMM_PB_OPENBLAS_REALPATHS}. "
      "This version is blocked by default because it has triggered unstable preconditioner solves on tested machines. "
      "Use a different OpenBLAS version such as 0.3.21+ or rerun CMake with -DFMM_PB_ALLOW_UNSTABLE_OPENBLAS_0320=ON only for controlled comparison work."
    )
  endif()
endif()

find_package(Threads REQUIRED)

target_link_libraries(fabipb
  PRIVATE
    LAPACK::LAPACK
    BLAS::BLAS
    Threads::Threads
)

if(FMM_PB_HAS_CUDA)
  target_link_libraries(fabipb PRIVATE CUDA::cudart)
endif()

if(UNIX AND NOT APPLE)
  target_link_libraries(fabipb PRIVATE m)
endif()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(fabipb PRIVATE -g)
endif()

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "BLA_VENDOR hint: ${BLA_VENDOR}")
message(STATUS "CUDA backend enabled: ${FMM_PB_HAS_CUDA}")
message(STATUS "BLAS libraries: ${BLAS_LIBRARIES}")
message(STATUS "LAPACK libraries: ${LAPACK_LIBRARIES}")
message(STATUS "Resolved OpenBLAS libraries: ${FMM_PB_OPENBLAS_REALPATHS}")
message(STATUS "Allow unstable OpenBLAS 0.3.20: ${FMM_PB_ALLOW_UNSTABLE_OPENBLAS_0320}")
