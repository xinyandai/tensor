cmake_minimum_required(VERSION 2.8)

project(tensor)
SET( CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -Ofast -lrt -lm -DNDEBUG -std=c++17 -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )

find_package(OpenMP)
if (OPENMP_FOUND)
    message("-- [Found PKG] Found OpenMP")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
find_package( BLAS REQUIRED )
message( STATUS BLAS found: ${BLAS_LIBRARIES} )

SET(EXAMPLES example)

FOREACH(EXAMPLE ${EXAMPLES} )
    ADD_EXECUTABLE(${EXAMPLE} examples/${EXAMPLE}.cc)
    target_link_libraries(${EXAMPLE} ${BLAS_LIBRARIES})
ENDFOREACH(EXAMPLE)
