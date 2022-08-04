# Copyright (c) zili zhang & fangyue liu @PKU.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(test_memory EXCLUDE_FROM_ALL test_memory.cpp)
target_link_libraries(test_memory PRIVATE faiss)

add_executable(test_avl EXCLUDE_FROM_ALL test_avl.cpp)
target_link_libraries(test_avl PRIVATE faiss)

add_executable(test_cpu_quantizer EXCLUDE_FROM_ALL test_cpu_quantizer.cpp)
target_link_libraries(test_cpu_quantizer PRIVATE faiss)

add_executable(test_profile_kernel EXCLUDE_FROM_ALL test_profile_kernel.cu)
target_link_libraries(test_profile_kernel PRIVATE faiss)

add_executable(test_heap EXCLUDE_FROM_ALL test_heap.cpp)
target_link_libraries(test_heap PRIVATE faiss)

add_executable(test_multi_sort EXCLUDE_FROM_ALL test_multi_sort.cpp)
target_link_libraries(test_multi_sort PRIVATE faiss)

add_executable(test_pipetensor EXCLUDE_FROM_ALL test_pipetensor.cpp)
target_link_libraries(test_pipetensor PRIVATE faiss)

add_executable(test_pipeline EXCLUDE_FROM_ALL test_pipeline.cu)
target_link_libraries(test_pipeline PRIVATE faiss)

add_executable(test_pthread EXCLUDE_FROM_ALL test_pthread.cpp)
target_link_libraries(test_pthread PRIVATE faiss)

add_executable(test_multi EXCLUDE_FROM_ALL test_multi_sort.cpp)
target_link_libraries(test_multi PRIVATE faiss)

