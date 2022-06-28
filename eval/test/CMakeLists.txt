# Copyright (c) Facebook, Inc. and its affiliates.
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

add_executable(test_gpu_quantizer EXCLUDE_FROM_ALL test_gpu_quantizer.cpp)
target_link_libraries(test_gpu_quantizer PRIVATE faiss)

add_executable(test_gpu_quantizer_bug EXCLUDE_FROM_ALL test_gpu_quantizer_bug.cpp)
target_link_libraries(test_gpu_quantizer_bug PRIVATE faiss)