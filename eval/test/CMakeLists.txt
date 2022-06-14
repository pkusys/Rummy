# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(test_memory EXCLUDE_FROM_ALL test_memory.cpp)
target_link_libraries(test_memory PRIVATE faiss)

add_executable(test_avl EXCLUDE_FROM_ALL test_avl.cpp)
target_link_libraries(test_avl PRIVATE faiss)