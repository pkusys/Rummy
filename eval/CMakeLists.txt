# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(test_pipe_cpu EXCLUDE_FROM_ALL test_pipe_cpu.cpp)
target_link_libraries(test_pipe_cpu PRIVATE faiss)

