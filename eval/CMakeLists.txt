# Copyright (c) zili zhang & fangyue liu @PKU.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(overall EXCLUDE_FROM_ALL overall.cpp)
target_link_libraries(overall PUBLIC faiss)

add_executable(process-data EXCLUDE_FROM_ALL process-data.cpp)
target_link_libraries(process-data PUBLIC faiss)