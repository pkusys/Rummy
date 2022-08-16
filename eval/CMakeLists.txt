# Copyright (c) zili zhang & fangyue liu @PKU.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(overall EXCLUDE_FROM_ALL overall.cu)
target_link_libraries(overall PRIVATE faiss)