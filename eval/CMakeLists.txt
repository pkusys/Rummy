# Copyright (c) zili zhang & fangyue liu @PKU.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(overall EXCLUDE_FROM_ALL overall.cu)
target_link_libraries(overall PRIVATE faiss)

add_executable(nomemman EXCLUDE_FROM_ALL nomemman.cu)
target_link_libraries(nomemman PRIVATE faiss)

add_executable(overall_query EXCLUDE_FROM_ALL overall_query.cu)
target_link_libraries(overall_query PRIVATE faiss)

add_executable(overall_no_pipeline EXCLUDE_FROM_ALL overall_no_pipeline.cu)
target_link_libraries(overall_no_pipeline PRIVATE faiss)

add_executable(overall_per_cluster_pipeline EXCLUDE_FROM_ALL overall_per_cluster_pipeline.cu)
target_link_libraries(overall_per_cluster_pipeline PRIVATE faiss)

add_executable(overall_no_reorder EXCLUDE_FROM_ALL overall_no_reorder.cu)
target_link_libraries(overall_no_reorder PRIVATE faiss)

add_executable(overall_no_pin EXCLUDE_FROM_ALL overall_no_pin.cu)
target_link_libraries(overall_no_pin PRIVATE faiss)

add_executable(overall_no_lru EXCLUDE_FROM_ALL overall_no_lru.cu)
target_link_libraries(overall_no_lru PRIVATE faiss)

add_executable(overall_billion EXCLUDE_FROM_ALL overall_billion.cu)
target_link_libraries(overall_billion PUBLIC faiss)

add_executable(overall_scalability EXCLUDE_FROM_ALL overall_scalability.cu)
target_link_libraries(overall_scalability PUBLIC faiss)