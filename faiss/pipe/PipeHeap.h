/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include <algorithm>
#include <iostream>

namespace faiss {

enum HeapType
{
	MAXHEAP, MINHEAP,
};

template <class keytype,
        class valtype>
class PipeHeap{

// The heap pair type
using HeapPair = std::pair<keytype, valtype>;

public:
    // Construct an empty heap
    PipeHeap(int cap, HeapType ht);

    ~PipeHeap();

    // Push an new element to the heap
    void push(HeapPair ele);

    // Pop the Min/Max element
    void pop();

    // Float an new element (adding)
    void floating(int index);

    // Sink an old element (delete)
    void sink(int index);

    // Swap two elemets
    void swap(int i, int j);

    // Dump to vector
    std::vector<valtype> dump(){
        std::vector<valtype> ret(size);
        for(int i = 0; i < size; i++)
            ret[i] = data_[i].second;
        
        // Sort and return
        // std::sort(ret.begin(), ret.end());
        return ret;
    }

    // Check if the heap is full
    bool isFull(){
		if (size >= cap_)
		{
			return true;
		}
		return false;
	}

    keytype read(){
        return data_[0].first;
    }

    int getSize(){
        return size;
    }

private:
    // Element storage
	std::vector<HeapPair> data_;

    // The current size of elements
	int size;

    // The capacity of the whole heap
	int cap_;

    // Heap type: Max or Min?
	HeapType type;

};

template <class keytype,class valtype>
PipeHeap<keytype, valtype>::PipeHeap(int cap, HeapType ht){
    // Initialize the attributes
    size = 0;
    cap_ = cap;
    type = ht;

    data_.resize(cap);
    // data_[0].first = initv;
}

template <class keytype,class valtype>
PipeHeap<keytype, valtype>::~PipeHeap(){}

template <class keytype,class valtype>
void PipeHeap<keytype, valtype>::push(
            PipeHeap<keytype, valtype>::HeapPair ele){
    // Check the remaining size
    if(isFull()){
        pop();
    }

    data_[size] = ele;
	size++;
	floating(size);
}

template <class keytype,class valtype>
void PipeHeap<keytype, valtype>::pop(){
    // Check the element size
    if(!size)
        FAISS_ASSERT(false);

    data_[0] = data_[size - 1];
	size--;
	sink(1);
}

template <class keytype,class valtype>
void PipeHeap<keytype, valtype>::swap(int i, int j){
    HeapPair tmp = data_[i];
    data_[i] = data_[j];
	data_[j] = tmp;
}

template <class keytype,class valtype>
void PipeHeap<keytype, valtype>::floating(int index){
    if (size == 1)
		return;
    if (type == HeapType::MINHEAP){
        for (int i = index; i > 0; i /= 2){
			if (data_[i - 1].first < data_[i/2 - 1].first)
				swap(i - 1, i/2 - 1);
			else
				break;
		}
    }
    else{
        for (int i = index; i > 0; i /= 2)
		{
			if (data_[i - 1].first > data_[i/2 - 1].first)
				swap(i - 1, i/2 - 1);
			else
				break;
		}
    }
}

template <class keytype,class valtype>
void PipeHeap<keytype, valtype>::sink(int index){
    if (type == HeapType::MINHEAP){
		while (index/2 <= size){
            // Lest node
			if (data_[index - 1] > data_[index * 2 - 1]){
				swap(index - 1, index * 2 - 1);

				if (index * 2 + 1 <= size && data_[index - 1] > data_[index * 2])
					swap(index - 1, index * 2);
				index *= 2;
			}
            // Right node
			else if (index * 2 + 1 <= size && data_[index - 1] > data_[index * 2]){
				swap(index - 1, index * 2);
				index = index * 2 + 1;
			}
			else
				break;
		}
	}
	else if (type == HeapType::MAXHEAP){
		while (index * 2 <= size){
			if (data_[index - 1] < data_[index * 2 - 1]){
				swap(index - 1, index * 2 - 1);

				if (index * 2 + 1 <= size && data_[index - 1]< data_[index * 2])
					swap(index - 1, index * 2);
				index *= 2;
			}
			else if (index * 2 + 1 <= size && data_[index - 1] < data_[index * 2]){
				swap(index - 1, index * 2);
				index = index * 2 + 1;
			}
			else
				break;
		}
	}
}


}