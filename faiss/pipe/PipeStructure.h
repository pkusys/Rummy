/**
 * Copyright (c) zili zhang & fangyue liu @PKU.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/FaissAssert.h>

#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip> 

namespace faiss {

// The Max/Min Heap Data Struture
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


template <class K, class V>
class AVLTreeNode{
public:
    K key;
    V val;
    int height;
    AVLTreeNode *left;
    AVLTreeNode *right;

    AVLTreeNode(K k, V v, AVLTreeNode *l = nullptr, 
            AVLTreeNode *r = nullptr):
            key(k), val(v), height(0),left(l),right(r){}
};

template <class K, class V>
class PipeAVLTree {

public:
    AVLTreeNode<K, V> *root;
    int size = 0;

public:
    PipeAVLTree();
    ~PipeAVLTree();

    // Get the height of the tree
    int height();

    // Pair compare func
    bool com(K k1, V v1, K k2, V v2);

    // For each the AVL tree in different orders
    void preOrder();
    void inOrder();
    void postOrder();

    // Recursive find
    AVLTreeNode<K, V>* Search(K k, V v);
    // Non-recursive find
    AVLTreeNode<K, V>* iterativeSearch(K k, V v);

    // Find the min node
    std::pair<K,V> minimum();
    // Find the max node
    std::pair<K,V> maximum();

    // Insert a node
    void insert(K k, V v);

    // Remove a node
    void remove(K k, V v);

    // Destroy the whole tree
    void destroy();

    // Print
    void print();

public:

    /// These operations focus on a subtree

    // Get the subtree height
    int height(AVLTreeNode<K, V>* tree) ;

    // Foreach the tree in different orders
    void preOrder(AVLTreeNode<K, V>* tree)  ;
    void inOrder(AVLTreeNode<K, V>* tree)  ;
    void postOrder(AVLTreeNode<K, V>* tree)  ;

    // Find the corresponding node in recursive and non-recursive ways
    AVLTreeNode<K, V>* Search(AVLTreeNode<K, V>* x, K k, V v)  ;
    AVLTreeNode<K, V>* iterativeSearch(AVLTreeNode<K, V>* x, K k, V v)  ;

    // Return Max/Min node
    AVLTreeNode<K, V>* minimum(AVLTreeNode<K, V>* tree);
    AVLTreeNode<K, V>* maximum(AVLTreeNode<K, V>* tree);

    // LL Rotate
    AVLTreeNode<K, V>* LLRotate(AVLTreeNode<K, V>* k2);

    // RR Rotate
    AVLTreeNode<K, V>* RRRotate(AVLTreeNode<K, V>* k1);

    // LR Rotate
    AVLTreeNode<K, V>* LRRotate(AVLTreeNode<K, V>* k3);

    // RL Rotate
    AVLTreeNode<K, V>* RLRotate(AVLTreeNode<K, V>* k1);

    // Insert a new pair to a subtree
    AVLTreeNode<K, V>* insert(AVLTreeNode<K, V>* &tree, K k, V v);

    // Delete a node
    AVLTreeNode<K, V>* remove(AVLTreeNode<K, V>* &tree, AVLTreeNode<K, V>* z);

    // Destroy a subtree
    void destroy(AVLTreeNode<K, V>* &tree);

    // Print the subtree
    void print(AVLTreeNode<K, V>* tree, K k, V v, int direction);
};

template <class K, class V>
PipeAVLTree<K, V>::PipeAVLTree(): root(nullptr) {}

template <class K, class V>
PipeAVLTree<K, V>::~PipeAVLTree() {destroy(root);}

template <class K, class V>
int PipeAVLTree<K, V>::height(AVLTreeNode<K, V>* tree) {
    if (tree != nullptr)
        return tree->height;

    return 0;
}

template <class K, class V>
int PipeAVLTree<K, V>::height() {
    return height(root);
}

template <class K, class V>
bool PipeAVLTree<K,V>::com(K k1, V v1, K k2, V v2) {
    if (k1 == k2)
        return v1 < v2;
    return k1 < k2;
}

template <class K, class V>
void PipeAVLTree<K, V>::preOrder(AVLTreeNode<K, V>* tree)  {
    if(tree != nullptr){
        std::cout<< "(" << tree->key << "," << tree->val << ") ";
        preOrder(tree->left);
        preOrder(tree->right);
    }
}

template <class K, class V>
void PipeAVLTree<K, V>::preOrder(){
    preOrder(root);
}

template <class K, class V>
void PipeAVLTree<K, V>::inOrder(AVLTreeNode<K, V>* tree)  {
    if(tree != nullptr){
        inOrder(tree->left);
        std::cout<< "(" << tree->key << "," << tree->val << ") ";
        inOrder(tree->right);
    }
}

template <class K, class V>
void PipeAVLTree<K, V>::inOrder(){
    inOrder(root);
}

template <class K, class V>
void PipeAVLTree<K, V>::postOrder(AVLTreeNode<K, V>* tree)  {
    if(tree != nullptr){
        postOrder(tree->left);
        postOrder(tree->right);
        std::cout<< "(" << tree->key << "," << tree->val << ") ";
    }
}

template <class K, class V>
void PipeAVLTree<K, V>::postOrder(){
    postOrder(root);
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::Search(AVLTreeNode<K, V>* x, K k, V v)  
{
    if (x == nullptr || (x->key==k && x->val == v))
        return x;

    if (com(k, v, x->key, x->val))
        return  Search(x->left, k, v);
    else
        return  Search(x->right, k ,v);
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::Search(K k, V v){
    return Search(root, k, v);
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::iterativeSearch(AVLTreeNode<K, V>* x, K k, V v)  {
    while ((x != nullptr) && (x->key != k || x->val != v)){
        if (com(k, v, x->key, x->val))
            x = x->left;
        else
            x = x->right;
    }
    return x;
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::iterativeSearch(K k, V v){
    return iterativeSearch(root, k, v);
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::minimum(AVLTreeNode<K, V>* tree){
    if (tree == nullptr)
        return nullptr;

    while(tree->left != nullptr)
        tree = tree->left;
    return tree;
}

template <class K, class V>
std::pair<K,V> PipeAVLTree<K, V>::minimum(){
    AVLTreeNode<K, V> *p = minimum(root);
    if (p != nullptr)
        return std::pair<K,V>(p->key, p->val);

    return std::pair<K,V>(K(-1), V(-1));
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::maximum(AVLTreeNode<K, V>* tree){
    if (tree == nullptr)
        return nullptr;

    while(tree->right != nullptr)
        tree = tree->right;
    return tree;
}

template <class K, class V>
std::pair<K,V> PipeAVLTree<K, V>::maximum(){
    AVLTreeNode<K, V> *p = maximum(root);
    if (p != nullptr)
        return std::pair<K,V>(p->key, p->val);

    return std::pair<K,V>(K(-1), K(-1));
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::LLRotate(AVLTreeNode<K, V>* k2){
    AVLTreeNode<K, V>* k1;

    k1 = k2->left;
    k2->left = k1->right;
    k1->right = k2;

    k2->height = std::max(height(k2->left), height(k2->right)) + 1;
    k1->height = std::max(height(k1->left), k2->height) + 1;

    return k1;
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::RRRotate(AVLTreeNode<K, V>* k1){
    AVLTreeNode<K, V>* k2;

    k2 = k1->right;
    k1->right = k2->left;
    k2->left = k1;

    k1->height = std::max(height(k1->left), height(k1->right)) + 1;
    k2->height = std::max(height(k2->right), k1->height) + 1;

    return k2;
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K,V>::LRRotate(AVLTreeNode<K, V>* k3)
{
    k3->left = RRRotate(k3->left);

    return LLRotate(k3);
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K,V>::RLRotate(AVLTreeNode<K, V>* k1)
{
    k1->right = LLRotate(k1->right);

    return RRRotate(k1);
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::insert(AVLTreeNode<K, V>* &tree, K k, V v){
    if (tree == nullptr){
        tree = new AVLTreeNode<K, V>(k, v, nullptr, nullptr);
        if (tree==nullptr){
            std::cout << "ERROR: create node failed!\n";
            FAISS_ASSERT(false);
        }
    }
    else if (com(k, v, tree->key, tree->val)){
        tree->left = insert(tree->left, k, v);
        if (height(tree->left) - height(tree->right) == 2){
            if (com(k, v, tree->left->key, tree->left->val))
                tree = LLRotate(tree);
            else
                tree = LRRotate(tree);
        }
    }
    else if (com(tree->key, tree->val, k, v)){
        tree->right = insert(tree->right, k, v);
        if (height(tree->right) - height(tree->left) == 2){
            if (com(tree->right->key, tree->right->val, k, v))
                tree = RRRotate(tree);
            else
                tree = RLRotate(tree);
        }
    }
    else
        FAISS_ASSERT_FMT(false, "%s","Do not insert an exsiting node");

    tree->height = std::max(height(tree->left), height(tree->right)) + 1;

    return tree;
}

template <class K, class V>
void PipeAVLTree<K, V>::insert(K k, V v){
    insert(root, k ,v);
    size++;
}

template <class K, class V>
AVLTreeNode<K, V>* PipeAVLTree<K, V>::remove(AVLTreeNode<K, V>* &tree, AVLTreeNode<K, V>* z)
{
    if (tree==nullptr || z==nullptr)
        return nullptr;

    if (com(z->key, z->val, tree->key, tree->val)){
        tree->left = remove(tree->left, z);
        if (height(tree->right) - height(tree->left) == 2){
            AVLTreeNode<K, V> *r = tree->right;
            if (height(r->left) > height(r->right))
                tree = RLRotate(tree);
            else
                tree = RRRotate(tree);
        }
    }
    else if (com(tree->key, tree->val, z->key, z->val)){
        tree->right = remove(tree->right, z);
        if (height(tree->left) - height(tree->right) == 2){
            AVLTreeNode<K, V> *l =  tree->left;
            if (height(l->right) > height(l->left))
                tree = LRRotate(tree);
            else
                tree = LLRotate(tree);
        }
    }
    else
    {
        if ((tree->left!=nullptr) && (tree->right!=nullptr)){
            if (height(tree->left) > height(tree->right)){
                AVLTreeNode<K, V>* maxv = maximum(tree->left);
                tree->key = maxv->key;
                tree->left = remove(tree->left, maxv);
            }
            else{
                AVLTreeNode<K, V>* minv = maximum(tree->right);
                tree->key = minv->key;
                tree->right = remove(tree->right, minv);
            }
        }
        else{
            AVLTreeNode<K, V>* tmp = tree;
            tree = (tree->left!=nullptr) ? tree->left : tree->right;
            delete tmp;
        }
    }
    return tree;
}

template <class K, class V>
void PipeAVLTree<K, V>::remove(K k, V v){
    AVLTreeNode<K, V>* z;
    if ((z = Search(root, k, v)) != nullptr){
        // std::cout << "Find \n";
        root = remove(root, z);
        size--;
    }
    else{
        // Do not delete an non-existing node
        FAISS_ASSERT(false);
    }
}

template <class K, class V>
void PipeAVLTree<K, V>::destroy(AVLTreeNode<K, V>* &tree)
{
    if (tree==nullptr)
        return;

    // Recursively destroy the tree
    if (tree->left != nullptr)
        destroy(tree->left);
    if (tree->right != nullptr)
        destroy(tree->right);

    delete tree;
}

template <class K, class V>
void PipeAVLTree<K, V>::destroy()
{
    destroy(root);
}

template <class K, class V>
void PipeAVLTree<K, V>::print(AVLTreeNode<K, V>* tree, K k, V v, int direction)
{
    if(tree != nullptr){
        if(direction==0)
            std::cout << std::setw(2) << tree->key << " is root" << std::endl;
        else
            std::cout << std::setw(2) << tree->key << " is " << std::setw(2) << k
                << "'s "  << std::setw(12) << (direction==1?"right child" : "left child") << std::endl;

        print(tree->left, tree->key, tree->val,-1);
        print(tree->right,tree->key, tree->val, 1);
    }
}

template <class K, class V>
void PipeAVLTree<K, V>::print()
{
    if (root != nullptr)
        print(root, root->key, root->val, 0);    
}

}