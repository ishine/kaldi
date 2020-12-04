// util/trie-tree.h

// Copyright 2015-2016   Shanghai Jiao Tong University (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_UTIL_TRIE_TREE_H_
#define KALDI_UTIL_TRIE_TREE_H_

#include <vector>
#include <list>
#include "util/stl-utils.h"

namespace kaldi {

typedef int TrieKey;
typedef float TrieWeight;
typedef int TrieClassType;
#define TRIE_NODE_SIZE  26
#define TRIE_NODE_MAX_SIZE  10000

class WordInfo {
public:
	int len_;
	TrieClassType class_;
	TrieWeight weight_;
	std::string word_;
	int	id_;
	WordInfo();
	std::string ToStr();
	void Clear();
};

class TrieNode {
public:
	std::unordered_map<TrieKey, TrieNode*> children_;
    std::vector<TrieNode*> children_vec_;
	TrieKey key_;
	bool is_word_;
    int num_child_;
    int layer_;
	WordInfo *info_;

	TrieNode();
	virtual ~TrieNode();
	TrieNode* GetNode(TrieKey key);
	TrieNode *AddNode(TrieKey key);
	int NumChild();
	std::string ToStr();
	void Clear();
};


class Trie {
public:
	Trie();
	virtual ~Trie();

	// If not present, inserts word ids into trie
	// If the word ids is prefix of trie node, just
	// marks leaf node
	TrieNode* Insert(std::vector<TrieKey> &bpe_ids, std::string& word,
			int word_id, TrieWeight weight, TrieClassType ctype);

	// Returns true if word presents in trie, else
	// false
	TrieNode* Search(std::vector<TrieKey> &word);

	bool LoadDict(std::string path);

	TrieNode *GetRootNode() { return &root_; }

	TrieNode* Trav(TrieNode* node, TrieKey key);


private:

	void split(std::string &str, const std::string &delim, std::vector<std::string> &strs);
	TrieNode root_;
};

}

#endif /* KALDI_UTIL_TRIE_TREE_H_ */
