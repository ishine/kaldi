// util/trie-tree.cc

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

#include "util/trie-tree.h"

namespace kaldi {

////////////
/// WordInfo
////////////
WordInfo::WordInfo() {
	Clear();
}

void WordInfo::Clear() {
	len_ = 0;
	id_ = -1;
	class_ = 0;
    weight_ = 0.0;
    word_.clear();
}

std::string WordInfo::ToStr() {
	char buffer[256] = {0};
	sprintf(buffer, "[word:%s, len=%d, id=%d, ctype=%d, weight=%.3f]",
			word_.c_str(), len_, id_, class_, weight_);
	return std::string(buffer);
}

////////////
/// TrieNode
////////////
TrieNode::TrieNode() {
	key_ = -1;
	is_word_ = false;
    num_child_ = 0;
    layer_ = -1;
	info_ = NULL;
}

TrieNode::~TrieNode() {
	if (info_ != NULL) {
		delete info_;
		info_ = NULL;
	}

    for (auto &node : children_)
        delete node.second;

    for (int i = 0; i < children_vec_.size(); i++) {
        if (children_vec_[i] != NULL)
            delete children_vec_[i];
    }
}

TrieNode *TrieNode::GetNode(TrieKey key) {
    TrieNode *node = NULL;

    if (num_child_ > TRIE_NODE_SIZE) {
        node = children_vec_[key];
    } else {
        auto it = children_.find(key);
	    if (it != children_.end())
            node = it->second;
    }
        
    return node;
}

TrieNode *TrieNode::AddNode(TrieKey key) {

    if (num_child_ == 0)
        children_.reserve(TRIE_NODE_SIZE);

	TrieNode *child = new TrieNode;
    child->key_ = key;
    num_child_++;

    if (num_child_ <= TRIE_NODE_SIZE) {
        children_[key] = child;
    } else {
        if (num_child_ == TRIE_NODE_SIZE+1) {
            children_vec_.resize(TRIE_NODE_MAX_SIZE, NULL);
            for (auto &node : children_)
                children_vec_[node.first] = node.second;
            children_.clear();
        }
        KALDI_ASSERT(key < TRIE_NODE_MAX_SIZE);
        children_vec_[key] = child; 
    }

	return child;
}

int TrieNode::NumChild() {
	return num_child_;
}

std::string TrieNode::ToStr() {
	char buffer[1024] = {0};
	sprintf(buffer, "[key: %d, is_word: %s, layer: %d, info: %s]",
			key_, is_word_?"true":"false", layer_, info_==NULL ? "NULL":info_->ToStr().c_str());
	return std::string(buffer);
}


//////////////
/// Trie tree
//////////////
Trie::Trie() {

}

Trie::~Trie() {

}

bool Trie::LoadDict(std::string path) {
	root_.key_ = '^';

	if (path == "")
		return false;

	std::ifstream ifs(path);
	if (!ifs.is_open())
		return false;

	std::string line, word;
	std::vector<std::string> strs, ids;
	std::vector<TrieKey> bpe_ids;
	TrieWeight weight;
	TrieClassType ctype;
	int num = 0, size, word_id = 0;
	while(getline(ifs, line)) {
		split(line, "\t", strs);
		size = strs.size();
		if (size < 2) {
            std::cout << "line err: " << num << ", " << line << std::endl;
			return false;
		}

        // word and bpe ids
		if (size >= 2) {
			word = strs[0];
			split(strs[1], " ", ids);
			bpe_ids.resize(ids.size());
			for (int i = 0; i < ids.size(); i++)
				bpe_ids[i] = std::stoi(ids[i]);
		}

        // word id
		word_id = -1;
		if (size >= 3) {
			word_id = std::stoi(strs[2]);
		}

        // word class type
		ctype = 0;
		if (size >= 4) {
			ctype = std::stoi(strs[3]);
		}

        // word weight
		weight = 0;
		if (size >= 5) {
			weight = std::stof(strs[4]);
		}
		/// build trie tree
		this->Insert(bpe_ids, word, word_id, ctype, weight);
	}

	ifs.close();
	return true;
}

void Trie::split(std::string &str, const std::string &delim, std::vector<std::string> &strs) {
	size_t start = 0, end = str.find(delim);
	strs.clear();
	while (end != std::string::npos) {
		strs.push_back(str.substr(start, end-start));
		start = end + delim.length();
		end = str.find(delim, start);
	}
	strs.push_back(str.substr(start, end-start));
}

TrieNode* Trie::Insert(std::vector<TrieKey> &bpe_ids, std::string& word,
		int word_id, TrieWeight weight, TrieClassType ctype) {
	TrieNode *p = &root_, *np = p;

	for (int i = 0; i < bpe_ids.size(); i++) {
		TrieKey key = bpe_ids[i];
		np = p->GetNode(key);
		if (np == NULL) {
			np = p->AddNode(key);
            np->layer_ = i;
        }
		p = np;
	}

	// mark last node as leaf
	if (p->is_word_ == false && p->info_ == NULL) {
		p->is_word_ = true;
		p->info_ = new WordInfo;
		p->info_->class_ = ctype;
		p->info_->len_ = bpe_ids.size();
		p->info_->weight_ = weight;
		p->info_->word_ = word;
		p->info_->id_ = word_id;
	}

	return p;
}

TrieNode* Trie::Search(std::vector<TrieKey> &word) {
	TrieNode *p = &root_;

	for (int i = 0; i < word.size(); i++) {
		TrieKey key = word[i];
		p = p->GetNode(key);
		if (p == NULL)
			return NULL;
	}
	return p;
}

TrieNode* Trie::Trav(TrieNode* node, TrieKey key) {
	if (node == NULL)
		return NULL;
	return node->GetNode(key);
}

}
