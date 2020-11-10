// util/trie-tree-inl.h

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

#ifndef KALDI_UTIL_TRIE_TREE_INL_H_
#define KALDI_UTIL_TRIE_TREE_INL_H_


namespace kaldi {

////////////
/// WordInfo
////////////
WordInfo::WordInfo() {
	Clear();
}

void WordInfo::Clear() {
	len_ = 0;
	class_ = 0;
    weight_ = 0.0;
    word_.clear();
}

std::string WordInfo::ToStr() {
	char buffer[256] = {0};
	sprintf(buffer, "[word:%s, len: %d, ctype=%d, weight=%.3f]",
			word_.c_str(), len_, class_, weight_);
	return std::string(buffer);
}

////////////
/// TrieNode
////////////
TrieNode::TrieNode() {
	key_ = 0;
	is_word_ = false;
	info_ = NULL;
}

TrieNode::~TrieNode() {
	if (info_ != NULL) {
		delete info_;
		info_ = NULL;
	}
}

TrieNode *TrieNode::GetNode(TrieKey key) {
	auto it = children_.find(key);
	if (it == children_.end()) {
		return NULL;
	} else {
		return &(it->second);
	}
}

TrieNode *TrieNode::AddNode(TrieKey key) {
	TrieNode child;
	child.key_ = key;
	children_[key] = child;
	return &children_[key];
}

int TrieNode::NumChild() {
	return children_.size();
}

std::string TrieNode::ToStr() {
	char buffer[1024] = {0};
	sprintf(buffer, "[key:%s, is_word: %s, info: %s]",
			key_, is_word_?"true":"false", info_==NULL ? "NULL":info_->ToStr().c_str());
	return std::string(buffer);
}

void TrieNode::Clear() {
	key_ = 0;
	is_word_ = false;
	if (info_ != NULL) {
		delete info_;
		info_ = NULL;
	}
	children_.clear();
}


//////////////
/// Trie tree
//////////////
Trie::Trie() {

}

Trie::~Trie() {

}

bool Trie::LoadDict(std::string path) {
	root_.Clear();
	root_.key_ = '^';

	if (path == "")
		return false;

	std::ifstream ifs(path);
	if (!ifs.is_open())
		return false;

	std::string line, word;
	std::vector<std::string> strs, ids;
	std::vector<TrieKey> word_ids;
	TrieWeight weight;
	TrieClassType ctype;
	int num = 0, size;
	while(getline(ifs, line)) {
		split(line, "\t", strs);
		size = strs.size();
		if (size < 2) {
			printf("line err: %d, %s\n", num, line);
			return false;
		}

		if (size >= 2) {
			word = strs[0];
			split(strs[1], " ", ids);
			word_ids.resize(ids.size());
			for (int i = 0; i < ids.size(); i++)
				word_ids[i] = std::stoi(ids[i]);
		}

		ctype = 0;
		if (size >= 3) {
			ctype = std::stoi(strs[2]);
		}

		weight = 0;
		if (size >= 4) {
			weight = std::stof(strs[3]);
		}
		/// build trie tree
		this->Insert(word_ids, word, ctype, weight);
	}

	ifs.close();
	return true;
}

void Trie::split(std::string &str, const std::string &delim, std::vector<std::string> &strs) {
	size_t start, end = str.find(delim);
	strs.clear();
	while (end != std::string::npos) {
		strs.push_back(str.substr(start, end-start));
		start = end + delim.length();
		end = str.find(delim);
	}
	strs.push_back(str.substr(start, end-start));
}

TrieNode* Trie::Insert(std::vector<TrieKey> &word_ids,
		std::string& word, TrieWeight weight, TrieClassType ctype) {
	TrieNode *p = &root_, *np = p;

	for (int i = 0; i < word_ids.size(); i++) {
		TrieKey key = word_ids[i];
		np = p->GetNode(key);
		if (np == NULL)
			np = p->AddNode(key);
		p = np;
	}

	// mark last node as leaf
	if (p->is_word_ == false && p->info_ == NULL) {
		p->is_word_ = true;
		p->info_ = new WordInfo;
		p->info_->class_ = ctype;
		p->info_->len_ = word_ids.size();
		p->info_->weight_ = weight;
		p->info_->word_ = word;
	}

	return p;
}

TrieNode* Trie::Search(std::vector<TrieKey> &word) {
	TrieNode *p = &root_, *np = p;

	for (int i = 0; i < word.size(); i++) {
		TrieKey key = word[i];
		p = p->GetNode(key);
		if (p == NULL)
			return NULL;
	}
	return p;
}

const TrieNode* Trie::Trav(TrieNode* node, TrieKey key) {
	if (node == NULL)
		return NULL;
	return node->GetNode(key);
}

}

#endif /* UTIL_TRIE_TREE_INL_H_ */
