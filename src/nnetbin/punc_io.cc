#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <vector>

using namespace std;

map<string, size_t> BuildVocab(ifstream &vocab_file) {
    map<string, size_t> word_to_id;
    string word;
    int i = 0;

    while (vocab_file >> word) {
        word_to_id[word] = i;
        i++;
    }

    return word_to_id;
}


vector<size_t> Transform(const string &words, const map<string, size_t> &vocab) {
    vector<size_t> ids;
    istringstream stream(words);
    string word;
    size_t id;

    while (stream >> word) {
        auto map_it = vocab.find(word);
        if (map_it == vocab.end()) {
            map_it = vocab.find("<unk>");
            if (map_it == vocab.end()) {
                std::cout << "Your Vocab Should include <unk>" << std::endl;
                exit(0);
            }
        }
        id = map_it->second;
        ids.push_back(id);
    }
    return ids;
}

void PrintVec(const vector<size_t> & ids) {
    for (int i = 0; i < ids.size(); ++i) {
        cout << ids[i] << " ";
    }
    cout << endl;
}

int main(int argc, char *argv[]) {
    // 1. Load vocab
    ifstream vocab_file(argv[1]);
    auto word_to_id = BuildVocab(vocab_file);

    for (auto b = word_to_id.begin(); b != word_to_id.end(); b++) {
        cout << b->first << " " << b->second << endl;
    }

    ifstream txt_file(argv[2]);
    string line;
    while (getline(txt_file, line)) {
        // 2. words to ids
        vector<size_t> ids;
        ids = Transform(line, word_to_id);
        PrintVec(ids);
    }

    return 0;
}
