// The unknown-word model: letters -> CMUdict phonemes with stress, in one
// forward pass. Letter embeddings (each letter has `slots` positions), a
// bidirectional GRU and a CTC output layer read greedily. Trained in Python;
// the arithmetic follows dizione's g2p.py.
//
// Part of the dizione C++ port. Standard library only.
#pragma once

#include "json.h"
#include "text.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace dizione {

// tiny-voca esp32: optional hook that runs fn(a) and fn(b), possibly on two cores at once
// (the two GRU directions are independent). Null: run one after the other.
extern void (*g2p_parallel)(void (*fn)(void *), void *a, void *b);

struct Tensor {
    std::vector<size_t> shape;
    std::vector<float> values;          // row-major
};
// By the names of g2p.npz: "embed__weight", "slot", "gru__weight_ih_l0", ...
using Tensors = std::unordered_map<std::string, Tensor>;

class G2p {
public:
    // `settings` is g2p.json.
    G2p(const json::Value & settings, Tensors weights);

    // The word reduced to the letters the model knows (accents removed).
    Str spellable(StrView word) const;
    // CMUdict symbols with stress digits for `word`.
    std::vector<std::string> predict(StrView word) const;

private:
    const Tensor & tensor(const std::string & name, std::vector<size_t> shape) const;
    std::vector<std::string> predict_spellable(const Str & word) const;

    std::unordered_map<char32_t, int> letters_;
    std::vector<std::string> symbols_;
    long slots_ = 0;
    long layers_ = 0;
    long hidden_ = 0;
    Tensors weights_;
    // tiny-voca esp32: predictions already made (bounded), for unknown words that repeat
    mutable std::unordered_map<Str, std::vector<std::string>> cache_;
};

}  // namespace dizione
