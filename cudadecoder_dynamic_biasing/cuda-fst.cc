// cudadecoder/cuda-fst.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun, Justin Luitjens, Ryan Leary
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright (c) 2022-2023, Idiap Research Institute (http://www.idiap.ch/)
//
// @author: Srikanth Madikeri (srikanth.madikeri@idiap.ch), Iuliia Nigmatulina (iuliia.nigmatulina@idiap.ch)
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation, either version 3 of the License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
// See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see https://www.gnu.org/licenses/.

#if HAVE_CUDA == 1

#include "cudadecoder/cuda-fst.h"

#include <cuda_runtime_api.h>
#include <nvToolsExt.h>

namespace kaldi {
namespace cuda_decoder {

void CudaFst::ComputeOffsets(const fst::Fst<StdArc> &fst) {
  // count states since Fst doesn't provide this functionality
  num_states_ = 0;
  for (fst::StateIterator<fst::Fst<StdArc> > iter(fst); !iter.Done();
       iter.Next())
    ++num_states_;

  // allocate and initialize offset arrays
  h_final_.resize(num_states_);
  h_e_offsets_.resize(num_states_ + 1);
  h_ne_offsets_.resize(num_states_ + 1);

  // iterate through states and arcs and count number of arcs per state
  e_count_ = 0;
  ne_count_ = 0;

  // Init first offsets
  h_ne_offsets_[0] = 0;
  h_e_offsets_[0] = 0;
  for (int i = 0; i < num_states_; i++) {
    h_final_[i] = fst.Final(i).Value();
    // count emiting and non_emitting arcs
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int32 ilabel = arc.ilabel;
      if (ilabel != 0) {  // emitting
        e_count_++;
      } else {  // non-emitting
        ne_count_++;
      }
    }
    h_ne_offsets_[i + 1] = ne_count_;
    h_e_offsets_[i + 1] = e_count_;
  }

  // We put the emitting arcs before the nonemitting arcs in the arc list
  // adding offset to the non emitting arcs
  // we go to num_states_+1 to take into account the last offset
  for (int i = 0; i < num_states_ + 1; i++)
    h_ne_offsets_[i] += e_count_;  // e_arcs before

  arc_count_ = e_count_ + ne_count_;
}

void CudaFst::AllocateData(const fst::Fst<StdArc> &fst) {
  d_e_offsets_ = static_cast<unsigned int *>(CuDevice::Instantiate().Malloc(
      (num_states_ + 1) * sizeof(*d_e_offsets_)));
  d_ne_offsets_ = static_cast<unsigned int *>(CuDevice::Instantiate().Malloc(
      (num_states_ + 1) * sizeof(*d_ne_offsets_)));
  d_final_ = static_cast<float *>(
      CuDevice::Instantiate().Malloc((num_states_) * sizeof(*d_final_)));

  h_arc_weights_.resize(arc_count_);
  h_arc_nextstate_.resize(arc_count_);
  // ilabels (id indexing)
  h_arc_id_ilabels_.resize(arc_count_);
  h_arc_olabels_.resize(arc_count_);

  d_arc_weights_ = static_cast<float *>(
      CuDevice::Instantiate().Malloc(arc_count_ * sizeof(*d_arc_weights_)));
  d_arc_nextstates_ = static_cast<StateId *>(
      CuDevice::Instantiate().Malloc(arc_count_ * sizeof(*d_arc_nextstates_)));

  // Only the ilabels for the e_arc are needed on the device
  d_arc_pdf_ilabels_ = static_cast<int32 *>(
      CuDevice::Instantiate().Malloc(e_count_ * sizeof(*d_arc_pdf_ilabels_)));
}

void CudaFst::PopulateArcs(const fst::Fst<StdArc> &fst) {
  // now populate arc data
  int e_idx = 0;
  int ne_idx = e_count_;  // starts where e_offsets_ ends
  for (int i = 0; i < num_states_; i++) {
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int idx;
      if (arc.ilabel != 0) {  // emitting
        idx = e_idx++;
      } else {
        idx = ne_idx++;
      }
      // KALDI_LOG << "PopulateArcs without rescoring: arc weights: " ;
      // KALDI_LOG << arc.weight.Value() ;
      h_arc_weights_[idx] = arc.weight.Value();
      h_arc_nextstate_[idx] = arc.nextstate;
      h_arc_id_ilabels_[idx] = arc.ilabel;
      // For now we consider id indexing == pdf indexing
      // If the two are differents, we'll call ApplyTransModelOnIlabels with a
      // TransitionModel
      h_arc_pdf_ilabels_[idx] = arc.ilabel;
      h_arc_olabels_[idx] = arc.olabel;
    }
  }
}

void CudaFst::PopulateArcs(const fst::Fst<StdArc> &fst,
                           const fst::Fst<fst::StdArc> &rescore_fst) {
  KALDI_LOG << "Partial boosting" ;
  // now populate arc data
  int e_idx = 0;
  int ne_idx = e_count_;  // starts where e_offsets_ ends
  // we only have one boosting fst
  h_boosted_arc_idx_.resize(1);

  // preprocess rescore fst properties
  int32 num_states_boosting_fst = 0;
  for (fst::StateIterator<fst::Fst<StdArc> > iter(rescore_fst); !iter.Done();
       iter.Next()) {
      num_states_boosting_fst++;
  }
  std::set<int32> unigram_tokens_to_boost;
  // get unique ilabels
  for (int i = 0; i < num_states_boosting_fst; i++) {
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(rescore_fst, i); !aiter.Done();
         aiter.Next()) {
        StdArc arc = aiter.Value();
        if (arc.weight != 0) {
          unigram_tokens_to_boost.insert(arc.ilabel);
        }        
    }
  }

  for (int i = 0; i < num_states_; i++) {
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int idx;
      if (arc.ilabel != 0) {  // emitting
        idx = e_idx++;
      } else {
        idx = ne_idx++;
      }

      h_arc_weights_[idx] = arc.weight.Value() ;
      if (unigram_tokens_to_boost.count(arc.olabel)){
        h_boosted_arc_idx_[0].push_back(idx);
      }
      h_arc_nextstate_[idx] = arc.nextstate;
      h_arc_id_ilabels_[idx] = arc.ilabel;
      // For now we consider id indexing == pdf indexing
      // If the two are differents, we'll call ApplyTransModelOnIlabels with a
      // TransitionModel
      h_arc_pdf_ilabels_[idx] = arc.ilabel;
      h_arc_olabels_[idx] = arc.olabel;
    }
  }
}


struct BFSState_ {
        // Empty constructor. Hopefully never used. But was required for compilation.
        BFSState_() {
            is_first_ = true;
            state_ = 0;
            prev_state_ = 0;
            output_ = 0;
            token_idx_ = 0;
        }

        BFSState_(uint32 state, uint32 prev_state, uint32 output, uint32 token_idx, bool is_first=false) {
            is_first_ = is_first;
            state_ = state;
            prev_state_ = prev_state;
            output_ = output;
            token_idx_ = token_idx;
        }

        bool operator< (const BFSState_& other) const {
            return state_ < other.state_ 
                    && prev_state_ < other.prev_state_
                    && output_ < other.output_
                    && token_idx_ < other.token_idx_;
        }

        uint32 state_;
        uint32 prev_state_;
        bool is_first_ = true;
        uint32 output_;
        uint32 token_idx_;
};
typedef struct BFSState_ BFSState;
typedef std::vector<BFSState> Path;


inline bool is_final_state(const fst::Fst<StdArc> &f, uint32 stateID) {
      return f.Final(stateID) != StdArc::Weight::Zero();
}


void reset_path(Path &p) {
      p.clear();
}

inline void refine_path(Path &p, const std::vector<uint32>& words, const fst::Fst<StdArc> &f) {
      // ignoring the last token because it will be a 0
      uint32 seq_len = words.size()-1;
      if (seq_len == 0) {
          reset_path(p);
          return;
      }
      uint32 max_idx = p.size()-1;
      while (max_idx>0) {
          if(is_final_state(f, p[max_idx].state_))
              break;
          max_idx--;
      }
      if (max_idx == 0) {
          // only one state
          reset_path(p);
          return;
      }
      else if(p[max_idx].token_idx_ >= seq_len) {
          p.resize(max_idx+1);
          return;
      }
      reset_path(p);
      return;
}

void update_arcs_to_boost_from_path(const fst::Fst<StdArc> &fst, const Path &curr_path, std::set<std::tuple<uint32, uint32, uint32, uint32> > &arcs_to_boost) {
  for(auto s: curr_path) {
    if(s.output_ == 0)
        continue;
    for(fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, s.prev_state_); !aiter.Done(); aiter.Next()) {
        StdArc arc = aiter.Value();
        if (arc.nextstate == s.state_ && arc.olabel == s.output_) {
            auto arc_id = std::make_tuple(s.prev_state_, arc.nextstate, arc.ilabel, arc.olabel);
            arcs_to_boost.insert(arc_id);
            break;
        }
    }
  }
}


void get_all_arcs_with_output(const fst::Fst<StdArc> &fst, const uint32 num_states, uint32 output, std::vector<std::tuple<uint32, uint32, uint32, uint32> > &arcs) {
  for (int i = 0; i < num_states; i++) {
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      if(arc.olabel != output)
        continue;
      auto arc_id = std::make_tuple(i, arc.nextstate, arc.ilabel, arc.olabel);
      arcs.push_back(arc_id);
    }
  }
}


void reach_final_state(const fst::Fst<StdArc> &fst, uint32 stateID, std::set<uint32> &final_states) {
  if(is_final_state(fst, stateID)) {
    final_states.insert(stateID);
    return;
  }
  std::queue<uint32> active_states;
  std::set<uint32> visited_states;
  active_states.push(stateID);
  visited_states.insert(stateID);
  while(active_states.size()>0) {
    auto next_state = active_states.front();
    active_states.pop();
    for(fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, next_state); !aiter.Done(); aiter.Next()) {
      StdArc arc = aiter.Value();
      if(visited_states.count(arc.nextstate) > 0)
        continue;
      if(arc.olabel != 0)
        continue;
      if(is_final_state(fst, arc.nextstate)) {
        final_states.insert(arc.nextstate);
      }
      else {
        active_states.push(arc.nextstate);
      }
      visited_states.insert(arc.nextstate);
    }
  }
}


void get_arcs_to_boost_unigram(const fst::Fst<StdArc> &fst, uint32 num_states, const std::vector<uint32>& words, std::set<std::tuple<uint32, uint32, uint32, uint32> > &arcs_to_boost) {
    if(words.size() != 1) {
      KALDI_WARN << "Calling unigram boosting function with more than one word. No words will be boosted";
      return;
    }
    auto first_word = words[0];
    std::vector<std::tuple<uint32, uint32, uint32, uint32> > arcs_with_first_word;
    get_all_arcs_with_output(fst, num_states, first_word, arcs_with_first_word);
    for(auto arc: arcs_with_first_word) arcs_to_boost.insert(arc);
}


void print_current_path(const Path &curr_path) {
  for(auto s: curr_path) {
    KALDI_WARN << "State: " << s.state_;
  }
}

// create a function that takes a vector of uint32 and returns a string concatenated with underscores between each word
// this function will be used to print the words in the sequence
std::string get_sequence_string(const std::vector<uint32>& words) {
  std::string seq_str = "";
  for(auto w: words) {
    seq_str += std::to_string(w) + "_";
  }
  return seq_str;
}

// in this function we have only one input sequence.
void get_arcs_to_boost_sequence(const fst::Fst<StdArc> &fst, const uint32 num_states, const std::vector<uint32>& words, std::set<std::tuple<uint32, uint32, uint32, uint32> > &arcs_to_boost) {
      if(words.size() == 1) {
        KALDI_WARN << "Input sequence has only one word. Calling unigram boosting function instead";
        get_arcs_to_boost_unigram(fst, num_states, words, arcs_to_boost);
      }
      else if(words.size() == 0) {
        KALDI_WARN << "Input sequence is empty. No words will be boosted.";
        return;
      }
      // step 1. for the first word in the sequence, get all arcs that start from the initial state
      auto first_word = words[0];
      std::set<uint32> states_reached_for_first_word;
      std::set<uint32> final_states_reached_after_first_word;
      std::vector<std::tuple<uint32, uint32, uint32, uint32> > arcs_with_first_word;
      get_all_arcs_with_output(fst, num_states, first_word, arcs_with_first_word);
      KALDI_LOG << "Got " << arcs_with_first_word.size() << " arcs with output " << first_word;
      for(auto arc: arcs_with_first_word) {
        states_reached_for_first_word.insert(std::get<0>(arc));
      }
      for(auto arc: arcs_with_first_word) {
        // for each arc go to the nearest final state
        std::set<uint32> final_states_reached_from_arc;
        reach_final_state(fst, std::get<1>(arc), final_states_reached_from_arc);
        if(final_states_reached_from_arc.size() > 0) {
          // add all states in final_states_reached_from_arc to final_states_reached_after_first_word
          final_states_reached_after_first_word.insert(final_states_reached_from_arc.begin(), final_states_reached_from_arc.end());
          // add the arc from std::get<0>(arc) to std::get<1>(arc) to arcs_to_boost
          arcs_to_boost.insert(arc);
        }
      }
      KALDI_LOG << "Got " << states_reached_for_first_word.size() << " unique states reached to start first word";
      KALDI_LOG << "180252 in those unique states: " << (states_reached_for_first_word.count(180252) > 0);


      // alternative step 2. instead of maintaing a queue for the whole sequence; go from 
      // the first word to the last word and maintain a visited set of states per word.
      {
        std::vector<uint32> rest_of_sequence;
        uint32 num_paths_found = 0;
        for(uint32 j=0; j<words.size(); j++) rest_of_sequence.push_back(words[j]);  
        rest_of_sequence.push_back(0);
        // declare a set of states reached for the previous token
        std::set<uint32> states_reached_for_prev_token;
        // copy the contents from states_reached_for_first_word to states_reached_for_prev_token
        states_reached_for_prev_token.insert(
          states_reached_for_first_word.begin(),
          states_reached_for_first_word.end()
        );
        for(uint32 token_idx = 1; token_idx < rest_of_sequence.size()-1; token_idx++) {
          uint32 curr_token = rest_of_sequence[token_idx];
          uint32 prev_token = rest_of_sequence[token_idx-1];
          KALDI_LOG << "Current token is " << curr_token << " and previous token is " << prev_token;
          std::set<uint32> states_reached_for_curr_token;
          std::set<uint32> visited;
          uint32 num_reachable = 0;
          for(auto start_state: states_reached_for_prev_token) {
            // get all arcs with output curr_token from start_state
            std::deque<uint32> queue;
            std::map<uint32, uint32> active_states;
            std::set<std::tuple<uint32, uint32, uint32, uint32> > possible_arcs_to_boost;
            for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, start_state); !aiter.Done(); aiter.Next()) {
                StdArc arc = aiter.Value();
                if(arc.olabel == prev_token) {
                  queue.push_back(arc.nextstate);
                  active_states[arc.nextstate] += 1;
                  possible_arcs_to_boost.insert(std::make_tuple(start_state, arc.nextstate, arc.ilabel, arc.olabel));
                }
            }
            bool reachable = false;
            while(queue.size()>0) {
              uint32 curr_state = queue.back();
              queue.pop_back();
              if(visited.count(curr_state)>0) {
                if(active_states[curr_state]>0) {
                  active_states[curr_state] -= 1;
                }
                if(active_states[curr_state] == 0) {
                  active_states.erase(curr_state);
                }
                continue;
              }
              visited.insert(curr_state);
              for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, curr_state); !aiter.Done(); aiter.Next()) {
                StdArc arc = aiter.Value();
                // it is quite possible that this state has been visited before. we don't want to progress in this
                // direction anymore since there may be another path that already explored this state
                if(arc.olabel == 0 && visited.count(arc.nextstate)==0) {
                    queue.push_back(arc.nextstate);
                    active_states[arc.nextstate] += 1;
                }
                else if(curr_token == arc.olabel) {
                    // we have successfully reached the next token
                    // add this arc to the arcs_to_boost
                    // add the state to states_reached_for_curr_token
                    states_reached_for_curr_token.insert(curr_state);
                    for(auto possible_arc: possible_arcs_to_boost) {
                      arcs_to_boost.insert(possible_arc);
                    }
                    reachable = true;
                }
              }
              active_states[curr_state] -= 1;
              if(active_states[curr_state] == 0) {
                active_states.erase(curr_state);
              }
            }
            if(reachable) {
              num_reachable += 1;
            }
          }
          // print no of new states reached
          KALDI_LOG << "Got " << states_reached_for_curr_token.size() << " unique states reached for token " << curr_token;
          KALDI_LOG << "Num reachable from " << prev_token << " to " << curr_token << " is " << num_reachable << " out of " << states_reached_for_prev_token.size();
          // clear states_reached_for_prev_token
          states_reached_for_prev_token.clear();
          // copy the contents from states_reached_for_curr_token to states_reached_for_prev_token
          states_reached_for_prev_token.insert(
            states_reached_for_curr_token.begin(),
            states_reached_for_curr_token.end()
          );
          states_reached_for_curr_token.clear();

          // for the last token, for each state in states_reached_for_prev_token, add an arc to the final state if the arc outputs curr_token
          if(token_idx == rest_of_sequence.size()-2) {
            for(auto start_state: states_reached_for_prev_token) {
              for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, start_state); !aiter.Done(); aiter.Next()) {
                StdArc arc = aiter.Value();
                if(arc.olabel == curr_token) {
                  // arcs_to_boost.insert(arc);
                  arcs_to_boost.insert(std::make_tuple(start_state, arc.nextstate, arc.ilabel, arc.olabel));

                  // now once we have all the arcs to boost, take the arcs that were boosted for the final token, and try to reach all possible final states from
                  // there. if we can reach a final state, then we can boost the arc to the final state as well
                  /*std::deque<uint32> queue;
                  std::map<uint32, uint32> active_states;
                  std::set<uint32> visited;
                  queue.push_back(arc.nextstate);
                  active_states[arc.nextstate] += 1;
                  while(queue.size()>0) {
                    uint32 curr_state = queue.back();
                    queue.pop_back();
                    if(visited.count(curr_state)>0) {
                      if(active_states[curr_state]>0) {
                        active_states[curr_state] -= 1;
                      }
                      if(active_states[curr_state] == 0) {
                        active_states.erase(curr_state);
                      }
                      continue;
                    }
                    visited.insert(curr_state);
                    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, curr_state); !aiter.Done(); aiter.Next()) {
                      StdArc arc = aiter.Value();
                      // it is quite possible that this state has been visited before. we don't want to progress in this
                      // direction anymore since there may be another path that already explored this state
                      if(arc.olabel == 0 && visited.count(arc.nextstate)==0) {
                          queue.push_back(arc.nextstate);
                          active_states[arc.nextstate] += 1;
                      }
                      else if(is_final_state(fst, arc.nextstate)) {
                          // we have successfully reached the next token
                          // add this arc to the arcs_to_boost
                          // add the state to states_reached_for_curr_token
                          arcs_to_boost.insert(std::make_tuple(curr_state, arc.nextstate, arc.ilabel, arc.olabel));
                      }
                    }
                    active_states[curr_state] -= 1;
                    if(active_states[curr_state] == 0) {
                      active_states.erase(curr_state);
                    }
                  }*/
                }
              }
            }
          }
        }

      }
}


void get_arcs_to_boost(const fst::Fst<StdArc> &f, const uint32 num_states, const std::vector<uint32>& words, std::set<std::tuple<uint32, uint32, uint32, uint32> > &arcs_to_boost) {
      if(words.size() == 0)
          return;
      else if(words.size() == 1) {
          // only one token
          get_arcs_to_boost_unigram(f, num_states, words, arcs_to_boost);
      }
      else {
          get_arcs_to_boost_sequence(f, num_states, words, arcs_to_boost);
      }
}


void CudaFst::PopulateArcs(const fst::Fst<StdArc> &fst,
                           uint32 ngram_size,
                           const std::vector<std::vector<uint32> > &rescore_str) {
  KALDI_LOG << "Partial boosting for str ";
  // now populate arc data
  int e_idx = 0;
  int ne_idx = e_count_;  // starts where e_offsets_ ends
  // we only have one boosting fst
  h_boosted_arc_idx_.resize(1);

  std::set<int32> tokens_to_boost;
  uint32 num_words = rescore_str.size();

  auto startId = fst.Start();
  std::set<std::tuple<uint32, uint32, uint32, uint32> > arcs_to_boost;
  // for each sequence to be boosted

  KALDI_LOG << "Getting boosting arcs for " << rescore_str.size() << " sequences";
  for(uint32 ridx=0; ridx<rescore_str.size(); ridx++) {
    // for each n-gram size from 1, 2, ..., ngram_size
    KALDI_LOG << "Getting boosting arc for sequence no." << ridx;
    num_words = rescore_str[ridx].size();
    get_arcs_to_boost(fst, num_states_, rescore_str[ridx], arcs_to_boost);
    KALDI_LOG << "NUmber of arcs to boost after sequence " << ridx << " is " << arcs_to_boost.size();
  }

  KALDI_LOG << "Num of arcs to be boosted : " << arcs_to_boost.size() ;
  int total = 0;
  for (int i = 0; i < num_states_; i++) {
    for (fst::ArcIterator<fst::Fst<StdArc> > aiter(fst, i); !aiter.Done();
         aiter.Next()) {
      StdArc arc = aiter.Value();
      int idx;
      if (arc.ilabel != 0) {  // emitting
        idx = e_idx++;
      } else {
        idx = ne_idx++;
      }

      h_arc_weights_[idx] = arc.weight.Value() ;
      auto arc_id = std::make_tuple(i, arc.nextstate, arc.ilabel, arc.olabel);
      if (arcs_to_boost.count(arc_id)) {
        h_boosted_arc_idx_[0].push_back(idx);
        total++;
      }
      h_arc_nextstate_[idx] = arc.nextstate;
      h_arc_id_ilabels_[idx] = arc.ilabel;
      // For now we consider id indexing == pdf indexing
      // If the two are differents, we'll call ApplyTransModelOnIlabels with a
      // TransitionModel
      h_arc_pdf_ilabels_[idx] = arc.ilabel;
      h_arc_olabels_[idx] = arc.olabel;
    }
  }
  KALDI_LOG << "TOtal arcs actually boosted " << total;
}

void CudaFst::ApplyTransitionModelOnIlabels(
    const TransitionModel &trans_model) {
  // Converting ilabel here, to avoid reindexing when reading nnet3 output
  // We only need to convert the emitting arcs
  // The emitting arcs are the first e_count_ arcs
  for (int iarc = 0; iarc < e_count_; ++iarc)
    h_arc_pdf_ilabels_[iarc] =
        trans_model.TransitionIdToPdf(h_arc_id_ilabels_[iarc]);
}

void CudaFst::CopyDataToDevice() {
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_e_offsets_, &h_e_offsets_[0], (num_states_ + 1) * sizeof(*d_e_offsets_),
      cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_ne_offsets_, &h_ne_offsets_[0],
      (num_states_ + 1) * sizeof(*d_ne_offsets_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(d_final_, &h_final_[0],
                                                num_states_ * sizeof(*d_final_),
                                                cudaMemcpyHostToDevice));

  KALDI_DECODER_CUDA_API_CHECK_ERROR(
      cudaMemcpy(d_arc_weights_, &h_arc_weights_[0],
                 arc_count_ * sizeof(*d_arc_weights_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_arc_nextstates_, &h_arc_nextstate_[0],
      arc_count_ * sizeof(*d_arc_nextstates_), cudaMemcpyHostToDevice));
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_arc_pdf_ilabels_, &h_arc_pdf_ilabels_[0],
      e_count_ * sizeof(*d_arc_pdf_ilabels_), cudaMemcpyHostToDevice));
}

void CudaFst::Initialize(const fst::Fst<StdArc> &fst,
                         const TransitionModel *trans_model) {
  nvtxRangePushA("CudaFst constructor");
  start_ = fst.Start();

  ComputeOffsets(fst);
  AllocateData(fst);
  // Temporarily allocating data for this vector
  // We just need it during CSR generation. We will clear it
  // at the end of Initialize
  h_arc_pdf_ilabels_.resize(arc_count_);
  PopulateArcs(fst);
  if (trans_model) ApplyTransitionModelOnIlabels(*trans_model);

  KALDI_ASSERT(d_e_offsets_);
  KALDI_ASSERT(d_ne_offsets_);
  KALDI_ASSERT(d_final_);
  KALDI_ASSERT(d_arc_weights_);
  KALDI_ASSERT(d_arc_nextstates_);
  KALDI_ASSERT(d_arc_pdf_ilabels_);

  CopyDataToDevice();

  // Making sure the graph is ready
  cudaDeviceSynchronize();
  KALDI_DECODER_CUDA_CHECK_ERROR();
  h_arc_pdf_ilabels_.clear();  // we don't need those on host
  nvtxRangePop();
}

void CudaFst::Initialize(const fst::Fst<StdArc> &fst, uint32 ngram_size, std::vector<std::vector<uint32> > &rescore_str, 
                         const TransitionModel *trans_model) {
  nvtxRangePushA("CudaFst constructor");
  start_ = fst.Start();

  ComputeOffsets(fst);
  AllocateData(fst);
  // Temporarily allocating data for this vector
  // We just need it during CSR generation. We will clear it
  // at the end of Initialize
  h_arc_pdf_ilabels_.resize(arc_count_);
  PopulateArcs(fst, ngram_size, rescore_str);
  CopyBoostingArcs();
  if (trans_model) ApplyTransitionModelOnIlabels(*trans_model);

  KALDI_ASSERT(d_e_offsets_);
  KALDI_ASSERT(d_ne_offsets_);
  KALDI_ASSERT(d_final_);
  KALDI_ASSERT(d_arc_weights_);
  KALDI_ASSERT(d_arc_nextstates_);
  KALDI_ASSERT(d_arc_pdf_ilabels_);

  CopyDataToDevice();

  // Making sure the graph is ready
  cudaDeviceSynchronize();
  KALDI_DECODER_CUDA_CHECK_ERROR();
  h_arc_pdf_ilabels_.clear();  // we don't need those on host
  nvtxRangePop();
}

void CudaFst::Initialize(const fst::Fst<StdArc> &fst,
                         const fst::Fst<fst::StdArc> &rescore_fst,
                         const TransitionModel *trans_model) {
  nvtxRangePushA("CudaFst constructor");
  start_ = fst.Start();

  ComputeOffsets(fst);
  AllocateData(fst);

  rescor_start_ = rescore_fst.Start();
/*  ComputeOffsets(rescore_fst);
  AllocateData(rescore_fst);*/
  // Temporarily allocating data for this vector
  // We just need it during CSR generation. We will clear it
  // at the end of Initialize
  h_arc_pdf_ilabels_.resize(arc_count_);
  // PopulateArcs(fst);
  PopulateArcs(fst, rescore_fst);
  CopyBoostingArcs();
  if (trans_model) ApplyTransitionModelOnIlabels(*trans_model);

  KALDI_ASSERT(d_e_offsets_);
  KALDI_ASSERT(d_ne_offsets_);
  KALDI_ASSERT(d_final_);
  KALDI_ASSERT(d_arc_weights_);
  KALDI_ASSERT(d_arc_nextstates_);
  KALDI_ASSERT(d_arc_pdf_ilabels_);

  CopyDataToDevice();

  // Making sure the graph is ready
  cudaDeviceSynchronize();
  KALDI_DECODER_CUDA_CHECK_ERROR();
  h_arc_pdf_ilabels_.clear();  // we don't need those on host
  nvtxRangePop();
}

void CudaFst::CopyBoostingArcs() {
  // first check how may graphs need to be copied
  int32 num_boosting_graphs = h_boosted_arc_idx_.size();
  /* if(num_boosting_graphs == 0) { */
  /*     KALDI_LOG << "WARNING num boosting graphs is 0"; */
  /*     return; */
  /* } */
  KALDI_LOG << "Num Boosting Graphs = " << num_boosting_graphs;
  // total number of arcs
  uint32 total_arcs = 0;
  for(uint32 i=0; i<num_boosting_graphs; i++) {
      total_arcs += static_cast<uint32>(h_boosted_arc_idx_[i].size());
  }
  /* KALDI_ASSERT(total_arcs>0); */
  int offset = 0;
  uint32 data_to_copy[total_arcs];
  for(int32 i=0; i<num_boosting_graphs; i++) {
      int num_arcs =h_boosted_arc_idx_[i].size(); 
      for(int j=0; j<num_arcs; j++) {
          data_to_copy[offset+j] = h_boosted_arc_idx_[i][j];
      }
      offset += h_boosted_arc_idx_[i].size();
  }
  d_boosted_arc_idx_ = static_cast<uint32  *>(CuDevice::Instantiate().Malloc(
      (total_arcs + 1) * sizeof(uint32)));
  // set the first value of d_boosted_arc_idx_ to total_arcs
  KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
      d_boosted_arc_idx_, &total_arcs, sizeof(uint32),
      cudaMemcpyHostToDevice));
  if(total_arcs>0)
      KALDI_DECODER_CUDA_API_CHECK_ERROR(cudaMemcpy(
          &d_boosted_arc_idx_[1], &data_to_copy, (total_arcs) * sizeof(uint32),
          cudaMemcpyHostToDevice));
}

void CudaFst::Finalize() {
  nvtxRangePushA("CudaFst destructor");

  // Making sure that Initialize was called before Finalize
  KALDI_ASSERT(d_e_offsets_ &&
               "Please call CudaFst::Initialize() before calling Finalize()");
  KALDI_ASSERT(d_ne_offsets_);
  KALDI_ASSERT(d_final_);
  KALDI_ASSERT(d_arc_weights_);
  KALDI_ASSERT(d_arc_nextstates_);
  KALDI_ASSERT(d_arc_pdf_ilabels_);

  CuDevice::Instantiate().Free(d_e_offsets_);
  CuDevice::Instantiate().Free(d_ne_offsets_);
  CuDevice::Instantiate().Free(d_final_);
  CuDevice::Instantiate().Free(d_arc_weights_);
  CuDevice::Instantiate().Free(d_arc_nextstates_);
  CuDevice::Instantiate().Free(d_arc_pdf_ilabels_);
  nvtxRangePop();
}

}  // end namespace cuda_decoder
}  // end namespace kaldi

#endif  // HAVE_CUDA == 1
