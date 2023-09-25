// cudadecoder/lattice-postprocessor.cc
//
// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
// Hugo Braun
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

#include "cudadecoder/lattice-postprocessor.h"

#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "util/common-utils.h"

using namespace std;

#if HAVE_CUDA == 1

namespace kaldi {
namespace cuda_decoder {

LatticePostprocessor::LatticePostprocessor(
    const LatticePostprocessorConfig &config)
    : config_(config), decoder_frame_shift_(0.0) {
  ApplyConfig();
}

void LatticePostprocessor::ApplyConfig() {
  // Lattice scale
  lattice_scales_.resize(2);
  lattice_scales_[0].resize(2);
  lattice_scales_[1].resize(2);
  lattice_scales_[0][0] = config_.lm_scale;
  lattice_scales_[0][1] = config_.acoustic2lm_scale;
  lattice_scales_[1][0] = config_.lm2acoustic_scale;
  lattice_scales_[1][1] = config_.acoustic_scale;

  use_lattice_scale_ =
      (config_.lm_scale != 1.0 || config_.acoustic2lm_scale != 0.0 ||
       config_.lm2acoustic_scale != 0.0 || config_.acoustic_scale != 1.0);

  // Word boundary
  if (!config_.word_boundary_rxfilename.empty())
    LoadWordBoundaryInfo(config_.word_boundary_rxfilename);
}

bool LatticePostprocessor::GetPostprocessedLattice(
    CompactLattice &clat, CompactLattice *out_clat) const {
  // Nothing to do for empty lattice
  if (clat.NumStates() == 0) return true;

  bool ok = true;
  // Scale lattice
  if (use_lattice_scale_) fst::ScaleLattice(lattice_scales_, &clat);

  // Word insertion penalty
  if (config_.word_ins_penalty > 0.0)
    AddWordInsPenToCompactLattice(config_.word_ins_penalty, &clat);

  // Word align
  int32 max_states;
  if (config_.max_expand > 0)
    max_states = 1000 + config_.max_expand * clat.NumStates();
  else
    max_states = 0;

  KALDI_ASSERT(tmodel_ &&
               "SetTransitionModel() must be called (typically by pipeline)");

  KALDI_ASSERT(decoder_frame_shift_ != 0.0 &&
               "SetDecoderFrameShift() must be called (typically by pipeline)");

  if (!word_info_)
    KALDI_ERR << "You must set --word-boundary-rxfilename in the lattice "
                 "postprocessor config";
  // ok &=
  // Ignoring the return false for now (but will print a warning),
  // because the doc says we can, and it can happen when using endpointing
  WordAlignLattice(clat, *tmodel_, *word_info_, max_states, out_clat);
  return ok;
}

bool LatticePostprocessor::GetPostprocessedLattice(
    CompactLattice &clat, fst::VectorFst<fst::StdArc> &rescore_fst,    
    CompactLattice *out_clat) const {
  // Nothing to do for empty lattice
  if (clat.NumStates() == 0) return true;

  bool ok = true;
  // Scale lattice
  // if (use_lattice_scale_) fst::ScaleLattice(lattice_scales_, &clat);

  // ------------------------------------------------------------
  // GPU boosting part:
  std::vector<vector<double> > scale = fst::LatticeScale(1.0, 1.0);  // GPU boosting
  fst::ScaleLattice(scale, &clat);

  Lattice lat;
  Lattice composed_lat;
  CompactLattice biased_clat;
  int32 num_states_cache = 50000;
  int32 phi_label = -1;  // GPU boosting: If >0, the label on backoff arcs of the LM

  // RemoveAlignmentsFromCompactLattice(&clat);
  ConvertLattice(clat, &lat);
  //fst::VectorFst<fst::StdArc> *rescore_fst_mut = NULL;  // GPU boosting
  //rescore_fst_mut = &rescore_fst;

  // ArcSort(rescore_fst, fst::ILabelCompare<fst::StdArc>());
  // mapped_fst2 is fst2 interpreted using the LatticeWeight semiring,
  // with all the cost on the first member of the pair (since we're
  // assuming it's a graph weight).
  fst::CacheOptions cache_opts(true, num_states_cache);
  fst::MapFstOptions mapfst_opts(cache_opts);
  fst::StdToLatticeMapper<BaseFloat> mapper;

  //fst::VectorFst<fst::StdArc>* rescore_fst_test;
  //rescore_fst_test = fst::ReadAndPrepareLmFst("/idiap/temp/inigmatulina/work/uniphore/online_recognition/cpu-online-decoding/fsts/biasing_test_liveatc_set_001410-002028.fst");
  
  //KALDI_LOG << "Before the composition!\n";
  fst::MapFst<fst::StdArc, LatticeArc, fst::StdToLatticeMapper<BaseFloat> >
      mapped_fst2(rescore_fst, mapper, mapfst_opts);
  //KALDI_LOG << "Mapping FST done\n";

      ArcSort(&lat, fst::OLabelCompare<LatticeArc>());
      if (phi_label > 0) PhiCompose(lat, mapped_fst2, phi_label, &composed_lat);
      else Compose(lat, mapped_fst2, &composed_lat);
      if (composed_lat.Start() == fst::kNoStateId) {
        KALDI_WARN << "Empty lattice for utterance ";
      } else {
          DeterminizeLatticePhonePrunedWrapper(*tmodel_, &composed_lat,
                                               config_.decoder_opts.lattice_beam,
                                               &biased_clat, config_.det_opts);
          // ConvertLattice(composed_lat, &biased_clat);
      }
  //KALDI_LOG << "Composition done\n";

  // GPU boosting part - END
  // ------------------------------------------------------------
  //
  // Scale lattice
  if (use_lattice_scale_) fst::ScaleLattice(lattice_scales_, &biased_clat);

  // Word insertion penalty
  if (config_.word_ins_penalty > 0.0)
    AddWordInsPenToCompactLattice(config_.word_ins_penalty, &biased_clat);  // GPU boosting

  // Word align
  int32 max_states;
  if (config_.max_expand > 0)
    max_states = 1000 + config_.max_expand * biased_clat.NumStates();  // GPU boosting
  else
    max_states = 0;

  KALDI_ASSERT(tmodel_ &&
               "SetTransitionModel() must be called (typically by pipeline)");

  KALDI_ASSERT(decoder_frame_shift_ != 0.0 &&
               "SetDecoderFrameShift() must be called (typically by pipeline)");

  if (!word_info_)
    KALDI_ERR << "You must set --word-boundary-rxfilename in the lattice "
                 "postprocessor config";

  //const std::string &key = "123";
  //kaldi::CompactLatticeWriter compact_lattice_writer;
  //compact_lattice_writer.Open("ark,t:/idiap/temp/inigmatulina/work/uniphore/online_recognition/lattices/test_lat_boosted.ark");
  //compact_lattice_writer.Write(key, biased_clat);

  // ok &=
  // Ignoring the return false for now (but will print a warning),
  // because the doc says we can, and it can happen when using endpointing
  WordAlignLattice(biased_clat, *tmodel_, *word_info_, max_states, out_clat);  // GPU boosting

  return ok;
}

bool LatticePostprocessor::GetCTM(CompactLattice &clat,                                    
                                  CTMResult *ctm_result) const {
  // Empty CTM output for empty lattice
  if (clat.NumStates() == 0) return true;

  CompactLattice postprocessed_lattice;
  GetPostprocessedLattice(clat, &postprocessed_lattice);

  // MBR
  MinimumBayesRisk mbr(postprocessed_lattice, config_.mbr_opts);
  ctm_result->conf = std::move(mbr.GetOneBestConfidences());
  ctm_result->words = std::move(mbr.GetOneBest());
  ctm_result->times_seconds = std::move(mbr.GetOneBestTimes());

  // Convert timings to seconds
  for (auto &p : ctm_result->times_seconds) {
    p.first *= decoder_frame_shift_;
    p.second *= decoder_frame_shift_;
  }

  return true;
}

bool LatticePostprocessor::GetCTM(CompactLattice &clat,
                                  fst::VectorFst<fst::StdArc> &rescore_fst,    // GPU boosting
                                  CTMResult *ctm_result) const {
  // Empty CTM output for empty lattice
  if (clat.NumStates() == 0) return true;

  CompactLattice postprocessed_lattice;
  GetPostprocessedLattice(clat, rescore_fst, &postprocessed_lattice);    // GPU boosting

  // MBR
  MinimumBayesRisk mbr(postprocessed_lattice, config_.mbr_opts);
  //MinimumBayesRisk mbr(clat, config_.mbr_opts);
  ctm_result->conf = std::move(mbr.GetOneBestConfidences());
  ctm_result->words = std::move(mbr.GetOneBest());
  ctm_result->times_seconds = std::move(mbr.GetOneBestTimes());
  
  KALDI_LOG << "CTM result words:" << ctm_result->words.size() << "\n" ;
  /*for (auto w: ctm_result->words)
      std::cerr << w << " ";
  std::cerr << "\n"; */

  // Convert timings to seconds
  for (auto &p : ctm_result->times_seconds) {
    p.first *= decoder_frame_shift_;
    p.second *= decoder_frame_shift_;
  }

  return true;
}


void SetResultUsingLattice(
    CompactLattice &clat, const int result_type,
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor,      
    CudaPipelineResult *result) {
  if (result_type & CudaPipelineResult::RESULT_TYPE_LATTICE) {
    if (lattice_postprocessor) {
      CompactLattice postprocessed_clat;
      lattice_postprocessor->GetPostprocessedLattice(clat, 
        &postprocessed_clat);
      result->SetLatticeResult(std::move(postprocessed_clat));
    } else {
      result->SetLatticeResult(std::move(clat));
    }
  }

  if (result_type & CudaPipelineResult::RESULT_TYPE_CTM) {
    CTMResult ctm_result;
    KALDI_ASSERT(lattice_postprocessor &&
                 "A lattice postprocessor must be set with "
                 "SetLatticePostprocessor() to use RESULT_TYPE_CTM");
    lattice_postprocessor->GetCTM(clat, &ctm_result);
    result->SetCTMResult(std::move(ctm_result));
  }
}

void SetResultUsingLattice(
    CompactLattice &clat, const int result_type,
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor,
    fst::VectorFst<fst::StdArc> &rescore_fst,    // GPU boosting
    CudaPipelineResult *result) {
  if (result_type & CudaPipelineResult::RESULT_TYPE_LATTICE) {
    if (lattice_postprocessor) {
      //KALDI_LOG << "Run GetPostprocessedLattice in\n";
      CompactLattice postprocessed_clat;
      lattice_postprocessor->GetPostprocessedLattice(clat, rescore_fst,
        &postprocessed_clat);    // GPU boosting
      result->SetLatticeResult(std::move(postprocessed_clat));
    } else {
      result->SetLatticeResult(std::move(clat));
    }
  }

  if (result_type & CudaPipelineResult::RESULT_TYPE_CTM) {
    CTMResult ctm_result;
    //KALDI_LOG << "Run GetPostprocessedCTM\n";
    KALDI_ASSERT(lattice_postprocessor &&
                 "A lattice postprocessor must be set with "
                 "SetLatticePostprocessor() to use RESULT_TYPE_CTM");
    lattice_postprocessor->GetCTM(clat, rescore_fst, &ctm_result);    // GPU boosting line
    result->SetCTMResult(std::move(ctm_result));
  }
}


}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // HAVE_CUDA
