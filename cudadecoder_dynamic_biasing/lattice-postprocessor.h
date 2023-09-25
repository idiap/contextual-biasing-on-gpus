// cudadecoder/lattice-postprocessor.h
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

#if HAVE_CUDA == 1

#ifndef KALDI_CUDA_DECODER_LATTICE_POSTPROCESSOR_
#define KALDI_CUDA_DECODER_LATTICE_POSTPROCESSOR_

#include "base/kaldi-common.h"
#include "base/kaldi-utils.h"
#include "cudadecoder/cuda-pipeline-common.h"
#include "cudamatrix/cu-device.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/sausages.h"
#include "lat/word-align-lattice.h"
#include "util/stl-utils.h"

#include "util/common-utils.h"

#include "lat/determinize-lattice-pruned.h"  // GPU boosting
#include "cudadecoder/cuda-decoder.h"  // GPU boosting

namespace kaldi {
namespace cuda_decoder {

using fst::SymbolTable;
using fst::VectorFst;
using fst::StdArc;

struct LatticePostprocessorConfig {
  std::string word_boundary_rxfilename;
  MinimumBayesRiskOptions mbr_opts;
  WordBoundaryInfoNewOpts wip_opts;
  CudaDecoderConfig decoder_opts;  // GPU boosting
  fst::DeterminizeLatticePhonePrunedOptions det_opts;  // GPU boosting
  BaseFloat max_expand = 0.0;

  // Lattice scale
  BaseFloat acoustic_scale = 1.0;
  BaseFloat lm_scale = 1.0;
  BaseFloat acoustic2lm_scale = 0.0;
  BaseFloat lm2acoustic_scale = 0.0;
  BaseFloat word_ins_penalty = 0.0;

  void Register(OptionsItf *po) {
    po->Register(
        "max-expand", &max_expand,
        "If >0, the maximum amount by which this "
        "program will expand lattices before refusing to continue.  E.g. 10."
        "This can be used to prevent this program consuming excessive memory "
        "if there is a mismatch on the command-line or a 'problem' lattice.");

    po->Register("acoustic-scale", &acoustic_scale,
                 "Scaling factor for acoustic likelihoods");
    po->Register("lm-scale", &lm_scale, "Scaling factor for graph/lm costs");
    po->Register("acoustic2lm-scale", &acoustic2lm_scale,
                 "Add this times original acoustic costs to LM costs");
    po->Register("lm2acoustic-scale", &lm2acoustic_scale,
                 "Add this times original LM costs to acoustic costs");
    po->Register("word-ins-penalty", &word_ins_penalty,
                 "Word insertion penalty.");
    po->Register("word-boundary-rxfilename", &word_boundary_rxfilename,
                 "Word boundary file");

    mbr_opts.Register(po);
    wip_opts.Register(po);
    det_opts.Register(po);  // GPU boosting
  }
};

class LatticePostprocessor {
  const LatticePostprocessorConfig config_;
  const TransitionModel *tmodel_;
  std::shared_ptr<const WordBoundaryInfo> word_info_;
  BaseFloat decoder_frame_shift_;
  // Params for align words
  int32 max_states_;
  // Params for ScaleLattice
  bool use_lattice_scale_;
  std::vector<std::vector<double> > lattice_scales_;

 public:
  LatticePostprocessor(const LatticePostprocessorConfig &config);
  void ApplyConfig();
  bool GetCTM(CompactLattice &clat, CTMResult *ctm_result) const;
  bool GetCTM(CompactLattice &clat, fst::VectorFst<fst::StdArc> &rescore_fst, CTMResult *ctm_result) const;  // GPU boosting
  bool GetPostprocessedLattice(CompactLattice &clat,
                               CompactLattice *out_clat) const;
  bool GetPostprocessedLattice(
    CompactLattice &clat, fst::VectorFst<fst::StdArc> &rescore_fst,  // GPU boosting
    CompactLattice *out_clat) const;

  void SetTransitionModel(const TransitionModel *tmodel) { tmodel_ = tmodel; }

  void SetWordBoundaryInfo(
      const std::shared_ptr<const WordBoundaryInfo> &word_info) {
    word_info_ = word_info;
  }

  void SetDecoderFrameShift(BaseFloat decoder_frame_shift) {
    decoder_frame_shift_ = decoder_frame_shift;
  }

  void Write(std::string fst_fpath);

  void LoadWordBoundaryInfo(const std::string &word_boundary_rxfilename) {
    word_info_ = std::make_shared<WordBoundaryInfo>(config_.wip_opts,
                                                    word_boundary_rxfilename);
  }
};

void SetResultUsingLattice(
    CompactLattice &clat, const int result_type,
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor,
    CudaPipelineResult *result);

void SetResultUsingLattice(
    CompactLattice &clat, const int result_type,
    const std::shared_ptr<LatticePostprocessor> &lattice_postprocessor,
    fst::VectorFst<fst::StdArc> &rescore_fst,  // GPU boosting
    CudaPipelineResult *result);

// Read lattice postprocessor config, apply it,
// and assign it to the pipeline
template <class PIPELINE>
void LoadAndSetLatticePostprocessor(const std::string &config_filename,
                                    PIPELINE *cuda_pipeline) {
  ParseOptions po("");  // No usage, reading from a file
  LatticePostprocessorConfig pp_config;
  pp_config.Register(&po);
  po.ReadConfigFile(config_filename);
  auto lattice_postprocessor =
      std::make_shared<LatticePostprocessor>(pp_config);
  cuda_pipeline->SetLatticePostprocessor(lattice_postprocessor);
}

}  // namespace cuda_decoder
}  // namespace kaldi

#endif  // KALDI_CUDA_DECODER_LATTICE_POSTPROCESSOR_
#endif  // HAVE_CUDA
