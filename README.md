# Contextual biasing on GPUs

The implementation of the contextual biasing for ASR decoding on GPUs without lattice generation. The code supports submission to Interspeech 2023 [Implementing contextual biasing in GPU decoder for online ASR](https://arxiv.org/abs/2306.15685).

We use the [Kaldi GPU decoder](https://github.com/kaldi-asr/kaldi/tree/master/src/cudadecoder) code as the base and implement contextual dynamic biasing on top of it.

## Biasing of lattices at the endpoints (post-processing)

Biasing of lattices at the endpoints (post-processing) can be found in the ```lattice-postprocess.cc``` script.

## Dynamic biasing of partial hypotheses without lattice generation

Dynamic biasing on GPUs is done in two steps. First, in order to enable contextual biasing directly on GPUs when the ```HCLG.fst``` graph is loaded, we record those arcs that correspond to the words and word sequences we need to boost. Then, the boosting itself is realised on kernels during decoding when the discount factor is added to the arcs to boost.

The first step is the core one in our implementation, as we have to choose the right arcs to boost when they come in a distributed way. The implementation of the algorithm can be found in the ```cuda-fst.cc``` script.

The second step can be found in the script ```cuda-decoder-kernels.cu```.

To run the implementation with the server-client setup, in addition to model and HCLG graph arguments, one needs to pass one more argument with the target contextual entities (=words or word sequences we want to boost). For the offline GPU boosting, this argument should be an FST graph built with the target entities (check the toy example ```context/bias_entities.fst```). For the online GPU boosting, the context argument should be a file where 1) all entities are listed with one entity per line and 2) all words are replaced with the word-IDs from the symbol table (words.txt) (check the toy example ```context/bias_entities_ids```, which corresponds to ```context/bias_entities_words```).

## The scripts modified comparing to the Kaldi original ones:

```batched-threaded-nnet3-cuda-online-pipeline.cc```\
```batched-threaded-nnet3-cuda-online-pipeline.h```\
```cuda-decoder-kernels-utils.h```\
```cuda-decoder-kernels.cu```\
```cuda-decoder-kernels.h```\
```cuda-decoder.cc```\
```cuda-fst.cc```\
```cuda-fst.h```\
```cuda-pipeline-common.h```\
```lattice-postprocessor.cc```\
```lattice-postprocessor.h```

#### Core scripts for online sequence boosting on GPUs:
```cuda-decoder-kernels.cu```\
```cuda-decoder.cc```\
```cuda-fst.cc```

#### Scripts for offline sequence boosting (at the endpoints):
```lattice-postprocessor.cc```\
```lattice-postprocessor.h```
