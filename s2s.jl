using Pkg
packages = ["Knet", "Test", "IterTools", "Random"]
for p in packages; Pkg.add(p); end
Pkg.update("Knet")
using Knet, Test, Base.Iterators, IterTools, Random # , LinearAlgebra, StatsBase
using AutoGrad: @gcheck  # to check gradients, usew with Float64
Knet.atype() = KnetArray{Float32}  # determines what Knet.param() uses.
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging
include("layers.jl", "tokenizer.jl", "NNLM.jl")

mutable struct S2S_v1
    srcembed::Embed     # source language embedding
    encoder::RNN        # encoder RNN (can be bidirectional)
    tgtembed::Embed     # target language embedding
    decoder::RNN        # decoder RNN
    projection::Linear  # converts decoder output to vocab scores
    dropout::Real       # dropout probability to prevent overfitting
    srcvocab::Vocab     # source language vocabulary
    tgtvocab::Vocab     # target language vocabulary
end

### S2S_v1 constructor
# Please review the RNN documentation using `@doc RNN`, paying attention to the following options in particular: `numLayers`, `bidirectional`, `dropout`, `dataType`, `usegpu`.
# The last two are important if you experiment with array types other than the default `KnetArray{Float32}`: make sure the RNNs use the same array type as the other layers.
# Note that if the encoder is bidirectional, its `numLayers` should be half of the decoder so that their hidden states match in size.

function S2S_v1(hidden::Int,         # hidden size for both the encoder and decoder RNN
                srcembsz::Int,       # embedding size for source language
                tgtembsz::Int,       # embedding size for target language
                srcvocab::Vocab,     # vocabulary for source language
                tgtvocab::Vocab;     # vocabulary for target language
                layers=1,            # number of layers
                bidirectional=false, # whether encoder RNN is bidirectional
                dropout=0)           # dropout probability

    # embeddings
    vocab_size_source = length(srcvocab.w2i)
    vocab_size_target = length(tgtvocab.w2i)
    srcembed = Embed(vocab_size_source, srcembsz)
    tgtembed = Embed(vocab_size_target, tgtembsz)

    if bidirectional
        encoder = RNN(srcembsz, hidden; numLayers = layers,   dropout = dropout, bidirectional = true)
        decoder = RNN(tgtembsz, hidden; numLayers = (layers*2), dropout = dropout, bidirectional = false)
    else
        encoder = RNN(srcembsz, hidden; numLayers = layers, dropout = dropout, bidirectional = false)
        decoder = RNN(tgtembsz, hidden; numLayers = layers, dropout = dropout, bidirectional = false)
    end

    projection = Linear(hidden, vocab_size_target)

    S2S_v1(srcembed, encoder, tgtembed, decoder, projection, dropout, srcvocab, tgtvocab)

end

### S2S_v1 loss function

# Define the S2S_v1 loss function that takes `src`, a source language minibatch, and `tgt`, a target language minibatch and returns either a `(total_loss, num_words)` pair if `average=false`, or `(total_loss/num_words)` average if `average=true`.
# Assume that `src` and `tgt` are integer arrays of size `(B,Tx)` and `(B,Ty)` respectively, where `B` is the batch size, `Tx` is the length of the longest source sequence, `Ty` is the length of the longest target sequence.
# The `src` sequences only contain words, the `tgt` sequences surround the words with `eos` tokens at the start and end. This allows columns `tgt[:,1:end-1]` to be used as the decoder input and `tgt[:,2:end]` as the desired decoder output.
# Assume any shorter sentences in the batches have been padded with extra `eos` tokens on the left for `src` and on the right for `tgt`. Don't worry about masking `src` for the encoder, it doesn't have a significant effect on the loss.
# However do mask `tgt` before `nll`: you do not want the padding tokens to be counted in the loss calculation.
# Please review `@doc RNN`: in particular the `r.c` and `r.h` fields can be used to get/set the cell and hidden arrays of an RNN (note that `0` and `nothing` act as special values).
# RNNs take a dropout value at construction and apply dropout to the input of every layer if it is non-zero. You need to handle dropout for other layers in the loss function or in layer definitions as necessary.

function (s::S2S_v1)(src, tgt; average=true)

    # init encoder
    s.encoder.h = 0
    s.decoder.c = 0

    # ENCODER
    src_embed_out = s.srcembed(src)
    encoder_out = s.encoder(src_embed_out)

    s.decoder.h = s.encoder.h
    s.decoder.c = s.encoder.c

    # DECODER
    tgt_embed_out = s.tgtembed(tgt[:,1:end-1])
    decoder_out = s.decoder(tgt_embed_out)

    # reshape
    decoder_out = reshape(decoder_out, (size(decoder_out)[1], size(decoder_out)[2] * size(decoder_out)[3]))

    # Linear
    projection_out = s.projection(decoder_out)

    # NLL
    scores = projection_out
    answers = tgt[:,2:end]
    answers = mask!(answers, s.tgtvocab.eos)

    answers = reshape(answers, (1, size(answers)[1] * size(answers)[2]))

    nll(scores, answers; dims=1, average=average)
end

### Loss for a whole dataset

# Define a `loss(model, data)` which returns a `(Σloss, Nloss)` pair if `average=false` and a `Σloss/Nloss` average if `average=true` for a whole dataset.
# Assume that `data` is an iterator of `(x,y)` pairs such as `MTData` and `model(x,y;average)` is a model like `S2S_v1` that computes loss on a single `(x,y)` pair.

function loss(model, data; average=true)
    if average; Knet.mean(model(x,y) for (x,y) in data)
    else
        total = 0
        counter = 0
        for (x,y) in data; out = model(x,y; average=average); total += out[1]; counter += out[2]; end
        return (total, counter)
    end
end

### Training SGD_v1
# `trn` is the training data, `dev` is used to determine the best model, `tst...` can be zero or more small test datasets for loss reporting.
# It returns the model that does best on `dev`.

function train!(model, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps=100) do y
        losses = [ loss(model, d) for d in (dev,tst...) ]
        if losses[1] < bestloss
            bestmodel, bestloss = deepcopy(model), losses[1]
        end
        return (losses...,)
    end
    return bestmodel
end

### Generating translations

# With a single argument, a `S2S_v1` object takes a batch of source sentences and generates translations for them.
# After passing `src` through the encoder and copying its hidden states to the decoder, the decoder is run starting with an initial input of all subsequent decoder steps.
# The decoder stops generating when all sequences in the batch have generated `eos` or when `stopfactor * size(src,2)` decoder steps are reached.
# Target language batch is returned.

function (s::S2S_v1)(src::Matrix{Int}; stopfactor = 3)

    #get source embeddings
    source_embed_out = s.srcembed(src)

    #init encoder
    s.encoder.h = 0
    s.encoder.c = 0

    #get encoder out
    encoder_out = s.encoder(source_embed_out)

    #set decoder to encoder out
    s.decoder.h = 0
    s.decoder.h = deepcopy(s.encoder.h)
    s.decoder.c = 0
    s.decoder.c = deepcopy(s.encoder.c)

    # get number of sentences in the batch
    batch_size = size(src)[1]

    # calculate stopping condition
    max_iters = stopfactor * size(src,2)

    # for each sentence, start with [eos]
    outputs = rand(s.tgtvocab.eos:s.tgtvocab.eos, (batch_size, 1))

    for i in 1:max_iters
        # get the embeddings of the current outputs, but only use the last timestep
        target_embed_out = s.tgtembed(outputs[:,i])

        # decoder forward pass
        decoder_out = s.decoder(target_embed_out)

        # projection forward pass
        proj_out = s.projection(decoder_out)

        # eliminate <unk>
        proj_out[s.tgtvocab.unk,1] = -10000

        best_words = (x->x[1]).(argmax(proj_out; dims=1))

        outputs = hcat(outputs, transpose(best_words))
    end

    return outputs
end

# BLEU is the most commonly used metric to measure translation quality.
# Takes a model and some data, generates translations and calculates BLEU.
function bleu(s2s,d::MTData)
    d = MTData(d.src,d.tgt,batchsize=32)
    reffile = d.tgt.file
    hypfile,hyp = mktemp()
    for (x,y) in progress(collect(d))
        g = s2s(x)
        for i in 1:size(y,1)
            println(hyp, int2str(g[i,:], d.tgt.vocab))
        end
    end
    close(hyp)
    isfile("multi-bleu.perl") || download("https://github.com/moses-smt/mosesdecoder/raw/master/scripts/generic/multi-bleu.perl", "multi-bleu.perl")
    run(pipeline(`cat $hypfile`,`perl multi-bleu.perl $reffile`))
    return hypfile
end
