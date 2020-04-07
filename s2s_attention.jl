import Pkg;
packages = ["Knet", "Test", "Printf", "LinearAlgebra", "Random", "CuArrays", "IterTools"]
for p in packages; Pkg.add(p); end
using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CuArrays, IterTools

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

struct S2S
    srcembed::Embed       # encinput(B,Tx) -> srcembed(Ex,B,Tx)
    encoder::RNN          # srcembed(Ex,B,Tx) -> enccell(Dx*H,B,Tx)
    memory::Memory        # enccell(Dx*H,B,Tx) -> keys(H,Tx,B), vals(Dx*H,Tx,B)
    tgtembed::Embed       # decinput(B,Ty) -> tgtembed(Ey,B,Ty)
    decoder::RNN          # tgtembed(Ey,B,Ty) . attnvec(H,B,Ty)[t-1] = (Ey+H,B,Ty) -> deccell(H,B,Ty)
    attention::Attention  # deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
    projection::Linear    # attnvec(H,B,Ty) -> proj(Vy,B,Ty)
    dropout::Real         # dropout probability
    srcvocab::Vocab       # source language vocabulary
    tgtvocab::Vocab       # target language vocabulary
end

function S2S(hidden::Int, srcembsz::Int, tgtembsz::Int, srcvocab::Vocab, tgtvocab::Vocab;
             layers=1, bidirectional=false, dropout=0)

    # embedding
    vocab_size_source = length(srcvocab.i2w)
    vocab_size_target = length(tgtvocab.i2w)
    srcembed = Embed(vocab_size_source, srcembsz)
    tgtembed = Embed(vocab_size_target, tgtembsz)
    if bidirectional
        encoder=RNN(srcembsz, hidden, numLayers=1/2*layers, dropout=dropout, bidirectional=bidirectional)
        memory=Memory(param(hidden,2*hidden))
        wattn=param(hidden,3*hidden)
    else
        memory=Memory(1)
        wattn=param(hidden,2*hidden)
    end

    wquery=1
    scale=param(1)
    encoder=RNN(srcembsz, hidden, numLayers=1/2*layers, dropout=dropout, bidirectional=bidirectional)
    decoder=RNN(tgtembsz+hidden, hidden, numLayers=layers, dropout=dropout, bidirectional=bidirectional)

    attention=Attention(wquery, wattn, scale)

    projection=Linear(hidden, vocab_size_target)
    S2S(srcembed, encoder, memory, tgtembed, decoder, attention, projection, dropout, srcvocab, tgtvocab)
end

function (m::Memory)(x)
    values=permutedims(x, (1,3,2))
    keys=mmul(m.w, values)
    return keys, values
end

function encode(s::S2S, src)
    s.encoder.h = 0
    s.encoder.c = 0

    src_embed_out = s.srcembed(src)
    encoder_out = s.encoder(src_embed_out)
    s.decoder.h = s.encoder.h
    s.decoder.c = s.encoder.c
    s.memory(encoder_out)
end

function (a::Attention)(cell, mem)
    query_tensor=a.wquery*cell
    keys, vals=mem

    query_tensor=permutedims(mmul(a.wquery, cell), (3,1,2))
    attention_scores=bmm(query_tensor, keys)
    scaled_attention_scores=attention_scores* a.scale[1]
    normalized_attention_scores=softmax(scaled_attention_scores, dims=2)
    permuted_vals=permutedims(vals, (2,1,3))

    context_tensor=bmm(normalized_attention_scores, permuted_vals)
    permuted_context_tensor=permutedims(context_tensor, (2,3,1))
    concatenated_context_tensor=vcat(cell, permuted_context_tensor)
    reshaped_context_tensor=reshape(concatenated_context_tensor, (size(a.wattn, 2),:))
    transformed_context_tensor=a.wattn*reshaped_context_tensor
    final_reshaped_context_tensor=reshape(transformed_context_tensor, size(cell))

    return final_reshaped_context_tensor
end

function decode(s::S2S, tgt, mem, prev)
    tgt_embed_out = s.tgtembed(tgt)
    tgt_embed_out = reshape(tgt_embed_out, size(tgt_embed_out)[1], size(tgt_embed_out)[2], 1)
    input_feeding=vcat(tgt_embed_out, prev)
    decoder_output=s.decoder(input_feeding)
    attention_vector=s.attention(decoder_output, mem)
    return attention_vector
end

function (s::S2S)(src, tgt; average=true)

    mem=encode(s, src)
    key, val=mem
    hidden=s.decoder.hiddenSize
    batchSize = size(src,1)
    targetLength = size(tgt,2)
    prev = zeros(Float32, size(s.encoder.h, 1), batchSize, 1)
    if (gpu() >= 0)
        prev = KnetArray(prev)
    end
    sumloss=0
    numwords=0

    to_be_masked=copy(tgt[:,2:end])
    mask!(to_be_masked, s.tgtvocab.eos)
    masked=to_be_masked

    for i in 1:targetLength-1
        decoder_output=decode(s, tgt[:, i], mem, prev)
        prev=decoder_output
        reshaped_decoder_output=reshape(decoder_output, :, batchSize)
        score=s.projection(reshaped_decoder_output)
        delta_loss, delta_numwords=nll(score, masked[:, i], average=false)
        sumloss=sumloss+delta_loss
        numwords=numwords+delta_numwords
    end

    if average
        return sumloss/numwords
    else
        return sumloss,numwords
    end
end

function (s::S2S)(src; stopfactor = 3)
    isDone = false
    batch_size = size(src,1)
    input = repeat([s.tgtvocab.eos], batch_size)
    is_all_finished = zeros(batch_size)
    translated_sentences = copy(input)
    max_length_output = 0

    mem = encode(s, src)

    prev_decoder_output = zeros(Float32, size(s.encoder.h, 1), batch_size, 1)
    if (gpu() >= 0)
        prev_decoder_output = KnetArray(prev_decoder_output)
    end
    input = reshape(input, (length(input), 1))

    while (!isDone && max_length_output < stopfactor*size(src,2))


        y = decode(s, input, mem, prev_decoder_output)
        prev_decoder_output = y


        hy, b ,ty = size(y)
        y = reshape(y, (hy, b*ty))

        scores = s.projection(y)

        output_words = reshape(map(x->x[1], argmax(scores, dims = 1)), batch_size)
        translated_sentences = hcat(translated_sentences, output_words)

        max_length_output = size(translated_sentences, 2)
        input = reshape(output_words, (length(output_words), 1))


        tmp_output_words = copy(output_words)
        tmp_output_words = tmp_output_words .== s.tgtvocab.eos
        is_all_finished += tmp_output_words
        if(sum(is_all_finished.==0)==0)
            isDone = true
        end
    end
    return translated_sentences[:, 2:end]
end

function trainmodel(trn,                  # Training data
                    dev,                  # Validation data, used to determine the best model
                    tst...;               # Zero or more test datasets, their loss will be periodically reported
                    bidirectional = true, # Whether to use a bidirectional encoder
                    layers = 2,           # Number of layers (use `layers÷2` for a bidirectional encoder)
                    hidden = 512,         # Size of the hidden vectors
                    srcembed = 512,       # Size of the source language embedding vectors
                    tgtembed = 512,       # Size of the target language embedding vectors
                    dropout = 0.2,        # Dropout probability
                    epochs = 0,           # Number of epochs (one of epochs or iters should be nonzero for training)
                    iters = 0,            # Number of iterations (one of epochs or iters should be nonzero for training)
                    bleu = false,         # Whether to calculate the BLEU score for the final model
                    save = false,         # Whether to save the final model
                    seconds = 60,         # Frequency of progress reporting
                    )
    @show bidirectional, layers, hidden, srcembed, tgtembed, dropout, epochs, iters, bleu, save; flush(stdout)
    model = S2S(hidden, srcembed, tgtembed, trn.src.vocab, trn.tgt.vocab;
                layers=layers, dropout=dropout, bidirectional=bidirectional)

    epochs == iters == 0 && return model

    (ctrn,cdev,ctst) = collect(trn),collect(dev),collect.(tst)
    traindata = (epochs > 0
                 ? collect(flatten(shuffle!(ctrn) for i in 1:epochs))
                 : shuffle!(collect(take(cycle(ctrn), iters))))

    bestloss, bestmodel = loss(model, cdev), deepcopy(model)
    progress!(adam(model, traindata), seconds=seconds) do y
        devloss = loss(model, cdev)
        tstloss = map(d->loss(model,d), ctst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (dev=devloss, tst=tstloss, mem=Float32(CuArrays.usage[]))
    end
    save && Knet.save("bestmodel.jld2", "bestmodel", bestmodel)
    bleu && Main.bleu(bestmodel,dev)
    return bestmodel
end
