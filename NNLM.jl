import Pkg
Packages = ["IterTools", "LinearAlgebra", "LinearAlgebra", "StatsBase", "Test"]
for p in Packages; Pkg.add(p); end;
using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test

struct NNLM; vocab; windowsize; embed; hidden; output; dropout; end

function NNLM(vocab::Vocab, windowsize::Int, embedsize::Int, hiddensize::Int, dropout::Real)

    vocabsize = length(vocab.w2i)

    embed = Embed(vocabsize, embedsize)
    hidden = Linear(windowsize * embedsize, hiddensize)
    output = Linear(hiddensize, vocabsize)

    NNLM(vocab, windowsize, embed, hidden, output, dropout)

end

function pred_v1(m::NNLM, hist::AbstractVector{Int})
    @assert length(hist) == m.windowsize
    embeds = m.embed(hist)
    embeds = reshape(embeds, size(embeds)[1] * size(embeds)[2])
    embeds = dropout(embeds, m.dropout)
    h = m.hidden(embeds)
    h = dropout(h, m.dropout)
    out = m.output(tanh.(h))
    out = reshape(out, size(out)[1])
end

## This predicts the scores for the whole sentence, will be used for later testing.
function scores_v1(model, sent)
    hist = repeat([ model.vocab.eos ], model.windowsize)
    scores = []
    for word in [ sent; model.vocab.eos ]
        push!(scores, pred_v1(model, hist))
        hist = [ hist[2:end]; word ]
    end
    hcat(scores...)
end

function generate(m::NNLM; maxlength=30)
    history = repeat([model.vocab.eos], model.windowsize)
    sentence_indexes = []
    sentence_words = []

    for i in 1:maxlength
        scores = softmax(pred_v1(m, history))
        scores[m.vocab.eos] = -10000
        scores[m.vocab.unk] = -10000

        best_index = argmax(softmax(scores))
        best_word = m.vocab.i2w[best_index]

        #handle history
        popfirst!(history)
        push!(history, i)

        #build the sentence
        push!(sentence_indexes, best_index)
        push!(sentence_words, best_word)
    end
    join(sentence_words, " ")
end

function loss_v1(m::NNLM, sent::AbstractVector{Int}; average = true)
    new_sent = deepcopy(sent)
    push!(new_sent, m.vocab.eos)

    pred = scores_v1(m, new_sent)[:,1:length(new_sent)]
    if average; nll(pred, new_sent); else; nll(pred, new_sent; average=false); end
end

function maploss(lossfn, model, data; average = true)
    if average
        losses = []
        for sent in data; push!(losses, lossfn(model, sent; average=average)); end
        return sum(losses)/length(losses)
    else
        losses = []
        for sent in data; push!(losses, lossfn(model, sent; average=average)); end
        total_loss = sum((x->x[1]).(losses))
        (total_loss, length(data) + sum(length.(data)))
    end
end

function pred_v2(m::NNLM, hist::AbstractMatrix{Int})
    embeds = m.embed(hist)
    embeds = reshape(embeds, size(embeds)[1]*size(embeds)[2], size(embeds)[3])
    embeds = dropout(embeds, m.dropout)
    h = m.hidden(embeds)
    h = tanh.(h)
    h = dropout(h, m.dropout)
    out = m.output(h)
end

function scores_v2(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    return pred_v2(model, hist)
end

function loss_v2(m::NNLM, sent::AbstractVector{Int}; average = true)
    losses = []
    push!(sent, m.vocab.eos)
    pred = scores_v2(m, sent)

    if average
        for i in (1:length(sent))
            predi = pred[:,i]
            lossi = nll(predi, [sent[i]])
            push!(losses, lossi)
        end
        return sum(losses)/length(losses)

    else
        for i in (1:length(sent))
            predi = pred[:,i]
            lossi = nll(predi, [sent[i]])
            push!(losses, lossi)
        end
        return (sum(losses), length(losses))
    end
end

#=
`pred_v3` takes a model `m`, a N×B×S dimensional history array `hist`, and returns a V×B×S dimensional score array, where N is `m.windowsize`, V is the vocabulary size, B is the batch size, and S is maximum sentence length in the batch + 1 for the final eos token.
First, the embeddings for all entries in `hist` are looked up, which results in an array of E×N×B×S where E is the embedding size.
The embedding array is reshaped to (E*N)×(B*S) and dropout is applied. It is then fed to the hidden layer which returns a H×(B*S) hidden output where H is the hidden size.
Following element-wise tanh and dropout, the output layer turns this into a score array of V×(B*S) which is reshaped and returned as a V×B×S dimensional tensor.
=#
function pred_v3(m::NNLM, hist::Array{Int})
    embeds = m.embed(hist)
    embeds = reshape(embeds, size(embeds)[1]*size(embeds)[2],size(embeds)[3]*size(embeds)[4])
    embeds = dropout(embeds, m.dropout)
    h = m.hidden(embeds)
    h = tanh.(h)
    h = dropout(h, m.dropout)
    out = m.output(h)
    out = reshape(out, size(out)[1], size(hist)[2], size(hist)[3])
end

function scores_v3(model, sent)
    hist = [ repeat([ model.vocab.eos ], model.windowsize); sent ]
    hist = vcat((hist[i:end+i-model.windowsize]' for i in 1:model.windowsize)...)
    @assert size(hist) == (model.windowsize, length(sent)+1)
    hist = reshape(hist, size(hist,1), 1, size(hist,2))
    return pred_v3(model, hist)
end

function mask!(a,pad)
    for row in 1:size(a)[1]
        count = 0
        for column in size(a)[2]:-1:1
            if a[row,column] == pad; count+=1; else break; end
        end
        for column in size(a)[2]:-1:1
            if count > 1; a[row,column] = 0; count-=1; end
        end
    end
    a
end

function loss_v3(m::NNLM, batch::AbstractMatrix{Int}; average = true)

    num_batch = size(batch)[1]

    losses = []

    for b in 1:num_batch
        sent = batch[b,:]
        pred = scores_v3(m, sent)[:,:,1:length(sent)]
        batch_loss = nll(pred, sent)
        push!(losses, batch_loss)
    end

    if average
        return sum(losses) / num_batch
    else
        return (sum(losses), length(losses))
    end

end

train(loss, model, data) = sgd!(loss, ((model,sent) for sent in data))
