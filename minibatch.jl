# ### Minibatching
#
# Below is a sample implementation of a sequence minibatcher. The `LMData` iterator wraps a
# TextReader and produces batches of sentences with similar length to minimize padding (too
# much padding wastes computation). To be able to scale to very large files, we do not want
# to read the whole file, sort by length etc. Instead `LMData` keeps around a small number
# of buckets and fills them with similar sized sentences from the TextReader. As soon as one
# of the buckets reaches the desired batch size it is turned into a matrix with the
# necessary padding and output. When the TextReader is exhausted the remaining buckets are
# returned (which may have smaller batch sizes).

struct LMData
    src::TextReader
    batchsize::Int
    maxlength::Int
    bucketwidth::Int
    buckets
end

function LMData(src::TextReader; batchsize = 64, maxlength = typemax(Int), bucketwidth = 10)
    numbuckets = min(128, maxlength ÷ bucketwidth)
    buckets = [ [] for i in 1:numbuckets ]
    LMData(src, batchsize, maxlength, bucketwidth, buckets)
end

Base.IteratorSize(::Type{LMData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{LMData}) = Base.HasEltype()
Base.eltype(::Type{LMData}) = Matrix{Int}

function Base.iterate(d::LMData, state=nothing)
    if state == nothing
        for b in d.buckets; empty!(b); end
    end
    bucket,ibucket = nothing,nothing
    while true
        iter = (state === nothing ? iterate(d.src) : iterate(d.src, state))
        if iter === nothing
            ibucket = findfirst(x -> !isempty(x), d.buckets)
            bucket = (ibucket === nothing ? nothing : d.buckets[ibucket])
            break
        else
            sent, state = iter
            if length(sent) > d.maxlength || length(sent) == 0; continue; end
            ibucket = min(1 + (length(sent)-1) ÷ d.bucketwidth, length(d.buckets))
            bucket = d.buckets[ibucket]
            push!(bucket, sent)
            if length(bucket) === d.batchsize; break; end
        end
    end
    if bucket === nothing; return nothing; end
    batchsize = length(bucket)
    maxlen = maximum(length.(bucket))
    batch = fill(d.src.vocab.eos, batchsize, maxlen + 1)
    for i in 1:batchsize
        batch[i, 1:length(bucket[i])] = bucket[i]
    end
    empty!(bucket)
    return batch, state
end


# Minibatching for s2s models
mutable struct MTData
    src::TextReader        # reader for source language data
    tgt::TextReader        # reader for target language data
    batchsize::Int         # desired batch size
    maxlength::Int         # skip if source sentence above maxlength
    batchmajor::Bool       # batch dims (B,T) if batchmajor=false (default) or (T,B) if true.
    bucketwidth::Int       # batch sentences with length within bucketwidth of each other
    buckets::Vector        # sentences collected in separate arrays called buckets for each length range
    batchmaker::Function   # function that turns a bucket into a batch.
end

function MTData(src::TextReader, tgt::TextReader; batchmaker = arraybatch, batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 10, numbuckets = min(128, maxlength ÷ bucketwidth))
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    MTData(src, tgt, batchsize, maxlength, batchmajor, bucketwidth, buckets, batchmaker)
end

function Base.iterate(d::MTData, state=nothing)
    numbuckets=length(d.buckets)
    batchsize=d.batchsize
    if (state===nothing)
        d.buckets=[ [] for i in 1:numbuckets ]
        src_sentence, src_state=iterate(d.src)
        tgt_sentence, tgt_state=iterate(d.tgt)
    else
        src_state=state[1]
        tgt_state=state[2]
        x=iterate(d.src, src_state)
        y=iterate(d.tgt, tgt_state)
        if (x==nothing || y==nothing)
            for i in 1:numbuckets
                if length(d.buckets[i])>0
                    batch=d.batchmaker(d, d.buckets[i])
                    d.buckets[i]=[]
                    return (batch, (src_state, tgt_state))
                end
            end
            return nothing
        else
            src_sentence,src_state=x
            tgt_sentence,tgt_state=y
        end
    end

    while (length(src_sentence)>d.maxlength)
        src_sentence, src_state=iterate(d.src, src_state)
        tgt_sentence, tgt_state=iterate(d.tgt, tgt_state)
    end


    for i in 1:numbuckets
        if (length(src_sentence) > length(d.buckets)*d.bucketwidth)
            push!(d.buckets[numbuckets], (src_sentence, tgt_sentence))
            batch=d.batchmaker(d, d.buckets[numbuckets])
            d.buckets[numbuckets]=[]
            return (batch, (src_state, tgt_state))
        elseif (length(src_sentence) in ((i-1)*d.bucketwidth+1):(i*d.bucketwidth))
            push!(d.buckets[i], (src_sentence, tgt_sentence))
            if (length(d.buckets[i])==d.batchsize)
                batch=d.batchmaker(d, d.buckets[i])
                d.buckets[i]=[]
                return (batch, (src_state, tgt_state))
            else
                return iterate(d, (src_state, tgt_state))
            end
        end
    end
end

function arraybatch(d::MTData, bucket)
    longest_src=0
    longest_tgt=0
    batchsize=length(bucket)
    for sentence_pairs in bucket
        if (length(sentence_pairs[1])>longest_src)
            longest_src=length(sentence_pairs[1])
        end
        if (length(sentence_pairs[2])>longest_tgt)
            longest_tgt=length(sentence_pairs[2])
        end
    end
    x= Array{Int64,2}(undef, batchsize, longest_src)
    y= Array{Int64,2}(undef, batchsize, longest_tgt+2)
    for (i, sentence_pair) in enumerate(bucket)
        for k in 1:(longest_src-length(sentence_pair[1]))
            prepend!(sentence_pair[1], d.src.vocab.eos)
        end
        for (j, word) in enumerate(sentence_pair[1])
            x[i,j]=word
        end
        prepend!(sentence_pair[2],d.tgt.vocab.eos)
        append!(sentence_pair[2], d.tgt.vocab.eos)
        for j in 1:(longest_tgt+2-length(sentence_pair[2]))
            append!(sentence_pair[2], d.tgt.vocab.eos)
        end
        for (j, word) in enumerate(sentence_pair[2])
            y[i,j]=word
        end
    end
    return (x,y)
end
