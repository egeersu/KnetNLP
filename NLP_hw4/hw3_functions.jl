
struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    io = open(file, "r")
    lines = readlines(io)
    close(io)
    
    reviews = []
    
    #tokenize each line with the function:tokenizer, add each tokenized line to reviews
    for line in lines; tokenized = push!(tokenizer(line), eos); push!(reviews, tokenized); end
      
    freq = Dict(); 
    
    #iterate over all words, count how many times they occur by adding them to w2i
    for review in reviews
        for word in review
            if word in keys(freq); freq[word] += 1; else; freq[word] = 1; end
        end
    end
    
    #sort the dictionary based on word frequencies, returns an array
    freq = sort(collect(freq), by=x->x[2])
    
    # Remove the least common word until we reach the specified vocabsize.
    # Keep track of total removed values
    total_removed = 0
    while (length(freq) > vocabsize)
        total_removed += freq[1][2]
        popfirst!(freq) 
    end
    
    # keep only the words that occur >= mincount times
    temp = []
    for (word, count) in freq
        if count >= mincount; push!(temp, (word, count)); end
    end
    freq = temp

    #turn array back into dictionary
    freq = Dict{String, Int}(freq)
    
    #add total removed values to <unk>    
    if !(unk in keys(freq)); freq[unk] = 0; end 
    freq[unk] += total_removed
    
    # Create i2w
    w2i = Dict(); i2w = []
    for (i,elt) in enumerate(keys(freq))
        w2i[elt] = i
        push!(i2w, elt)
    end
    i2w = Vector{String}(i2w)
    
    Vocab(w2i, i2w, w2i[unk], w2i[eos], tokenizer)
end


function words_to_ints(vocab::Vocab, sentence)
    [get(vocab.w2i, word, vocab.unk) for word in vocab.tokenizer(sentence)]
end

struct TextReader
    file::String
    vocab::Vocab
end

function Base.iterate(r::TextReader, s=nothing)
    if s==nothing
        state=open(r.file)
    else 
        state=s
    end
    if eof(state)
        close(state)
        return nothing
    else
        line= readline(state)
        return words_to_ints(r.vocab, line), state
    end
end


# These are some optional functions that can be defined for iterators. They are required for
# `collect` to work, which converts an iterator to a regular array.

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}


struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    l.w[:,x]
end

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize), param0(outputsize))
end

function (l::Linear)(x)
    l.w * mat(x,dims=1) .+ l.b
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

Base.IteratorSize(::Type{MTData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MTData}) = Base.HasEltype()
Base.eltype(::Type{MTData}) = NTuple{2}


function Base.iterate(d::MTData, state=nothing)
    # Your code here
    numbuckets=length(d.buckets)
    batchsize=d.batchsize
    if (state===nothing)
#         buckets=[ [] for i in 1:numbuckets ]
#         d=MTData(d.src, d.tgt, d.batchsize, d.maxlength, d.batchmajor, d.bucketwidth, buckets , d.batchmaker)
        d.buckets=[ [] for i in 1:numbuckets ]
        src_sentence, src_state=iterate(d.src)
        tgt_sentence, tgt_state=iterate(d.tgt)
#         println(src_sentence)
#         println(tgt_sentence)
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
            #print(d.buckets) this prints the buckets just before it ends we expect all empty buckets
            return nothing
        else
            src_sentence,src_state=x
            tgt_sentence,tgt_state=y
        end
    end
    
    while (length(src_sentence)>d.maxlength)
        #print("the src sentence is skipped")
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
#you still havent done batchmajor

function arraybatch(d::MTData, bucket)
    # Your code here
    longest_src=0
    longest_tgt=0
    batchsize=length(bucket)
#     x=[[] for i in 1:batchsize]
#     y=[[] for i in 1:batchsize]
    for sentence_pairs in bucket
        if (length(sentence_pairs[1])>longest_src)
            longest_src=length(sentence_pairs[1])
        end
        if (length(sentence_pairs[2])>longest_tgt)
            longest_tgt=length(sentence_pairs[2])
        end
    end
#     print(longest_src)
#     print(longest_tgt)
    x= Array{Int64,2}(undef, batchsize, longest_src)
    y= Array{Int64,2}(undef, batchsize, longest_tgt+2)
    for (i, sentence_pair) in enumerate(bucket)
        for k in 1:(longest_src-length(sentence_pair[1]))
            prepend!(sentence_pair[1], d.src.vocab.eos)
        end
        for (j, word) in enumerate(sentence_pair[1])
            x[i,j]=word
        end
#         print(x[i,:])
        prepend!(sentence_pair[2],d.tgt.vocab.eos)
        append!(sentence_pair[2], d.tgt.vocab.eos)
        for j in 1:(longest_tgt+2-length(sentence_pair[2]))
            append!(sentence_pair[2], d.tgt.vocab.eos)
        end
        for (j, word) in enumerate(sentence_pair[2])
            y[i,j]=word
        end
#         print(y[i,:])
    end
#     print(longest_src)
#     print(longest_tgt)
#     return (reshape(x, (batchsize, longest_src)), reshape(y, (batchsize, longest_tgt)))
#     print(size(x))
    return (x,y)
end


function loss(model, data; average=true)
    if average; Knet.mean(model(x,y) for (x,y) in data)
    else
        total = 0
        counter = 0
        for (x,y) in data; out = model(x,y; average=average); total += out[1]; counter += out[2]; end
        return (total, counter)
    end
end

# Utility to convert int arrays to sentence strings
function int2str(y,vocab)
    y = vec(y)
    ysos = findnext(w->!isequal(w,vocab.eos), y, 1)
    ysos == nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1+length(y))
    join(vocab.i2w[y[ysos:yeos-1]], " ")
end


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
