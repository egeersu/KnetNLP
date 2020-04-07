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