import Pkg
Packages = ["IterTools", "LinearAlgebra", "LinearAlgebra", "StatsBase", "Test"]
for p in Packages; Pkg.add(p); end;
using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test

function tokenize(str)
    out = []
    str = split(str)
    symbols = ['(', ':',';',',', ')', '.', '?', '!']
    finish_symbols = [')', ',', '.']
    bad_strings = ["<br", "/><br", "/>", "/><br", ".<br", "\"", "..."]
    for word in str
        f = true
        word = lowercase(word)
        put_symbols = []
        for symbol in symbols
            if occursin(string(symbol), word); word = strip(word, [symbol]); push!(put_symbols, symbol); f=false; end;
        end
        for bad in bad_strings
            if occursin(bad, word); word = replace(word, bad=>""); f=false; break; end
        end
        for s in put_symbols
            if !(s in finish_symbols); push!(out, s); end
        end
        push!(out, word)
        for s in put_symbols
            if (s in finish_symbols); push!(out, s); end
        end
    end
    #remove zero-length strings
    out = [word for word in out if length(word) > 0]
end

# Utility to convert int arrays to sentence strings
function int2str(y,vocab)
    y = vec(y)
    ysos = findnext(w->!isequal(w,vocab.eos), y, 1)
    ysos == nothing && return ""
    yeos = something(findnext(isequal(vocab.eos), y, ysos), 1+length(y))
    join(vocab.i2w[y[ysos:yeos-1]], " ")
end
