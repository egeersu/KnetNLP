import Pkg
Packages = ["IterTools", "LinearAlgebra", "LinearAlgebra", "StatsBase", "Test"]
for p in Packages; Pkg.add(p); end;
using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test

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
