import Pkg
Packages = ["IterTools", "LinearAlgebra", "LinearAlgebra", "StatsBase", "Test"]
for p in Packages; Pkg.add(p); end;
using Knet, Base.Iterators, IterTools, LinearAlgebra, StatsBase, Test

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    Linear(param(outputsize, inputsize), param0(outputsize))
end

function (l::Linear)(x)
    l.w * mat(x,dims=1) .+ l.b

end


struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    Embed(param(embedsize,vocabsize))
end

function (l::Embed)(x)
    l.w[:,x]
end
