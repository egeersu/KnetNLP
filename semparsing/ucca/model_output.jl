import Pkg; 
pkgs = ["LightGraphs", "MetaGraphs", "JSON", "DataFrames"]
for p in pkgs; Pkg.add(p); end
using LightGraphs, MetaGraphs, DataFrames;
import JSON;

mutable struct model_output
    sentence
    tokens
    annotations
end    

function model_output(file_path)
    out = read_output(file_path)
    sentence = generate_sentence(out)
    tokens = generate_nodes(out)
    annotations = generate_annotations(out)
    model_output(sentence, tokens, annotations)
end

function read_output(file_path)
    trn = open(file_path, "r")
    output = JSON.parse(trn)
end

function generate_nodes(out)
    nodes = Dict()
    for (i,d) in enumerate(out["tokens"])
        nodes[i] = d["text"]
    end
    nodes
end

function generate_sentence(out)
    nodes = generate_nodes(out)
    lst = []; Str = ""
    for k in keys(nodes); push!(lst, k); end
    for i in sort(lst); Str *= (nodes[i] * " "); end
    Str
end

function generate_annotations(out)
    nodenum = length(generate_nodes(out))
    edges = []
    for i in (2:nodenum)
        units = out["annotation_units"][i]
        category = out["annotation_units"][i]["categories"][1]["name"]   
        for target in  units["children_tokens"]
            new_target = target["id"]
            origin = i-1
            new_target = out["annotation_units"][i]["children_tokens"][1]["id"]
            new_edge = (category, origin, new_target); push!(edges, new_edge)
        end
    end
    edges
end