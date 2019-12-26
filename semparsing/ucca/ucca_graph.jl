mutable struct UCCA_graph
    sentence
    tokens
    annotations
end 

function UCCA_graph(file_path)
    out1 = model_output(file_path)
    tokens = out1.tokens
    annotations = out1.annotations
    sentence = out1.sentence
    UCCA_graph(sentence, tokens, annotations)
end

function get_relations(graph, index)
    annotations = []
    for n in graph.annotations
        if n[2] == index
            an = (n[1], n[2], graph.tokens[n[2]], n[3], graph.tokens[n[3]])
            push!(annotations, an);
        end
    end
    annotations
end

function PrintGraph(graph)
    println("Sentence: ", graph.sentence)
    println("...Printing Graph...")
 end