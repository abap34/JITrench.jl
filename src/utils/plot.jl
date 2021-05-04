function get_value_type(nested_var)
    if eltype(nested_var) <: Real
        return eltype(nested_var)
    else
        get_value_type(eltype(nested_var))
    end
end

const tmp_dir = joinpath(expanduser("~"), ".JITrench")
const dot_file_path = joinpath(tmp_dir, "tmp_graph.dot")

struct PNGContainer
    content
end

function Base.show(io::IO, ::MIME"image/png", c::PNGContainer)
    write(io, c.content)
end


function _dot_var(var::Variable, show_value)
    name = (var.name == "") ? "" : var.name * ":"
    value = (show_value == 0 ? var.data : var.grad.data)
    if var.data !== nothing
        var_size = size(value)
        if isempty(var_size)
            try var.data !== nothing
                name *= "$(value)"
            catch
                name *= "nothing"
            end
        else    
            name *= "shape: $(var_size) \n type: $(get_value_type(value))"
        end
    end
    dot_var = "$(objectid(var)) [label=\"$name\", color=orange, style=filled]\n"
end


function _dot_func(f::Functional)
    f_type = typeof(f)
    if f_type == JITrench.Functional.Pow
        f_type = "$(f_type)($(f.c))"
    end
    txt = "$(objectid(f)) [label=\"$(f_type)\", color=lightblue, style=filled, shape=box]\n"
    for x in f.inputs
        txt *= "$(objectid(x)) -> $(objectid(f))\n"
    end
    for y in f.outputs
        txt *= "$(objectid(f)) -> $(objectid(y))\n"
    end
    return txt
end



function get_dot_graph(var, show_value, title)
    txt = ""
    funcs = []
    seen_set = Set{Func}()
    push!(funcs, var.creator)
    txt = _dot_var(var, show_value)
    while !(isempty(funcs))
        f = pop!(funcs)
        txt *= _dot_func(f)
        for x in f.inputs
            txt *= _dot_var(x, show_value)
            if x.creator !== nothing && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                push!(funcs, x.creator)
            end
        end
    end
    return "digraph g {
            graph [
                labelloc=\"t\";
                label= \"$(title)\"
            ];
                $txt 
            }"
end


function plot_tmp_dir(extension)
    extension = "png"
    to_file = joinpath(tmp_dir, "graph.png")       
    cmd = `dot $(dot_file_path) -T $(extension) -o $(to_file)`
    run(cmd)
    return to_file
end

function plot_graph(var::Variable; to_file="", show_value="data", title="")
    if show_value == "grad"
        show_value = 1
    elseif show_value == "data"
        show_value = 0
    else
        error("The argument to show_value is invalid.")
    end

    dot_graph = get_dot_graph(var, show_value, title)
    (!(ispath(tmp_dir))) && (mkdir(tmp_dir))
    open(dot_file_path, "w") do io
        write(io, dot_graph)
    end
    
    if to_file == ""
        png_file_path = plot_tmp_dir(".png")
        c = open(png_file_path) do io
            PNGContainer(read(io))
        end
        return c
    else
        extension = split(to_file, ".")[2]
        cmd = `dot $(dot_file_path) -T $(extension) -o $(to_file)`
        run(cmd)
    end
end