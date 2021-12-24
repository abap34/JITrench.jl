const tmp_dir = joinpath(expanduser("~"), ".JITrench")
const dot_file_path = joinpath(tmp_dir, "tmp_graph.dot")
colors = Dict(
    "func" => "lightblue",
    "var" => "orange",
    "user_defined_var" => "orange2"
)

"""
    PNGContainer
A structure for handling images.
You can display an image by using it as a return value of a cell or by explicitly `display(::PNGContainer)` in Jupyter.


"""
struct PNGContainer
    content
end

function Base.show(io::IO, ::MIME"image/png", c::PNGContainer)
    write(io, c.content)
end


function _dot_var(var::Variable{T}) where T
    name = (var.name === nothing) ? "" : var.name 
    value = var.values
    if value !== nothing
        var_size = size(value)
        if isempty(var_size)
            try value !== nothing
                name *= "\n$(value)"
            catch
                name *= "nothing"
            end
        else    
            name *= "shape: $(var_size) \n type: $(T))"
        end
    end
    if var.creator === nothing
        color = colors["user_defined_var"]
        shape = "doublecircle"
    else
        color = colors["var"]
        shape = "circle"
    end
    dot_var = "$(objectid(var)) [shape=$shape, label=\"$name\", color=\"$color\", style=filled]\n"
    return dot_var
end


function _dot_func(f::DiffableFunction)
    f_type = typeof(f)
    txt = "$(objectid(f)) [label=\"$(f_type)\", color=\"$(colors["func"])\", style=filled, shape=box]\n"
    for x in f.grad_field.inputs
        txt *= "$(objectid(x)) -> $(objectid(f))\n"
    end
    for y in f.grad_field.outputs
        txt *= "$(objectid(f)) -> $(objectid(y))\n"
    end
    return txt
end



function get_dot_graph(var, title)
    txt = ""
    funcs = []
    seen_set = Set{DiffableFunction}()
    push!(funcs, var.creator)
    txt = _dot_var(var)
    while !(isempty(funcs))
        f = pop!(funcs)
        txt *= _dot_func(f)
        for x in f.grad_field.inputs
            txt *= _dot_var(x)
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


function plot_tmp_dir()
    extension = "png"
    to_file = joinpath(tmp_dir, "graph." * extension)       
    cmd = `dot $(dot_file_path) -T $(extension) -o $(to_file)`
    run(cmd)
    return to_file
end



"""
    plot_graph(var::Variable; to_file="", title="")
Draw a computational graph with y as the end. This requires the installation of graphviz.

# Arguments
- `to_file`: Specifies the file name where the image will be saved.
If it is an empty string, it will return an image of type `PNGContainer`. See also: [`PNGContainer`](@ref)
- `title`: Title of graph.


```julia-repl
julia> x = Variable(10)
name: nothing 
values: 10
creator: User-Defined(nothing)

julia> y = x + 3
name: nothing 
values: 13
creator: JITrench.Add

julia> JITrench.plot_graph(y, to_file="graph.png") 
```
"""
function plot_graph(var::Variable; to_file="", title="")
    dot_graph = get_dot_graph(var, title)
    (!(ispath(tmp_dir))) && (mkdir(tmp_dir))
    open(dot_file_path, "w") do io
        write(io, dot_graph)
    end
    
    if to_file == ""
        png_file_path = plot_tmp_dir()
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
