# JITrench


<h1 align="center">
  <img src=https://cdn.discordapp.com/attachments/810478331790491681/855768153913425930/unknown.png  width=450><br/>
  <img src=https://cdn.discordapp.com/attachments/810478331790491681/855763093072904192/unknown.png width=400>
</h1>
<p align="center">JITrench.jl is lightweight, <br>scalable, <br>and all the action is in your hands.<br>Let's dive into the deep trench of the loss function, <br>with JITrench.jl.</b></p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://abap34.github.io/JITrench.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://abap34.github.io/JITrench.jl/dev)
[![Build Status](https://travis-ci.com/abap34/JITrench.jl.svg?branch=master)](https://travis-ci.com/abap34/JITrench.jl)


---

## Install
```
]add https://github.com/abap34/JITrench.jl
```

## First Step
```julia                                                 
julia> using JITrench
[ Info: Precompiling JITrench [573e55e6-eb53-4bcf-a884-6670806246ed]

julia> x = Variable(3)
name: nothing
values: 3
creator: User-Defined (nothing)

julia> f(x) = 2x^2 + 5x + 10
f (generic function with 1 method)

julia> y = f(x)
name: nothing
values: 43
creator: JITrench.Add

julia> backward!(y)

julia> x.grad
name: nothing
values: 17
creator: JITrench.Add
```


# Display a calculation graph

```julia
julia> x = Variable(2.5, name="x")
name: x values: 2.5
creator: User-Defined(nothing)

julia> y = Variable(3.5, name="y")
name: y values: 3.5
creator: User-Defined(nothing)

julia> goldstain(x, y) = 
       (1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6x*y + 3y^2)) *  (30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36x*y + 27*y^2))
goldstain (generic function with 1 method)

julia> z = goldstain(x, y)
name: nothing values: 1.260939725e7
creator: JITrench.Mul

julia> JITrench.plot_graph(z, to_file="graph.png")
Process(`dot /file/to/dir/.JITrench/tmp_graph.dot -T png -o graph.png`, ProcessExited(0))
```

![](https://cdn.discordapp.com/attachments/810478331790491681/916949639222149160/graph.png)