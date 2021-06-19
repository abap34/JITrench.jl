JITrench

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://abap34.github.io/JITrench.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://abap34.github.io/JITrench.jl/dev)
[![Build Status](https://travis-ci.com/abap34/JITrench.jl.svg?branch=master)](https://travis-ci.com/abap34/JITrench.jl)

<img src=https://cdn.discordapp.com/attachments/810478331790491681/855696564952367134/unknown.png  width=400>

##### <b>Let's dive with JITrench.jl into the deep trench of the loss function.</b>

```julia
MacBookPro $ julia                                                   
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.6.0 (2021-03-24)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

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

