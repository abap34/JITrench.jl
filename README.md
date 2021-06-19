JITrench

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://abap34.github.io/JITrench.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://abap34.github.io/JITrench.jl/dev)
[![Build Status](https://travis-ci.com/abap34/JITrench.jl.svg?branch=master)](https://travis-ci.com/abap34/JITrench.jl)
<h1 align="center">
  <img src=https://cdn.discordapp.com/attachments/810478331790491681/855696564952367134/unknown.png  width=250><br/>
  <img src=https://cdn.discordapp.com/attachments/810478331790491681/855763093072904192/unknown.png width=400>
</h1>
<p align="center">JITrench.jl is lightweight, <br>scalable, <br>and all the action is in your hands.<br>Let's dive into the deep trench of the loss function, <br>with JITrench.jl.</b></p>




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
