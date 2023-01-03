# JITrench


<h1 align="center">
  <img src=https://cdn.discordapp.com/attachments/810478331790491681/855768153913425930/unknown.png  width=450><br/>
</h1>
<p align="center">lightweight, <br>scalable, <br>and affordable deep learning framework.<br>Let's dive into the deep trenches of the loss function <br>with JITrench.jl.</b></p>

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://abap34.github.io/JITrench.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://abap34.github.io/JITrench.jl/dev)
[![Build Status](https://travis-ci.com/abap34/JITrench.jl.svg?branch=master)](https://travis-ci.com/abap34/JITrench.jl)


---

## Install
```
]add https://github.com/abap34/JITrench.jl
```

## Automatic gradient calculation


```julia                                                 
julia> using JITrench

julia> f(x) = sin(x) + 1
f (generic function with 1 method)

julia> @diff! f(x)
f′ (generic function with 1 method)

julia> f′(π)
-1.0
```


## Compute gradients for "Deep" functions, visualize computational graphs

```julia
julia> x = AutoDiff.Scalar(2.5, name="x")
name: x 
values: 2.5
creator: User-Defined(nothing)

julia> y = AutoDiff.Scalar(3.5, name="y")
name: y 
values: 3.5
creator: User-Defined(nothing)

julia> goldstain(x, y) = (1 + (x + y + 1)^2 * (19 - 14x + 3x^2 - 14y + 6x*y + 3y^2)) *  (30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36x*y + 27*y^2))
goldstain (generic function with 1 method)

julia> z = goldstain(x, y)
name: nothing 
values: 1.260939725e7
creator: JITrench.Mul

julia> backward!(z)

julia> x.grad
-5.324409e6

julia> y.grad
3.3109701e7

julia> JITrench.plot_graph(z, to_file="example/visualize/goldstain.png")
```

![](example/visualize/goldstain.png)

## Compute gradients for operations on multidimensional arrays


```julia
julia> A = AutoDiff.Tensor([1 2; 3 4; 5 6])
name: nothing 
values: [1 2; 3 4; 5 6]
creator: User-Defined(nothing)

julia> B = reshape(A, (2, 3))
name: nothing 
values: [1 5 4; 3 2 6]
creator: JITrench.ArrOperator.Reshape{Tuple{Int64, Int64}, Tuple{Int64, Int64}}

julia> C = B[1, :]
name: nothing 
values: [1, 5, 4]
creator: JITrench.ArrOperator.GetIndex{Tuple{Int64, Int64}, Tuple{Int64, Colon}}

julia> y = sum(C)
name: nothing 
values: 10
creator: JITrench.ArrOperator.Sum

julia> backward!(y)

julia> A.grad
3×2 Matrix{Float64}:
 1.0  0.0
 0.0  1.0
 1.0  0.0
```