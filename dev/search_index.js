var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = JITrench","category":"page"},{"location":"#JITrench","page":"Home","title":"JITrench","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [JITrench]","category":"page"},{"location":"#JITrench.DiffableFunction","page":"Home","title":"JITrench.DiffableFunction","text":"DiffableFunction\n\nAn abstract type that is the parent type of differentiable functions;\n\nall function in JITrench must be children of this type.\n\nExamples\n\njulia> subtypes(DiffableFunction)\n24-element Vector{Any}:\n JITrench.Add\n JITrench.BroadcastTo\n JITrench.Broadcasting\n JITrench.Cos\n JITrench.Div\n JITrench.Exp\n JITrench.Flatten\n JITrench.GetIndex\n JITrench.GetIndexGrad\n JITrench.Log\n ⋮\n JITrench.Reshape\n JITrench.Sigmoid\n JITrench.Sin\n JITrench.Sub\n JITrench.Sum\n JITrench.SumTo\n JITrench.Tan\n JITrench.Transpose\n JITrench._Linear\n\n\n\n\n\n","category":"type"},{"location":"#JITrench.GradField","page":"Home","title":"JITrench.GradField","text":"GradField\n\nA structure that contains all the necessary items for automatic differentiation. All function types in JITrench have this type as a field named \"grad_field\".\n\n\n\n\n\n","category":"type"},{"location":"#JITrench.Model","page":"Home","title":"JITrench.Model","text":"Model\n\nAbstract type representing a model.\n\nChild types of this abstract type are recognized as models by JITrench.\n\nFor more information, refer to the \"Neural Networks\" section of the documentation (here).\n\n\n\n\n\n","category":"type"},{"location":"#JITrench.PNGContainer","page":"Home","title":"JITrench.PNGContainer","text":"PNGContainer\n\nA structure for handling images. You can display an image by using it as a return value of a cell or by explicitly display(::PNGContainer) in Jupyter.\n\n\n\n\n\n","category":"type"},{"location":"#JITrench.TrenchObject","page":"Home","title":"JITrench.TrenchObject","text":"TrenchObject\n\nAn abstract type that is the root of an object as implemented in  Specifically, see also subtypes(foo).\n\njulia> subtypes(TrenchObject)\n2-element Vector{Any}:\n DiffableFunction\n Variable\n\n\n\n\n\n","category":"type"},{"location":"#JITrench.Variable","page":"Home","title":"JITrench.Variable","text":"Variable(values, [creator, grad, generation, name])\n\nA type that is a variable in JITrench. \n\nFor a detailed explanation of this, see the documentation(here).\n\nOnly real numbers or arrays consisting of real numbers can be stored.\n\nExamples\n\njulia> Variable(2)\nname: nothing \nvalues: 2\ncreator: User-Defined(nothing)\n\njulia> Variable([1.2, 3.5])\nname: nothing \nvalues: [1.2, 3.5]\ncreator: User-Defined(nothing)\n\n\n\n\n\n","category":"type"},{"location":"#Base.getindex-Tuple{Variable, Vararg{Any, N} where N}","page":"Home","title":"Base.getindex","text":"Base.getindex(x::Variable, ind...)\n\nreturn x.values[ind...] as Variable.\n\nExamples\n\njulia> x = Variable(rand(2, 2))\nname: nothing \nvalues: [0.7050007249265509 0.5075375401538957; 0.9953109600473362 0.8447135817368259]\ncreator: User-Defined(nothing)\n\njulia> x[1, 2]\nname: nothing \nvalues: 0.5075375401538957\ncreator: JITrench.GetIndex\n\njulia> x[1, :]\nname: nothing \nvalues: [0.7050007249265509, 0.5075375401538957]\ncreator: JITrench.GetIndex\n\n\n\n\n\n","category":"method"},{"location":"#Base.sum-Tuple{Variable}","page":"Home","title":"Base.sum","text":"sum(x::Variable; dims=nothing, keepdims=false)\n\nArguments\n\nkeepdims\n\nWhen true, the number of dimensions in the input and output arrays is guaranteed to match.\n\nExamples\n\njulia> x = Variable([1, 2, 3])\nname: nothing \nvalues: [1, 2, 3]\ncreator: User-Defined(nothing)\n\njulia> y = sum(x)\nname: nothing \nvalues: 6\ncreator: JITrench.Sum\n\njulia> x = Variable(rand(2, 2, 2))\nname: nothing \nvalues: [0.8036752887616154 0.918293421092931; 0.9092024610524445 0.8788506330169976]\n\n[0.7726260228472543 0.6884237844746439; 0.38214845033860456 0.660455509851009]\ncreator: User-Defined(nothing)\n\njulia> y = sum(x, keepdims=true)\nname: nothing \nvalues: [6.013675571435501]\ncreator: JITrench.Sum\n\njulia> y.values\n1×1×1 Array{Float64, 3}:\n[:, :, 1] =\n 6.013675571435501\n\n\n\n\n\n","category":"method"},{"location":"#Base.transpose-Tuple{Variable}","page":"Home","title":"Base.transpose","text":"transpose(x::Variable)\n\nExamples\n\njulia> x = Variable([1 2; 3 4])\nname: nothing \nvalues: [1 2; 3 4]\ncreator: User-Defined(nothing)\n\njulia> transpose(x)\nname: nothing \nvalues: [1 3; 2 4]\ncreator: JITrench.Transpose\n\n\n\n\n\n","category":"method"},{"location":"#JITrench._broadcast_to-Union{Tuple{R}, Tuple{R, Any}} where R<:Real","page":"Home","title":"JITrench._broadcast_to","text":"_broadcast_to(A, shape)\n\nApply broadcast to make size(x.values) as shape.\n\nExamples\n\njulia> _broadcast_to([1, 2, 3], (3, 3))\n3×3 Matrix{Float64}:\n 1.0  1.0  1.0\n 2.0  2.0  2.0\n 3.0  3.0  3.0\n\njulia> _broadcast_to(1, (2, 2))\n2×2 Matrix{Float64}:\n 1.0  1.0\n 1.0  1.0\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.backward!-Tuple{Variable}","page":"Home","title":"JITrench.backward!","text":"backward(y::Variable; retain_grad=false)\n\nCompute the back propagation, y as the end of the computational graph. For a more detailed explanation, see the documentation (here).\n\njulia> x\nname: nothing \nvalues: 1\ncreator: User-Defined(nothing)\n\njulia> x = Variable(1)\nname: nothing \nvalues: 1\ncreator: User-Defined(nothing)\n\njulia> y = x +  2\nname: nothing \nvalues: 3\ncreator: JITrench.Add\n\njulia> backward!(y)\n\njulia> x.grad\nname: nothing \nvalues: 1\ncreator: User-Defined(nothing)\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.broadcast_to-Tuple{Variable, Any}","page":"Home","title":"JITrench.broadcast_to","text":"broadcast_to(x::Variable, shape)\n\nApply broadcast to make size(x.values) as shape.　\n\nExamples\n\njulia> x = Variable([1, 2, 3])\nname: nothing \nvalues: [1, 2, 3]\ncreator: User-Defined(nothing)\n\njulia> JITrench.broadcast_to(x, (3, 2))\nname: nothing \nvalues: [1.0 1.0; 2.0 2.0; 3.0 3.0]\ncreator: JITrench.BroadcastTo\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.cleargrad!-Tuple{Variable}","page":"Home","title":"JITrench.cleargrad!","text":"cleargrad!(x::Variable)\n\nReset the Variable's gradient.\n\njulia> x\nname: nothing \nvalues: 1\ngrad: Variable(1)\ncreator: User-Defined(nothing)\n\njulia> x.grad\nname: nothing \nvalues: 1\ncreator: User-Defined(nothing)\n\njulia> cleargrad!(x)\n\njulia> x.grad === nothing\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.cleargrads!-Tuple{Model}","page":"Home","title":"JITrench.cleargrads!","text":"cleargrads!(model::Model; skip_uninitialized=false)\n\nclear information about grad in all parameters in model. if skip_uninitialized is true, parameters which is uninitialized(i.e nothing) is skipped.\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.flatten-Tuple{Variable}","page":"Home","title":"JITrench.flatten","text":"flatten(x::Variable)\n\nThe function corresponding to vcat(x...).\n\nExample\n\njulia> x = Variable([1 2; 3 4; 5 6]) name: nothing  values: [1 2; 3 4; 5 6] creator: User-Defined(nothing)\n\njulia> JITrench.flatten(x) name: nothing  values: [1, 3, 5, 2, 4, 6] creator: JITrench.Flatten\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.matmul-Tuple{Any, Any}","page":"Home","title":"JITrench.matmul","text":"matmul(x, W)\n\nMatrix multiplication.\n\nExample\n\njulia> x = Variable([1 1; 0 1])\nname: nothing \nvalues: [1 1; 0 1]\ncreator: User-Defined(nothing)\n\njulia> W = Variable([1 0; 1 1])\nname: nothing \nvalues: [1 0; 1 1]\ncreator: User-Defined(nothing)\n\njulia> matmul(x, W)\nname: nothing \nvalues: [2 1; 1 1]\ncreator: MatMul\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.mean_squared_error-Tuple{Any, Any}","page":"Home","title":"JITrench.mean_squared_error","text":"mean_squared_error(y_true, y_pred)\n\nExamples\n\njulia> y_true = Variable([10, 20, 30])\nname: nothing \nvalues: [10, 20, 30]\ncreator: User-Defined(nothing)\n\njulia> y_pred = Variable([12, 20, 25])\nname: nothing \nvalues: [12, 20, 25]\ncreator: User-Defined(nothing)\n\njulia> mean_squared_error(y_true, y_pred)\nname: nothing \nvalues: 9.666666666666666\ncreator: JITrench.MeanSquaredError\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.plot_graph-Tuple{Variable}","page":"Home","title":"JITrench.plot_graph","text":"plot_graph(var::Variable; to_file=\"\", title=\"\")\n\nDraw a computational graph with y as the end. This requires the installation of graphviz.\n\nArguments\n\nto_file: Specifies the file name where the image will be saved.\n\nIf it is an empty string, it will return an image of type PNGContainer. See also: PNGContainer\n\ntitle: Title of graph.\n\njulia> x = Variable(10)\nname: nothing \nvalues: 10\ncreator: User-Defined(nothing)\n\njulia> y = x + 3\nname: nothing \nvalues: 13\ncreator: JITrench.Add\n\njulia> JITrench.plot_graph(y, to_file=\"graph.png\") \n\n\n\n\n\n","category":"method"},{"location":"#JITrench.plot_model-Tuple{Model, Vararg{Any, N} where N}","page":"Home","title":"JITrench.plot_model","text":"plot_model(model::Model, inputs...,  to_file=\"model.png\")\n\nplot model by graphviz. Note that JITrench builds the graph at the time of calculation, so some kind of input is required.\n\n\n\n\n\n","category":"method"},{"location":"#JITrench.sum_to-Tuple{Any, Any}","page":"Home","title":"JITrench.sum_to","text":"sum_to(x, shape)\n\nTake the sum of each axis to form a shape.\n\nExamples\n\njulia> x = Variable(rand(2, 3))\nname: nothing \nvalues: [0.2397911359535343 0.34270903251201146 0.699060178623987; 0.2345132451371843 0.21845435948625758 0.2924942369518322]\ncreator: User-Defined(nothing)\n\njulia> JITrench.sum_to(x, (2, 1))\nname: nothing \nvalues: [1.2815603470895327; 0.7454618415752741]\ncreator: JITrench.SumTo\n\n\n\n\n\n","category":"method"}]
}
