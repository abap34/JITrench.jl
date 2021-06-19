using JITrench: plot_graph
using JITrench

f(x) = 2x^2 + 5x + 10

x = Variable(2.0, name="x")
#=
name: x 
values: 2.0
creator: User-Defined (nothing)
=#


y = f(x)
#=
name: nothing 
values: 28.0
creator: JITrench.Add
=#


plot_graph(y, to_file="graph.png")


