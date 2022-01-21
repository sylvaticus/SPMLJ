
cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()

module TestModule
export plusOne, multiply # functions, structs, and other objects that will be directly available once `using ModuleName` is typed

plusOne(x) = x + 1
multiply(x,y) = x * y 

end

using .TestModule

plusOne(1.0)
plusOne(1)

multiply(2,3)

include("includedfoo.jl") # which strings will be printed ?

x # error not defined
foo.x
using foo # error: looking up for a package and of course can't find it
using .foo
x
foo.z()
foo.r()