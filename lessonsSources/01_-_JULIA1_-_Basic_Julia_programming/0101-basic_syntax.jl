cd(@__DIR__)                    #src
using Pkg                       #src
Pkg.activate("../..")           #src


# # 0101 - Basic Syntax

# ## Comments

## This is a comment
a = 1 # also this one
a = #= also this one =# 1
#= also
#= this =#
one 
=#


# ## Code organisation

## Semicolon:
a = 1
a = 1;
for i in 1:3
    println("i is $i")
end # Keyword `end` to finish a block
println("Hello world!")
## println ("This would error!")


# ## Types

## 1 in not 1.0:
a = 1
b = 1.0
typeof(a) # type is inferred !
typeof(b)
## Convert type (cast)
a = 1
b = convert(Float64,a)
typeof(b)

# Type hierarchy in Julia:
# 
# - Any
#  - AbstractString     # We'll see what all these "abstract" mean....
#    - String
#    - ...
#  - AbstractArray
#    - Array
#    - ....
#  - Number
#    - Complex
#    - Real
#      - Rational
#      - Integer
#        - Unsigned
#          - UInt64
#          - ...
#        - Signed
#          - Int32
#          - Int64
#          - BigInt
#          - ...
#        - Bool
#      - FixedPoints
#        - ...
#      - AbstractIrrational
#        - Irrational
#      - AbstractFloat
#        - Float32
#        - Float64
#        - BigFloat
#        - ...
#  - ...
# Complete Number hierarchy: https://upload.wikimedia.org/wikipedia/commons/d/d9/Julia-number-type-hierarchy.svg

## Everythong is an object, i.e. of some "type"
c = typeof(a)
typeof(c)
d = sin
typeof(d) <: Function
typeof(+) <: Function

## Operators are just functions:
1 + 2
+(1,2) # this call the function "+"

# ## Unicode support
# Actually you can use any fancy unicode character and modifiers in the names of variable, type, funcion..
using Statistics # for the `mean` function, in the Standard Library
σ²(x) = sum( (x .- mean(x)).^2 )/length(x) 
σ²([1,2,3])
x̄ₙ = 10

# ## Broadcasting
10 .+ [1,2,3] 
add2(x) = x + 2
## add2([1,2,3]) # would return an error
add2.([1,2,3])  # any, including user defined functions, can be broadcasted. No need for map, for loops, etc..

# 1 based arrays
a = [1,2,3]
a[1]
# Should array indices start at 0 or 1?  My compromise of 0.5 was rejected without, I thought, proper consideration.

# ## Memory issues in assignments/copying

a = [[[1,2],3],4] # First element of a is said to be "mutable", second one is not:
isimmutable(a[1])
isimmutable(a[2])
## mutable objects are stored in memory directly, while for mutable objects it is its memory address to be stored
b = a            # name binding: it binds (assign) the entity (object) referenced by a to the b identifier (the variable name)
c = copy(a)      # create a new copy of a and binds it to c
d = deepcopy(a)  # copy all the references recursively and assign this new object to d
c == a           # are the two objects equal ?
c === a          # are the two identifiers binding the same identical object in memory ?
a[2] = 40        # rebinds a[2] to an other objects and at the same time mutates object a:
b
c
d
a[1][2] = 30     # rebinds a[1][2] and at the same time mutates both a and a[1]
b
c
d
a[1][1][2] = 20
b
c
d
a = 5            # rebinds a:
b
c
d
# !!! note: Consider these memory isues when we'll discuss calling a function by reference/value !


# ## Missingness implementations
a = nothing # C-style, "software engineer's null → run-time error
b = missing # Data scientist's null → silent propagation
c = NaN     # Not a number → silent propagation
typeof(a)
typeof(b)
typeof(c)
d = 0/0
##a2 = mean([1,a,3]) # would error
b2 = mean([1,b,3])
c2 = mean([1,c,3])
b3 = mean(skipmissing([1,b,3]))
b == missing
ismissing(b)
b4 = [1,missing,3]
typeof(b4)
eltype(b4)
nonmissingtype(eltype(b4))

# ## Random values
rand()       # [0,1] continuous
rand(30:40)  # [30,40] integer
rand(30:0.01:40) # [30,40] with precision to the second digit
using Distributions
rand(Exponential(10)) # We'll see Distributions more in detail in the Scientific Programming lesson
rand(30:40,10) # A vector of 10 random numbers.
using Random
myRNG = MersenneTwister(123) # use StableRNG for a RNG guaranteed to remain stable between Julia-versions
a1 = rand(myRNG,10:1000,5)
a2 = rand(myRNG,10:1000,5)
a1 == a2
myRNG = MersenneTwister(123)
b1 = rand(myRNG,10:1000,5)
b2 = rand(myRNG,10:1000,5)
b1 == b2
a1 == b1
a2 == b2
