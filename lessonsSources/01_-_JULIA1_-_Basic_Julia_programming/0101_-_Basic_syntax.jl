################################################################################ #src
###  Introduction to Scientific Programming and Machine Learning with Julia  ### #src
###                                                                          ### #src
### Run each script on a new clean Julia session                             ### #src
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ### #src
### Licence (apply to all material of the course: scripts, videos, quizes,..)### #src
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ### #src
################################################################################ #src


# # 0101 - Basic Syntax Elements


# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
ENV["PYTHON"] = ""  # This will be needed in a further segment  
ENV["R_HOME"] = "*" # This will be needed in a further segment
## If using a Julia version different than 1.10 please uncomment and run the following line (reproducibility guarantee will however be lost)
## Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)


# ## Comments

# Similar to many other languages, everything that folows a hash symbol (`#`), up to the end of the line, is considered by Julia as a comment.
# Julia supports also multi-line, middle-of-the-line and nested comments using the `#= ...comment... =#` syntax: 

## This is a comment
a = 1 # also this one
a = #= also this one =# 1
#= also
#= this =#
one 
=#


# ## Code organisation

# Delimiting the end of a programming statement with a semi-colon, is optional, and has the effect to block the normal display on the terminal of the output of that statement.

## Semicolon:
a = 1
a = 1;

# All code blocks (like the foor loop that we'll study in detail in the [Control flow and functions](@ref control_flow) segment) ends with the `end` keyword, like in the example below:
for i in 1:3
   println("i is $i")
end # Keyword `end` to finish a block

# Function calls require to specify the name of the function to call followed straight away (without spaces) with the function arguments inside round brackets:
println("Hello world!")
##println ("This would error!")


# ## Unicode support

# We can can use any fancy Unicode symbol, including modifiers, in the names of variables, types, functions..
using Statistics # for the `mean` function, in the Standard Library
σ²(x) = sum( (x .- mean(x)).^2 )/length(x) # The superscript `²` is just another character, it has no syntactic value.
σ²([1,2,3])
x̄ₙ = 10
vàlidVarNαme! = 2
🥞 = 8
🍴(🥫,parts=4) = 🥫/parts
🍰 = 🍴(🥞)

# ## Broadcasting

# _Broadcasting_ refers to the capacity to dispatch a function that accepts a scalar argument to a whole collection. Broadcasting will then call the function with each individual element of the collection.
# It is implemented by postfixing a function name (or prefixing an operator) with a single dot at call time:

10 .+ [1,2,3] 
add2(x) = x + 2
add2(10)
## add2([1,2,3]) # would return an error
add2.([1,2,3])  # any, including user defined functions, can be broadcasted. No need for map, for loops, etc..
add2.((1,2,3)) # not only for arrays, here is a Tuple
add2.(Set([1,2,3,2])) # and here a Set
.+([1,2,3],[10,20,30]) # fine here
## .+([1,2,3],[10,20]) # DimensionMismatch error: the input of the the broadcasted arguments must have the same size or be a scalar

# To "protect" one specific argument to be broadcasted, use `Ref(arg)`:
foo(x,y::AbstractArray) = [yi+x for yi in y] # a function that wants the first argument as scalar and the second as vector
foo(1,[10,20])
## foo.([1,2],[10,20])  # error, as we try to broadcast also the seond element, but we can't call `foo` with two integers
foo.([1,2],Ref([10,20])) # now it is fine, we broadcast only the first argument

# ## 1 based arrays
# Arrays start indexing at 1 (like R, Fortran, Matlab,...) instead of 0 (like in C, Python, Java...)
a = [1,2,3]
a[1]
collect(1:3) # in ranges, both extremes are included

# !!! joke "0 or 1 ?"
#     Should array indices start at 0 or 1?  My compromise of 0.5 was rejected without, I thought, proper consideration. --Stan Kelly-Bootle


# ## Basic Mathematic operations

# All standard mathematical arithmetic operators (`+`,`-`,`*`,`/`) are supported in the obvious way:

a = 2^4         # rise to power
b = ℯ^2; #= or =# b = exp(2) # Exponential with base ℯ 
d = log(7.3890) # base ℯ
e = log(10,100) # custom base
f = 5 ÷ 2       # integer division
e = 5 % 2       # reminder (modulo operator)
a = 2//3 + 1//3     # rational numbers
typeof(a)
π == pi         # some irrational constants
typeof(ℯ)
convert(Float64,a)

# ## Quotation

a = 'k'              # single quotation mark: a single Char
b = "k"              # double quotation mark: a (Unicode) String
#c = 'hello'         # error !
c = "hello"
d = `echo hello`     # backtick: define a command to (later) run (e.g. on the OS)
e = """a
multiline
string"""
println(c)
println(e)
using Markdown
f = md"""a
**markdown**
_string_"""


# ## [Missingness implementations](@id missingness_implementations)

# Attention to these three different implementations (of 3 different concepts) of "missingness":

a = nothing # C-style, "software engineer's null → run-time error
b = missing # Data scientist's null → silent propagation
c = NaN     # Not a number → silent propagation
typeof(a)
typeof(b)
typeof(c)
d = 0/0
## a2 = mean([1,a,3]) # would error
b2 = mean([1,b,3])
c2 = mean([1,c,3])
b3 = mean(skipmissing([1,b,3]))
b == missing # propagate
ismissing(b)
isequal(b,2)  # exception to the propagation rule, it allows a comparison when one of the item may be missing
isequal(b,missing)
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
rand(30:40,3) # A vector of 3 random numbers.
rand(Exponential(10),3,4) # Same syntax for sampling from any distribution ! 
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

a = rand(myRNG,Exponential(10),5)

