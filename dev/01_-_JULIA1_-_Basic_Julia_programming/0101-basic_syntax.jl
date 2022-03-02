################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################


# # 0101 - Basic Syntax Elements


# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
ENV["PYTHON"] = ""  # This will be needed in a further segment  
ENV["R_HOME"] = "*" # This wil lbe needed in a further segment
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)


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
## println("This would error!")


# ## Unicode support

# Actually you can use any fancy unicode character and modifiers in the names of variable, type, funcion..
using Statistics # for the `mean` function, in the Standard Library
σ²(x) = sum( (x .- mean(x)).^2 )/length(x) 
σ²([1,2,3])
x̄ₙ = 10
vàlidVarNαme! = 2

# ## Broadcasting

10 .+ [1,2,3] 
add2(x) = x + 2
add2(10)
## add2([1,2,3]) # would return an error
add2.([1,2,3])  # any, including user defined functions, can be broadcasted. No need for map, for loops, etc..


# ## 1 based arrays

a = [1,2,3]
a[1]
# !!! joke "0 or 1 ?"
#     Should array indices start at 0 or 1?  My compromise of 0.5 was rejected without, I thought, proper consideration. --Stan Kelly-Bootle


# ## Basic Mathematic operations

# ## All standard mathepatical arithmetic operators (`+`,`-`,`*`,`/`) are supported in the obvious way. 
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


# ## Missingness implementations

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
isequal(b,2)  # exception to the propagation rule, it allows comparations when one of the item may be missing
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
rand(30:40,10) # A vector of 10 random numbers.
rand(Exponential(10),10,23)
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

