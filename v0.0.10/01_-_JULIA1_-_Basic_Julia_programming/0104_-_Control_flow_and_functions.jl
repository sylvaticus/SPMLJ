################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################

# ## 0104 Control Flow and Functions

# ## Some stuff to set-up the environment..
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")     
## If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
## Pkg.resolve()   
## Pkg.instantiate() # run this if you didn't in Segment 01.01
using Random
Random.seed!(123)
using InteractiveUtils # loaded automatically when working... interactively

# ## Variables scope

# The scope of a variable is the region of code where the variable can be accessed directly (without using prefixes).
# Modules, functions, `for` and other blocks (but notably not "if" blocks) introduce an inner scope that hinerit from the scope where the block or function is defined (but not, for function, from the caller's scope).
# Variables that are defined outside any block or function are _global_ for the module where they are defined (the `Main` module if outside any other module, e.g. on the REPL), the others being _local_.
# Variables defined in a `for` block that already exists as global behave differently depending if we are working interactively or not:

g  = 2
g2 = 20
for i in 1:2
    l1 = g2                                       # l1: local, g2: global (read only)
    l1 += i
    g = i                                         # REPL/INTERACTIVE: global (from Julia 1.5), FILE MODE: local by default (with a warning `g` being already defined)
    g += i                                       
    println("i: $i")
    println("l1: $l1")
    println("g: $g")
    for j in 1:2
        l1 += j                                   # still the local in outer loop, not a new local one 
        l2 = j
        g  += j
        println("j: $j")
        println("l1 inside inner loop: $l1")
        println("l2 inside inner loop: $l2")
        println("g inside inner loop: $g")
    end
    ## println("l2 post: $l2")                     # error: l2 not defined in this scope 
    println("l1 post: $l1")
    println("g post: $g")
end
## println("l1 global $l1")                        # error; l1 is not defined in the global scope
println("g in global: $g")                        # REPL/INTERACTIVE: "7", FILE MODE: "2"

function foo(i)
    l1 = g2                                       # l1: local, g2: global (read only)
    l1 += i
    g = i                                         # REPL/INTERACTIVE and FILE MODE: local by default (with a warning `g` being already defined)
    g += i                                       
    println("i: $i")
    println("l1: $l1")
    println("g: $g")
    for j in 1:2
        l1 += j                                   # still the local in outer loop, not a new local one 
        l2 = j
        g  += j
        println("j: $j")
        println("l1 inside inner loop: $l1")
        println("l2 inside inner loop: $l2")
        println("g inside inner loop: $g")
    end
    ## println("l2 post: $l2")                     # error: l2 not defined in this scope 
    println("l1 post: $l1")
    println("g post: $g")
end

println("Calling foo..")
foo(10)
println("g in global: $g")                        # REPL/INTERACTIVE: "7", FILE MODE: "2"

g = 2
include("010401-varScopeExample.jl.txt")            # gives a warning !


# ## Repeated iterations: `for` and `while` loops, List Comprehension, Maps

for i in 1:2, j in 3:4             # j is the inner loop
    println("i: $i, j: $j")
end
a = 1
while true             # or condition, e.g. while a == 10
    global a += 1
    println("a: $a")
    if a == 10
        break 
    else 
        continue
    end
    println("This is never printed")
end

# ### List Comprehension
[ i+j for i in 1:2, j in 3:4 if j >= 4]

# ### Maps

# Apply a (possible anonymous) function to a list of arguments:

map((name,year) -> println("$name is $year year old"), ["Marc","Anna"], [25,22])

# !!! warninng
#     Don't confuse the single-line arrows used in anonymous functions (`->`) with the double-line arrow used to define a Pair (`=>`)

# We can use maps to substitute values of an array based on a dictionary:
countries = ["US","UK","IT","UK","UK"]
countryNames = Dict("IT" => "Italy", "UK" => "United Kngdom",  "US"=>"United States")
countryLongNames = map(cShortName -> countryNames[cShortName], countries)

# ## Conditional statements: if blocks and ternary operators

a = 10

if a < 4                      # use `!`, `&&` and `||` for "not", "and" and  "or" conditions
    println("a < 4")
elseif a < 8
    println("a < 8")
else 
    println("a is big!")
end

a = 10

if a < 5
  b = 100
else 
  b = 200
end

b

# Ternary operators
b =  a < 5 ? 100 : 200   # ? condition : if true : if false

# Short-circuit evaluation
b = 100
(a < 5) || (b = 200)    # replace an if: the second part is executed unless the first part is already true
b
(a < 50) || (b = 500)   # here is never executed
b

# !!! warning 
#     Don't confuse boolean operators `&&` and `||` with their analogous `&` and `|` bitwise operators

a = 3  
b = 2 
bitstring(a)
bitstring(b)

##a && b     # error non boolean used in boolean context
a & b       
a | b 


# ## Functions

function foo(x)             # function definition
    x+2
end
foo(2)                      # function call
inlineFunction(x) = x+2
foo2 = x -> x+2             # anonymous function (aka "lambda function") and assignment to the variable foo2
foo2(2)

# A nested function:
function f1(x)
    function f2(x,y)
        x+y
    end
    f2(x,2)
end

f1(2)

# A recursive function:
function fib(n)                   # This is a naive implementation. Much faster implementations of the Fibonacci numbers exist
    if n == 0  return 0
    elseif n == 1 return 1
    else
     return fib(n-1) + fib(n-2)
    end
end
fib(4)

# ### Function arguments

# #### Positional vs keyword arguments
f(a,b=1;c=1) = a+10b+100c  # `a` and `b` are positional arguments (`b` with a default provided), `c` is a keyword argument
f(2)
f(2,c=3)

foo(a, args...;c=1) = a + length(args) + sum(args) + c
foo(1,2,3,c=4)

# Rules for positional and keyword arguments:
# - keyword arguments follow a semicolon `;` in the parameters list of the function definition 
# - a positional argument without a default can not follow a positional argument with a default provided
# - the splat operator to define variable number of arguments must be the last positional argument 
# - the function call must use positional arguments by position and keyword arguments by name


## foo(a::String="aaa",b::Int64) = "$a "+string(b) # error! Optional positional argument before a mandatory positional one  

# ### Argument types and multiple dispatch

# Simple to understand the usage, complex to understand the deep implications
foo3(a::Int64,b::String) = a + parse(Int64,b)
foo3(2,"3")
foo3(a::String,b::Int64) = parse(Int64,a) + b
foo3("3",2)
methods(foo3)

# Multiple dispatch allows to compile a _specialised_ version JIT at run time, on the first call with the given parameters type
# We will see it again when dealing with type inheritance
# In general, unless we need to write specialised methods, no need to specify the type of the parameters. No influence on performances, this is automatically inferred (and the funciton compiled) based on the run-time type of the argument

# !!! tip Functions performances tip
#     The most important things for performances are (1) that the function is _type stable_, that is, that conditional to a specific combination of the types of the parameters the function returns the same type. This is a condition necessary to have a working chain of type inference across function calls; (2) that no (non constant) global constants are used in the function and indeed all the required information for the functio ndoing its work is embedded in the function parameters


# ### Function templates
foo3(a::T,b::String) where {T<: Number} = a + parse(T,b)             # can use T in the function body
foo3(2,"1")
foo3(1.5,"1.5")
foo4(a::Int64,b::T where T <: Number) = a + b                        # ok not used in functio nbody
foo4(a::Int64,b::Array{T} where T <: Number) = a .+ fill(T,b,2)      # wil lerror, can't use T in the function body
## foo4(2,[1,2])                                                     # run time error, T not defined 

# ### Call by reference vs. call by value 

# How the variable used as function argument within the function body relates to the variable used in calling the function ?
# - **call by value**: the value of the argument is copied and the function body works on a copy of the value
# - **call by reference**: the function works on the same object being referenced by the caller variable and the function argument
# - **call by sharing** (Julia): the arguments are just new local variables that bind the same object. The effects of "modifications" on the local variable on the caller's one depends on the mutability property of the object as we saw in the _Types and objects_ segment:
#   - immutable objects: we can only have that the argument is rebinded to other objects. No effects on the original caller object
#   - mutable objects: if the argument is rebinded to an other object, no effects on the caller object. If the object is modified, the caller object (being the same object) is also modified

x = 10
foo(y) = y = 1
foo(x)
x
foo(x) = x[1] = 10
x = [1,2]
foo(x)
x
foo3(x) = x = [10,20]
foo3(x)
x


# !!! info
#     Functions that modify at least one of their arguments are named, by convention, with an exclamation mark at the end of their name and the argument(s) that is (are) modified set as the first(s) argument(s) 

foo!(x) = x[1] = 10 # to follow the convention


# ## `do` blocks

# Functions that accept an other function as their first parameter can be rewritten with the function itself defined in a `do` block:
using Statistics
pool(f,x,poolSize=3) = [f(x[i:i+poolSize-1]) for i in 1:length(x)-poolSize+1] # a real case, used in neural networks as pooling layer
pool(mean,[1,2,3,4,5,6])
pool(maximum,[1,2,3,4,5,6])
pool([1,2,3,4,5]) do x      # x is a local variable within the do block. We need as many local variables as the number of parameters of the inner function
    sum(x)/length(x)
end
# Using the `do` block we can call the outer function and define the inner function at the same time.
# `do` blocks are frequently used in input/output operations
