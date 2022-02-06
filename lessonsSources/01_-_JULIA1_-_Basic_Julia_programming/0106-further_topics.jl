################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################

# # 0106 Further Topics


# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")     
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will however be lost)
# Pkg.resolve()   
# Pkg.instantiate()
using Random
Random.seed!(123)
using InteractiveUtils # loaded automatically when working... interactively

# ## Metaprogramming and macros

# "running" some code include the following passages (roughly):
# - parsing of the text defining the code and its translation in hierarchical expressions to the Abstract syntax Tree (AST) (syntax errors are caugth at this time)
# - on the first instance required ("just in time") compilation of the AST expressions into object code (using the LLVM compiler)
# - execution of the compiled object code

# "Macros" in many other language (e.g. C or C++) refer to the possibility to "pre-process" the textual representation of the code statements before it is parsed. In julia instead it refers to the possibility to alter the expression once has already being parsed in the AST, allowing a greater expressivity as we are no longer limited by the parsing syntax

# The AST is organised in a hierarchical tree of _expressions_ where each element (including the operators) is a _symbol_
# For variables, you can use symbols to refer to the actual identifiers instad to the variable's value
# Expressions themselves are objects representing unevaluated computer expressions

# ### Expressions and symbols

expr1 = Meta.parse("a = b + 2") # What the parser do when reading the source code. b doesn't need to actually been defined, it's just a namebinding without the reference to any object, not even `nothing`
typeof(expr1) # expressions are first class objects
expr2 = :(a = b + 1)
expr3 = quote a = b + 1 end
expr3
dump(expr1)   # The AST ! Note this is already a nested statement, an assignment of the result of an expression (the sum call between the symbol `:b` and 1) to the symbol `a`  
expr4 = Expr(:(=),:a,Expr(:call,:+,:b,1)) # The AST using the "Expr" constructor

symbol1   = :(a)               # as for expressions
symbol2   = Meta.parse("a")    # as for expressions
symbol3   = Symbol("a")        # specific for symbols only
othsymbol = Symbol("aaa",10,"bbb")
typeof(symbol1)

# I can access any parts of my expression before evaluating it (indeed, that's what macro will do...)
myASymbol = expr1.args[2].args[1]
expr1.args[2].args[1] = :(*)
b = 2
## a # error, a not defined
eval(expr1)
a # here now is defined and it has an object associated... 4!

# !!! danger
#     The capability to evaluate expressions is very powerfull but due to obvious secutiry implications never evaluate expressions you aren't sure of their provenience. For example if you develop a Julia web app (e.g. using [Genie.jl](https://github.com/GenieFramework/Genie.jl)) never evaluate user provided expressions.

# Note that evaluation of expressions happens always at global scope, even if it done inside a function:
function foo()
    locVar = 1
    expr = :(locVar + 1)
    return eval(expr)
end
## a = foo() # error locVar not defined 

# To refer to the _value_ of a variable rather than the identifier itself within an expression, interpolate the variable using the dollar sign:

expr = :($a + b) # here the identifier 'a' has been replaced with its numerical value, `4`
dump(expr)
eval(expr)
a = 10
eval(expr) # no changes
b = 100
eval(expr) # here it change, as it is at eval time that the identifier `b` is "replaced" with its value

# ### Macros

# One of the best usage of macros is they allow package developers to provide a very flexible API to their package, suited for the specific needs of the package, making life easier for the users. Compare for example [the API of JuMP](https://jump.dev/JuMP.jl/stable/manual/constraints/) with [those of Pyomo](https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Constraints.html) to define model constraints !

# Some examples of macros:
# [MultiDimEquations.jl](https://github.com/sylvaticus/MultiDimEquations.jl)
# - from: `@meq par1[d1 in DIM1, d2 in DIM2, dfix3] =  par2[d1,d2]+par3[d1,d2]`
# - to:   `[par1[d1,d2,dfix3] =  par2[d1,d2]+par3[d1,d2] for d1 in DIM1, d2 in DIM2]`
# [Pipe.jl:](https://github.com/oxinabox/Pipe.jl)
# - from: `@pipe  10 |> foo(_,a) |> foo2(b,_,c) |> foo3(_)`
# - to:   `foo3(foo2(b,foo(10,a),c))`
# Brodcasting (Base):
# - from: `@. a + b * D^2`
# - to:   `a .+ b .* D.^2`

# Defining a macro...

# Like functions, but both the arguments and the retuned output are expressions
macro customLoop(controlExpr,workExpr) 
    return quote
      for i in $controlExpr
        $workExpr
      end
    end
end

# Invoking a macro....
a = 5
@customLoop 1:4 println(i) #note that "i" is in the macro
@customLoop 1:a println(i)
@customLoop 1:a if i > 3 println(i) end
@customLoop ["apple", "orange", "banana"]  println(i)
@customLoop ["apple", "orange", "banana"]  begin print("i: "); println(i)  end
@macroexpand @customLoop 1:4 println(i) # print what the macro does with the specific expressions provided


# String macros (aka "non-standard string literals")
# Invoked with the syntax `xxx" ...text..."` or `xxx""" ...multi-line text..."""` where `xxx` is the name of the macro and the macro must be defined as `macro  xxx_str`.
# Used to perform textual modification o nthe given text, for example this print the given text on a 8 characters  

macro print8_str(mystr)                                 # input here is a string, not an expression
    limits = collect(1:8:length(mystr))
    for (i,j) in enumerate(limits)
      st = j
      en = i==length(limits) ? length(mystr) : j+7
      println(mystr[st:en])
    end
end

print8"123456789012345678"
print8"""This is a text that once printed in 8 columns with terminal will be several lines. Ok, no rammar rules relating to carriage returns are emploied here..."""

# While normally used to modify text, string macros are "true" macros:

macro customLoop_str(str) 
    exprs = Meta.parse(str)
    controlExpr,workExpr = exprs.args[1],exprs.args[2]
    return quote
      for i in $controlExpr
        $workExpr
      end
    end
end

customLoop"""1:4; println(i)"""

# ## Interfacing with other languages

# There are 3 ways to interface Julia with programs or libraries wrote in other languages.
# At the lowest level, Julia allows to directly interface with C or Fortran libraries, and this means, aside using directly libraries written in C, to be able to interface with any programming language that offer also a C interface (R, Python...)
# Using this low level C Interface, users have created specific packages to interface many languages using a simple, Julian-way syntax. We will see these interfaces for R and Python.
# Finally, at the highest level, many common packages of other languages have been already "interfaced", so that the user can use the Julia Package without even knowing that this is an interface for an other package, for example `SymPy.jl` is a large interface to the Python package `SymPy`.

# ### Using C libraries

# Let's start by seing how to use a C library. For this example to work you will need to have the GCC compiler installed on your machine
# First let's write the header and source C files and write them to the disk:

cheader = """
extern int get5();
extern double mySum(float x, float y);
"""
csource = """
int get5(){
    return 5;
}

double mySum(float x, float y){
    return x+y;
}
"""

open(f->write(f,cheader),"myclib.h","w")  # We open a stream to file with the "w" parameter as for "writing", and we pass the stream to the anonymous function to actually write to the stream. If this funcitons is many lines of code, consider rewriting the `open` statement using a `do` block
open(f->write(f,csource),"myclib.c","w") 

# Now let's run the command to compile the C code we saved as shared library using gcc, a C compiler.
# The following example assume that GCC is installed in the machine where this example is run and available as `gcc`.

compilationCommand1 = `gcc -o myclib.o -c myclib.c` # the actua lcompilation, note the backticks used to define a command
compilationCommand2 = `gcc -shared -o libmyclib.so myclib.o -lm -fPIC` # the linking into a shared library
run(compilationCommand1)
run(compilationCommand2)

# This should have created the C library `libmyclib.so` on disk. Let's gonna use it:
const myclib = joinpath(@__DIR__, "libmyclib.so")  # we need the full path
# ccall arguments:
# 1. A tuple with the funcion name to call and the library path. For both, if embedded in a variable, the variable must be set constant.  
# 2. The Julia type that map to the C type returned by the function.
#    - `int` → `Int32` or `Int64` (or the easy-to remmeber `Cint` alias)
#    - `float` → `Float32` (or the `Cfloat` alias)
#    - `double` → `Float64` (or the `Cdouble` alias)
# 3. A tuple with the Julia types of the parameters passed to the C function
# 4. Any other argument are the values of the parameter passed
a = ccall((:get5,myclib), Int32, ())       
b = ccall((:mySum,myclib), Float64, (Float32,Float32), 2.5, 1.5)

# More details on calling C or Fortran code can be obtained [in the official Julia documentation](https://docs.julialang.org/en/v1/manual/calling-c-and-fortran-code/).  

# ### Using Python in Julia 

# The "default" way to use Python code in Julia is trough the [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) package. It automatically take care of convert between Python types (including numpy arrays) and Julia types (types that can not be converted automatically are converted to the generic `PyObject` type).
ENV["PYTHON"] = "" # will force PyCall to download and use a "private to Julia" (conda based) version of Python. use "/path/to/python" if you want to reuse a version already installed on your system
## using Pkg
## Pkg.add("PyCall")
## Pkg.build("PyCall")
using PyCall

# #### Embed short python snippets in Julia
py"""
def sumMyArgs (i, j):
  return i+j
def getNthElement (vec,n):
  return vec[n]
"""
a = py"sumMyArgs"(3,4)             # 7 - here we call the Python object (a function) with Julia parameters
b = py"getNthElement"([1,2,3],1)   # 2 - attention to the diffferent convention for starting arrays!. Note the Julia Array ahas been converted automatically to a Python list
d = py"getNthElement([1,$a,3],1)"  # 7 - here we interpolate the Python call

# Alternativly, use `@pyinclude("pythonScript.py")`
pythonCode = """
def sumMyArgs (i, j, z):
  return i+j+z
"""
open(f->write(f,pythonCode),"pythonScript.py","w")
@pyinclude("pythonScript.py")
a = py"sumMyArgs"(3,4,5)

# !!! tip
#     Note thaat the 3 arguments definition of `sumMyArgs` has _replaced_ the 3-arguments one. This would now error `py"sumMyArgs"(3,4)` 


# #### Use Python libraries

# Add a package to the local Python installation using Conda:
pyimport_conda("ezodf", "ezodf", "conda-forge") # pyimport_conda(module, package, channel)

const ez = pyimport("ezodf")  # Equiv. of Python `import ezodf as ez`
destDoc  = ez.newdoc(doctype="ods", filename="anOdsSheet.ods")
# Both `ez` and `destDoc` are `PyObjects` for which we can access attributes and call the methods using the usual `obj.method()` syntax as we would do in Python
sheet    = ez.Sheet("Sheet1", size=(10, 10))
destDoc.sheets.append(sheet)
## dcell1 = sheet[(2,3)] # This would error because the index is a tuple. Let's use directly the `get(obj,key)` function instead:
dcell1   = get(sheet,(2,3)) # Equiv. of Python `dcell1 = sheet[(2,3)]`. Attention again to Python indexing from zero: this is cell "D3", not "B3" !
dcell1.set_value("Hello")
get(sheet,"A9").set_value(10.5) # Equiv. of Python `sheet['A9'].set_value(10.5)`
destDoc.backup = false
destDoc.save()

# ### Using Julia in Python

# #### Installation of the Python package `PyJulia`

# [PyJulia](https://github.com/JuliaPy/pyjulia) can be installed using `pip`, taking note that its name using `pip` is `julia` not `PyJulia`:
# ```$ python3 -m pip install --user julia```
 
# We can now open a Python terminal and initialise PyJulia to work with our Julia version:
# ```python
# >>> import julia
# >>> julia.install() # Only once to set-up in julia the julia packages required by PyJulia
# ```
 
# If we have multiple Julia versions, we can specify the one to use in Python passing julia="/path/to/julia/binary/executable" (e.g. julia = "/home/myUser/lib/julia-1.1.0/bin/julia") to the install() function.

# #### Running Julia libraries and code in Python

# On each Python session we need to run the following code: 
# ```python
# from julia import Julia
# Julia(compiled_modules=False)
# ```

# This is a workaround to the common situation when the Python interpreter is statically linked to libpython, but it will slow down the interactive experience, as it will disable Julia packages pre-compilation, and every time we will use a module for the first time, this will need to be compiled first. Other, more efficient but also more complicate, workarounds are given in the package documentation, under the [Troubleshooting section](https://pyjulia.readthedocs.io/en/stable/troubleshooting.html).

# We can now direcltly load a Julia module, including `Main`, the global namespace of Julia’s interpreter, with `from julia import ModuleToLoad` and access the module objects directly or using the `Module.evel()` interface.

# ##### Add a Julia package...
# ```python
# >>> from julia import Pkg
# >>> Pkg.add("BetaML")
# ```
# Of course we can add a package alternatively from within Julia

# ##### "Direct" calling of Julia functions...
# ```python
# >>> from julia import BetaML
# >>> import numpy as np
# >>> model = BetaML.buildForest([[1,10],[2,12],[12,1]],["a","a","b"])
# >>> predictions = BetaML.predict(model,np.array([[2,9],[13,0]]))
# >>> predictions
# [{'b': 0.36666666666666664, 'a': 0.6333333333333333}, {'b': 0.7333333333333333, 'a': 0.26666666666666666}]
# ```

# ##### Access using the `eval()` interface...
# If we are using the jl.eval() interface, the objects we use must be already known to julia. To pass objects from Python to Julia, we can import the julia Main module (the root module in julia) and assign the needed variables, e.g.

# ```python
# >>> X_python = [1,2,3,2,4]
# >>> from julia import Main
# >>> Main.X_julia = X_python
# >>> Main.eval('BetaML.gini(X_julia)')
# 0.7199999999999999
# >>> Main.eval("""
# ...   function makeProd(x,y)
# ...       return x*y
# ...   end
# ...   """
# ... )
# >>> Main.eval("makeProd(2,3)") # or Main.makeProd(2,3)
# ```
# For large scripts instead of using `eval()` we can equivalently use `Main.include("aJuliaScript.jl")`

# ### Using R in Julia 

# To use R from within Julia we use the [RCall](https://github.com/JuliaInterop/RCall.jl) package. 
ENV["R_HOME"] = "*" #  # will force RCall to download and use a "private to Julia" (conda based) version of R. use "/path/to/R/directory" (e.g. `/usr/lib/R`) if you want to reuse a version already installed on your system
## using Pkg
## Pkg.add("RCall")
## Pkg.build("RCall")
using RCall


R"""
sumMyArgs <- function(i,j) i+j
getNthElement <- function(vec,n) {
  return(vec[n])
}
"""
a = rcopy(R"sumMyArgs"(3,4))             # 7 - here we call the R object (a function) with Julia parameters
b = rcopy(R"getNthElement"([1,2,3],1))   # 1 - no differences in array indexing here
d = rcopy(R"as.integer(getNthElement(c(1,$a,3),2))")  # 7 - here we interpolate the R call
d = convert(Int64,R"getNthElement(c(1,$a,3),2)")  

# While we don't have here the problem of different array indexing convention (both Julia and R start indexing arrays at 1), we have the "problem" that the output returned by using `R"..."` is not yet an exploitable Julia object but it remains as an `RObject` that we can convert with `rcopy()` or explicitly with `convert(T,obj)`. Also, R elements are all floats by default, so if we need an integer in Julia we need to explicitly convert it, either in R or in Julia.

# If the R code is on a script, we don't have here a sort of @Rinclude macro, so let's implement it ourselves by loading the file content as a file and evaluating it using the function `reval` provided by `RCall`:

macro Rinclude(fname)
    quote
        rCodeString = read($fname,String)
        reval(rCodeString)
        nothing
    end
end

RCode = """
sumMyArgs <- function(i, j, z) i+j+z
"""
open(f->write(f,RCode),"RScript.R","w")
@Rinclude("RScript.R")
a = rcopy(R"sumMyArgs"(3,4,5))  # 12
## a = rcopy(R"sumMyArgs"(3,4)) # error !  The 3-arguments version of `sumMyArgs` has _replaced_ the 2-arguments one

# ### Using Julia in R

# #### Installation of the R package `JuliaCall`

# [JuliaCall](https://github.com/Non-Contradiction/JuliaCall) can be installed from CRAN:

# ```{r}
# > install.packages("JuliaCall")
# > library(JuliaCall)
# install_julia()
# ```

# `install_julia()`` will force R download and install a private copy of julia. If you prefer to use instead an existing version of julia and having R default to download a private version only if it can't find a version already installed, use `julia_setup(installJulia = TRUE)` instead of `install_julia()`, eventually passing the `JULIA_HOME = "/path/to/julia/binary/executable/directory"` (e.g. `JULIA_HOME = "/home/myUser/lib/julia-1.7.0/bin"`) parameter to the `julia_setup` call

# `JuliaCall` depends for some things (like object conversion between Julia and R) from the Julia `RCall` package. If we don't already have it installed in Julia, it will try to install it automatically.

# #### Running Julia libraries and code in R

# On each R session we need to run the `julia_setup` function:
# ```R
# library(JuliaCall)
# julia_setup() # If we have already downloaded a private version of Julia for R it will be retrieved automatically
# ```

# We can now load a Julia module and access the module objects directly or using the `Module.evel()` interface.

# ##### Add a Julia package...

# ```{r}
# > julia_eval('using Pkg; Pkg.add("BetaML")')
# ```
# Of course we can add a package alternatively from within Julia


# Let's load some data from R and do some work with this data in Julia:

# ```{r}
# > library(datasets)
# > X <- as.matrix(sapply(iris[,1:4], as.numeric))
# > y <- sapply(iris[,5], as.integer)
# ```

# ##### Calling of Julia functions with `julia_call`...

# With `JuliaCall`, differently than `PyJulia`, we can't call direclty the julia functions but we need to employ the R function `julia_call("juliaFunction",args)`:

# ```{r}
# > julia_eval("using BetaML")
# > yencoded <- julia_call("integerEncoder",y)
# > ids      <- julia_call("shuffle",1:length(y))
# > Xs       <- X[ids,]
# > ys       <- yencoded[ids]
# > cOut     <- julia_call("kmeans",Xs,3L)    # kmeans expects K to be an integer
# > y_hat    <- sapply(cOut[1],as.integer)[,] # We need a vector, not a matrix
# > acc      <- julia_call("accuracy",y_hat,ys)
# > acc
# [1] 0.8933333
# ```
# ##### Access using the `eval()` interface...

# As alternative, we can embed Julia code directly in R using the `julia_eval()` function:

# ```{r}
# > kMeansR  <- julia_eval('
# +      function accFromKmeans(x,k,y_true)
# +        cOut = kmeans(x,Int(k))
# +        acc = accuracy(cOut[1],y_true)
# +        return acc
# +      end
# + ')
# ```

# We can then call the above function in R in one of the following three ways:
# 1. `kMeansR(Xs,3,ys)`
# 2. `julia_assign("Xs_julia", Xs); julia_assign("ys_julia", ys); julia_eval("accFromKmeans(Xs_julia,3,ys_julia)")`
# 3. `julia_call("accFromKmeans",Xs,3,ys)`.

# While other "convenience" functions are provided by the package, using  `julia_call` or `julia_assign` followed by `julia_eval` should suffix to accomplish most of the task we may need in Julia.


# ## Some performance tips

# # ### Type stability

# "Type stable" functions guarantee to the compiler that given a certain method (i.e. with the arguments being of a given type) the object returned by the function is also of a certain fixed type. Type stability is fundamental to allow type inference continue across the function call stack.

function f1(x)    # Type unstable
    outVector = [1,2.0,"2"]
    if x < 0
        return outVector[1]
    elseif x == 0
        return outVector[2]
    else
        return outVector[3]
    end
end

function f2(x)   # Type stable
    outVector = [1,convert(Int64,2.0),parse(Int64,"2")]
    if x < 0
        return outVector[1]
    elseif x == 0
        return outVector[2]
    else
        return outVector[3]
    end
end

a = f1(0)
b = f1(1)
typeof(a)
typeof(b)

c = f2(0)
d = f2(1)
typeof(c)
typeof(d)

using BenchmarkTools
@btime f1(0) # 661 ns 6 allocations
@btime f2(0) #  55 ns 1 allocations

@code_warntype f1(0) # Body::Any 
@code_warntype f2(0) # Body::Int64 

# While in general is NOT important to annotate function parameters for performance, it is important to annotate struct fields with concrete types 
abstract type Goo end
struct Foo <: Goo
    x::Number
end
struct Boo <: Goo
    x::Int64
end

function f1(o::Goo)
    return o.x +2
end

fobj = Foo(1)
bobj = Boo(1)
@btime f1($fobj) # 17.1 ns 0 allocations
@btime f1($bobj) #  2.8 ns 0 allocations

# Here the same function under some argument types is type stable, under other argument types is not
@code_warntype f1(fobj)
@code_warntype f1(bobj) 

# # #### Avoid (non-constant) global variables

g        = 2
const cg = 1   # you can't change the _type_ of the object binded to a constant variable 
cg       = 2   # you can rebind to an other object of the same type
## cg    = 2.5 # this would error !
f1(x,y) = x+y
f2(x)   = x + g
f3(x)   = x + cg

@btime f1(3,2)  
@btime f2(3)    # 22 times slower !!!
@btime f3(3)    # as f1

# #### Loop arrays with the inner loop by rows

# Julia is column mayor (differently than Python) so arrays of bits types are contiguous in memory across the different rows of the same column

a = rand(1000,1000)

function f1(x)
    (R,C) = size(x)
    cum = 0.0
    for r in 1:R
        for c in 1:C
            cum += x[r,c]
        end
    end
    return cum
end
function f2(x)
    (R,C) = size(x)
    cum = 0.0
    for c in 1:C
        for r in 1:R
            cum += x[r,c]
        end
    end
    return cum
end
@btime f1($a) # 2.3 ms 0 allocations
@btime f2($a) # 1.3 ms 0 allocations

# ## Profiling the code to discover bootlenecks

# We already see `@btime` and `@benchmark` from the package [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl)
# Remember to quote the global variables used as parameter of your function with the dollar sign to have accurate benchmarking of the function execution.
# Julia provide the macro `@time` but we should run on a second call to a given function (with a certain parameter types) or it will include compilation time in its output:

function fb(x)
    out = Union{Int64,Float64}[1,2.0,3]
    push!(out,4)
    if x > 10
        if ( x > 100)
            return [out[1],out[2]] |>  sum
        else
            return [out[2],out[3]] |>  sum
        end
    else
        return [out[1],out[3]] |>  sum
    end
end


@time fb(3)
@time fb(3)

# We can use `@profile function(x,y)` to use a sample-based profiling

using Profile # in the stdlib
function foo(n)
    a = rand(n,n)
    b = a + a
    c = b * b
    return c
end
@profile (for i = 1:100; foo(1000); end) # too fast otherwise
Profile.print() # on my pc: 243 rand, 174 the sum, 439 the matrix product
Profile.clear()

# ## Introspection and debugging

# To discover problems on the code more in general we can use several introspection functions that Julia provide us (some of which we have already saw):

## @less foo(3)  # Show the source code of the specific method invoked - use `q` to quit
## @edit foo(3)  # Like @loss but it opens the source code in an editor
methods(foo)
@which foo(2)          # which method am I using when I call foo with an integer?
typeof(a)
eltype(a)
fieldnames(Foo)
dump(fobj)
names(Main, all=false) # available (e.g. exported) identifiers of a given module
sizeof(2)              # bytes
typemin(Int64)
typemax(Int64)
bitstring(2)
# Various low-level interpretation of an expression
@code_native foo(3)
@code_llvm foo(3)
@code_typed foo(3)
@code_lowered foo(3)

# We can use a debugger, like e.g. the one integrated in Juno or VSCode.
# Graphical debuggers allow to put a _breakpoint_ on some specific line of code, run the code in debug mode (yes, it will be slower), let the program arrive to the breakpoint and inspect the state of the system at that point of the code, including local variables. In Julia we can also _change_ the program interactively !
# Other typycal functions are running a single line, running inside a function, running until the current function return, ecc..

# ## Runtime exceptions

# As many (all?) languages, Julia when "finds" an error issues an exception, that if it is not caugth at higher level in the call stack (i.e. recognised and handled) lead to an error and return to the prompt or termination of the script (and rarely with the Julia process crashing altogether).

# The idea is that we _try_ some potentially dangerous code and if some error is raised in this code we _catch_ it and handle it.

function customIndex(vect,idx;toReturn=0)
    try
        vect[idx]
    catch e  
        if isa(e,BoundsError)
            return toReturn
        end
        rethrow(e)
    end
end

a = [1,2,3]
## a[4] # Error ("BoundsError" to be precise)
customIndex(a,4)

# Note that handling exceptions is computationally expensive, so do not use exceptions in place of conditional statements

# ## Distributed computation

# Finally one note on distributed computation. We see only some basic usage of multithreading and multiprocesses in this course, but with Julia it is relativelly easy to parallelise the code either using multiple threads or multiple processes. What's the difference ?
# - **multithread**
#   - advantages: computationally "cheap" to create (the memory is shared)
#   - disadvantages: limited to the number of cores within a CPU, require attention in not overwriting the same memory or doing it at the intended order ("data race"), we can't add threads dynamically (within a script)
# - **multiprocesses**
#   - advantages: unlimited number, can be run in different CPUs of the same machine or differnet nodes of a cluster, even using SSH on different networks, we can add processes from withi nour code with `addprocs(nToAdd)`
#   - disadvantages: the memory being copied (each process wil lhave its own memory) are computationally expensive (you need to have a gain higher than the cost on setting a new process) and require attention to select which memory a given process will need to "bring with it" for its functionality

# Note that if you are reading this document on the github pages, this script is compiled using GitHub actions where a single thread and process are available, so you will not see performance gains.

# ### Multithreading

# !!! warning
#     It is not possible to add threads dinamically, either we have to start Julia with the parameter `-t` (e.g. `-t 8` or `-t auto`) in the command line or use the VSCode Julia externsion setting `Julia: Num Threads`

function inner(x)
    s = 0.0
    for i in 1:x
        for j in 1:i
            if j%2 == 0
                s += j
            else
                s -= j
            end
        end
    end
    return s
end

function parentSingleThread(x,y)
    toTest = x .+ (1:y)
    out = zeros(length(toTest))
    for i in 1:length(toTest)
        out[i] = inner(toTest[i])
    end
    return out
end
function parentThreaded(x,y)
    toTest = x .+ (1:y)
    out = zeros(length(toTest))
    Threads.@threads for i in 1:length(toTest)
        out[i] = inner(toTest[i])
    end
    return out
end


x = 100
y = 20

str = parentSingleThread(x,y)
mtr = parentThreaded(x,y)   

str == mtr # true
Threads.nthreads() # 4 in my case 
Threads.threadid()
@btime parentSingleThread(100,20) # 140 μs on my machine
@btime parentThreaded(100,20)     #  47 μs 

# ### Multiprocessing

using Distributed     # from the Standard Library
addprocs(3)           # 2,3,4
# The first process is considered a sort of "master" process, the other one are the "workers"
# We can add processes on other machines by providing the SSH connection details directly in the `addprocs()` call (Julia must be installed on that machines as well)
# We can alternativly start Julia directly with _n_ worker processes using the armument `-p n` in the command line.
println("Worker pids: ")
for pid in workers()  # return a vector to the pids
    println(pid)      # 2,3,4
end
rmprocs(workers()[2])    #  remove process pid 3
println("Worker pids: ")
for pid in workers() 
    println(pid) # 2,4 are left
end
@everywhere begin using Distributed end # this is needed only in GitHub action
@everywhere println(myid()) # 2,4


# #### Run heavy tasks in parallel

using Distributed, BenchmarkTools
a = rand(1:35,100)
@everywhere function fib(n)
    if n == 0 return 0 end
    if n == 1 return 1 end
    return fib(n-1) + fib(n-2)
end
# The macro `@everywhere` make available the given function (or functions with `@everywhere begin [shared function definitions] end` or `@everywhere include("sharedCode.jl")`) to all the current workers.
result  = pmap(fib,a)
# The pmap function ("parallel" map) automatically pick up the free processes, assign them the job prom the "input" array and merge the results in the returned array. Note that the order is preserved:
result2 = pmap(fib,a)
result == result2
@btime map(fib,$a)  # serialised:   median time: 514 ms    1 allocations
@btime pmap(fib,$a) # parallelised: median time: 265 ms 4220 allocations # the memory of `a` need to be copied to all processes


# #### Divide and Conquer

# Rather than having a "heavy operation" and being interested in the individual results, here we have a "light" operation and we want to aggregate the results of the various computations using some aggreagation function.
# We can then use `@distributed (aggregationfunction) for [forConditions]` macro:

using Distributed, BenchmarkTools
function f(n)   # our single-process benchmark
  s = 0.0
  for i = 1:n
    s += i/2
  end
    return s
end
function pf(n)
  s = @distributed (+) for i = 1:n # aggregate using sum on variable s
        i/2                        # the last element of the for cycle is used by the aggregator
  end
  return s
end
@btime  f(10000000) # median time: 11.1 ms   0 allocations
@btime pf(10000000) # median time:  5.7 ms 145 allocations

# Note that also in this case the improvement is less than proportional with the number of processes we add

# Details on parallel comutation can be found [on the official documentation](https://docs.julialang.org/en/v1/manual/parallel-computing/), including information to run nativelly Julia on GPUs or TPUs.