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
using Pkg
#Pkg.add("PyCall")
#Pkg.build("PyCall")
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

# Add a Julia package...
# ```python
# >>> from julia import Pkg
# >>> Pkg.add("BetaML")
# ````
# Of course we can add a apckage alternatively from within Julia

# "Direct" access of Julia objects...
# ```python
# >>> from julia import BetaML
# >>> import numpy as np
# >>> model = BetaML.buildForest([[1,10],[2,12],[12,1]],["a","a","b"])
# >>> predictions = BetaML.predict(model,np.array([[2,9],[13,0]]))
# >>> predictions
# [{'b': 0.36666666666666664, 'a': 0.6333333333333333}, {'b': 0.7333333333333333, 'a': 0.26666666666666666}]
# ```

# Access using the `eval()` interface...
# If we are using the jl.eval() interface, the objects we use must be already known to julia. To pass objects from Python to Julia, we can import the julia Main module (the root module in julia) and assign the needed variables, e.g.

# ```
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


# ### Using Julia in R



# ## Some performance tips
