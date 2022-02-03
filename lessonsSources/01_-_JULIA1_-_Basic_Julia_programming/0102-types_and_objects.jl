################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################

# # 0102 Types and objects



# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")     
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
# Pkg.instantiate()
using Random
Random.seed!(123)


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
#   - AbstractString     # We'll see what all these "abstract" mean....
#     - String
#     - ...
#   - AbstractArray
#     - Array
#     - ....
#   - Number
#     - Complex
#     - Real
#       - Rational
#       - Integer
#         - Unsigned
#           - UInt64
#           - ...
#         - Signed
#           - Int32
#           - Int64
#           - BigInt
#           - ...
#         - Bool
#       - FixedPoints
#         - ...
#       - AbstractIrrational
#         - Irrational
#       - AbstractFloat
#         - Float32
#         - Float64
#         - BigFloat
#         - ...
#   - ...
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
import Base.+
## +(a,b,c) = a*b*c  # Defining my new crazy addition operation with 3 arguments
10+20+30            # This call it
10+20               # The addition with two parameters remains the same
10+20+30+40         # Also this one remains with the standard addition..

# !!! warning
#     After you tested this crazy addition, please restart julia or norhing will work.
#     With great power come great responsability.. (..if you change the meaning of addition it is difficult you will not run into problems...)


# ## Objects and variables

k = 10    # "create" an object Int64 in memory and binds (assign) it to the `k` identifier (the variable name)
typeof(k)
sizeof(k)  # bytes (1 byte is 8 bits)
bitstring(k)
0*2^0+1*2^1+0*2^2+1*2^3
m = k      # name binding: it binds (assign) the entity (object) referenced by a to the b identifier (the variable name)
m == k     # are the two objects equal ?
m === k    # are the two identifiers binding the same identical object in memory ?


# ## Mutability property of Julia objects

k = 10
v = [1,2]
p = 'z'
g = "hello"
ismutable(k)
ismutable(v)
ismutable(p)
ismutable(g)
## mutable objects are stored in memory "directly", while for mutable objects it is its memory address to be stored


# ## Three different ways to "copy" objects...

a = [[[1,2],3],4] # First element of a is said to be "mutable", second one is not:
ismutable(a[1])
ismutable(a[2])
b = a            # binding to a new identifier
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
# !!! note
#     Consider these memory isues when we'll discuss calling a function by reference/value !