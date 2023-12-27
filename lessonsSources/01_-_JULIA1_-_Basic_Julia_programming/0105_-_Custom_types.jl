################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################

# # 0105 Custom Types


# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")     
## If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will however be lost)
## Pkg.resolve()   
## Pkg.instantiate() # run this if you didn't in Segment 01.01
using Random
Random.seed!(123)
using InteractiveUtils # loaded automatically when working... interactively

# ## "Type" of types

# In Julia primitive, composite and abstract types can all be defined by the user

primitive type APrimitiveType 819200 end # name and size in bit - multiple of 8 and below 8388608 (1MB)
primitive type APrimitiveType2 819200 end
819200/(8*1024)
struct ACompositeType end                # fields, constructors.. we'll see this later in details
abstract type AnAbstractType end         # no objects, no instantialisation of objects


# ## Composite types
mutable struct Foo # starting with a capital letter
    field1
    field2::String
    field3::ACompositeType
end

# mutable struct Foo # error! can't change a struct after I defined it
# end

fieldnames(Foo)

o = Foo(123,"aaa",ACompositeType()) # call the default constructor (available automatically) - order matters!
typeof(o)


# Outer constructor
function Foo(f2,f3=ACompositeType()) # "normal" functions just happens it has the name of the object to create
    if startswith(f2,"Booo")         # put whatever logic you wish
        return nothing
    end
    return Foo(123,f2,f3)            # call the default constructor
end

o = Foo(123,"aaa", ACompositeType()) # call the default constructor
o = Foo("blaaaaa")                   # call the outer constructor we defined

o.field1       # access fields
o.field1 = 321 # modify field (because type defined as "mutable" !!!)
o


function Base.show(io::IO, x::Foo)             # function(o) rather than o.function() 
    println(io,"My custom representation for Foo objects")
    println(io,"Field1: $(o.field1)")
    println(io,"Field2: $(o.field2)")
end
o

# Inner constructor
mutable struct Foo2
    field1::Int64
    field2::String
    function Foo2(f1,f2,f3)
        ## ... logic
        return new(f1+f2,f3)
    end
end
Foo2(1,2,"aaa")

# !!! tip
#     If any inner constructor method is defined, no default constructor method is provided.

## Foo2(1,"aaa") # Error, no default constructor !

# You can also use a macro (requires Julia 1.1) to automatically define an (outer) keyword_based constructor with support for optional arguments:

Base.@kwdef struct Kfoo
   x::Int64 = 1
   y = 2
   z
end
Kfoo(z=3)

# Note that, at time of writing, `@kwdef` is not exported by Julia base, meaning that while widely used, it is considered still "experimental" and its usage may change in future Julia minor versions.



# ## Custom pretty-printing 


# We can customise the way our custom type is rendered by overriding the `Base.show` function for our specific type.

# We first need to import `Base.show`
import Base.show

mutable struct FooPoint
    x::Int64
    y::Int64
end

function show(io::IO, ::MIME"text/plain", x::FooPoint)
    ## if get(io, :compact, true) ... # we can query the characteristics of the output
    print(io,"A FooPoint struct")
end
function show(io::IO, x::FooPoint) # overridden by print
    print(io, "FooPoint x is $(x.x) and y is $(x.y) \nThat's all!")
end

foo_obj=FooPoint(1,2)
display(foo_obj)
show(foo_obj)
print(foo_obj)
println(foo_obj)


# ## Parametric types


struct Point{T<:Number} # T must be a child of type "Number"
   x::T
   y::T
end

o = Point(1,2)
Point(1.0,2.)
## Point(1,2.0) # error !

function Point(x::T, y::T=zero(T)) where {T}
    return Point(x,y)
end
Point(2)
Point(1.5)

abstract type Figure{T<:Number} end

a = Array{Int64,2}(undef,2,2) # Array is nothing else than a parametric type with 2 parameters
typeof(a)
eltype(a)

# As we see for arrays, parameters doesn't need to be _types_, but can be any value of a bits type (in practice an integer value) :

struct MyType{T,N}
  data::Array{T,N}
end

intMatrixInside = MyType([1 2 3; 4 5 6])
floatVectorInside = MyType([1 2 3])

function getPlane(o::MyType{T,N},dim,pos) where {T,N}
sizes = size(o.data)
if length(sizes) > N
  error("Dim over the dimensions of the data")
elseif sizes[dim] < pos
  error("Non enought elements in dimension $dim to cut at $pos")
end
return selectdim(o.data,dim,pos)
end

getPlane(intMatrixInside,1,2)

# A package where non-type parameters are emploied to boost speed is [StaticArray.jl](https://github.com/JuliaArrays/StaticArrays.jl) where one parameter is the _size_ of the array that hence become known at compile time


# ## Inheritance

abstract type MyOwnGenericAbstractType end                       # the highest-level
abstract type MyOwnAbstractType1 <: MyOwnGenericAbstractType end # child of MyOwnGenericAbstractType 
abstract type MyOwnAbstractType2 <: MyOwnGenericAbstractType end # also child of MyOwnGenericAbstractType
mutable struct AConcreteTypeA <: MyOwnAbstractType1
  f1::Int64
  f2::Int64
end
mutable struct AConcreteTypeB <: MyOwnAbstractType1
  f1::Float64
end
mutable struct AConcreteTypeZ <: MyOwnAbstractType2
  f1::String
end
oA = AConcreteTypeA(2,10)
oB = AConcreteTypeB(1.5)
oZ = AConcreteTypeZ("aa")

supertype(AConcreteTypeA)
subtypes(MyOwnAbstractType1)


# !!! tip
#     When multiple methods are available for an object, function calls are dispatched to the most stricter method, i.e. the one defined over the exact parameter's type or their immediate parents

function foo(a :: MyOwnGenericAbstractType)                      # good for everyone 
  println("Default implementation: $(a.f1)")
end
foo(oA) # Default implementation: 2
foo(oB) # Default implementation: 1.5
foo(oZ) # Default implementation: aa
function foo(a :: MyOwnAbstractType1)                            # specialisation for MyOwnAbstractType1
  println("A more specialised implementation: $(a.f1*4)")
end
foo(oA) # A more specialised implementation: 8
foo(oB) # A more specialised implementation: 6.0
foo(oZ) # Default implementation: aa                             # doesn't match the specialisation, default to foo(a :: MyOwnGenericAbstractType)
function foo(a :: AConcreteTypeA)
     println("A even more specialised implementation: $(a.f1 + a.f2)")
end
foo(oA) # A even more specialised implementation: 12
foo(oB) # A more specialised implementation: 6.0
foo(oZ) # Default implementation: aa

# !!! warning 
#     Attention to the inheritance for parametric types. If it is true that `Vector{Int64} <: AbstractVector{Int64}` and `Int64 <: Number`, it is FALSE that `AbstractVector{Int64} <: AbstractVector{Number}`. If you want to allow a function parameter to be a vector of numbers, use instead templates explicitly, e.g. `foo(x::AbstractVector{T}) where {T<:Number} = return sum(x)`

Vector{Int64} <: AbstractVector{Int64}
Int64 <: Number
Vector{Int64} <: Vector{Number}
AbstractVector{Int64} <: AbstractVector{Number}


# ## Object-oriented model

# OO model based on _composition_ 

struct Shoes
   shoesType::String
   colour::String
end
struct Person
  myname::String
  age::Int64
end
struct Student
   p::Person        # by referencing a `Person`` object, we do not need to repeat its fields
   school::String
   shoes::Shoes     # same for `shoes`
end
struct Employee
   p::Person
   monthlyIncomes::Float64
   company::String
   shoes::Shoes
end
gymShoes = Shoes("gym","white")
proShoes = Shoes("classical","brown")
Marc     = Student(Person("Marc",15),"Divine School",gymShoes)
MrBrown  = Employee(Person("Brown",45),3200.0,"ABC Corporation Inc.", proShoes)

function printMyActivity(self::Student)
   println("Hi! I am $(self.p.myname), I study at $(self.school) school, and I wear $(self.shoes.colour) shoes") # I can use the dot operator chained...
end
function printMyActivity(self::Employee)
  println("Good day. My name is $(self.p.myname), I work at $(self.company) company and I wear $(self.shoes.colour) shoes")
end

printMyActivity(Marc)     # Hi! I am Marc, ...
printMyActivity(MrBrown)  # Good day. My name is MrBrown, ...

# OO models based on Specialisation (Person → Student) or Weack Relation (Person → Shoes) instead of Composition (Person → Arm) can be implemented using third party packages, like e.g. [SimpleTraits.jl](https://github.com/mauro3/SimpleTraits.jl) or [OOPMacro.jl](https://github.com/ipod825/OOPMacro.jl)
