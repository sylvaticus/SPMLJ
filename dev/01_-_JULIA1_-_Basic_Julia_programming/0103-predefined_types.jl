################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################

# # 0103 Predefined types


# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")     
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
# Pkg.instantiate()
using Random
Random.seed!(123)
using InteractiveUtils # loaded automatically when working... interactively

# ## Primitive types

# Primitive types have a fixed number of bits associated to them. Examples of them are `Int64`, `Float64`, `Char`, `UInt64`, `UFloat64`, `Int32`, `Float32`,...
# Even primitive types can be custom defined. See the "custom types" segment !

# ### Char and Strings - `Char`, `String`

a = "Hello World "
b = a[2]
typeof(b)
b = a[3:end]
typeof(b)
#a[2] = 'E' # error !

## info...
ismutable(a) # long story.... https://github.com/JuliaLang/julia/issues/30210
length(a)
findfirst(isequal('o'), a)
findnext(isequal('o'), a, 6)
occursin("world",a)
occursin(lowercase("world"),lowercase(a))
using Unicode
occursin(Unicode.normalize("world", casefold=true),Unicode.normalize(a, casefold=true)) 
endswith(a,"ld ")
startswith(a,"Mooo")
occursin(r"H*.d ", a)

## modifications..
lowercase(a)
lowercasefirst(a)
split(a, " ")
replace(a,"World" => "Universe")
strip(a)

## concatenation..
a = "Hello"; b= "World"
c = join([a,b]," ")
c = a*" "*b
c = string(a," ",b)
c = "$a $b"              # interpolation

## Conversion..
a = parse(Int64,"2012")
b = string(2019)

# !!! warning
#     Attention not to confuse the `string` function with the `String` type and the `String()` constructor!

## to know more...
methodswith(String,supertypes=true); # where any argument is a String or any parent type (such e.g. AbstractString)
## See also https://docs.julialang.org/en/v1/manual/strings/

# ## Arrays - `Array{T,N}`

# `Array{T,NDims}` A _parameteric_ type where the type of the content and the number of dimensions define the specific type

# !!! tip
#     `Vector{T}` is an alias for `Array{T,1}` and `Matrix{T}` is an alias for `Array{T,2}`, but there isn't anything "special" for 1 or 2 dimensions compared to more dimensions 

# ### Vectors - `Array{T,1}`

# One-dimensions arrays in julia are treated as column vector, and, depending on the inner type, they can be stored efficiently contiguously in memory. However they are NOT the same of a single column of a two dimensions array.
# A row vector is necessarily instead a single row of a 2 dimensions array.

a = [1,2,3]
b = [1 ; 2 ; 3 ;;] # This syntax requires Julia >= 1.7
a == b

# #### Initialisation
a = [1,2,3]; #= or =# a = [1;2;3] 
a = [1; 6:-2:2; 10] # notes: (a) the semicolon, (b) the range includes BOTH the extremes
a = [[1,2],[3,4]]   # nested vectors. Each elements can have a different lenght, but rules of linear algebra doesn't apply
# !!! danger
#     Don't confuse nested vectors with multi-dimensional arrays!

# Empty (zero-elements) arrays:
a = []
a = Int64[]
a = [1,2,3]
b = [1,"pizza","beer"]
a = Array{Int64,1}()
# !!! warning
#     Watch out for the difference between `a = Array{Int64,1}()` and `a = Array{Int64,1}`
a = Vector{Int64}()
# n-elements initialisation:
n = 3
T = Int64
zeros(n)            # zeros (Float64)
zeros(T,n)          # zeros (casted as type T)
ones(n)             # ones  (Float64)
ones(T,n)           # ones  (casted of type T)
Array{T,1}(undef,n) # garbage
fill(2,3)

# #### Accessing Vectors
a = [101:200;] 
a[1]
a[end]
a[[1; 6:-2:2; 10]] # indicised by a vector of positions 


# #### Collecting iterators into vectors
aRange = 3:2:7
a = collect(aRange)
typeof(aRange)
typeof(aRange) <: AbstractArray # Everywhere an AbstractArray is expected, you can provide a range instead

# #### Common operations with vectors
a = [1,2,3]
reverse(a), a[end:-1:1] # Other way to revert an array
vcat([1,2,3],[4,5],6)

push!(a,4)  # add as individual elements
append!(a,5) # add as many new elements 

# !!! tip "Functions with exclamation marks"
#     By **convention** functions that modify one of their parameters (and usually this is the first one) are named with an exclamation mark at the end. Remember (most) unicode characters are valid in functions or variable names.

push!([[1,2],[3,4,5]],[6,7])
append!([1,2,3,4,5],[6,7])

pop!(a)
a
popfirst!(a)
deleteat!(a,2)
pushfirst!([2,3],1)
a = [2,1,3,1]
sort(a)   # also `sort!(a)``
unique(a) # also `unique!(a)`
in(1,a)   # also available as operator: `if 2 in a [...] end`
length(a) # number of elements contained in all the dimensions
size(a),size(a,1) # number of elements by dimension
minimum(a)
min(a...)
# !!! tip "..."
#     "..." is called the *splat operator* and it is used to convert the elements in a vector into a tuple of separate elements in a function call, like the example above
min(4,7,3)
minimum([4,7,9])
argmin([4,2,5,2])
sum([1,2,3])
cumsum([1,2,3])
empty!(a) # only for Vectors
using Random
shuffle([1,2,3]) # also shuffle!([1,2,3])
isempty(a)
findall(x -> x == 1, [2,1,3,1]) # anonymous function returning an array of bools, findall then return the indexes
findfirst(x -> x == 1, [2,1,3,1])
myComparitionWith1(i) = i==1
findall(x -> myComparitionWith1(x), [2,1,3,1])

## delete [7,2,5] from an 1:10 array:
data     = [1:10;]
toDelete = [7,5,2]
deleteat!(data, findall(x -> x in toDelete, data))

for (i,value) in enumerate([10,20,30]) # iterator that returns an index/element tuple
    println("$i - $value")
end

names = ["Marc", "Anne"]
sex   = ['M','F']
age   = [25,20]

for zippedElements in zip(names,sex,age) # iterator that returns tuples made with one element per argument of zip
    println(zippedElements)
end

# ### Multidimensional arrays - `Array{T,N}`

# #### Initialisation
a = [[1,2,3] [4,5,6]] # By column, i.e. elements of the first column, elements of the second column, ...
a = [1 4; 2 5; 3 6]   # By row, i.e. elements of the first row, elements of the second row, ... 

## Empty (zero-elements) arrays:
a = Array{Float64}(undef, 0, 0, 0) # using explicitly the constructor and explicitly giving zero for each wanted dimension 
## n-elements initialisation:
(T,n,m,g,j) = (Int64,1,2,3,'a')
a = zeros(n,m,g)            # n,m,g-elements zeros array
a = ones(n,m,g)             # n,m,g-elements ones array
a = Array{T,3}(undef,n,m,g) # n,m,g-elements array whose content is garbage
a = fill(j,n,m,g)           # n,m,g-elements array of identical j elements
a = rand(n,m,g)             # n,m,g-elements array of of random numbers 
a = [3x + 2y + z for x in 1:2, y in 2:3, z in 1:2] # from using list comprehension 

# #### Accessing n-dimensional arrays
# Access by indicating position
a = [1 2 3 4; 5 6 7 8; 9 10 11 12]
a[2,1]   # comma to separate the dimensions
# !!! warning
#     Don't confuse `a[i,j]` for selecting an element of a Matrix with `a[i][j]` to select the inner component of a nested array
a[1,2:3]   # with a range on the second dimension
a[[1,3],1] # with a vector of positions in the first dimension
a[2,:]     # with a full range (all values) in the second dimension, i.e. all columns value for row 2
# !!! warning
#     Note that when the data as only one element on a given dimension, julia reduces the dimensions automatically: the result of `a[2,:]` is NOT a row vector (that is a one-row matrix) but a one dimensional array
# Access by a mask (boolean selection)
b = [true false true false; true true true false; true false true false]
a[b] # always flatted array returned (need eventually reshaping, see later)

# #### Funcionality related to dimensions

size(a)              # returns a tuple (i.e. an immutable list) with the sizes of the n dimensions
ndims(a)             # return the number of dimensions of the array (e.g. `2` for a matrix)
reshape(a, 2,3,2) 
2*3*2 == length(a)
b = rand(2,1,3)
dropdims(b,dims=(2)) # remove the specified dimensions, provided that the specified dimension have only a single element
permutedims(a)  # "swap" the dimensions
reshape(a,4,3)  # keep the column mayor order
for slice in eachslice(a,dims=1)
    println(slice)
end
a = reshape(1:24, 3,4,2)
for slice in eachslice(a,dims=1)
    println(slice)
end
a = [1 2;3 4; 5 6]
selectdim(a,1,3) # Select an hyperplane on dimension 1 (rows) at position 3. Returns a view


# #### Flat to vector..
a = [1 2; 3 4]
vec(a)        # shadow copy (different view of the underlying data)
reshape(a,4)  # shadow copy
a[:]          # allocate, as all slice operations do

# #### Other functionality related to Arrays
vcat([1 2; 3 4], [5 6; 7 8]) # works also for DataFrames
hcat([1,2,3],[4,5,6])        # works also for DataFrames
a = [1 2; 3 4]
b = similar(a) # garbage inside
cat(a,a,a,dims=3)

# Sort by column (field)
a = [[3,2,1] [20,30,20] [1000,3000,2000] [300,100,200]]
idx = sortperm(a[:,3], rev=true) # return the positions that sort the 3rd column 
sortedMatrix = a[idx,:] # selected by using the sorted positions array on the row dimension 
sortslices(a, dims=2)   # by cols, using the first row to sort
sortslices(a, dims=1)   # by rows, using the first column to sort
sortslices(a, dims=1, by = x -> (x[2],x[4])) # by rows, using second and fourth columns

# #### Basic linear algebra
using LinearAlgebra
a = [-1,2,3]
b = [4,5,6]
transpose(a)
a'
norm(a)   # l-2 by default
norm(a,1)

# Vector products:
dot(a,b)    # dot, aka "inner" product
a' * b  == dot(a,b)
cross(a,b)  # cross product
a .* b      # element-wise product
a * b'

A = [1 2 3; 6 4 5; 7 8 9]

A^(-1) # inverse
A^2
det(A) # determinant
transpose(A)
A'
# !!! warning
#     Be aware that `transpose` works only for numerical types. When the matrix contains other types (e.g. strings), use `permutedims` 

diag(A)
I # operator that automatically scale to the context without actually building the matrix
A*I
B = [1 2; 3 4]; B*I
(evalues, evectors) = eigen(A)

# ## Tuples - `Tuple{T1,T2,...}`

# A "collection" similar to Array but:
# - Immutable
# - Can efficiently host heterogeneous types, as type information is stored for each individual element
# - Linear algebra doesn't apply (use StaticArray.jl package for that)

# Can be tought as anonymous (immutable) structures
# Used to unpack multiple values, e.g. to store on inddividual variables the output of functions with multiple return value 

# ### Initialisation

t = (1,2.5,"a",3)
t = 1,2.5,"a",3
typeof(t)

# ### Indexing
t[1]

# ### Conversion
v = [1,2,3]
t = (v...,) # note the comma
v2 = [t...]
v3 = [i[1] for i in t]
v4 = collect(t)
v == v2 == v3 == v4

# ## Named tuples - `NamedTuple{T1,T2,...}`

# As the name suggests, named tuples are collection similar to ordinary tuples, but whose indexing can accept also a name:

nt = (a=1, b=2.5)
#nt = ("a"=1, "b"=2.5)    # Error !
typeof(nt)
nt[1]
nt.a
keys(nt)
values(nt)

for (k,v) in pairs(nt)
    println("$k - $v")
end

# !!! warning
#     The keys of NamedTuples are _symbols_, not strings. We'll see symbols in the metaprogramming segment.

# ### Conversion
k = [:a,:b,:c]
v = [1,2,3]
nt = NamedTuple(Dict(:a=>1,:b=>2,:c=>3))             # Order not guaranteed! We are "lucky" here
nt = NamedTuple(Dict([k=>v for (k,v) in zip(k,v)]))  # Same...
v2 = [nt...]
v3 = [i[1] for i in nt]
v4 = collect(nt)
v == v2 == v3 == v4

# ## Dictionaries - `Dict{Tkey,TValue}`

# Dictionary are mutable, key-referenced containers:
# 
# |               | Mutable      | Immutable    |
# | ------------- | ------------ | ------------ | 
# | Use position  | Arrays       | Tuples       |
# | Use keys      | Dictionaries | Named tuples |

# !!! warning
#     Note that order is not preserved. For insertion-order preservation see `OrderedDict` and for sorted dictionaries see `SortedDict`, both from the [DataStructures.jl](https://github.com/JuliaCollections/DataStructures.jl) package.


# ### Initialisation

mydict = Dict(); #= or better =#  mydict = Dict{String,Int64}()
mydict = Dict('a'=>1, 'b'=>2, 'c'=>3)

# ### Indexing
mydict['a']
#mydict['d']     # error!
get(mydict,'d',0) # specific a default if key not found

# ### Adding/deleting/checking
mydict['d'] = 4

delete!(mydict,'d')
haskey(mydict, 'a')
in(('a' => 1), mydict)
typeof('a' => 1)

# ### Conversion
# Array - > Dictionary
map((i,j) -> mydict[i]=j, ['e','f','g'], [4,5,6])
mydict
k = [:a,:b,:c]
v = [1,2,3]
mydict = Dict([k=>v for (k,v) in zip(k,v)])
# Dictionary -> Arrays
collect(keys(mydict)) # keys or values alore return an iterator
collect(values(mydict)) 

# ### Iteration
for (k,v) in mydict
   println("$k is $v")
end

# ## Sets - `Set{T}`
s = Set(); #= or better =# Set{Int64}()
s = Set([1,2,3,4]) # Only a single `2` will be stored
s
push!(s,5)
delete!(s,1)
s2 = Set([4,5,6,7])
intersect(s,s2)
union(s,s2)
setdiff(s,s2)

# ## Date and time - `Date`, `DateTime`

using Dates # a standard library module dealing with dates and times, including periods and calendars

# While a `DateTime` is a more informative object it is also a much more complex one, as it has to deal with problems
# as the time zones and the daylight saving


# ### Creation of a date or time object ("input")
# From current (local) date/time...
todayDate = today()
nowTime = now()
typeof(todayDate)

typeof(nowTime) 
Date     <: Dates.AbstractTime
DateTime <: Dates.AbstractTime
nowTimeUnix = time()  # The so-called "Unix time, a 64bit integer counting the number of seconds since the beginning of the year 1970 
nowTime = Dates.unix2datetime(nowTimeUnix) # attention this is not local but UTC (Coordinated Universal Time - the Greenwitch time )!
nowTime = Dates.now(Dates.UTC) # an other way to have UTC time

# !!! tip
#     For Time Zone functionalities and conversion, use the external package [TimeZone.jl](https://github.com/JuliaTime/TimeZones.jl/)


# From a String...
christmasDay      = Date("25 Dec 2030", "d u yyyy")
newYearDay        = Date("2031/01/01", "yyyy/m/d")
christmasLunch    = DateTime("2030-12-25T12:30:00", ISODateTimeFormat)   # well known string datetime ISO8601 Format
newYearEvenDinner = DateTime("Sat, 30 Dec 2030 21:30:00", RFC1123Format) # an othe well known format

# Date and time formatters:
# - y  Year digit (ef yyyy => 2030, yy => 30)
# - m  Month digit (eg m => 3, mm => 03)
# - u  Month name (eg "Jan")
# - U  Month name long (eg "January")
# - e  Day of week (eg "Tue")
# - E  Day of week long (eg "Tuesday")
# - d  Day of month (eg d => 3, dd => 03)
# - H  Hour digit (eg H => 8, HH => 08)
# - M  Minute digit (eg M => 0, MM => 00)
# - S  Second digit (eg S => 0, SS => 00)
# - s  Millisecond digit (eg .000, fixed 3 digits)
#
# Note that the doubling for the digits matters only for using the formatters in the output (see later)

# From a tuple of integers: y, m, d, H, M, S, s ...
d  = Date(2030, 12)  # no need to give it all
dt = DateTime(2030, 12, 31, 9, 30, 0, 0) 

# ### Date/Time extraction of information ("output")...

# To String represerntation...
Dates.format(newYearDay, "dd/m/yy")
Dates.format(christmasLunch, "dd/mm/yy H:M:SS")

# Other...
## Date and DateTime...
year(christmasDay)
isleapyear(christmasDay)
month(christmasLunch)
monthname(christmasDay)
day(christmasDay)
dayofweek(christmasDay)
dayname(christmasDay)
daysofweekinmonth(christmasDay) # there are 4 Wednesdays in December 2030
dayofweekofmonth(christmasDay)  # and the 25th is the 4th of them

## Only datetime..
hour(christmasLunch)
minute(christmasLunch)
second(christmasLunch)


# ### Periods and datetime arithmetics
hollidayPeriod = newYearDay - christmasDay  # between dates is in days
longPeriod = Date(2035,6,1) - christmasDay
mealPeriod = DateTime(2030,12,31,23,30) - newYearEvenDinner # between datetime is in milliseconds
#newYearDay - newYearEvenDinner # error! no mixed
convert(DateTime,newYearDay)
convert(Date,newYearEvenDinner) # possible information loss
mealPeriod = convert(DateTime,newYearDay) - newYearEvenDinner
typeof(hollidayPeriod)
typeof(mealPeriod)

# Period hierarchy:
# - `Period`
#   - `DatePeriod`
#     - `Year`
#     - `Month`
#     - `Week`
#     - `Day`
#   - `TimePeriod`
#     - `Hour`
#     - `Minute`
#     - `Second`
#     - `Millisecond`
#     - `Microsecond`
#     - `Nanosecond`


#convert(Dates.Year,longPeriod)      # going up: error or inexacterror
convert(Dates.Millisecond,longPeriod) # going down:  fine
convert(Dates.Millisecond,mealPeriod)

canLongPeriod = Dates.canonicalize(longPeriod)
typeof(canLongPeriod) 

# That the best we can get. We can't "easily" decompose a "period" in  years or months... how many days in a month ?
# 31 or 30 ? And in an year ? A `Period` doesn't store information on when it starts.
# However we can make math with periods based on a specific date/time:

nextChristmas                = christmasDay + Year(1) # We can use the constructors of the various periods
christmasPresentsOpeningTime = christmasLunch + Hour(3)
thisWeekdayNextCentury       = dayname(today()+Year(100))

# Ranges
semesters = Dates.Date(2020,1,1):Dates.Month(6):Dates.Date(2022,1,1)
collect(semesters)

# ### Adjustments

# Iterate the past/future days of a date untill some condition is true
sundayBefChristmas = toprev(d -> Dates.dayname(d) == "Sunday", christmasDay)
lastDayOfThisMonth = tonext(d -> Dates.day(d+Day(1)) == 1, today())

# Find first or last weekday of {month,year} of a given date:
lastTuesdayOfThisMonth = tolast(today(), 2, of=Month) # "2" stands for Tuesday
firstSundayOfThisYear  = tofirst(today(), 7, of=Year) # "7" stands for Sunday
