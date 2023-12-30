################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################


# # 0201 - Data Wrangling 

# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
## If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
#Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)


# ## Introduction

# This segment will be mostly based on [DataFrames](https://github.com/JuliaData/DataFrames.jl) and related packages

# !!! info DataFrames vs Matrix 
#     DataFrames are popular format for **in-memory tabular data**. Their main advantages over Arrays are that they can efficiently store different types of data on each column (indeed each column is a wrapper over an `Array{T,1}` where `T` is specific to each column) and, thanks also to their named columns, provide convenient API for data operations, like indexing, querying , joining, split-apply-combine, etc. 

# !!! info
#     In most circunstances we can refer to dataframe columns either by using their name as a string, e.g. `"Region"`, or as symbol, e.g. `:Region`. In the rest of the segment we will use the string approach.


# ## Data import

# Our example: Forest volumes and area by country and year
# Source: Eurostat; units: forarea: Milion hectars, forvol: Milion cubic metres

# ### Built-in solution: CSV --> Matrix
using DelimitedFiles  # in the stdlib
data = convert(Array{Float64,2},readdlm("data.csv",';')[2:end,3:end]) # Skip the first 1 row and the first 2 columns - out is a Matrix

# ### CSV.jl: {CSV file, hardwritten data} --> DataFrame
using CSV, DataFrames
data = CSV.read("data.csv",DataFrame) # source, sink, kword options
data = CSV.read("data.csv",NamedTuple)

data = CSV.read(IOBuffer("""
Country	Year	forarea	forvol
Germany	2000	11.354	3381
France	2000	15.288	2254.28
Italy	2000	8.36925	1058.71
Sweden	2000	28.163	3184.67
Germany	2020	11.419	3663
France	2020	17.253	3055.83
Italy	2020	9.56613	1424.4
Sweden	2020	27.98	3653.91
"""), DataFrame, copycols=true)

# Some common CSV.jl options: `delim` (use `'\t'` for tab delimited files), `quotechar`, `openquotechar`, `closequotechar`, `escapechar`, `missingstring`, `dateformat`, `append`, `writeheader`, `header`, `newline`, `quotestrings`, `decimal`, `header`, `normalizenames`, `datarow`, `skipto`, `footerskip`, `limit`, `transpose`, `comment`, `use_mmap`, `type`, `types` (e.g. `types=Dict("fieldFoo" => Union{Missing,Int64})`), `typemap`, `pool`, `categorical`, `strict`, `silencewarnings`, `ignorerepeated`

# ### XLSX.jl: xlsx -> {Matrix, DataFrame}
using XLSX
sheetNames = XLSX.sheetnames(XLSX.readxlsx("data.xlsx"))
data = XLSX.readxlsx("data.xlsx")["Sheet1"]["A1:D9"]
data = XLSX.readxlsx("data.xlsx")["Sheet1"][:]
data = XLSX.readdata("data.xlsx", "Sheet1", "A1:D9")
XLSX.readtable("data.xlsx", "Sheet1") # a specific XLSX data structure usable as DF constructor
data = DataFrame(XLSX.readtable("data.xlsx", "Sheet1")) 

# ### OdsIO.jl: ods -> {Matrix, DataFrame}
using OdsIO
ods_read("data.ods";sheetName="Sheet1",retType="DataFrame")
ods_read("data.ods";sheetName="Sheet1",retType="Matrix",range=[(2,3),(9,4)]) # [(tlr,tlc),(brr,brc)]

# ### HTTP.jl: from internet
import HTTP, Pipe.@pipe
using ZipFile, Tar
urlData  = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/raw/main/lessonsSources/02_-_JULIA2_-_Scientific_programming_with_Julia/data.csv"
urlDataZ = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/raw/main/lessonsSources/02_-_JULIA2_-_Scientific_programming_with_Julia/data.zip"


data = @pipe HTTP.get(urlData).body        |>
             replace!(_, UInt8(';') => UInt8(' ')) |>  # if we need to do modifications to the file before importing 
             CSV.File(_, delim=' ')                |>
             DataFrame;

# ### ZipFile.jl : from a single zipped csv file..
# ...on disk... 
data = @pipe ZipFile.Reader("data.zip").files[1] |>
             CSV.File(read(_), delim=';') |>
             DataFrame  

# ...or on internet:
data = @pipe HTTP.get(urlDataZ).body       |>
             IOBuffer(_)                      |>
             ZipFile.Reader(_).files[1]    |>
             CSV.File(read(_), delim=';')  |>
             DataFrame 

            
# ### DataFrame constructor
## named tuple of (colName,colData)..
data = DataFrame(
    country = ["Germany", "France", "Italy", "Sweden", "Germany", "France", "Italy", "Sweden"],
    year    = [2000,2000,2000,2000,2020,2020,2020,2020],
    forarea = [11.354, 15.288, 8.36925, 28.163, 11.419, 17.253, 9.56613, 27.98],
    forvol  = [3381, 2254.28, 1058.71, 3184.67, 3663, 3055.83, 1424.4, 3653.91]
)

# ### Matrix -> DataFrame

# Headers and data separated:

M = ["Germany"	2000	11.354	3381
     "France"	2000	15.288	2254.28
     "Italy"	2000	8.36925	1058.71
     "Sweden"	2000	28.163	3184.67
     "Germany"	2020	11.419	3663
     "France"	2020	17.253	3055.83
     "Italy"	2020	9.56613	1424.4
     "Sweden"	2020	27.98	3653.91]
headers = ["Country", "Year", "forarea", "forvol"]
## array of colData arrays, array of headers
data = DataFrame([[M[:,i]...] for i in 1:size(M,2)], Symbol.(headers))

# Headers on the first row of the matrix:
M = ["Country"	"Year"	"forarea" "forvol"
     "Germany"	2000	11.354	  3381
     "France"	2000	15.288	  2254.28
     "Italy"	2000	8.36925	  1058.71
     "Sweden"	2000	28.163	  3184.67
     "Germany"	2020	11.419	  3663
     "France"	2020	17.253	  3055.83
     "Italy"	2020	9.56613	  1424.4
     "Sweden"	2020	27.98	  3653.91]
data = DataFrame([[M[2:end,i]...] for i in 1:size(M,2)], Symbol.(M[1,:])) # note the autorecognision of col types


# ## Getting insights on the data

# In VSCode, we can also use the workpanel for a nice sortable tabular view of a df

show(data,allrows=true,allcols=true)
first(data,6)
last(data, 6)
describe(data)
nR,nC = size(data)
names(data)
for r in eachrow(data)
    println(r) # note is it a "DataFrameRow"
end
for c in eachcol(data)
    println(c) # an array
    println(nonmissingtype(eltype(c)))
end

# ## Selection and querying data

# ### Column(s) selection
# In general we can select: (a) by name (strings or symbols) or by position; (b) a single or a vector of columns, (c) copying or making a view (a reference without copying) of the underlying data 
data[:,["Country","Year"]] # copy, using strings
data[!,[:Country,:Year]]   # view, using symbols
data.Year                  # equiv. to `data[!,Year]`
data[:,1]                  # also `data[!,1]`
data[:,Not(["Year"])]

# ### Row(s) selection
data[1,:]    # DataFrameRow
data[1:3,:]  # DataFrame
# Note rows have no title names as colums do.

# ### Cell(s) selection
data[2,[2,4]]
data[2,"forarea"]

# Note that the returned selection is:
# * an `Array{T,1}` if it is a single column;
# * a `DataFrameRow` (similar in behaviour to a `DataFrame`) if a single row;
# * `T` if a single cell;
# * an other `DataFrame` otherwise.

# ### Boolean selection
# Both rows and column of a DataFrame (but also of an Matrix) can be selected by passing an array of booleans as column or row mask (and, only for Matrices, also a matrix of booleans)

mask = [false, false, false, false, true, true, true, true]
data[mask,:]
mask = fill(false,nR,nC)
mask[2:end,2:3] .= true
mask
## data[mask] # error !
Matrix(data)[mask] 

# Boolean selection can be used to filter on conditions, e.g.:
data[data.Year .>= 2020,:]
data[[i in ["France", "Italy"] for i in data.Country] .&& (data.Year .== 2000),:] # note the parhenthesis

# ### Filtering using the @subset macro from the `DataFramesMacro` package
using DataFramesMeta
@subset(data, :Year .> 2010 )
colToFilter = :Country
@subset(data, :Year .> 2010, cols(colToFilter) .== "France" ) # Conditions are "end" by default. If the column name is embedded in a varaible we eed to use `cols(varname)`

# ### Filtering using the `Query` package
using Query 
dfOut = @from i in data begin # `i` is a single row
    @where i.Country == "France" .&& i.Year >= 2000
    ## Select a group of columns, eventually changing their name:
    @select {i.Year, FranceForArea=i.forarea} # or just `i` for the whole row
    @collect DataFrame
end
# long but flexible

# ### Managing missing values

# !!! tip
#     See also the section [`Missingness implementations`](@ref) for a general discussion on missing values. [BetaML](https://github.com/sylvaticus/BetaML.jl) has now several algorithms for missing imputation.

df = copy(data)
## df[3,"forarea"]  = missing # Error, type is Flat64, not Union{Float64,Missing}
df.forarea = allowmissing(df.forarea) # also disallowmissing
allowmissing!(df)
df[3,"forarea"]  = missing
df[6,"forarea"]  = missing
df[6,"Country"]  = missing
nMissings        = length(findall(x -> ismissing(x), df.forarea)) # Count `missing` values in a column.
dropmissing(df)
dropmissing(df[:,["forarea","forvol"]])
collect(skipmissing(df.forarea))
completecases(df)
completecases(df[!,["forarea","forvol"]])
[df[ismissing.(df[!,col]), col] .= 0 for col in names(df) if nonmissingtype(eltype(df[!,col])) <: Number] # Replace `missing` with `0` values in all numeric columns, like `Float64` and `Int64`;
[df[ismissing.(df[!,col]), col] .= "" for col in names(df) if nonmissingtype(eltype(df[!,col])) <: AbstractString] # Replace `missing` with `""` values in all string columns;
df

# ## Editing data
df = copy(data)
df[1,"forarea"] = 11.3
df[[2,4],"forarea"] .= 10
df
push!(df,["UK",2020,5.0,800.0]) # add row
sort!(df,["Country","Year"], rev=false)
df2 = similar(df) # rubish inside
df = similar(df,0) # empty a dataframe. The second parameter is the number of rows desired

# ## Work on dataframe structure

df       = copy(data)
df.foo   = [1,2,3,4,5,6,7,8] # existing or new column
df.volHa = data.forvol ./ data.forarea
df.goo   = Array{Union{Missing,Float64},1}(missing,size(data,1))
select!(df,Not(["foo","volHa","goo"])) # remove  cols by name
rename!(df, ["Country2", "Year2", "forarea2", "forvol2"])
rename!(df, Dict("Country2" => "Country"))
df       = df[:,["Year2", "Country","forarea2","forvol2"] ] #  change column order
insertcols!(df, 2, :foo => [1,2,3,4,5,6,7,8] ) # insert a column at position 2

df.namedYear = map(string,df.Year2)
stringToInt(str) = try parse(Int64, str) catch; return(missing) end; df.Year3 = map(stringToInt, df.namedYear)

df2 = hcat(df,data,makeunique=true) 
df3 = copy(data)
df4 = vcat(data,df3)

# ### Categorical data
using CategoricalArrays # You may want to consider also PooledArrays
df.Year2 = categorical(df.Year2)
transform!(df, names(df, AbstractString) .=> categorical, renamecols=false) # transform to categorical all string columns

# !!! warning
#     Attention that while the memory to store the data decreases, and grouping is way more efficient, filtering with categorical values is not necessarily quicker (indeed it can be a bit slower)

levels(df.Year2)
levels!(df.Country,["Sweden","Germany","France","Italy"]) # Let you define a personalised order, useful for ordered data
sort(df.Country)
sort!(df,"Country")
df.Years2 = unwrap.(df.Year2) # convert a categorical array into a normal one.

# ### Joining dataframes
df1,df2 = copy(data),copy(data)
push!(df1,["US",2020,5.0,1000.0])
push!(df2,["China",2020,50.0,1000.0])
rename!(df2,"Year"=>"year")
innerjoin(df1,df2,on=["Country","Year"=>"year"],makeunique=true) # common records only 
# Also available: `leftjoin` (all records on left df), `rightjoin` (all on right df), `outerjoin`(all records returned), `semijoin` (like inner join by only with columns from the right df), `antijoin` (left not on right df) and `crossjoin` (like the cartesian product, each on the right by each on the left)

# ## Pivoting data

# _long_ and _wide_ are two kind of layout for equivalent representation of tabled data.
# In a _long_ layout we have each row being an observation of a single variable, and each record is represented as dim1, dim2, ..., value. As the name implies, "long" layouts tend to be relativelly long and hard to analyse by an human, but are very easity to handle.
# At the opposite, A _wide_ layout represents multiple observations on the same row, eventually using multiple horizontal axis as in the next figure (but Julia dataframes handle only a single horizzontal axis):

# |          |              |       |                  |      |
# | -------- | ------------ | ----- | ---------------- | ---- |
# |          | Forest Area  |       |   Forest Volumes |      |
# |          | 2000         | 2020  | 2000             | 2020 |
# | Germany  | 11.35        | 11.42 | 3381             | 3663 |
# | France   | 15.29        | 17.25 | 2254             | 3056 |

# _wide layout is easier to visually analise for a human mind but much more hard to analyse in a sistermatic way. We will learn now how to move from one type of layout to the other.

# ### Stacking columns: from _wide_ to _long_
longDf  = stack(data,["forarea","forvol"])    # we specify the variable to stack (the "measured" variables)
longDf2 = stack(data,Not(["Country","Year"])) # we specify the variables _not_ to stack (the id variables)
longDf3 = stack(data)                         # automatically stack all numerical variables
longDf == longDf2 == longDf3
# Note how the columns `variable` and `value` have been added automatically to host the stachked data

# ### Unstacking columns: from _wide_ to _long_
wideDf = unstack(longDf,["Country","Year"],"variable","value") # args: df, [cols to remains cols also in the wide layout], column with the ids to expand horizontally and column with the relative values
wideDf2 = unstack(longDf,"variable","value") # cols to remains cols also in the wide layout omitted: all cols not to expand and relative value col remains as col
wideDf == wideDf2 == data

# While the DataFrames package doesn't support multiple axis we can still arrive to the table below with a little bit of work by unstacking different columns in separate wide dataframes and then joining or horizontally concatenating them:
wideArea = unstack(data,"Country","Year","forarea")
wideVols = unstack(data,"Country","Year","forvol")
rename!(wideArea,["Country","area_2000","area_2020"])
rename!(wideVols,["Country","vol_2000","vol_2020"])
wideWideDf = outerjoin(wideArea,wideVols,on="Country") 

# ## The Split-Apply-Combine strategy

# Aka "divide and conquer". Rather than try to modify the dataset direclty, we first split it in subparts, we work on each subpart and then we recombine them in a target dataset

using Statistics # for `mean`
groupby(data,["Country","Year"]) # The "split" part

# Aggregation:
combine(groupby(data,["Year"]) ,  "forarea" => sum => "sum_area", "forvol" => sum => "sum_vol", nrow)
# ...or...
combine(groupby(data,["Year"])) do subdf # slower
    (sumarea = sum(subdf.forarea), sumvol = sum(subdf.forvol), nCountries = size(subdf,1))
end
# Cumulative computation:
a = combine(groupby(data,["Year"])) do subdf # slower
    (country = subdf.Country, area = subdf.forarea, cumArea = cumsum(subdf.forarea))
end

# Note in these examples that while in the aggregation we was returning a _single record_ for each subgroup (hence we did some dimensionality reduction) in the cumulative compuation we still output the whole subgroup, so the combined dataframe in output has the same number of rows as the original dataframe.

# An alternative approach is to use the `@linq` macro from the `DatAFrameMEta` package that provide a R's `dplyr`-like query language using piped data: 
using DataFramesMeta
dfCum = @linq data |>
            groupby([:Year]) |>
            transform(:cumArea = cumsum(:forarea))

# ## Export and saving 

# ### DataFrame to Matrix

M = Matrix(data)
# !!! warning
#     Attention that if the dataframe contains different types across columns, the inner type of the matrix will be `Any`

M = Matrix{Union{Float64,Int64,String}}(data)

# ### DataFrame to Dictionary

function toDict(df, dimCols, valueCol)
    toReturn = Dict()
    for r in eachrow(df)
        keyValues = []
        [push!(keyValues,r[d]) for d in dimCols]
        toReturn[(keyValues...,)] = r[valueCol]
    end
    return toReturn
end
dict = toDict(data,["Country","Year"],["forarea","forvol"])
dict["Germany",2000][1]
dict["Germany",2000]["forvol"]
toDict(data,["Country","Year"],"forarea")

# ### DataFrame to NamedTuple

nT = NamedTuple(Dict([Symbol(c) => data[:,c]  for c in names(data)]))        # order not necessarily preserved
using DataStructures
nT = NamedTuple(OrderedDict([Symbol(c) => data[:,c]  for c in names(data)])) # order preserved

# ### Saving as CSV file
rm("outdata.csv", force=true)
CSV.write("outdata.csv",data) # see options at the beginning of segment in the import section and `? CSV.write` for specific export options

# ### Saving as OpenDocument spreadsheet
rm("outdata.ods", force=true)
ods_write("outdata.ods",Dict(("myData",3,2) => data)) # exported starting on cell B3 of sheet "myData"

# ### Saving as Excel spreadsheet
rm("outdata.xlsx", force=true)
XLSX.writetable("outdata.xlsx",myData = (collect(eachcol(data)),names(data)))

