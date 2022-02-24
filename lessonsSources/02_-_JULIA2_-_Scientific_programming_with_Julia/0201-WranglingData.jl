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
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)
using DelimitedFiles            # stdlib
using CSV, DataFrames, OdsIO, XLSX



# ## Data import

# Our example: Forest volumes and area by country and year
# Source: Eurostat; units: forarea: Milion hectars, forvol: Milion cubic metres

# ### Built-in solution: CSV --> Matrix
data = convert(Array{Float64,2},readdlm("data.csv",';')[2:end,3:end]) # Skip the first 1 row and the first 2 columns - out is a Matrix
# ### CSV.jl: {CSV file, hardwritten data} --> DataFrame
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
sheetNames = XLSX.sheetnames(XLSX.readxlsx("data.xlsx"))
data = XLSX.readxlsx("data.xlsx")["Sheet1"]["A1:D9"]
data = XLSX.readxlsx("data.xlsx")["Sheet1"][:]
data = XLSX.readdata("data.xlsx", "Sheet1", "A1:D9")
XLSX.readtable("data.xlsx", "Sheet1") # tuple vector of (data) vectors, vector of symbols, usable as DF constructor
data = DataFrame(XLSX.readtable("data.xlsx", "Sheet1")...) 
# ### OdsIO.jl: ods -> {Matrix, DataFrame}
ods_read("data.ods";sheetName="Sheet1",retType="DataFrame")
ods_read("data.ods";sheetName="Sheet1",retType="Matrix",range=[(2,3),(9,4)]) # [(tlr,tlc),(brr,brc)]


# ### HTTP.jl: from internet
urlDataOriginal = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original"
data = @pipe HTTP.get(urlDataOriginal).body                                                |>
             replace!(_, UInt8('\t') => UInt8(' '))                                        |>
             CSV.File(_, delim=' ', missingstring="NA", ignorerepeated=true, header=false) |>
             DataFrame;



# !!! info DataFrames vs Matrix 
#     DataFrames are popular format for **in-memory tabular data**. Their main advantages over Arrays are that they can efficiently store different types of data on each column (indeed each column is a wrapper over an `Array{T,1}` where `T` is specific to each column) and, thanks also to their named columns, provide convenient API for data operations, like indexing, querying , joining, split-apply-combine, etc.   





