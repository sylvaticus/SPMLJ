################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################


# # 0202 - Miscellaneous topics

# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
## If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
## Pkg.resolve()   
## Pkg.instantiate() # run this if you didn't in Segment 02.01
using Random
Random.seed!(123)


# ## Plotting

# Within the many possible packages to plot in Julia we use here the `Plots` package that allows at run time to choose the plot's backend.
# The defauld backend is `gr` (that, if you want to go back to it after you choosen another backend, you can activate with `gr()`). Other common backends are the Python MatplotLib (`pyplot()`, requiring package `PyPlot`) and Plotly (`plotlyjs()` from package `PlotlyJS`).
# Actually we use `StatsPlots` that just adds a set of convenient functionalities (in `plots` terminology called "_recipes_") on top of `Plots`.

using StatsPlots # no need to `using Plots` as `Plots` elements are reexported by StatsPlots

# The basic idea is that we can draw different "graphical elements" in a Plot figure. The `plot(.)` function will first create a new figure, while `plot!(.)` will modify an existing plot (by drawing new elements on it, such as a new serie) taking the "current" plot as default if no plot object is passed as first argument.

# ### Plotting functions

# Let's start plotting a function. The FIRST TIME you invoke `plot` will take a while. This is the famous "time to first plot" problem due to the JIT compilation, but it refer only to the first plotting in a working session 
plot(cos) # default [-5,+5] range
savefig("currentPlot1.svg"); #src
# ![](currentPlot.svg)
plot!(x->x^2, -2,2 ) # more explicit, with ranges
savefig("currentPlot1.svg"); #src
# ![](currentPlot.svg)
plot!(x->max(x,2), label="max function", linestyle=:dot, color=:black, title="Chart title", xlabel= "X axis", ylabel="Y axis", legend=:topleft) # a bit of design
savefig("currentPlot2.svg"); #src
# ![](currentPlot2.svg)
plot!(twinx(),x->20x,colour=RGB(20/255,120/255,13/255)) # secondary axis
savefig("currentPlot3.svg"); #src
# ![](currentPlot3.svg)

# ### Plotting data

using DataFrames
x = 11:15
data = DataFrame(a=[4,8,6,6,3], b=[2,4,5,8,6], c=[20,40,15,5,30])
plot(x, Matrix(data)) # x, series (in column)
savefig("currentPlot4.svg"); #src
# ![](currentPlot4.svg)
@df data plot(x, :a, seriestype=:bar, legend=:topleft)
savefig("currentPlot5.svg"); #src
# ![](currentPlot5.svg)
plot!(x, data.b, seriestype=:line)
savefig("currentPlot6.svg"); #src
# ![](currentPlot6.svg)
scatter!(twinx(), x, data.c) # alias for `plot!(..., seriestype=:scatter)`
savefig("currentPlot7.svg"); #src
# ![](currentPlot7.svg)

# ### Layouts with multiple plots

l = @layout [a ; b c] # a,b,c, here are just placeholders, not related with the df column names..
p1 = plot(x, data.a)
p2 = scatter(x, data.b)
p3 = plot(x, data.c)
plot(p1, p2, p3, layout = l)
savefig("currentPlot8.svg"); #src
# ![](currentPlot8.svg)

# ### Saving the plot..
savefig("myplot.png")
savefig("myplot.pdf")
savefig("myplot.svg")

# ## Probabilistic analysis

# Julia has a very elegant package to deal with probability analysis, the `Distributions` package.

using Distributions

# The idea is that we first create a `Distribution` object, of the distribution family and with the parameter required, and then we operate on it, for example to sample or to retrieve a quantile.
# The following table gives the distribution constructors for some of the most common distributions:


# | Discrete distr.   | Constructor                       |  Continuous distr. | Constructor              |
# | ----------------- | --------------------------------- | ------------------ | ------------------------ |
# | Discrete uniform  | `DiscreteUniform(lRange,uRange)`  | Uniform            | `Uniform(lRange,uRange)` |
# | Bernoulli         | `Bernoulli(p)`                    | Exponential        | `Exponential(rate)`      |
# | Binomial          | `Binomial(n,p)`                   | Laplace            | `Laplace(loc, scale)`    |
# | Categorical       | `Categorical(ps)`                 | Normal             | `Normal(μ,σ)`            |
# | Multinomial       | `Multinomial(n, ps)`              | Erlang             | `Erlang(n,rate)`         |
# | Geometric         | `Geometric(p)`                    | Cauchy             | `Cauchy(μ, σ)`           |
# | Hypergeometric    | `Hypergeometric(nS, nF, nTrials)` | Chisq              | `Chisq(df)`              |
# | Poisson           | `Poisson(rate)`                   | T Dist             | `TDist(df)`              |
# | Negative Binomial | `NegativeBinomial(nSucc,p)`       | F Dist             | `FDist(df1, df2)`        |
# |                   |                                   | Beta Dist          | `Beta(shapeα,shapeβ)`    |
# |                   |                                   | Gamma Dist         | `Gamma(shapeα,1/rateβ)`  |

d = Normal(10,3) # note that the parameter is the standard deviation, not the variance

mean(d)
var(d)
median(d)
quantile(d,0.9)
cdf(d,13.844)
pdf(d,0)
rand(d,3,4,2)

sample = rand(d,1000);
density(sample)
savefig("currentPlot9.svg"); #src
# ![](currentPlot9.svg)
plot!(d)
savefig("currentPlot10.svg"); #src
# ![](currentPlot10.svg)
fit(Normal, sample) # using MLE 

# ## Curve fitting

# `LsqFit` is a very flexible data fitting package for linear and nonlinear arbitrary functions (using least squares).
# Let's use it to fit a logistic growth curve (Verhulst model) of volumes for a forest stand.
using LsqFit
data = DataFrame(
    age = 20:5:90,
    vol = [64,112,170,231,293,352,408,459,505,546,582,613,640,663,683]
) # Scots Pine, data from the UK Forestry Commission https://web.archive.org/web/20170119072737/http://forestry.gov.uk/pdf/FCBK048.pdf/$FILE/FCBK048.pdf
plot(data.vol)
savefig("currentPlot11.svg"); #src
# ![](currentPlot11.svg)

logisticModel(age,parameters) = parameters[1]/(1+exp(-parameters[2] * (age-parameters[3]) ))
logisticModelVec(age,parameters) = logisticModel.(age,Ref(parameters))

initialParameters = [1000,0.02,50] #max growth; growth rate, mid age
fitobject         = curve_fit(logisticModelVec, data.age, data.vol, initialParameters)
fitparams         = fitobject.param

fitobject.resid
residuals = logisticModelVec(data.age,fitparams) .- data.vol 
fitobject.resid == residuals

sigma            = stderror(fitobject)
confidence_inter = confidence_interval(fitobject, 0.05) # 5% significance level

x = 0:maximum(data.age)*1.5
plot(x->logisticModel(x,fitparams),0,maximum(x), label= "Fitted vols", legend=:topleft)
plot!(data.age, data.vol, seriestype=:scatter, label = "Obs vols")
plot!(data.age, residuals, seriestype=:bar, label = "Residuals")
savefig("currentPlot12.svg"); #src
# ![](currentPlot12.svg)

# ## Constrained optimisation

# `JuMP` is the leading library to express complex optimisation problems in a clear, mathematical friendly syntax, compute the information required by the solver engines to solve the optimisation problem, pass the problem to the aforementioned solver engines and retrieve the solutions.

# JuMP has the same flexibility and expressivity of dedicated _algeabric modelling languages_ as `GAMS` or `AMPL` but with the advantage of being a library within a much more general programming language, with larger community, development tools, language constructs and possibility to interface the specific "optimisation component" of a model with the rest of the model.

# We will see how to specify the variables, the constraints and the objective function of an optimisation model, how to "solve" it and how to retrieve the optimal values

# ### A linear example: the classical "transport" problem

# Obj: minimise transport costs $c$ from several plants $p$ to several markets $m$ under the contraint to satisfy the demand $d_m$ at each market while respecting the production capacity $c_p$ of each plant:
# $min_{x_{p,m}} \sum_p \sum_m c_{p,m} * x_{p,m}$
# subject to:
# $\sum_m x_{p,m} \leq d_m$
# $\sum_p x_{p,m} \geq c_m$

using  JuMP, GLPK, DataFrames, CSV

# #### "Sets" and exogenous parameters definition

# index sets
orig = ["Epinal", "Bordeaux", "Grenoble"] # plant/sawmills origin of timber
dest = ["Paris", "Lyon", "Nantes", "Tulouse", "Lille", "Marseille", "Strasbourg"] # markets
prod = ["Fuelwood", "Sawnwood", "Pannels"]

# Read input data into DataFrames and then into Dictionaries with 2D or 3D tuples of index sets as keys

# supply(prod, orig) amounts available at origins
supplytable = CSV.read(IOBuffer("""
prod     Epinal Bordeaux Grenoble
Fuelwood 400    700      800
Sawnwood 800    1600     1800
Pannels  200    300      300
"""), DataFrame, delim=" ", ignorerepeated=true,copycols=true)
supply = Dict( (r[:prod],o) => r[Symbol(o)] for r in eachrow(supplytable), o in orig)

# demand(prod, dest) amounts required at destinations
demandtable = CSV.read(IOBuffer("""
prod      Paris Lyon Nantes Tulouse Lille Marseille Strasbourg
Fuelwood  300   300  100    75      650   225       250
Sawnwood  500   750  400    250     950   850       500
Pannels   100   100  0      50      200   100       250
"""), DataFrame, delim=" ", ignorerepeated=true,copycols=true)
demand = Dict( (r[:prod],d) => r[Symbol(d)] for r in eachrow(demandtable), d in dest)

# limit(orig, dest) of total units from any origin to destination
defaultlimit = 625.0
limit = Dict((o,d) => defaultlimit for o in orig, d in dest)

# cost(prod, orig, dest) Shipment cost per unit
costtable = CSV.read(IOBuffer("""
prod     orig     Paris Lyon Nantes Tulouse Lille Marseille Strasbourg
Fuelwood Epinal   30    10   8      10      11    71        6
Fuelwood Bordeaux 22    7    10     7       21    82        13
Fuelwood Grenoble 19    11   12     10      25    83        15

Sawnwood Epinal   39    14   11     14      16    82        8
Sawnwood Bordeaux 27    9    12     9       26    95        17
Sawnwood Grenoble 24    14   17     13      28    99        20

Pannels  Epinal   41    15   12     16      17    86        8
Pannels  Bordeaux 29    9    13     9       28    99        18
Pannels  Grenoble 26    14   17     13      31    104       20
"""), DataFrame, delim=" ", ignorerepeated=true,copycols=true)
cost = Dict( (r[:prod],r[:orig],d) => r[Symbol(d)] for r in eachrow(costtable), d in dest)

# #### Optimisation model definition

trmodel = Model(GLPK.Optimizer)
set_optimizer_attribute(trmodel, "msg_lev", GLPK.GLP_MSG_ON)

# #### Model's endogenous variables definition

@variables trmodel begin
    x[p in prod, o in orig, d in dest] >= 0
end;

# #### Constraints definition

@constraints trmodel begin
    supply[p in prod, o in orig], # observe supply limit at plant/sawmill origin o
        sum(x[p,o,d] for d in dest) <= supply[p,o]
    demand[p in prod, d in dest], # satisfy demand at market dest d
        sum(x[p,o,d] for o in orig) >= demand[p,d]
    c_total_shipment[o in orig, d in dest],
        sum(x[p,o,d] for p in prod) <= limit[o,d]
end;

# #### Objective function definition
@objective trmodel Min begin
    sum(cost[p,o,d] * x[p,o,d] for p in prod, o in orig, d in dest)
end

# #### Human-readable visualisation of the model

print(trmodel)

# #### Model resolution
optimize!(trmodel)
status = termination_status(trmodel)

# #### Post-resolution information retrieval 

# Here, after the model has been "solved", we can retrieve information as the optimal level of the endogenous variables, the value of the opjective function at these optimal levels and the shadow costs of the contraints.

if (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.TIME_LIMIT) && has_values(trmodel)
    println("#################################################################")
    if (status == MOI.OPTIMAL)
        println("** Problem solved correctly **")
    else
        println("** Problem returned a (possibly suboptimal) solution **")
    end
    println("- Objective value (total costs): ", objective_value(trmodel))
    println("- Optimal routes:\n")
    optRoutes = value.(x)
    for p in prod
      println("\n* $(p):")
      [println("$o --> $d: $(optRoutes[p,o,d])") for o in orig, d in dest]
      println("- Shadow prices of supply:")
      [println("$o = $(dual(supply[p,o]))") for o in orig]
      println("- Shadow prices of demand:")
      [println("$d = $(dual(demand[p,d]))") for d in dest]
    end
else
    println("The model was not solved correctly.")
    println(status)
end


# ### A nonlinear example: portfolio optimisation

# The problem objective is to choose the shares of different assets in the portfolio (here forest species, but the example is exactly the same considering other assets, for example financial investments) that maximise the portfolio expected returns while minimising its expected variance under the portfolio owner risk aversion risk.
# Here the "returns" are based on the timber production and the covariance between individual species of the portfolio is based on the observed volume growth covariances.
# The idea is that within the infinite possible allocations, the locus of those allocations for which is not possible to increase the portfolio profitability without increasing also its variance and the converse whose variance can not be lowered without at the same time lower its expected profitability are efficient in the Pareto meaning and form an "efficient frontier". Within this frontier the problem is to find the unique point that maximise the utility of the portfolio's owner given its risk aversion characteristic.
# Graphically the problem is depicted i nthe following picture:

# ![The efficient frontier and the owner utility curves](https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsSources/02_-_JULIA2_-_Scientific_programming_with_Julia/graph_eff_frontier_v2.png)


# Data originally from the Institut national de l'information géographique et forestière (IGN) of France. See the paper [A. Dragicevic, A. Lobianco, A. Leblois (2016), ”Forest planning and productivity-risk trade-off through the Markowitz mean-variance model“, Forest Policy and Economics, Volume 64](http://dx.doi.org/10.1016/j.forpol.2015.12.010) for a thorough discussion of this model.

# Declare the packages we are going to use:
using JuMP, Ipopt, StatsPlots

# Forest species names
species   = ["Chêne pédonculé", "Chêne sessile", "Hêtre", "Pin sylvestre"]
nSpecies  = length(species)
# Average productiities by specie
# This is implemented in a dictionary: key->value
y   = Dict( "Chêne pédonculé" => 1.83933333333333,
            "Chêne sessile"   => 2.198,
            "Hêtre"           => 3.286,
            "Pin sylvestre"   => 3.3695)

# Covariance matrix between species
σtable = [[0.037502535947712	0.016082745098039	0.027797176470588	-0.025589882352942]
          [0.016082745098039	0.015177019607843	0.018791960784314	-0.102880470588234]
          [0.027797176470588	0.018791960784314	0.031732078431373	-0.166391058823529]
          [-0.025589882352942	-0.102880470588234	-0.166391058823529	2.02950454411765]]
# We reshape the covariance matrix in a dictionary (sp1,sp2) -> value
# The function (ix,x) = enumerate(X) returns a tuple of index position and element
# for each element of an array
σ = Dict((i,j) => σtable[i_ix,j_ix] for (i_ix,i) in enumerate(species), (j_ix,j) in enumerate(species))

################################################################################
###### Showing the possible mean/variance of the portfolio by simulation #######
################################################################################

nSamples = 1000
shares   = rand(nSamples,nSpecies);
# Converting to probabilities:
import BetaML.Utils:softmax
[shares[i,:]  = softmax(shares[i,:], β=one.(shares[i,:]) .* 5) for i in 1:nSamples]

pScores = Array{Float64,2}(undef,0,2)
for i in 1:nSamples
    global pScores
    pVar    = sum(shares[i,j1] * shares[i,j2] * σ[species[j1],species[j2]] for j1 in 1:nSpecies, j2 in 1:nSpecies)
    pY      = sum(shares[i,j]*y[species[j]] for j in 1:nSpecies)
    pScores = vcat(pScores,[pVar pY])
end

scatter(pScores[:,1],pScores[:,2],colour=:blue)
savefig("currentPlot13.svg"); #src
# ![](currentPlot13.svg)

################################################################################
### Finding (one) optimal portfolio ############################################
################################################################################

# Risk aversion coefficient
α = 0.1

# We declare an optimisation problem, we name it "m" and we let JuMP associate it with the most
# suitable solver within the one installed:
port = Model(Ipopt.Optimizer)

# We declare a set of variables, indicized by the species name:
@variables port begin
    x[i in species] >= 0
end

# We declare the constraint shat the sum of shares must be equal to 1
@constraint(port, c_share, sum(x[i] for i in species) == 1)

#=
@objective port Min begin
  α *  sum(x[i] * x[j] * σ[i,j] for i in species for j in species) - sum(x[i] * y[i] for i in species)
end
=#

@NLobjective port Min α *  sum(x[i] * x[j] * σ[i,j] for i in species for j in species) - sum(x[i] * y[i] for i in species)

# Print the optimisation model in nice human-readable format:
print(port)

# Solve the model and return the solving status:
optimize!(port)
status = termination_status(port)

# Return the objective:
println("Objective value: ", objective_value(port))
# Return the value of the decision variable:
optShares = value.(x)
[println("$sp = $(optShares[sp])") for sp in species];
pOptVar = sum(optShares[species[j1]] * optShares[species[j2]] * σ[species[j1],species[j2]] for j1 in 1:nSpecies, j2 in 1:nSpecies)
pOptY   = sum(optShares[species[j]]*y[species[j]] for j in 1:nSpecies)


function computeOptimalPortfolio(species,y,σ,α)
    port = Model(Ipopt.Optimizer)
    set_optimizer_attribute(port, "print_level", 0)
    @variables port begin
        x[i in species] >= 0
    end
    @constraint(port, c_share, sum(x[i] for i in species) == 1)
    @NLobjective port Min α *  sum(x[i] * x[j] * σ[i,j] for i in species for j in species) - sum(x[i] * y[i] for i in species)
    optimize!(port)
    status = termination_status(port)
    optShares = value.(x)
    pOptVar = sum(optShares[species[j1]] * optShares[species[j2]] * σ[species[j1],species[j2]] for j1 in 1:nSpecies, j2 in 1:nSpecies)
    pOptY   = sum(optShares[species[j]]*y[species[j]] for j in 1:nSpecies)
    return (pOptVar,pOptY)
end

αs = [1000,100,10,1,0.1,0.05,0.02,0.01]
pOptScores = Array{Float64,2}(undef,0,2)
for α in αs
    global pOptScores
    pVar,pY =computeOptimalPortfolio(species,y,σ,α)
    pOptScores = vcat(pOptScores,[pVar pY])
end
scatter!(pOptScores[:,1],pOptScores[:,2],colour=:red)
savefig("currentPlot14.svg"); #src
# ![](currentPlot14.svg)

αs = [82.45,50,30,20,15,12,10,9,8,7,6,5]
pOptScores = Array{Float64,2}(undef,0,2)
for α in αs
    global pOptScores
    pVar,pY =computeOptimalPortfolio(species,y,σ,α)
    pOptScores = vcat(pOptScores,[pVar pY])
end

scatter(pOptScores[:,1],pOptScores[:,2],colour=:red)
savefig("currentPlot15.svg"); #src
# ![](currentPlot15.svg)


