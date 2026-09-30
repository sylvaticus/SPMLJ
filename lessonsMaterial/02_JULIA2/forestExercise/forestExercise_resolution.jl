################################################################################
# forestGrowthFitting Problem
#
# The objective of this problem is to use the so-called "raw data" that the National Forest Inventory of France make available at the level of the individual inventoried trees and plots to fit a generic growth model of the forest stands in terms of volumes with respect to the age of the trees.
#

# ### Environment set-up and data loading

# Start by setting the working directory to the directory of this file and activate it. If you have the `Manifest.toml` and `Project.toml` files in the directory, run `instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots.

# Load the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots


cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate() 
using Random
Random.seed!(123)

using Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots

# Import data from remote..
ltURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_foret_2012.csv?raw=true"
dtURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_morts_foret_2012.csv?raw=true"
pointsURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/placettes_foret_2012.csv?raw=true"
docURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/documentation_2012.csv?raw=true"

lt     = @pipe HTTP.get(ltURL).body |> CSV.File(_) |> DataFrame
dt     = @pipe HTTP.get(dtURL).body |> CSV.File(_) |> DataFrame
points = @pipe HTTP.get(pointsURL).body |> CSV.File(_) |> DataFrame
doc    = @pipe HTTP.get(docURL).body |> CSV.File(_) |> DataFrame

#= # Import from local...
lt     = CSV.read("data/arbres_foret_2012.csv",DataFrame)
dt     = CSV.read("data/arbres_morts_foret_2012.csv",DataFrame)
points = CSV.read("data/placettes_foret_2012.csv",DataFrame)
doc    = CSV.read("data/documentation_2012.csv",DataFrame)
=#

lt     = lt[:,["idp","c13","v"]]
dt     = dt[:,["idp","c13","v"]]
trees  = vcat(lt,dt)
points = points[:,["idp","esspre","cac"]]

"""
    vHaContribution(volume,circonference)

Return the contribution in terms of m³/ha of the tree.

The French inventory system is based on a concentric sample method: small trees are sampled on a small area (6 metres radius), intermediate trees on a concentric area of 9 metres and only large trees (with a circonference larger than 117.5 cm) are sampled on a concentric area of 15 metres of radius.
This function normalise the contribution of each tree to m³/ha.
"""
function vHaContribution(v,c13)
    if c13 < 70.5
        return v/(6^2*pi/(100*100))
    elseif c13 < 117.5
        return v/(9^2*pi/(100*100))
    else 
        return v/(15^2*pi/(100*100))
    end
end

trees.vHa  = vHaContribution.(trees.v,trees.c13)
pointsVols = combine(groupby(trees,["idp"]) ,  "vHa" => sum => "vHa", nrow => "ntrees")
points     = innerjoin(points,pointsVols,on="idp")
filter_nTrees           = points.ntrees .> 5 # we skip points with few trees 
filter_IHaveAgeClass    = .! in.(points.cac,Ref(["AA","NR"]))
filter_IHaveMainSpecies = .! ismissing.(points.esspre) 
filter_overall          = filter_nTrees .&& filter_IHaveAgeClass .&& filter_IHaveMainSpecies
points                  = points[filter_overall,:] 
points.cac              = (parse.(Int64,points.cac) .- 1 ) .* 5 .+ 2.5

logisticModel(age,parameters) = parameters[1]/(1+exp(-parameters[2] * (age-parameters[3]) ))
logisticModelVec(age,parameters) = logisticModel.(age,Ref(parameters))

initialParameters = [1000,0.05,50] #max growth; growth rate, mid age
fitobject         = curve_fit(logisticModelVec, points.cac, points.vHa, initialParameters)
fitparams         = fitobject.param

fitobject.resid

sigma            = stderror(fitobject)
confidence_inter = confidence_interval(fitobject, 0.1) # 10% significance level

x = 0:maximum(points.cac)*1.5
plot(x->logisticModel(x,fitparams),0,maximum(x), label= "Fitted vols", legend=:topleft)
plot!(points.cac, points.vHa, seriestype=:scatter, label = "Obs vHa")


# Extra part: looking to the species individually..
speciesCount = combine(groupby(points, :esspre), nrow => :count)
sort!(speciesCount,"count",rev=true)

spLabel        = doc[doc.donnee .== "ESSPRE",:]
spLabel.spCode = parse.(Int64,spLabel.code)
speciesCount = leftjoin(speciesCount,spLabel, on="esspre" => "spCode")


# plot the 5 main species separately
for (i,sp) in enumerate(speciesCount[1:5,"esspre"])
    local fitobject, fitparams, x
    spLabel    = speciesCount[i,"libelle"]
    pointsSp   = points[points.esspre .== sp, : ]
    fitobject  = curve_fit(logisticModelVec, pointsSp.cac, pointsSp.vHa, initialParameters)
    fitparams  = fitobject.param
    x = 0:maximum(points.cac)*1.5
    println(i)
    println(sp)
    println(fitparams)
    if i == 1
        myplot = plot(x->logisticModel(x,fitparams),0,maximum(x), label= spLabel, legend=:topleft)
    else
        myplot = plot!(x->logisticModel(x,fitparams),0,maximum(x), label= spLabel, legend=:topleft)
    end
    display(myplot)
end


