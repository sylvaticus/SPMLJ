################################################################################
# forestGrowthFitting Problem
#
# How fast are French forests growing? How much timber can they provide in one hectare?
# Surely forests provide multiple ecosystem services, but in this (simplified) exercise we focus on the above questions by looking at the so-called "raw data" from the French National Forest Inventory, both in terms of individual trees and in terms of inventoried plots, to fit a generic growth model of the forest stands in terms of volumes with respect to the age of the trees.
#

# ### Environment set-up and data loading

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots.

# 2) Load the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots

# 3) Load from internet or from local files the following data:

ltURL     = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_foret_2012.csv?raw=true" # live individual trees data
dtURL     = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_morts_foret_2012.csv?raw=true" # dead individual trees data
pointsURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/placettes_foret_2012.csv?raw=true" # plot level data
docURL    = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/documentation_2012.csv?raw=true" # optional, needed for the species label

# If you choosen to download the data from internet, you can make for each of the dataset a `@pipe` macro starting with `HTTP.get(URL).body`, continuing the pipe with `CSV.File(_)` and end the pipe with a DataFrame object.

# 4) These datasets have many variable we are not using in this exercise.
# Out of all the variables, select only for the `lt` and `dt` dataframes the columns "idp" (pixel id), "c13" (circumference at 1.30 meters) and "v" (tree's volume). Then vertical concatenate the two dataset in an overall `trees` dataset.
# For the `points` dataset, select only the variables "idp" (pixel id), "esspre" (code of the main forest species in the stand) and "cac" (age class).


# 5) As the French inventory system is based on a concentric sample method (small trees are sampled on a small area (6 metres radius), intermediate trees on a concentric area of 9 metres and only large trees (with a circonference larger than 117.5 cm) are sampled on a concentric area of 15 metres of radius), define the following function to compute the contribution of each tree to the volume per hectare:

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
# Use the above function to compute `trees.vHa` based on `trees.v` and `trees.c13`.

# 6) Aggregate the `trees` dataframe by the `idp` column to retrieve the sum of `vHa` and the number of trees for each point, calling these two columns `vHa` and `ntrees`.




# 7) Join the output of step 6 (the trees dataframe aggregated "by point") with the original points dataframe using the column `idp`.

# 8) Use boolean selection to apply the following filters:
filter_nTrees           = points.ntrees .> 5
filter_IHaveAgeClass    = .! in.(points.cac,Ref(["AA","NR"]))
filter_IHaveMainSpecies = .! ismissing.(points.esspre) 
filter_overall          = filter_nTrees .&& filter_IHaveAgeClass .&& filter_IHaveMainSpecies

# 9) Run the following line to parse the age class (originally as a string indicating the 5-ages group) to an integer and compute the mid-range of the class in years. For example, class "02" will become 7.5 years.
points.cac              = (parse.(Int64,points.cac) .- 1 ) .* 5 .+ 2.5

# 10) Define the following logistic model of the growth relation with respect to the age with 3 parameters and make its vectorised form:
logisticModel(age,parameters) = parameters[1]/(1+exp(-parameters[2] * (age-parameters[3]) ))
logisticModelVec(age,parameters) = # .... complete

# 11) Set `initialParameters` to 1000,0.05 and 50 respectivelly

# 12) Perform the fittin of the model using the function `curve_fit(model,X,Y,initial parameters)` and obtain the fitted parameter fitobject.param

# 13) Compute the standard error for each estimated parameter and the confidence interval at 10% significance level

# 14) Plot a chart of fitted (y) by stand age (x) (i.e. the logisticModel with the given parameters)

# 15) Add to the plot a scatter chart of the actual observed VHa

# 16) [OPTIONAL] Look at the growth curves of individual species.
# Try to perform the above analysis for individual species, for example plot the fitted curves for the 5 most common species



