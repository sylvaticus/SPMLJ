################################################################################
# forestGrowthFitting Problem
#
# The objective of this problem is to use the so-called "raw data" that the National Forest Inventory of France make availmable at the level of the individual inventoried trees and plots to fit a geric growth model of the forest stands in terms of volumes with respect to the trees age.
#

# ### Environment set-up and data loading

# Start by setting the working directory to the directory of this file and activate it. If you have the `Manifest.toml` and `ProjecT.toml` files in the directory, run `instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots.

# Load the packages Pipe, HTTP, CSV, DataFrames, LsqFit, StatsPlots

# Load from internet or from local files the following data:

ltURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_foret_2012.csv?raw=true" # live individual trees data
dtURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/arbres_morts_foret_2012.csv?raw=true" # dead individual trees data
pointsURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/placettes_foret_2012.csv?raw=true" # plot level data
docURL = "https://github.com/sylvaticus/IntroSPMLJuliaCourse/blob/main/lessonsMaterial/02_JULIA2/forestExercise/data/documentation_2012.csv?raw=true" # optional, needed for the species label

# If you go for downloading the data from internet, make for each of them a `@pipe` macro from `HTTP.get(URL).body` to `CSV.File(_)` to dfinally DataFrame

# Out of all the variables in these dataset, select only for the `lt` and `dt` dataframes the columns "idp" (pixel id),"c13" (circumference ad 1.30 meters) and "v" (tree's volume). Vertical concatenate the two dataset in a overall `trees` dataset.
# For the `points` dataset, select only the variables "idp" (pixel id), "esspre" (code of the main forest species in the stand) and "cac" (age class)?


# Define the following function to compute the contribution of each tree to the volume per hectar measure of the plot

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

# Use the above function to compute `trees.vHa`
# Aggregate the `trees` dataframe by the `idp` column to retrieve the sum of `vHa` and the number of trees for each point, callign these two columns `vHa` and `ntrees`.
# Join this aggregated "by point" dataframe with the original points dataframe using the column `idp`.

# Use boolean selection to apply the following filters:
filter_nTrees           = points.ntrees .> 5
filter_IHaveAgeClass    = .! in.(points.cac,Ref(["AA","NR"]))
filter_IHaveMainSpecies = .! ismissing.(points.esspre) 
filter_overall          = filter_nTrees .&& filter_IHaveAgeClass .&& filter_IHaveMainSpecies

# Run the followinf line to parse the age class (originally as a string indicating the 5-ages group) as an integer and computing the mid-range of the class in years. For example, class "02" will become 7.5 anni.

points.cac              = (parse.(Int64,points.cac) .- 1 ) .* 5 .+ 2.5

# Define the following logistic model of the growth relation with respect to the age with 3 parameters and make its vectorised form:
logisticModel(age,parameters) = parameters[1]/(1+exp(-parameters[2] * (age-parameters[3]) ))

logisticModelVec(age,parameters) = # .... complete

# Set `initialParameters` to 1000,0.05 and 50 respectivelly

# Perform the fittin of the model using the function `curve_fit(model,X,Y,initial parameters)` and obtain the fitted parameter fitobject.param

# Compute the standard error for each estimated parameter and the confidence interval at 10% significance level

# Plot a chart of fitted (y) by stand age (X) (i.e. the logisticModel with the given parameters):

# Add to the plot a scatter chart of the actual observed VHa:


# ### Optional part: looking to the species individually..
# Try to look and plot the 5 most common species and display their specific plot



