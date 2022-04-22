# # Predicting credit approval by applicants' characteristics

# ### Instructions

# In this exercise we will predict the approval (yes/no) for credit applications from the applicant characteristics.

# As the data comes from a real-world log from a financial institution, both fields' names and values have been replaced with meaningless symbols to preserve anonymity. 

# In detail, the attributes of this dataset are:
# - A1: b, a.
# - A2: continuous.
# - A3: continuous.
# - A4: u, y, l, t.
# - A5: g, p, gg.
# - A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
# - A7: v, h, bb, j, n, z, dd, ff, o.
# - A8: continuous.
# - A9: t, f.
# - A10: t, f.
# - A11: continuous.
# - A12: t, f.
# - A13: g, p, s.
# - A14: continuous.
# - A15: continuous.
# - A16: +,- (class attribute) - what we want to predict

# Further information concerning this dataset can be found online on the [UCI Machine Learning Repository dedicated page](https://archive.ics.uci.edu/ml/datasets/Credit+Approval)

# Our prediction concerns the positive or negative outcome of the credit application.

# While you can use any supervised ML algorithm, I suggest the [`Random Forests`](https://sylvaticus.github.io/BetaML.jl/dev/Trees.html) from BetaML because of their ease of use and the presence of numerous categorical data and missing data that would require additional work with most other algorithms.

# ------------------------------------------------------------------------------
# ### 1) Start by setting the working directory to the directory of this file and activate it.
# If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, Plots and BetaML
# Also, seed the random seed with the integer `123`.
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)

# ------------------------------------------------------------------------------
# ### 2) Load the packages/modules Pipe, HTTP, CSV, DataFrames, Plots, BetaML
using  Pipe, HTTP, CSV, DataFrames, Plots, BetaML


# ### 3) Load from internet or from local file the input data.
# You can use a pipeline from HTTP.get() to CSV.File to finally a DataFrame.
# Use the parameter `missingstring="?"` in the `CSV.File()` call.

dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

# [...] write your code here...

# ------------------------------------------------------------------------------
# ### 4) Now create the X matrix and Y vector
# Create the X matrix of features using the first to the second-to-last column of the data you loaded above and the Y vector by taking the last column.
# If you use the random forests algorithm suggested above, the only data preprocessing you need to do is to convert the X from a DataFrame to a Matrix and to `collect` the Y to a vector. Otherwise be sure to encode the categorical data, skip or impute the missing data and scale the feature matrix as required by the algorithm you employ.

# [...] write your code here...

# ------------------------------------------------------------------------------
# ### 5) Partition your data in (xtrain,xtest) and (ytrain,ytest)
# (e.g. using 80% for the training and 20% for testing)
# You can use the BetaML [`partition()`](https://sylvaticus.github.io/BetaML.jl/dev/Utils.html#BetaML.Api.partition-Union{Tuple{T},%20Tuple{AbstractVector{T},%20AbstractVector{Float64}}}%20where%20T%3C:AbstractArray) function.
# Be sure to shuffle your data if you didn't do it earlier! (that's done by default)

# [...] write your code here...

# ------------------------------------------------------------------------------
# ### 6) (optional but suggested) Find the best hyper-parameters for your model, i.e. the ones that lead to the highest accuracy under the records not used for training.
# You can use the [`crossValidation`](https://sylvaticus.github.io/BetaML.jl/dev/Utils.html#BetaML.Utils.crossValidation) function here.
# The idea is that for each hyper-parameter you have a range of possible values, and for each hyper-parameter, you first set `bestAcc=0.0` and then loop on each possible value, you run crossValidation with that particular value to compute the average training accuracy with that specific value under different data samples, and if it is better than the current `bestAcc` you save it as the new `bestAcc` and the parameter value as the best value for that specific hyper-parameter.
# After you have found the best hyper-parameter value for one specific hyper-parameter, you can switch to the second hyper-parameter repeating the procedure but using the best value for the first hyper-parameter that you found earlier, and you continue with the other hyper-parameters.
# Note that if you limit the hyper-parameter space sufficiently you could also directly loop over all the possible combinations of hyper-parameters.
    
# If you use the Random Forests from BetaML consider the following hyper-parameter ranges:

nTrees_range             = 20:5:60
splittingCriterion_range = [gini,entropy]
maxDepth_range           = [10,15,20,25,30,500]
minRecords_range         = [1,2,3,4,5]
maxFeatures_range        = [2,3,4,5,6]
β_range                  = [0.0,0.5,1,2,5,10,20,50,100]

# To train a Random Forest in BetaML use:
# `myForest = buildForest(xtrain,ytrain, nTrees; <other hyper-parameters>)`
# And then to predict and compute the accuracy use:
# ```julia
# ŷtrain=predict(myforest,xtrain)
# trainAccuracy = accuracy(ŷtrain,ytrain)
# ```
# This activity is "semi-optional" as Random Forests have very good default values, so the gain you will likely obtain with tuning the various hyper-parameters is not expected to be very high.  But it is a good exercise to arrive at this result by yourself !

# [...] write your code here...

# ------------------------------------------------------------------------------
# ### 7) Perform the final training with the best hyperparameters and compute the accuracy on the test set
# If you have chosen good hyperparameters, your accuracy should be in the 98%-99% range for training and 81%-89% range for testing 

# [...] write your code here...
