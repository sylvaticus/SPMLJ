# # Predicting credit approval by applicants' characteristics

# ### Instructions

# In this exercise we will try to predict the approval (yes/no) for credit applications from the applicant characteristics.

# Note that both fields' names and values have been replaced with meaningless symbols to preserve anonymity. 

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
# If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)

# ------------------------------------------------------------------------------
# ### 2) Load the packages/modules Pipe, HTTP, CSV, DataFrames, Plots, BetaML
using Pipe, HTTP, CSV, DataFrames, Plots, BetaML


# ### 3) Load from internet or from local file the input data.
# You can use a pipeline from HTTP.get() to CSV.File to finally a DataFrame.
# Use the parameter `missingstring="?"` in the `CSV.File()` call.
dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_,missingstring="?") |> DataFrame

# ------------------------------------------------------------------------------
# ### 4) Now create the X matrix and Y vector
# Create the X matrix of features using the first to the second-to-last column of the data you loaded above and the Y vector by taking the last column.
# If you use the random forests algorithm suggested above, the only data preprocessing you need to do is to convert the X from a DataFrame to a Matrix and to `collect` the Y to a vector. Otherwise be sure to encode the categorical data, skip or impute the missing data and scale the feature matrix as required by the algorithm you employ.

(nR,nD) = size(data)

describe(data)
data = data[shuffle(1:nR),:]
X    = Matrix(data[:,1:end-1])
Y    = convert(Vector{String},collect(data[:,end]))

# ------------------------------------------------------------------------------
# ### 5) Partition your data in (xtrain,xtest) and (ytrain,ytest)
# (e.g. using 80% for the training and 20% for testing)
# You can use the BetaML [`partition()`](https://sylvaticus.github.io/BetaML.jl/dev/Utils.html#BetaML.Api.partition-Union{Tuple{T},%20Tuple{AbstractVector{T},%20AbstractVector{Float64}}}%20where%20T%3C:AbstractArray) function.
# Be sure to shuffle your data if you didn't do it earlier! (that's done by default)

((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])

# ------------------------------------------------------------------------------
# ### 6) (optional but suggested) Find the best hyper-parameters for your model, i.e. the ones that lead to the highest accuracy under the records not used for training.
# You can use the [`cross_validation`](https://sylvaticus.github.io/BetaML.jl/stable/Utils.html#BetaML.Utils.cross_validation) function here.
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
# `myForest = RandomForestEstimator([hyper-parameters])
# `fit!(myForest,xtrain,ytrain)`
# And then to predict and compute the accuracy use:
# ```julia
# ŷtrain        = predict(myForest,xtrain)
# trainAccuracy = accuracy(ytrain,ŷtrain)
# ```

# This activity is "semi-optional" as Random Forests have very good default values, so the gain you will likely obtain with tuning the various hyper-parameters is not expected to be very high.  But it is a good exercise to arrive at this result by yourself !

# Alternatively, since BetaML v0.8, the best model hyperparameters can be automatically selected using the model option `autotune`, where the hyperparapeters ranges to test can be specified in `tunemethod` :

forest = RandomForestEstimator(autotune=true,tunemethod=SuccessiveHalvingSearch(
    hpranges=Dict(
      "n_trees"             => collect(nTrees_range),
      "splitting_criterion" => collect(splittingCriterion_range),
      "max_depth"           => maxDepth_range,
      "min_records"         => minRecords_range,
      "max_features"        => maxFeatures_range,
      "beta"                => β_range   
    )))

# !!! warning
#     Training with the above autotune may take a few hours on a pc
fit!(forest,xtrain,ytrain)

sampler = KFold(nsplits=10,nrepeats=2)

nTrees_best             = 20
splittingCriterion_best = "gini"
maxDepth_best           = 10
minRecords_best         = 2
maxFeatures_best        = 3
β_best                  = 0.0

# Looking for hyper-parameters one at a time

# #### Number of trees
bestAcc = 0.0
accuracies = []
for nt in nTrees_range
    global nTrees_best, bestAcc, accuracies
    local acc
    print("Accuracy for $nt nTrees: ")
    (acc,σ)    = cross_validation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = RandomForestEstimator(n_trees=nt)
                    fit!(forest,xtrain,ytrain)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(collect(yval),ŷval)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc = acc
        nTrees_best = nt
    end
    push!(accuracies,acc)
    println("$acc (σ: $σ)")
end
plot(nTrees_range,accuracies,legend=nothing,ylabel="accuracy",xlabel="nTrees")


# #### Splitting criterion
bestAcc = 0.0
accuracies = []
for sc in splittingCriterion_range
    global splittingCriterion_best, bestAcc, accuracies
    local acc
    print("Accuracy for $sc splittingCriterion: ")
    (acc,σ)    = cross_validation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = RandomForestEstimator(splitting_criterion=sc)
                    fit!(forest,xtrain,ytrain)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(yval,ŷval)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc = acc
        splittingCriterion_best = sc
    end
    push!(accuracies,acc)
    println("$acc (σ: $σ)")
end
bar(string.(splittingCriterion_range),accuracies,legend=nothing,ylabel="accuracy",xlabel="splititngCriterion")

# #### Max (tree) depth
bestAcc = 0.0
accuracies = []
for md in maxDepth_range
    global maxDepth_best, bestAcc, accuracies
    local acc
    print("Accuracy for $md maxDepth: ")
    (acc,σ)    = cross_validation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = RandomForestEstimator(max_depth=md)
                    fit!(forest,xtrain,ytrain)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(collect(yval),ŷval)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc = acc
        maxDepth_best = md
    end
    push!(accuracies,acc)
    println("$acc (σ: $σ)")
end
plot(maxDepth_range,accuracies,legend=nothing,ylabel="accuracy",xlabel="maxDepth")
plot(maxDepth_range[1:end-1],accuracies[1:end-1],legend=nothing,ylabel="accuracy",xlabel="maxDepth")


# #### Min records per leaf
bestAcc = 0.0
accuracies = []
for mr in minRecords_range
    global minRecords_best, bestAcc, accuracies
    local acc
    print("Accuracy for $mr minRecords: ")
    (acc,σ)    = cross_validation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = RandomForestEstimator(min_records=mr)
                    fit!(forest,xtrain,ytrain)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(collect(yval),ŷval)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc = acc
        minRecords_best = mr
    end
    push!(accuracies,acc)
    println("$acc (σ: $σ)")
end
plot(minRecords_range,accuracies,legend=nothing,ylabel="accuracy",xlabel="minRecords")

# #### Max features to consider in a tree
bestAcc = 0.0
accuracies = []
for mf in maxFeatures_range
    global mmaxFeatures_best, bestAcc, accuracies
    local acc
    print("Accuracy for $mf maxFeatures: ")
    (acc,σ)    = cross_validation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = RandomForestEstimator(max_features=mf)
                    fit!(forest,xtrain,ytrain)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(collect(yval),ŷval)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc = acc
        maxFeatures_best = mf
    end
    push!(accuracies,acc)
    println("$acc (σ: $σ)")
end
plot(maxFeatures_range,accuracies,legend=nothing,ylabel="accuracy",xlabel="maxFeatures")

# #### Weigth for best trees representation
bestAcc = 0.0
accuracies = []
for b in β_range
    global β_best, bestAcc, accuracies
    local acc
    print("Accuracy for $b β: ")
    (acc,σ)    = cross_validation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = RandomForestEstimator(beta=b)
                    fit!(forest,xtrain,ytrain)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(collect(yval),ŷval)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc = acc
        β_best = b
    end
    push!(accuracies,acc)
    println("$acc (σ: $σ)")
end
plot(β_range,accuracies,legend=nothing,ylabel="accuracy",xlabel="β")



# ------------------------------------------------------------------------------
# ### 7) Perform the final training with the best hyperparameters and compute the accuracy on the test set
# If you have chosen good hyperparameters, your accuracy should be in the 98%-99% range for training and 81%-89% range for testing (and up to 91% if you used autotune)

forest = RandomForestEstimator(
      n_trees             = nTrees_best,
      splitting_criterion = splittingCriterion_best,
      max_depth           = maxDepth_best,
      min_records         = minRecords_best,
      max_features        = maxFeatures_best,
      beta                = β_best) # skip if autotune has been used
fit!(forest,xtrain,ytrain)          # skip if autotune has been used
ŷtrain        = predict(forest,xtrain)
ŷtest         = predict(forest,xtest)
trainAccuracy = accuracy(ytrain,ŷtrain)
testAccuracy  = accuracy(ytest,ŷtest)

# To compare, using the default values and a single Decision Tree:

# default values..
forest        = RandomForestEstimator()
fit!(forest,xtrain, ytrain)
ŷtrain        = predict(forest,xtrain)
ŷtest         = predict(forest,xtest)
trainAccuracy = accuracy(ytrain,ŷtrain)
testAccuracy  = accuracy(ytest,ŷtest)

# single decision tree..
tree          = DecisionTreeEstimator()
fit!(tree,xtrain, ytrain)
ŷtrain        = predict(tree,xtrain)
ŷtest         = predict(tree,xtest)
trainAccuracy = accuracy(ytrain,ŷtrain)
testAccuracy  = accuracy(ytest,ŷtest)
