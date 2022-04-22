# EXERCISE 5.1: Predicting credit approval with a Random Forest model

```@raw html
<p>&nbsp;</p>
<img src="imgs/errorByNumberOfTrees.png" alt="Error per number of trees" style="height:250px;"> 
<img src="imgs/errorByMaxFeatures.png" alt="Error per maxFeatures" style="height:250px;"> 
<p>&nbsp;</p>
```
In this exercise we will implement a Machine Learning workflow to predict the positive or negative outcome of credit applications on the basis of the applicant characteristics. As the data comes from a real-world log from a financial institution, both fields' names and values have been replaced with meaningless symbols to preserve anonymity. 

In detail, the attributes of this dataset are:
- A1: b, a.
- A2: continuous.
- A3: continuous.
- A4: u, y, l, t.
- A5: g, p, gg.
- A6: c, d, cc, i, j, k, m, r, q, w, x, e, aa, ff.
- A7: v, h, bb, j, n, z, dd, ff, o.
- A8: continuous.
- A9: t, f.
- A10: t, f.
- A11: continuous.
- A12: t, f.
- A13: g, p, s.
- A14: continuous.
- A15: continuous.
- A16: +,- (class attribute) - what we want to predict

Further information concerning this dataset can be found online on the [UCI Machine Learning Repository dedicated page](https://archive.ics.uci.edu/ml/datasets/Credit+Approval)

Our prediction concerns the positive or negative outcome of the credit application.

While we could have used any supervised ML algorithm, it is suggested to work here with [`Random Forests`](https://sylvaticus.github.io/BetaML.jl/dev/Trees.html) from BetaML because of their ease of use and the presence of numerous categorical data and missing data that would require additional work with most other algorithms.

**Skills employed:**
- download and import data from internet
- train a Random Forest model for classification using `BetaML`
- tune the various Random Forest hyper-parameters using cross-validation
- use the additional `BetaML` functions `partition` and `accuracy`,


## Instructions

If you have already cloned or downloaded the whole [course repository](https://github.com/sylvaticus/SPMLJ/) the folder with the exercise is on `[REPOSITORY_ROOT]/lessonsMaterial/05_DT/creditApproval`.
Otherwise download a zip of just that folder [here](https://downgit.github.io/#/home?url=https://github.com/sylvaticus/SPMLJ/tree/main/lessonsMaterial/05_DT/creditApproval).

In the folder you will find the file `creditApproval.jl` containing the Julia file that **you will have to complete to implement the missing parts and run the file** (follow the instructions on that file). 
In that folder you will also find the `Manifest.toml` file. The proposal of resolution below has been tested with the environment defined by that file.  
If you are stuck and you don't want to lookup to the resolution above you can also ask for help in the forum at the bottom of this page.
Good luck! 

## Resolution

Click "ONE POSSIBLE SOLUTION" to get access to (one possible) solution for each part of the code that you are asked to implement.

--------------------------------------------------------------------------------
### 1) Setting up the environment...
Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages `Pipe`, `HTTP`, `Plots`, `CSV`, `DataFrames`, and `BetaML`.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate() 
using Random
Random.seed!(123)
```
```@raw html
</details>
```


--------------------------------------------------------------------------------
### 2) Load the packages 
Load the packages `Pipe`, `HTTP`, `Plots`, `CSV`, `DataFrames` and `BetaML`.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
using Pipe, HTTP, CSV, DataFrames, Plots, BetaML
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 3) Load the data
Load from internet or from local file the input data.
You can use a pipeline from HTTP.get() to CSV.File to finally a DataFrame.
Use the parameter `missingstring="?"` in the `CSV.File()` call.

```julia
dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"
```

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_,missingstring="?") |> DataFrame
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 4) Write the feature matrix and the the label vector
Create the `X` matrix of features using the first to the second-to-last column of the data you loaded above and the `Y` vector by taking the last column.
If you use the random forests algorithm suggested above, the only data preprocessing you need to do is to convert the X from a `DataFrame` to a `Matrix` and to `collect` the `Y` to a vector. Otherwise be sure to encode the categorical data, skip or impute the missing data and scale the feature matrix as required by the algorithm you employ.

_[...] write your code here..._

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
(nR,nD) = size(data)

describe(data)
data = data[shuffle(1:nR),:]
X    = Matrix(data[:,1:end-1])
Y    = collect(data[:,end])
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 5) Partition the data
Partition your data in (`xtrain`,`xtest`) and (`ytrain`,`ytest`)
(e.g. using 80% for the training and 20% for testing)
You can use the BetaML [`partition()`](https://sylvaticus.github.io/BetaML.jl/dev/Utils.html#BetaML.Api.partition-Union{Tuple{T},%20Tuple{AbstractVector{T},%20AbstractVector{Float64}}}%20where%20T%3C:AbstractArray) function.
Be sure to shuffle your data if you didn't do it earlier! (that's done by default)

_[...] write your code here..._

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 6) (optional but suggested) Tune the hyper-parameters
Find the best hyper-parameters for the model, i.e. the ones that lead to the highest accuracy under the records not used for training.

We can use the [`crossValidation`](https://sylvaticus.github.io/BetaML.jl/dev/Utils.html#BetaML.Utils.crossValidation) function here.

The idea is that for each hyper-parameter we have a range of possible values, and for each hyper-parameter, we first set `bestAcc=0.0` and then loop on each possible value, we run `crossValidation` with that particular value to compute the average training accuracy with that specific value under different data samples, and if it is better than the current `bestAcc`, we save it as the new `bestAcc` and the parameter value as the best value for that specific hyper-parameter.
After we have found the best hyper-parameter value for one specific hyper-parameter, we can switch to the second hyper-parameter repeating the procedure but using the best value for the first hyper-parameter that we have found earlier, and we continue with the other hyper-parameters.
Note that if we limit the hyper-parameter space sufficiently, we could also directly loop over all the possible combinations of hyper-parameters.
    
If you use the Random Forests from BetaML, consider the following hyper-parameter ranges:

```julia
nTrees_range             = 20:5:60
splittingCriterion_range = [gini,entropy]
maxDepth_range           = [10,15,20,25,30,500]
minRecords_range         = [1,2,3,4,5]
maxFeatures_range        = [2,3,4,5,6]
β_range                  = [0.0,0.5,1,2,5,10,20,50,100]
```

To train a Random Forest in BetaML use:
`myForest = buildForest(xtrain,ytrain, nTrees; <other hyper-parameters>)`.

And then to predict and compute the accuracy use:

```julia
ŷtrain=predict(myforest,xtrain)
trainAccuracy = accuracy(ŷtrain,ytrain)
```

This activity is "semi-optional", becauses Random Forests have already very good default values, so the gain we will likely obtain with tuning the various hyper-parameters is not expected to be very high. But it is a good exercise to arrive at this result by yourself !

_[...] write your code here..._

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
sampler = KFold(nSplits=10,nRepeats=2)

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
    (acc,σ)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = buildForest(xtrain, ytrain, nt)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(ŷval,collect(yval))
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
    (acc,σ)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = buildForest(xtrain, ytrain, nTrees_best, splittingCriterion=sc)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(ŷval,collect(yval))
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
    (acc,σ)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = buildForest(xtrain, ytrain, nTrees_best, splittingCriterion=splittingCriterion_best, maxDepth=md)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(ŷval,collect(yval))
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
    (acc,σ)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = buildForest(xtrain, ytrain, nTrees_best, splittingCriterion=splittingCriterion_best, maxDepth=maxDepth_best, minRecords=mr)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(ŷval,collect(yval))
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
    (acc,σ)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = buildForest(xtrain, ytrain, nTrees_best, splittingCriterion=splittingCriterion_best, maxDepth=maxDepth_best, maxFeatures=mf)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(ŷval,collect(yval))
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
    (acc,σ)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData, rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    forest          = buildForest(xtrain, ytrain, nTrees_best, splittingCriterion=splittingCriterion_best, maxDepth=maxDepth_best, maxFeatures=maxFeatures_best, β=b)
                    ŷval            = predict(forest,xval)
                    valAccuracy     = accuracy(ŷval,collect(yval))
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
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 7) Train and evaluate the final model
Perform the final training with the best hyperparameters and compute the accuracy on the test set
If you have chosen good hyperparameters, your accuracy should be in the 98%-99% range for training and 81%-89% range for testing 

_[...] write your code here..._

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
forest = buildForest(xtrain, ytrain, nTrees_best, splittingCriterion=splittingCriterion_best, maxDepth=maxDepth_best, maxFeatures=maxFeatures_best)
ŷtrain = predict(forest,xtrain)
ŷtest  = predict(forest,xtest)
trainAccuracy = accuracy(ŷtrain,ytrain)
testAccuracy  = accuracy(ŷtest,ytest)

# To compare, using the default values and a single Decision Tree:

# default values..
forest = buildForest(xtrain, ytrain)
ŷtrain = predict(forest,xtrain)
ŷtest  = predict(forest,xtest)
trainAccuracy = accuracy(ŷtrain,ytrain)
testAccuracy  = accuracy(ŷtest,ytest)

# single decision tree..
tree = buildTree(xtrain, ytrain)
ŷtrain = predict(tree,xtrain)
ŷtest  = predict(tree,xtest)
trainAccuracy = accuracy(ŷtrain,ytrain)
testAccuracy  = accuracy(ŷtest,ytest)
```
```@raw html
</details>
```
