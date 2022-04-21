################################################################################
# Breast Cancer diagnosis problem
#
# In this problem we are given a dataset containing real world characteristics of observed breast cancer (size, pattern,..) together with the associated diagnosis in terms of malignity or benignity of the cancer.
# Our task is to build a linear classifier using the perceptron algorithm that we studied and train it in order to make diagnosis based on the cancer characteristics.

# ### Environment set-up and data loading

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, StatsPlots and BetaML.
# Also, seed the random number generator with the integer `123`.



# 2) Load the packages/modules Statistics, DelimitedFiles, LinearAlgebra, Pipe, HTTP, StatsPlots, BetaML



# 3) Load from internet or from localfile the input data and shuffle its rows (records)

dataURL = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/03_ML1/BreastCancerDiagnosisWithPerceptron/data/wdbc.data.csv"

# Source: [Breast Cancer Wisconsin (Diagnostic) Data Set, UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)


# 4) Map the data to (X,y)
# The data you have loaded contains the actual diagnosis for the cancer in the second column, coded with a string "B" for "Benign" and "M" for "Malign", and the characteristics of the cancer foir the next 30 columns.
# Save the diagnosis to the vector `y`, coding malign cancers with `1` and benign cancers with `-1`
# Save the characteristics to the feature matrix `X` (and be sure it is made of Float64)


# 5) (this task is provided) Plot the first 2 attributes of training points and define the function plot2DClassifierWithData()
colors = [y == 1 ? "red" : "green" for y in y]
labels = [y == 1 ? "malign" : "benign" for y in y]
scatter(X[:,1],X[:,2], colour=colors, title="Classified tumors",xlabel="Tumor Radius", ylabel="Tumor Texture", group=labels)


function plot2DClassifierWithData(X,y,θ;d1=1,d2=2,origin=false,xlabel="Dimx: $(d1)",ylabel="Dimy: $(d2)")
    nR    = size(X,1)
    X     = hcat(ones(nR),X)
    X     = scale(X) # for visualisation
    d1    += 1
    d2    += 1
    colors = [y == 1 ? "red" : "green" for y in y]
    labels = [y == 1 ? "malign" : "benign" for y in y]
    minD1,maxD1 = extrema(X[:,d1])
    minD2,maxD2 = extrema(X[:,d2])
    myplot = scatter(X[:,d1],X[:,d2], colour=colors, title="Linear classifier in 2D",xlabel=xlabel, ylabel=ylabel, group=labels)
    d2Class(x) = -θ[1]/θ[d2] -x * θ[d1]/θ[d2]
    if θ[d2] == 0
        vline!([0], color= "blue",label="",linewidth=5)
    else
        plot!(d2Class,minD1,maxD1, color= "blue",label="",linewidth=5)
    end
    display(myplot)
end

# 6) (provided) Define the Model and training options structures

abstract type SupervisedModel end
abstract type TrainingOptions end

mutable struct Perceptron <: SupervisedModel
    θ::Vector{Float64}
end

mutable struct PerceptronTrainingOptions <: TrainingOptions
    epochs::Int64
    verbose::Bool
    shuffle::Bool
    function PerceptronTrainingOptions(;epochs=1,verbose=false,shuffle=false)
        return new(epochs,verbose,shuffle)
    end
end

# 7) (provided) Implement the functions `predict()` and `update()`
function predict(model::Perceptron,x::AbstractVector)
    x = vcat(1.0,x)
    x' * model.θ > eps() ? (return 1) : (return -1)
end

function predict(model::Perceptron,X::AbstractMatrix)
    return [predict(model,r) for r in eachrow(X)]
end

function update!(model::Perceptron,X::Vector,y)
    X       = vcat(1.0,X)
    model.θ = model.θ .+ y .* X
    return model.θ
end

# 8) Implement the function `train!(model::Perceptron,X,y,ops=PerceptronTrainingOptions()::TrainingOptions)`
# Compared to the function we saw in the 0302-perceptron.jl file, add, if you wish, a counter to eventually return early if there are no more errors in an epoch (i.e., all points are correctly classified)
function train!(model::Perceptron,X,y,ops=PerceptronTrainingOptions()::TrainingOptions)
    #...
    for t in 1:epochs
        #...
        if ops.shuffle   
           #...
        end
        for n in 1:nR
           #...
        end
        #...
    end
    #...
    return model.θ
end

# 9) Instanziate a `Perceptron` object with a parameter vector of nD+1 zeros and a `PerceptronTrainingOption` object with 5 epochs and shuffling, use the options to train the model on the whole dataset, compute the model predictions and the accuracy relative to the whole sample.

m   = Perceptron(zeros(size(X,2)+1))
ops = #...
train!(m,X,y,ops)
plot2DClassifierWithData(X,y,m.θ,d1=1,d2=2,xlabel="Tumor Radius", ylabel="Tumor Texture")
ŷ           = #...
inSampleAcc = accuracy(#= ... =#) # 0.91

# 10) Partition the data in `(xtrain,xtest)` and `(ytrain,ytest)` keeping 65% of the data for training and reserving 35% for testing

((xtrain,xtest),(ytrain,ytest)) = partition(#=...=#)

# 11) Using a 10-folds cross-validation strategy, find the best hyperparameters within the following ranges :

sampler = KFold(#=...=#)

epochsSet  = 1:5:150
shuffleSet = [false,true]

bestE       = 0
bestShuffle = false
bestAcc     = 0.0

accuraciesNonShuffle = []
accuraciesShuffle    = []

for e in epochsSet, s in shuffleSet
    global bestE, bestShuffle, bestAcc, accuraciesNonShuffle, accuraciesShuffle
    local acc
    local ops  = PerceptronTrainingOptions(#=...=#)
    (acc,_)    = crossValidation([xtrain,ytrain],sampler) do trainData,valData,rng
                    (xtrain,ytrain) = trainData; (xval,yval) = valData
                    m               = Perceptron(zeros(size(xtrain,2)+1))
                    train!(#=...=#)
                    ŷval            = predict(#=...=#)
                    valAccuracy     = accuracy(#=...=#)
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc     = acc
        bestE       = e
        bestShuffle = s
    end
    if s
        push!(accuraciesShuffle,acc)
    else
        push!(accuraciesNonShuffle,acc)
    end
end

bestAcc # 0.91
bestE
bestShuffle

plot(epochsSet,accuraciesNonShuffle,label="Val accuracy without shuffling", legend=:bottomright)
plot!(epochsSet,accuraciesShuffle, label="Val accuracy with shuffling")

# 12) Using the "best" hyperparameters found in the previous step, instantiate a new model and options, train the model using `(xtrain,ytrain)`, make your predicitons for the testing features (`xtest`) and compute your output accuracy compared with those of the true `ytest` (use the BetaML function `accuracy`)

ops = PerceptronTrainingOptions(#=...=#)
m   = Perceptron(zeros(size(xtest,2)+1))
train!(#=...=#)
ŷtest           = predict(#=...=#)
testAccuracy    = accuracy(#=...=#) # 0.89

plot2DClassifierWithData(xtest,ytest,m.θ,xlabel="Tumor Radius", ylabel="Tumor Texture")
plot2DClassifierWithData(xtest,ytest,m.θ,d1=3,d2=4)

# 13) (optional) Add a scaling passage to the workflow and test it with cross-validation if it improves the accuracy

epochsSet  = 1:10:150
shuffleSet = [false,true]
scalingSet = [false,true]

bestE       = 0
bestShuffle = false
bestScaling = false
bestAcc     = 0.0

accNShuffleNSc = Float64[]
accNShuffleSc  = Float64[]
accShuffleNSc  = Float64[]
accShuffleSc   = Float64[]


scfactors = getScaleFactors(xtrain)

for e in epochsSet, s in shuffleSet, sc in scalingSet
    global bestE, bestShuffle, bestAcc, accNShuffleNSc, accNShuffleSc, accShuffleNSc, accShuffleSc
    local acc
    local ops  = PerceptronTrainingOptions(#=...=#)
    xtrainsc= copy(xtrain)
    if(sc)
        xtrainsc =scale(xtrain,scfactors)
    end
    (acc,_)    = crossValidation([xtrainsc,ytrain],sampler) do trainData,valData,rng
                    #...
                    return valAccuracy
                end
    if acc > bestAcc
        bestAcc     = acc
        bestE       = e
        bestShuffle = s
        bestScaling = sc
    end
    if s && sc
        push!(accShuffleSc,acc)
    elseif  s && !sc
        push!(accShuffleNSc,acc)
    elseif  !s && sc
        push!(accNShuffleSc,acc)
    elseif  !s && !sc
        push!(accNShuffleNSc,acc)
    else
        @error "Something wrong here"
    end
end

bestAcc # 0.97
bestE
bestShuffle
bestScaling

plot(epochsSet,accShuffleSc,label="Val accuracy shuffling scaling", legend=:bottomright)
plot!(epochsSet,accNShuffleSc,label="Val accuracy NON shuffling scaling", legend=:bottomright)
plot!(epochsSet,accShuffleNSc,label="Val accuracy shuffling NON scaling", legend=:bottomright)
plot!(epochsSet,accNShuffleNSc,label="Val accuracy NON shuffling NON scaling", legend=:bottomright)

ops = PerceptronTrainingOptions(#=...=#)
m   = Perceptron(#=...=#)
if bestScaling
    train!(m,scale(xtrain),ytrain,ops)
    ŷtest  = predict(m,scale(xtest))
else
    train!(m,xtrain,ytrain,ops)
    ŷtest  = predict(m,xtest)
end
testAccuracy    = accuracy(#=...=#) # 0.96

plot2DClassifierWithData(xtest,ytest,m.θ,xlabel="Tumor Radius", ylabel="Tumor Texture")