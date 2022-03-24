################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################


# # 0402 - Neural network implementations 

# ## Some stuff to set-up the environment..

cd(@__DIR__)    
using Pkg      
Pkg.activate(".")  
# If using a Julia version different than 1.7 please uncomment and run the following line (the guarantee of reproducibility will however be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"


# We will _not_ run cross validation here to find the optimal hypermarameters. The process will not be different than those we saw in the lesson on the Perceptron. Instead we focus on creating neural network models, train them based on data and evaluationg their predictions.
# For feed-forward neural networks (both for classification and regression) we will use [BetaML](https://github.com/sylvaticus/BetaML.jl), while for Convolutional Neural Networks and Recursive Neural NEtworks we will use the [Flux.jl](https://github.com/FluxML/Flux.jl) package.

# ## Feed-forward neural networks

# ### Binary classification

# Data loading...
using BetaML, DelimitedFiles
data  = readdlm(joinpath(dirname(pathof(BetaML)),"..","test","data","binary2DData.csv"),'\t')
nR   = size(data,1)
idx  = shuffle(1:nR)
data = data[idx,:]
X    = copy(data[:,[2,3]])
y    = max.(0,convert(Array{Int64,1},copy(data[:,1]))) # Converting labels from {-1,1} to {0,1}
((xtrain,xtest),(ytrain,ytest)) = partition([X,y],[0.7,0.3])

# #### Using defaults - hidding complexity

# Model definition...
l1   = DenseLayer(2,5,f=tanh)
l2   = DenseLayer(5,5,f=relu) 
l3   = DenseLayer(5,1,f=sigmoid)
mynn = buildNetwork([l1,l2,l3],squaredCost)

# Training...
train!(mynn,xtrain,ytrain)

ŷtrain         = predict(mynn, xtrain) |> makeColVector .|> round .|> Int
ŷtest          = predict(mynn, xtest)  |> makeColVector .|> round .|> Int
trainAccuracy  = accuracy(ŷtrain,ytrain) 
testAccuracy   = accuracy(ŷtest,ytest)

# #### Specifying all options

# Model definition...
l1   = DenseLayer(2,5,f=tanh, df= dtanh,rng=copy(FIXEDRNG))
l2   = DenseLayer(5,5,f=relu,df=drelu,rng=copy(FIXEDRNG))
l3   = DenseLayer(5,1,f=sigmoid,df=dsigmoid,rng=copy(FIXEDRNG))
mynn = buildNetwork([l1,l2,l3],squaredCost,dcf=dSquaredCost,name="A classification task") # or crossEntropy / dCrossEntropy

# Training...

function myOwnTrainingInfo(nn,x,y;n,nBatches,epochs,verbosity,nEpoch,nBatch)
    if verbosity == NONE
        return false # doesn't stop the training
    end
    nMsgDict = Dict(LOW => 0, STD => 10,HIGH => 100, FULL => n)
    nMsgs = nMsgDict[verbosity]
    batchSize = size(x,1)
    if verbosity == FULL || ( nBatch == nBatches && ( nEpoch == 1  || nEpoch % ceil(epochs/nMsgs) == 0))
 
       ϵ = loss(nn,x,y)
       println("Training.. \t avg ϵ on (Epoch $nEpoch Batch $nBatch): \t $(ϵ)")
    end
    return false
 end
train!(mynn,xtrain,ytrain,epochs=300,batchSize=6,sequential=false,verbosity=STD,cb=myOwnTrainingInfo,optAlg=ADAM(η=t -> 0.001, λ=1.0, β₁=0.9, β₂=0.999, ϵ=1e-8),rng=copy(FIXEDRNG))

ŷtrain         = predict(mynn, xtrain) |> makeColVector .|> round .|> Int
ŷtest          = predict(mynn, xtest)  |> makeColVector .|> round .|> Int
trainAccuracy  = accuracy(ŷtrain,ytrain)
testAccuracy   = accuracy(ŷtest,ytest)

# ### Multinomial classification

# We want to determine the plant specie given some bothanic measures of the flower

iris     = readdlm(joinpath(dirname(Base.find_package("BetaML")),"..","test","data","iris.csv"),',',skipstart=1)
iris     = iris[shuffle(axes(iris, 1)), :] # Shuffle the records, as they aren't by default
x        = convert(Array{Float64,2}, iris[:,1:4])
y        = map(x->Dict("setosa" => 1, "versicolor" => 2, "virginica" =>3)[x],iris[:, 5]) # Convert the target column to numbers

((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.8,0.2],shuffle=false)

ytrain_oh = oneHotEncoder(ytrain) # Convert to One-hot representation (e.g. 2 => [0 1 0], 3 => [0 0 1])

# Define the Artificial Neural Network model
l1   = DenseLayer(4,10,f=relu) # Activation function is ReLU
l2   = DenseLayer(10,3)        # Activation function is identity by default
l3   = VectorFunctionLayer(3,f=softmax) # Add a (parameterless) layer whose activation function (softMax in this case) is defined to all its nodes at once
mynn = buildNetwork([l1,l2,l3],crossEntropy,name="Multinomial logistic regression Model Sepal") # Build the NN and use the squared cost (aka MSE) as error function (crossEntropy could also be used)

# Training it (default to ADAM)
res = train!(mynn,scale(xtrain),ytrain_oh,batchSize=6) # Use optAlg=SGD() to use Stochastic Gradient Descent instead

# Test it
ŷtrain        = predict(mynn,scale(xtrain))   # Note the scaling function
ŷtest         = predict(mynn,scale(xtest))
trainAccuracy = accuracy(ŷtrain,ytrain)
testAccuracy  = accuracy(ŷtest,ytest,tol=1,ignoreLabels=false)  

cm = ConfusionMatrix(ŷtest,ytest, labels=["setosa", "versicolor", "virginica"])
println(cm)

# ### Regression

# Data Loading and processing..
using Pipe, HTTP, CSV, Plots, DataFrames
urlData = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
data = @pipe HTTP.get(urlData).body |> CSV.File(_, delim='\t') |> DataFrame
sex_oh = oneHotEncoder(data.SEX) 
X = hcat(data.AGE, Matrix(data[:,3:10]),sex_oh)
y = data.Y
(xtrain,xval),(ytrain,yval) = partition([X,y],[0.8,0.2])

# Model definition...
l1   = DenseLayer(11,20,f=relu)
l2   = DenseLayer(20,20,f=relu) 
l3   = DenseLayer(20,1,f=relu) # y is positive
mynn = buildNetwork([l1,l2,l3],squaredCost)

# Training...
trainingLogs = train!(mynn,scale(xtrain),ytrain,batchSize=6,epochs=600)

ŷtrain   = predict(mynn, scale(xtrain)) |> makeColVector 
ŷval     = predict(mynn, scale(xval))  |> makeColVector
trainRME = meanRelError(ŷtrain,ytrain,normRec=false) 
testRME  = meanRelError(ŷval,yval,normRec=false)
plot(trainingLogs.ϵ_epochs[10:end])
scatter(yval,ŷval,xlabel="obs",ylabel="est",legend=nothing)

# ## Convolutional neural networks

println("Working with a convolutional neural network...")
using Flux, MLDatasets, Statistics, Plots

x_train, y_train = MLDatasets.MNIST.traindata()
x_train          = permutedims(x_train,(2,1,3)) # For correct img axis
x_train          = convert(Array{Float32,3},x_train)
x_train          = reshape(x_train,(28,28,1,60000))
y_train          = Flux.onehotbatch(y_train, 0:9)
train_data       = Flux.Data.DataLoader((x_train, y_train), batchsize=128)
x_test, y_test   = MLDatasets.MNIST.testdata()
x_test           = permutedims(x_test,(2,1,3)) # For correct img axis
x_test           = convert(Array{Float32,3},x_test)
x_test           = reshape(x_test,(28,28,1,10000))
y_test           = Flux.onehotbatch(y_test, 0:9)

myaccuracy(ŷ, y) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))
myloss(x, y)     = Flux.crossentropy(model(x), y)

model = Chain(
    # 28x28 => 14x14
    Conv((5, 5), 1=>8, pad=2, stride=2, relu),
    # 14x14 => 7x7
    Conv((3, 3), 8=>16, pad=1, stride=2, relu),
    # 7x7 => 4x4
    Conv((3, 3), 16=>32, pad=1, stride=2, relu),
    # 4x4 => 2x2
    Conv((3, 3), 32=>32, pad=1, stride=2, relu),
    # Average pooling on each width x height feature map
    GlobalMeanPool(),
    Flux.flatten,
    Dense(32, 10),
    Flux.softmax
)

opt = Flux.ADAM()
ps  = Flux.params(model)
number_epochs = 4
println("mydebug a")
Flux.@epochs number_epochs Flux.train!(myloss, ps, train_data, opt)
println("mydebug b")

ŷtrain =   model(x_train)
println("mydebug c")
ŷtest  =   model(x_test)
println("mydebug d")
myaccuracy(ŷtrain, y_train)
println("mydebug e")
myaccuracy(ŷtest, y_test)
println("mydebug f")
plot(Gray.(x_train[:,:,1,1]))
println("mydebug g")

# ## Recursive neural networks


# Generating simulated data
nSeeds    = 5
seqLength = 10
nTrains   = 1000
nTest     = 100
println("mydebug h")
nTot = nTrains+nTest
makeSeeds(nSeeds) = 2 .* (rand(nSeeds) .- 0.5) # [-1,+1]
function makeSequence(seeds,seqLength)
  seq = Vector{Float32}(undef,seqLength+nSeeds) # Flux Works with Float32 for performance reasons
  [seq[i] = seeds[i] for i in 1:nSeeds]
  for i in nSeeds:(seqLength+nSeeds)
    seq[i] = seq[i-1] + seeds[1]*0.1*seq[i-1] +seeds[2]*seeds[3]*seq[i-1]*0.4+seeds[4]*seeds[5]*(seq[i-3]-seq[i-4])
    #seq[i] = seq[i-1] + mean(seeds)
  end
  return seq
  return seq[nSeeds+1:end]
end

seq=makeSequence(makeSeeds(nSeeds),seqLength)
plot(seq)

x0   = [makeSeeds(nSeeds) for i in 1:nTot]
seqs = makeSequence.(x0,seqLength)
seqs_vectors = [[[e] for e in seq] for seq in seqs]
seqs_vectors[1][1]
y    = seqs_vectors # y here is the value of the sequence itself
m    = Chain(Dense(1,5,σ),LSTM(5, 5), Dense(5, 1))
#σ
function myloss(x, y)
    Flux.reset!(m)               # Reset the state (not the weigtht!)
    #[m(x[i]) for i in 1:nSeeds]  # Ignores the output but updates the hidden states
    sum(Flux.mse(m(xi), yi) for (xi, yi) in zip(x[1:end], y[1:end]))
end

ps  = params(m)
opt = ADAM()
println("mydebug i")
trainxy = zip(seqs_vectors,seqs_vectors)
println("mydebug l")
# Actual training
#=
Flux.train!(myloss, ps, trainxy, opt)


function predictSequence(m,seeds,seqLength)
    seq = Vector{Vector{Float32}}(undef,seqLength+length(seeds))
    Flux.reset!(m) # Reset the state (not the weigtht!)
    [seq[i] = m([convert(Float32, seeds[i])]) for i in 1:nSeeds]
    [seq[i] = m(seq[i-1]) for i in nSeeds+1:nSeeds+seqLength]
    [s[1] for s in seq]
end 

a = predictSequence(m,x0[1],seqLength)

i = 2
trueseq = makeSequence(x0[i],seqLength)
estseq  = predictSequence(m,x0[i],seqLength)

plot(trueseq[nSeeds+1:end])
plot!(estseq[nSeeds+1:end])
=#