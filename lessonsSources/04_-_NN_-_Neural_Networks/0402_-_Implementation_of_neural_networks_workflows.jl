################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################


# # 0402 - Implementation of Neural network workflows

# ## Some stuff to set-up the environment..

cd(@__DIR__)    
using Pkg      
Pkg.activate(".")  
## If using a Julia version different than 1.10 please uncomment and run the following line (the guarantee of reproducibility will however be lost)
## Pkg.resolve()   
Pkg.instantiate()
using Random, Plots
Random.seed!(123)
ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"


# We will _not_ run cross validation here to find the optimal hyper-parameters. The process will not be different than those we saw in the lesson on the Perceptron. Instead we focus on creating neural network models, train them based on data and evaluating their predictions.
# For feed-forward neural networks (both for classification and regression) we will use [BetaML](https://github.com/sylvaticus/BetaML.jl), while for Convolutional Neural Networks example we will use the [Flux.jl](https://github.com/FluxML/Flux.jl) package.

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
mynn = NeuralNetworkEstimator()

# Training...
fit!(mynn,xtrain,ytrain)

ŷtrain         = predict(mynn, xtrain) |> makecolvector .|> round .|> Int
ŷtest          = predict(mynn, xtest)  |> makecolvector .|> round .|> Int
trainAccuracy  = accuracy(ytrain,ŷtrain) 
testAccuracy   = accuracy(ytest,ŷtest)

# #### Specifying all options

# Creating a custim callback function to receive info during training...
function myOwnTrainingInfo(nn,xbatch,ybatch,x,y;n,n_batches,epochs,epochs_ran,verbosity,n_epoch,n_batch)
    if verbosity == NONE
        return false # doesn't stop the training
    end

    nMsgDict = Dict(LOW => 0, STD => 10,HIGH => 100, FULL => n)
    nMsgs = nMsgDict[verbosity]

    if verbosity == FULL || ( n_batch == n_batches && ( n_epoch == 1  || n_epoch % ceil(epochs/nMsgs) == 0))

        ϵ = BetaML.Nn.loss(nn,x,y)
        println("Training.. \t avg loss on epoch $n_epoch ($(n_epoch+epochs_ran)): \t $(ϵ)")
    end
    return false
end

# Model definition...
l1   = DenseLayer(2,5,f=tanh, df= dtanh,rng=copy(FIXEDRNG))
l2   = DenseLayer(5,5,f=relu,df=drelu,rng=copy(FIXEDRNG))
l3   = DenseLayer(5,1,f=sigmoid,df=dsigmoid,rng=copy(FIXEDRNG))
mynn = NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost,dloss=dsquared_cost,descr="A classification task", cb=myOwnTrainingInfo,epochs=300,batch_size=6,opt_alg=ADAM(η=t -> 0.001, λ=1.0, β₁=0.9, β₂=0.999, ϵ=1e-8),rng=copy(FIXEDRNG),verbosity=STD)

# Training...
fit!(mynn,xtrain,ytrain)

ŷtrain         = predict(mynn, xtrain) |> makecolvector .|> round .|> Int
ŷtest          = predict(mynn, xtest)  |> makecolvector .|> round .|> Int
trainAccuracy  = accuracy(ŷtrain,ytrain)
testAccuracy   = accuracy(ŷtest,ytest)

# ### Multinomial classification

# We want to determine the plant specie given some bothanic measures of the flower
iris     = readdlm(joinpath(dirname(Base.find_package("BetaML")),"..","test","data","iris.csv"),',',skipstart=1)
iris     = iris[shuffle(axes(iris, 1)), :] # Shuffle the records, as they aren't by default
x        = convert(Array{Float64,2}, iris[:,1:4])
#y       = map(x->Dict("setosa" => 1, "versicolor" => 2, "virginica" =>3)[x],iris[:, 5]) # Convert the target column to numbers
ystring  = String.(iris[:, 5])
iemod  = OrdinalEncoder() 
y      = fit!(iemod,ystring) 

((xtrain,xtest),(ytrain,ytest)) = partition([x,y],[0.8,0.2],shuffle=false)

ohmod = OneHotEncoder()
ytrain_oh = fit!(ohmod,ytrain) # Convert to One-hot representation (e.g. 2 => [0 1 0], 3 => [0 0 1])

# Define the Artificial Neural Network model
l1   = DenseLayer(4,10,f=relu) # Activation function is ReLU
l2   = DenseLayer(10,3)        # Activation function is identity by default
l3   = VectorFunctionLayer(3,f=softmax) # Add a (parameterless) layer whose activation function (softMax in this case) is defined to all its nodes at once
mynn = NeuralNetworkEstimator(layers=[l1,l2,l3],loss=crossentropy,batch_size=6,descr="Multinomial logistic regression Model Sepal") # Build the NN and use the squared cost (aka MSE) as error function (crossEntropy could also be used)

# Training it (default to ADAM)
fit!(mynn,fit!(Scaler(),xtrain),ytrain_oh) # Use optAlg=SGD() to use Stochastic Gradient Descent instead

# Test it
ŷtrain        = predict(mynn,fit!(Scaler(),xtrain))   # Note the scaling model
ŷtest         = predict(mynn,fit!(Scaler(),xtest)) 
trainAccuracy = accuracy(ytrain,ŷtrain)
testAccuracy  = accuracy(ytest,ŷtest,tol=1,ignorelabels=false)  

cm = ConfusionMatrix()
fit!(cm,inverse_predict(iemod,ytrain),inverse_predict(iemod,mode(ŷtrain))) 
println(cm)

res = info(cm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")
savefig("cm_iris.svg");

# ![](cm_iris.svg)

# ### Regression

# Data Loading and processing..
using Pipe, HTTP, CSV, DataFrames
urlData = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt"
data = @pipe HTTP.get(urlData).body |> CSV.File(_, delim='\t') |> DataFrame
ohmod = OneHotEncoder()
sex_oh = fit!(ohmod,data.SEX) 
X = hcat(data.AGE, Matrix(data[:,3:10]),sex_oh)
y = data.Y
(xtrain,xval),(ytrain,yval) = partition([X,y],[0.8,0.2])

# Model definition...
l1   = DenseLayer(11,20,f=relu)
l2   = DenseLayer(20,20,f=relu)  
l3   = DenseLayer(20,1,f=relu) # y is positive
mynn = NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost, batch_size=6,epochs=600)

# Training...
fit!(mynn,fit!(Scaler(),xtrain),ytrain)

ŷtrain   = predict(mynn, fit!(Scaler(),xtrain))
ŷval     = predict(mynn, fit!(Scaler(),xval))
trainRME = relative_mean_error(ytrain,ŷtrain)
testRME  = relative_mean_error(yval,ŷval)

plot(info(mynn)["loss_per_epoch"][10:end])
savefig("loss_per_epoch.svg");

# ![](loss_per_epoch.svg)

scatter(yval,ŷval,xlabel="obs",ylabel="est",legend=nothing)
savefig("obs_vs_est.svg");

# ![](obs_vs_est.svg)

# ## Convolutional neural networks

using LinearAlgebra, Statistics,Flux, MLDatasets, Plots

x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_train          = permutedims(x_train,(2,1,3)); # For correct img axis
#x_train          = convert(Array{Float32,3},x_train);
x_train          = reshape(x_train,(28,28,1,60000));
y_train          = Flux.onehotbatch(y_train, 0:9)
train_data       = Flux.Data.DataLoader((x_train, y_train), batchsize=128)
#x_test, y_test   = MLDatasets.MNIST.testdata(dir = "data/MNIST")
x_test, y_test   = MLDatasets.MNIST(split=:test)[:]
x_test           = permutedims(x_test,(2,1,3)); # For correct img axis
#x_test           = convert(Array{Float32,3},x_test);
x_test           = reshape(x_test,(28,28,1,10000));
y_test           = Flux.onehotbatch(y_test, 0:9)

model = Chain(
    ## 28x28 => 14x14
    Conv((5, 5), 1=>8, pad=2, stride=2, relu),
    ## 14x14 => 7x7
    Conv((3, 3), 8=>16, pad=1, stride=2, relu),
    ## 7x7 => 4x4
    Conv((3, 3), 16=>32, pad=1, stride=2, relu),
    ## 4x4 => 2x2
    Conv((3, 3), 32=>32, pad=1, stride=2, relu),
    ## Average pooling on each width x height feature map
    GlobalMeanPool(),
    Flux.flatten,
    Dense(32, 10),
    Flux.softmax
)

myaccuracy(y,ŷ) = (mean(Flux.onecold(ŷ) .== Flux.onecold(y)))
myloss(x, y)     = Flux.crossentropy(model(x), y)

opt = Flux.ADAM()
ps  = Flux.params(model)
number_epochs = 4

[(println(e); Flux.train!(myloss, ps, train_data, opt)) for e in 1:number_epochs]

ŷtrain =   model(x_train)
ŷtest  =   model(x_test)
myaccuracy(y_train,ŷtrain)
myaccuracy(y_test,ŷtest)

plot(Gray.(x_train[:,:,1,2]))

cm = ConfusionMatrix()
fit!(cm,Flux.onecold(y_test) .-1, Flux.onecold(ŷtest) .-1 )
println(cm)

res = info(cm)
heatmap(string.(res["categories"]),string.(res["categories"]),res["normalised_scores"],seriescolor=cgrad([:white,:blue]),xlabel="Predicted",ylabel="Actual", title="Confusion Matrix (normalised scores)")

savefig("cm_digits.svg")
# ![](cm_digits.svg)
