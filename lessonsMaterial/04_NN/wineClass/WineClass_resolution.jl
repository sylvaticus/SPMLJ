# In this exercise we will try to predict the quality class of wines given some chemical characteristics

# In detail, the attributes of this dataset are:
#   1) Alcohol
#   2) Malic acid
#   3) Ash
#   4) Alcalinity of ash  
#   5) Magnesium
#   6) Total phenols
#   7) Flavanoids
#   8) Nonflavanoid phenols
#   9) Proanthocyanins
#   10) Color intensity
#   11) Hue
#   12) OD280/OD315 of diluted wines
#   13) Proline 

# Further information concerning this dataset can be found online on the [UCI Machine Learning Repository dedicated page](https://archive.ics.uci.edu/ml/datasets/wine) or in particular on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names)

# Our prediction concerns the quality class of the wine (1, 2 or 3) that is given as first column of the data.

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, Plots and BetaML.
# Also, seed the random seed with the integer `123`.
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)


# 2) Load the packages/modules DelimitedFiles, Pipe, HTTP, Plots, BetaML
using DelimitedFiles, Pipe, HTTP, Plots, BetaML


# 3) Load from internet or from local file the input data as a Matrix.
# You can use `readdlm`` using the comma as field separator.
dataURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
data    = @pipe HTTP.get(dataURL).body |> readdlm(_,',')


# 4) Now create the X matrix of features using the second to final columns of the data you loaded above and the Y vector by taking the 1st column. Transform the Y vector to a vector of integers using the `Int()` function (broadcasted). Make shure you have a 178×13 matrix and a 178 elements vector
X = data[:,2:end]
Y = Int.(data[:,1] )

# 4bis)
# Alternatively, data can be loaded from the saved CSV file (but it seems to miss 2 features):
using CSV, DataFrames
data2 = CSV.read("data/winequality-red.csv", DataFrame)
X2 = Matrix(data2[:,1:11])
Y2 = data2[:,12]

# 5) Partition the data in (`xtrain`,`xtest`) and (`ytrain`,`ytest`) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.
((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])


# 6) As the output is multinomial we need to encode `ytrain`. We use the `OneHotEncoder()` model to make `ytrain_oh`
ytrain_oh = fit!(OneHotEncoder(),ytrain) 

# 7) Define a `NeuralNetworkEstimator` model with the following characteristics:
#   - 3 dense layers with respectively 13, 20 and 3 nodes and activation function relu
#   - a `VectorFunctionLayer` with 3 nodes and `softmax` as activation function
#   - `crossentropy` as the neural network cost function
#   - training options: 100 epochs and 6 records to be used on each batch
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn= NeuralNetworkEstimator(layers=[l1,l2,l3,l4],loss=crossentropy,batch_size=6,epochs=100)

# 8) Train your model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviation)
fit!(mynn,fit!(Scaler(),xtrain),ytrain_oh)

# 9) Predict the training labels `ŷtrain` and the test labels `ŷtest`. Recall you did the training on the scaled features!
ŷtrain   = predict(mynn, fit!(Scaler(),xtrain)) 
ŷtest    = predict(mynn, fit!(Scaler(),xtest)) 


# 10) Compute the train and test accuracies using the function `accuracy`
trainAccuracy  = accuracy(ytrain,ŷtrain)
testAccuracy   = accuracy(ytest,ŷtest)  


# 11) Compute and print a Confusion Matrix of the test data true vs. predicted
cm = ConfusionMatrix()
fit!(cm,ytest,ŷtest)
println(cm)


# 12) Run the following commands to plots the average loss per epoch 
plot(info(mynn)["loss_per_epoch"])


# 13) (Optional) Run the same workflow without scaling the data or using `squared_cost` as cost function. How this affect the quality of your predictions ? 
Random.seed!(123)
((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])
ytrain_oh = fit!(OneHotEncoder(),ytrain) 
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn= NeuralNetworkEstimator(layers=[l1,l2,l3,l4],loss=crossentropy,batch_size=6,epochs=100)
fit!(mynn,xtrain,ytrain_oh)
ŷtrain   = predict(mynn, xtrain) 
ŷtest    = predict(mynn, xtest) 
trainAccuracy  = accuracy(ytrain,ŷtrain)
testAccuracy   = accuracy(ytest,ŷtest)  
plot(info(mynn)["loss_per_epoch"])

Random.seed!(123)
((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])
ytrain_oh = fit!(OneHotEncoder(),ytrain) 
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn= NeuralNetworkEstimator(layers=[l1,l2,l3,l4],loss=squared_cost,batch_size=6,epochs=100)
fit!(mynn,fit!(Scaler(),xtrain),ytrain_oh)
ŷtrain   = predict(mynn, fit!(Scaler(),xtrain)) 
ŷtest    = predict(mynn, fit!(Scaler(),xtest)) 
trainAccuracy  = accuracy(ytrain,ŷtrain)
testAccuracy   = accuracy(ytest,ŷtest)  
plot(info(mynn)["loss_per_epoch"])

