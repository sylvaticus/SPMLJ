# EXERCISE 4.2: Wine class prediction with Neural Networks (multinomial classification)

```@raw html
<p>&nbsp;</p>
<img src="imgs/errorPerEpoch.png" alt="Error per epoch" style="height:170px;"> 
<img src="imgs/errorPerEpoch_unscaled.png" alt="Error per epoch (uscaled)" style="height:170px;"> 
<img src="imgs/errorPerEpoch_squaredCost.png" alt="Error per epoch (squaredCost" style="height:170px;"> 
<p>&nbsp;</p>
```

In this problem, we are given a dataset containing the quality class of some Italian wines, together with their chemical characteristics (alcohol content, flavonoids,  colour intensity...)
Our task is to build a neural network model and train it in order to predict the wine quality class.
This is an example of a _multinomial regression_.

In detail, the attributes of this dataset are:
1. Alcohol
2. Malic acid
3. Ash
4. Alcalinity of ash  
5. Magnesium
6. Total phenols
7. Flavanoids
8. Nonflavanoid phenols
9. Proanthocyanins
10. Color intensity
11. Hue
12. OD280/OD315 of diluted wines
13. Proline 

Further information concerning this dataset can be found online on the [UCI Machine Learning Repository dedicated page](https://archive.ics.uci.edu/ml/datasets/wine) or in particular on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.names)

Our prediction concerns the quality class of the wine (1, 2 or 3) that is given in the first column of the data.

**Skills employed:**
- download and import data from internet
- design and train a Neural Network for multinomial classification using `BetaML`
- use the additional `BetaML` functions `partition`, `oneHotEncoder`, `scale`, `accuracy`, `ConfusionMatrix`

## Instructions

If you have already cloned or downloaded the whole [course repository](https://github.com/sylvaticus/SPMLJ/) the folder with the exercise is on `[REPOSITORY_ROOT]/lessonsMaterial/04_NN/wineClass`.
Otherwise download a zip of just that folder [here](https://downgit.github.io/#/home?url=https://github.com/sylvaticus/SPMLJ/tree/main/lessonsMaterial/04_NN/wineClass).

In the folder you will find the file `WineClass.jl` containing the julia file that **you will have to complete to implement the missing parts and run the file** (follow the instructions on that file). 
In that folder you will also find the `Manifest.toml` file. The proposal of resolution below has been tested with the environment defined by that file.  
If you are stuck and you don't want to lookup to the resolution above you can also ask for help in the forum at the bottom of this page.
Good luck! 

## Resolution

Click "ONE POSSIBLE SOLUTION" to get access to (one possible) solution for each part of the code that you are asked to implement.

--------------------------------------------------------------------------------
### 1) Setting up the environment...
Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages `Pipe`, `HTTP`, `Plots` and `BetaML`.

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
Load the packages `DelimitedFiles`, `Pipe`, `HTTP`, `Plots` and `BetaML`.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
using DelimitedFiles, Pipe, HTTP, Plots, BetaML
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 3) Load the data

Load from internet or from local file the input data as a Matrix.
You can use `readdlm`` using the comma as field separator.

```julia
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
```

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
data    = @pipe HTTP.get(dataURL).body |> readdlm(_,',')
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 4) Write the feature matrix and the the label vector
Now create the X matrix of features using the second to final columns of the data you loaded above and the Y vector by taking the 1st column. Transform the Y vector to a vector of integers using the `Int()` function (broadcasted). Make sure you have a 178×13 matrix and a 178 elements vector

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
X = data[:,2:end]
Y = Int.(data[:,1] )
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 5) Partition the data
Partition the data in (xtrain,xval) and (ytrain,yval) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 6) Implement one-hot encoding of categorical variables
As the output is multinomial we need to encode ytrain. We use the `oneHotEncoder()` function to make `ytrain_oh`

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
ytrain_oh = oneHotEncoder(ytrain) 
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 7) Define the neural network architecture
Define a Neural Network model with the following characteristics:
- 3 dense layers with respectively 13, 20 and 3 nodes and activation function relu
- a `VectorFunctionLayer` with 3 nodes and `softmax` as activation function
- `crossEntropy` as the neural network cost function

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3,l4],crossEntropy)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 8) Train the model
Train the model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviaiton) for 100 epochs and use a batch size of 6 records.
Save the output of your training function to `trainingLogs`

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
trainingLogs = train!(mynn,scale(xtrain),ytrain_oh,batchSize=6,epochs=100)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 9) Predict the labels 
Predict the training labels ŷtrain and the validation labels ŷval. Recall you did the training on the scaled features!

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
ŷtrain   = predict(mynn, scale(xtrain)) 
ŷval     = predict(mynn, scale(xval))  
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 10) Evaluate the model
Compute the train and test accuracies using the function `accuracy`

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
trainAccuracy = accuracy(ŷtrain,ytrain)
valAccuracy   = accuracy(ŷval,yval)  
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 11) Evaluate the model more in detail
Compute and print a confution matrix of the validation data true vs. predicted

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
cm = ConfusionMatrix(ŷval,yval)
println(cm)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 12) Plot the errors
Run the following commands to plots the average loss per epoch 

```julia
plot(trainingLogs.ϵ_epochs, label="ϵ per epoch")
```

--------------------------------------------------------------------------------
### 13) (Optional) Use unscaled data
Run the same workflow without scaling the data or using `squaredCost` as cost function. How this affect the quality of your predictions ? 

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
Random.seed!(123)
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
ytrain_oh = oneHotEncoder(ytrain) 
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3,l4],crossEntropy)
trainingLogs = train!(mynn,xtrain,ytrain_oh,batchSize=6,epochs=100)
ŷtrain   = predict(mynn, xtrain)
ŷval     = predict(mynn, xval) 
trainAccuracy = accuracy(ŷtrain,ytrain)
valAccuracy   = accuracy(ŷval,yval)  
plot(trainingLogs.ϵ_epochs, label="ϵ per epoch (unscaled version)")

Random.seed!(123)
((xtrain,xval),(ytrain,yval)) = partition([X,Y],[0.8,0.2])
ytrain_oh = oneHotEncoder(ytrain) 
l1 = DenseLayer(13,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,3,f=relu)
l4 = VectorFunctionLayer(3,f=softmax)
mynn = buildNetwork([l1,l2,l3,l4],squaredCost)
trainingLogs = train!(mynn,xtrain,ytrain_oh,batchSize=6,epochs=100)
ŷtrain   = predict(mynn, xtrain)
ŷval     = predict(mynn, xval) 
trainAccuracy = accuracy(ŷtrain,ytrain)
valAccuracy   = accuracy(ŷval,yval)  
plot(trainingLogs.ϵ_epochs, label="ϵ per epoch (squaredCost version)")
```
```@raw html
</details>
```