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


# 2) Load the packages/modules DelimitedFiles, Pipe, HTTP, Plots, BetaML


# 3) Load from internet or from local file the input data as a Matrix.
# You can use `readdlm`` using the comma as field separator.
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"


# 4) Now create the X matrix of features using the second to final columns of the data you loaded above and the Y vector by taking the 1st column. Transform the Y vector to a vector of integers using the `Int()` function (broadcasted). Make shure you have a 178×13 matrix and a 178 elements vector


# 5) Partition the data in (`xtrain`,`xtest`) and (`ytrain`,`ytest`) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.


# 6) As the output is multinomial we need to encode `ytrain`. We use the `OneHotEncoder()` model to make `ytrain_oh`


# 7) Define a `NeuralNetworkEstimator` model with the following characteristics:
#   - 3 dense layers with respectively 13, 20 and 3 nodes and activation function relu
#   - a `VectorFunctionLayer` with 3 nodes and `softmax` as activation function
#   - `crossentropy` as the neural network cost function
#   - training options: 100 epochs and 6 records to be used on each batch


# 8) Train your model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviaiton) 


# 9) Predict the training labels `ŷtrain` and the test labels `ŷtest`. Recall you did the training on the scaled features!


# 10) Compute the train and test accuracies using the function `accuracy`


# 11) Compute and print a Confusion Matrix of the test data true vs. predicted


# 12) Run the following commands to plots the average loss per epoch 
plot(info(mynn)["loss_per_epoch"])


# 13) (Optional) Run the same workflow without scaling the data or using `squared_cost` as cost function. How this affect the quality of your predictions ? 
