# In this exercise we will try to predict the average housing value in suburbs of Boston given some characteristics of the suburb.

# These are the detailed attributes of the dataset:
#   1. CRIM      per capita crime rate by town
#   2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
#   3. INDUS     proportion of non-retail business acres per town
#   4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#   5. NOX       nitric oxides concentration (parts per 10 million)
#   6. RM        average number of rooms per dwelling
#   7. AGE       proportion of owner-occupied units built prior to 1940
#   8. DIS       weighted distances to five Boston employment centres
#   9. RAD       index of accessibility to radial highways
#   10. TAX      full-value property-tax rate per $10,000
#   11. PTRATIO  pupil-teacher ratio by town
#   12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#   13. LSTAT    % lower status of the population
#   14. MEDV     Median value of owner-occupied homes in $1000's

# Further information concerning this dataset can be found on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)

# Our prediction concern the median value (column 14 of the dataset)

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages Pipe, HTTP, CSV, DataFrames, Plots and BetaML.
# Also, seed the random seed with the integer `123`.


# 2) Load the packages/modules Pipe, HTTP, CSV, DataFrames, BetaML


# 3) Load from internet or from local file the input data into a DataFrame or a Matrix.
# You will need the CSV options `header=false` and `ignorerepeated=true``
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"


# 4) The 4th column is a dummy related to the information if the suburb bounds a certain Boston river. Use the BetaML function `oneHotEncoder` to encode this dummy into two separate vectors, one for each possible value. Note that you will need to transform the range {0,1} into {1,2} before running the oneHotEncoder function (this can be done by simply uinsg `data[:,4] .+1`)


# 5) Now create the X matrix of features concatenating horizzontaly the 1st to 3rd column of `data`, the 5th to 13th columns and the two columns you created with the one hot encoding. Make shure you have a 506×14 matrix.


# 6) Similarly define Y to be the 14th column of data


# 7) Partition the data in (xtrain,xval) and (ytrain,yval) keeping 80% of the data for training and reserving 35% for testing. Keep the default option to shuffle the data, as the input data isn't.


# 8) Define a Neural Network model with the following characteristics:
#   - 3 dense layers with respectively 14, 20 and 1 nodes and activation function relu
#   - cost function `squaredCost` 


# 9) Train your model using `ytrain` and a scaled version of xtrain (where all columns have zero mean and 1 standard deviaiton) for 400 epochs and use a batch size of 6 records.
# Save the output of your training function to `trainingLogs`


# 10) Predict the training labels ŷtrain and the validation labels ŷval. Recall you did the training on the scaled features!


# 11) Compute the train and test relative mean error using the function `meanRelError` with the parameter `normRec` set to `false`


# 12) Run the following commands to plots the average loss per epoch and the true vs estimation validation values 
plot(trainingLogs.ϵ_epochs)
scatter(yval,ŷval,xlabel="true values", ylabel="estimated values", legend=nothing)


# 13) (Optional) Run the same workflow without scaling the data. How this affect the quality of your predictions ? 
