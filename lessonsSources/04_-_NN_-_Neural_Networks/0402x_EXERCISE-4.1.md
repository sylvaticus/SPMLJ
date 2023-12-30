# EXERCISE 4.1: House value prediction with Neural Networks (regression)


```@raw html
<p>&nbsp;</p>
<img src="imgs/bostonHousingErrorPerEpoch.png" alt="Error per epoch" style="height:250px;"> 
<img src="imgs/bostonHousingEstVsTrueValues.png" alt="Estimated vs True house values" style="height:250px;"> 
<p>&nbsp;</p>
```

In this problem, we are given a dataset containing average house values in different Boston suburbs, together with the suburb characteristics (proportion of owner-occupied units built prior to 1940, index of accessibility to radial highways, etc...)
Our task is to build a neural network model and train it in order to predict the average house value on each suburb.

The detailed attributes of the dataset are:
  1. CRIM      per capita crime rate by town
  2. ZN        proportion of residential land zoned for lots over 25,000 sq.ft.
  3. INDUS     proportion of non-retail business acres per town
  4. CHAS      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
  5. NOX       nitric oxides concentration (parts per 10 million)
  6. RM        average number of rooms per dwelling
  7. AGE       proportion of owner-occupied units built prior to 1940
  8. DIS       weighted distances to five Boston employment centres
  9. RAD       index of accessibility to radial highways
  10. TAX      full-value property-tax rate per \$10,000
  11. PTRATIO  pupil-teacher ratio by town
  12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
  13. LSTAT    % lower status of the population
  14. MEDV     Median value of owner-occupied homes in \$1000's


Further information concerning this dataset can be found on [this file](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names)

Our prediction concern the median value (column 14 of the dataset)


**Skills employed:**
- download and import data from internet
- design and train a Neural Network for regression tasks using `BetaML`
- use the additional `BetaML` functions `partition`, `oneHotEncoder`, `scale`, `meanRelError`


## Instructions

If you have already cloned or downloaded the whole [course repository](https://github.com/sylvaticus/SPMLJ/) the folder with the exercise is on `[REPOSITORY_ROOT]/lessonsMaterial/04_NN/bostonHousing`.
Otherwise download a zip of just that folder [here](https://downgit.github.io/#/home?url=https://github.com/sylvaticus/SPMLJ/tree/main/lessonsMaterial/04_NN/bostonHousing).

In the folder you will find the file `BostonHousingValue.jl` containing the julia file that **you will have to complete to implement the missing parts and run the file** (follow the instructions on that file). 
In that folder you will also find the `Manifest.toml` file. The proposal of resolution below has been tested with the environment defined by that file.  
If you are stuck and you don't want to lookup to the resolution above you can also ask for help in the forum at the bottom of this page.
Good luck! 

## Resolution

Click "ONE POSSIBLE SOLUTION" to get access to (one possible) solution for each part of the code that you are asked to implement.

--------------------------------------------------------------------------------
### 1) Setting up the environment...
Start by setting the working directory to the directory of this file and activate it. If you have the provided `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages `Pipe`, `HTTP`, `CSV`, `DataFrames`, `Plots` and `BetaML`.


```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
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
Load the packages `Pipe`, `HTTP`, `CSV`, `DataFrames`, `Plots` and `BetaML`.

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
Load from internet or from local file the input data into a DataFrame or a Matrix.
You will need the CSV options `header=false` and `ignorerepeated=true`

```julia
dataURL="https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
```

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
data    = @pipe HTTP.get(dataURL).body |> CSV.File(_, delim=' ', header=false, ignorerepeated=true) |> DataFrame
```
```@raw html
</details>
```


--------------------------------------------------------------------------------
### 4) Implement one-hot encoding of categorical variables
The 4th column is a dummy related to the information if the suburb bounds a certain Boston river. Use the BetaML model `OneHotEncoder` to encode this dummy into two separate vectors, one for each possible value.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
riverDummy = fit!(OneHotEncoder(),data[:,4])
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 5) Put together the feature matrix
Now create the X matrix of features concatenating horizzontaly the 1st to 3rd column of `data`, the 5th to 13th columns and the two columns you created with the one hot encoding. Make sure you have a 506×14 matrix.

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
X = hcat(Matrix(data[:,[1:3;5:13]]),riverDummy)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 6) Build the label vector
Similarly define Y to be the 14th column of data

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
Y = data[:,14]
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 7) Partition the data
Partition the data in (`xtrain`,`xtest`) and (`ytrain`,`ytest`) keeping 80% of the data for training and reserving 20% for testing. Keep the default option to shuffle the data, as the input data isn't.

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
### 8) Define the neural network architecture
Define a `NeuralNetworkEstimator` model with the following characteristics:
  - 3 dense layers with respectively 14, 20 and 1 nodes and activation function relu
  - cost function `squared_cost` 
  - training options: 400 epochs and 6 records to be used on each batch

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
l1 = DenseLayer(14,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,1,f=relu)
mynn= NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost,batch_size=6,epochs=400)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 9) Train the model
Train the model using `ytrain` and a scaled version of `xtrain` (where all columns have zero mean and 1 standard deviation).

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
fit!(mynn,fit!(Scaler(),xtrain),ytrain)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 10) Predict the labels
Predict the training labels `ŷtrain` and the test labels `ŷtest`. Recall you did the training on the scaled features!

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
ŷtrain   = predict(mynn, fit!(Scaler(),xtrain)) 
ŷtest    = predict(mynn, fit!(Scaler(),xtest))    
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 11) Evaluate the model
Compute the train and test relative mean error using the function `relative_mean_error`

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
trainRME = relative_mean_error(ytrain,ŷtrain) 
testRME  = relative_mean_error(ytest,ŷtest)
```
```@raw html
</details>
```

--------------------------------------------------------------------------------
### 12) Plot the errors and the estimated values vs the true ones
Run the following commands to plots the average loss per epoch and the true vs estimated test values:

```julia
plot(info(mynn)["loss_per_epoch"])
scatter(ytest,ŷtest,xlabel="true values", ylabel="estimated values", legend=nothing)
```
--------------------------------------------------------------------------------
### 13) (Optional) Use unscaled data
Run the same workflow without scaling the data. How this affect the quality of your predictions ? 

```@raw html
<details><summary>ONE POSSIBLE SOLUTION</summary>
```
```julia
Random.seed!(123)
((xtrain,xtest),(ytrain,ytest)) = partition([X,Y],[0.8,0.2])
l1 = DenseLayer(14,20,f=relu)
l2 = DenseLayer(20,20,f=relu)
l3 = DenseLayer(20,1,f=relu)
mynn= NeuralNetworkEstimator(layers=[l1,l2,l3],loss=squared_cost,batch_size=6,epochs=400)
fit!(mynn,xtrain,ytrain)
ŷtrain   = predict(mynn, xtrain) 
ŷtest    = predict(mynn, xtest)
trainRME = relative_mean_error(ytrain,ŷtrain) 
testRME  = relative_mean_error(ytest,ŷtest)
plot(info(mynn)["loss_per_epoch"])
scatter(ytest,ŷtest,xlabel="true values", ylabel="estimated values", legend=nothing)
```
```@raw html
</details>
```

---------
```@raw html
<div id="pd_rating_holder_8962705"></div>
<script type="text/javascript">
const pageURL = window.location.href;
PDRTJS_settings_8962705 = {
"id" : "8962705",
"unique_id" : "/home/lobianco/CloudFiles/lef-nancy-sync/Documents/Teaching/2021-2022/Introduction to Scientific Programming and Machine Learning with Julia/SPMLJ/lessonsSources/04_-_NN_-_Neural_Networks/0402x_EXERCISE-4.1.md",
"title" : "0402x_EXERCISE-4.1.md",
"permalink" : pageURL
};
</script>
```
```@raw html
<div class="addthis_inline_share_toolbox"></div>
```

---------
```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```
```@raw html
<script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
```
```@raw html
<!-- Go to www.addthis.com/dashboard to customize your tools -->
<script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
```

---------
```@raw html
<div id="pd_rating_holder_8962705"></div>
<script type="text/javascript">
const pageURL = window.location.href;
PDRTJS_settings_8962705 = {
"id" : "8962705",
"unique_id" : "/home/lobianco/CloudFiles/lef-nancy-sync/Documents/Teaching/2021-2022/Introduction to Scientific Programming and Machine Learning with Julia/SPMLJ/lessonsSources/04_-_NN_-_Neural_Networks/0402x_EXERCISE-4.1.md",
"title" : "0402x_EXERCISE-4.1.md",
"permalink" : pageURL
};
</script>
```
```@raw html
<div class="addthis_inline_share_toolbox"></div>
```

---------
```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```
```@raw html
<script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
```
```@raw html
<!-- Go to www.addthis.com/dashboard to customize your tools -->
<script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
```

---------
```@raw html
<div id="pd_rating_holder_8962705"></div>
<script type="text/javascript">
const pageURL = window.location.href;
PDRTJS_settings_8962705 = {
"id" : "8962705",
"unique_id" : "/home/lobianco/CloudFiles/lef-nancy-sync/Documents/Teaching/2021-2022/Introduction to Scientific Programming and Machine Learning with Julia/SPMLJ/lessonsSources/04_-_NN_-_Neural_Networks/0402x_EXERCISE-4.1.md",
"title" : "0402x_EXERCISE-4.1.md",
"permalink" : pageURL
};
</script>
```
```@raw html
<div class="addthis_inline_share_toolbox"></div>
```

---------
```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```
```@raw html
<script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
```
```@raw html
<!-- Go to www.addthis.com/dashboard to customize your tools -->
<script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
```

---------
```@raw html
<div id="pd_rating_holder_8962705"></div>
<script type="text/javascript">
const pageURL = window.location.href;
PDRTJS_settings_8962705 = {
"id" : "8962705",
"unique_id" : "/home/lobianco/CloudFiles/lef-nancy-sync/Documents/Teaching/2021-2022/Introduction to Scientific Programming and Machine Learning with Julia/SPMLJ/lessonsSources/04_-_NN_-_Neural_Networks/0402x_EXERCISE-4.1.md",
"title" : "0402x_EXERCISE-4.1.md",
"permalink" : pageURL
};
</script>
```
```@raw html
<div class="addthis_inline_share_toolbox"></div>
```

---------
```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```
```@raw html
<script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
```
```@raw html
<!-- Go to www.addthis.com/dashboard to customize your tools -->
<script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
```

---------
```@raw html
<div id="pd_rating_holder_8962705"></div>
<script type="text/javascript">
const pageURL = window.location.href;
PDRTJS_settings_8962705 = {
"id" : "8962705",
"unique_id" : "/home/lobianco/CloudFiles/lef-nancy-sync/Documents/Teaching/2021-2022/Introduction to Scientific Programming and Machine Learning with Julia/SPMLJ/lessonsSources/04_-_NN_-_Neural_Networks/0402x_EXERCISE-4.1.md",
"title" : "0402x_EXERCISE-4.1.md",
"permalink" : pageURL
};
</script>
```
```@raw html
<div class="addthis_inline_share_toolbox"></div>
```

---------
```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```
```@raw html
<script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
```
```@raw html
<!-- Go to www.addthis.com/dashboard to customize your tools -->
<script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
```

---------
```@raw html
<div id="pd_rating_holder_8962705"></div>
<script type="text/javascript">
const pageURL = window.location.href;
PDRTJS_settings_8962705 = {
"id" : "8962705",
"unique_id" : "/home/lobianco/CloudFiles/lef-nancy-sync/Documents/Teaching/2021-2022/Introduction to Scientific Programming and Machine Learning with Julia/SPMLJ/lessonsSources/04_-_NN_-_Neural_Networks/0402x_EXERCISE-4.1.md",
"title" : "0402x_EXERCISE-4.1.md",
"permalink" : pageURL
};
</script>
```
```@raw html
<div class="addthis_inline_share_toolbox"></div>
```

---------
```@raw html
<script src="https://utteranc.es/client.js"
        repo="sylvaticus/SPMLJ"
        issue-term="title"
        label="💬 website_comment"
        theme="github-dark"
        crossorigin="anonymous"
        async>
</script>
```
```@raw html
<script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
```
```@raw html
<!-- Go to www.addthis.com/dashboard to customize your tools -->
<script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
```
