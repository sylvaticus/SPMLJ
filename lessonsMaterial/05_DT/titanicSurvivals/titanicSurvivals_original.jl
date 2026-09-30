# # Predicting titanic survivals

# Add packages
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")  
#Pkg.instantiate()

# Import modules
using DataFrames, CSV, DecisionTree, ScikitLearn.CrossValidation 


# Load training data to data frame
train_df = CSV.read("data/train.csv", DataFrame)
train_df = dropmissing(train_df,"Embarked")
train_df.Age = replace(train_df.Age,missing=>28)
train_df = select(train_df, Not("Cabin"))
train_df = select(train_df,Not(["PassengerId","Name"]));
train_df.Embarked = Int64.(
    replace(train_df.Embarked, 
        "S" => 1, "C" => 2, "Q" => 3
    )
)
train_df.Sex = Int64.(
    replace(train_df.Sex, 
        "female" => 1, "male" => 2
    )
)
train_df = select(train_df,Not("Ticket"))
y = train_df[:,"Survived"]
X = Matrix(train_df[:,Not(["Survived"])])
model = RandomForestClassifier(n_trees=100)
fit!(model,X,y)

# Evaluate the accuracy of predictions 
# using Cross Validation
accuracy = minimum(
    cross_val_score(model, X, y, cv=5)
)

test_df = CSV.read("data/test.csv",DataFrame)
PassengerId = test_df[:,"PassengerId"]
# Repeat the same transformations as we did for training dataset
test_df = select(test_df,
    Not(
        ["PassengerId","Name","Ticket","Cabin"]
    )
)
test_df.Age = replace(test_df.Age,missing=>28)
test_df.Embarked = replace(
    test_df.Embarked,"S" => 1, "C" => 2, "Q" => 3
)
test_df.Embarked = convert.(Int64,test_df.Embarked)
test_df.Sex = replace(
    test_df.Sex,"female" => 1,"male" => 2
)
test_df.Sex = convert.(Int64,test_df.Sex)

# In addition, replace missing value
# in 'Fare' field with median
test_df.Fare = replace(
    test_df.Fare,
    missing=>14.4542
)

Survived = predict(model, Matrix(test_df)) 


submit_df = DataFrame(PassengerId=PassengerId,Survived=Survived)
CSV.write("submission.csv",submit_df)

