# # Predicting titanic survivals

cd(@__DIR__)  
using Pkg             
Pkg.activate(".")  
#Pkg.instantiate()

# Import modules
using DataFrames, CSV, BetaML

# Load training data to data frame
train_df  = CSV.read("data/train.csv", DataFrame)
train_df  = select(train_df,Not(["PassengerId","Name","Ticket"]));
y         = train_df[:,"Survived"]
X_partial = Matrix(train_df[:,Not(["Survived"])])
impMod    = RFImputer(n_trees=60,recursive_passages=2)
X         = fit!(impMod,X_partial)

sampler   = KFold(nsplits=5,nrepeats=2);
(μ,σ) = cross_validation([X,y],sampler) do trainData,valData,rng
                (xtrain,ytrain) = trainData; (xval,yval) = valData
                model = RandomForestEstimator(n_trees=100,force_classification=true)
                fit!(model,xtrain,ytrain)
                ŷ     =  predict(model,xval)  
                return accuracy(collect(yval),ŷ)
        end # (0.826, 0.038)

# Actual model training
model            = RandomForestEstimator(n_trees=100,force_classification=true)
ŷ                =  fit!(model,X,y)
inSampleAccuracy = accuracy(y,ŷ) # 0.9472

# Saving models
model_save("titanic_model.jld";imputation_model=impMod,titanic_model=model)

# Test (possibly in production...)
test_df         = CSV.read("data/test.csv", DataFrame)
impModel, model = model_load("titanic_model.jld","imputation_model","titanic_model")
PassengerId     = test_df[:,"PassengerId"]
test_df         = select(test_df,Not(["PassengerId","Name","Ticket"]));
Xtest_partial   = Matrix(test_df)
Xtest           = predict(impModel,Xtest_partial)
Survived        = mode(predict(model,Xtest))
submit_df       = DataFrame(PassengerId=PassengerId,Survived=Survived)
CSV.write("submission_betaml.csv",submit_df) # 0.766
