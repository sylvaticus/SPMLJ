# # Predicting titanic survivals using BetaML NeuraLNetworkEstimator

cd(@__DIR__)  
using Pkg             
Pkg.activate(".")  
#Pkg.instantiate()

# Import modules
using DataFrames, CSV, BetaML

# Load training data to data frame
train_df  = CSV.read("data/train.csv", DataFrame)
y         = collect(train_df.Survived) .+1

encoder_y, encoder_sex, encoder_embarked = OneHotEncoder(), OneHotEncoder(), OneHotEncoder()
y_oh        = fit!(encoder_y,y)
sex_oh      = fit!(encoder_sex,train_df.Sex) 
embarked_oh = fit!(encoder_embarked,train_df.Embarked)

train_df  = select(train_df,Not(["PassengerId","Name","Ticket","Cabin","Sex","Embarked","Survived"]));
X_partial = hcat(fit!(Scaler(),Matrix(train_df)),sex_oh,embarked_oh)
impMod    = RFImputer(n_trees=60,recursive_passages=2)
X         = fit!(impMod,X_partial)
(N,D)     = size(X)
DY        = size(y_oh,2)

sampler   = KFold(nsplits=5,nrepeats=2);
(μ,σ) = cross_validation([X,y],sampler) do trainData,valData,rng
                (xtrain,ytrain) = trainData; (xval,yval) = valData
                innerD = Int(round(D*2)) # These two hyper-parameters you would most likely want to tune by running different cross validations 
                epochs = 100
                layers  = [DenseLayer(D,innerD,f=relu),DenseLayer(innerD,DY,f=relu),VectorFunctionLayer(DY,f=softmax)];
                model   = NeuralNetworkEstimator(layers=layers,opt_alg=ADAM(),epochs=epochs,verbosity=NONE)
                ytrain_oh = predict(encoder_y,ytrain)
                fit!(model,xtrain,ytrain_oh)
                ŷval_oh = predict(model,xval)
                return accuracy(yval,ŷval_oh) 
        end # (0.8192, 0.018)

# Actual model training
innerD  = Int(round(D*2))
epochs  = 100
layers  = [DenseLayer(D,innerD,f=relu),DenseLayer(innerD,DY,f=relu),VectorFunctionLayer(DY,f=softmax)];
model   = NeuralNetworkEstimator(layers=layers,opt_alg=ADAM(),epochs=epochs)
ŷ_oh    =  fit!(model,X,y_oh)
inSampleAccuracy = accuracy(y,ŷ_oh) # 0.8406
hcat(y,mode(ŷ_oh))

# Saving models
model_save("titanic_model_nn.jld"; encoder_y, encoder_sex, encoder_embarked, imputation_model=impMod, titanic_model_nn=model)

# Test (possibly in production...)
test_df       = CSV.read("data/test.csv", DataFrame)
encoder_y, encoder_sex, encoder_embarked, impModel, model = model_load("titanic_model_nn.jld","encoder_y","encoder_sex", "encoder_embarked", "imputation_model","titanic_model_nn")
PassengerId   = test_df[:,"PassengerId"]

sex_oh      = predict(encoder_sex,test_df.Sex) 
embarked_oh = predict(encoder_embarked,test_df.Embarked)

test_df       = select(test_df,Not(["PassengerId","Name","Ticket","Cabin","Sex","Embarked"]));
Xtest_partial = hcat(fit!(Scaler(),Matrix(test_df)),sex_oh,embarked_oh)
Xtest         = predict(impMod,Xtest_partial)

Survived      = mode(predict(model,Xtest))
submit_df     = DataFrame(PassengerId=PassengerId,Survived=(Survived .- 1))
CSV.write("submission_betaml_nn.csv",submit_df) # 0.78


