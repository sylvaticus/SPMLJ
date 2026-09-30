cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using Random
Random.seed!(123)

using DelimitedFiles, CSV, HTTP, Pipe, DataFrames, Flux

# Load the data and shuffle it in case it isn't..
dataURL   = "https://raw.githubusercontent.com/sylvaticus/SPMLJ/main/lessonsMaterial/04_NN/sentimentAnalysis/productReviews.csv"
data      = @pipe HTTP.get(dataURL).body |> CSV.File(_,delim='\t') |> DataFrame
stopwords = convert(Array{String,1},readdlm("data/stopwords.csv")[:,1])
stopwords = String[]
data      = data[shuffle(1:size(data,1)),:] # Shuffle the data in case it isn't..
# data = data[1:500,:] # let's work fist on  a subsample... 
data.sentiment = max.(0,data.sentiment) # Converting the sentiment label from {-1,1} to {0,1}



"""
    extractWords(inputString)
Inputs a text string, returns a list of lowercase words in the string.
Punctuation and digits are separated out into their own words.
"""
function extractWords(inputString,stopwords)
    punctuation = "!\"#\$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    digits = "0123456789"
    for c in punctuation * digits
        inputString = replace(inputString, c => " " * c * " ")
    end
    words = lowercase(inputString) |> split
    return filter(w -> ! (w in stopwords), words)
end

# Extract a vocaboulary of unique words
vocabulary = unique(vcat(extractWords.(data.text,Ref(stopwords))...));
in("emptysequence",vocabulary)
nV         = length(vocabulary)

traindata,valdata = data[1:4000,:],data[4001:end,:]

# nRecords size vector of nSeq size vector of nVocabulary vector of booleans 
xtrain = [Flux.onehot.(extractWords(traindata.text[j],stopwords),Ref(vocabulary)) for j in 1:size(traindata,1)]
xval   = [Flux.onehot.(extractWords(valdata.text[i],stopwords),Ref(vocabulary)) for i in 1:size(valdata,1)]

ytrain = traindata.sentiment
yval   = valdata.sentiment



m  = Chain(Dense(nV,15,σ),LSTM(15, 15), Dense(15, 1))

function loss(x, y)
    nSeq = length(x)
    Flux.reset!(m) # Reset the state (not the weigtht!)
    [m(x[i]) for i in 1:nSeq-1]  # Ignores the output but updates the hidden states
    Flux.Losses.logitbinarycrossentropy(m(x[end]),y) # Compute the loss only on the decoding of the final sequence
end

ps  = params(m)
opt = ADAM()

function predictSentiment(m,x)
    nSeq = length(x)
    Flux.reset!(m) # Reset the state (not the weigtht!)
    [m(x[i]) for i in 1:nSeq-1]  # ignores the output but updates the hidden states
    return Int64(round(sigmoid(m(x[end])[1])))
end 

epochs = 40
trainAccs = Float64[]
valAccs   = Float64[]
for e in 41:epochs
    print("Epoch $e ")
    # Shuffling at each epoch
    ids = shuffle(1:length(xtrain))
    xtrain  = xtrain[ids]
    ytrain  = ytrain[ids]
    trainxy = zip(xtrain,ytrain)
    # Actual training
    Flux.train!(loss, ps, trainxy, opt)
    # Making prediction on the trained model and computing accuracies
    ŷtrain        = predictSentiment.(Ref(m),xtrain)
    ŷval          = predictSentiment.(Ref(m),xval)
    trainaccuracy =  sum(ŷtrain .== ytrain)/length(xtrain)
    valaccuracy   =  sum(ŷval   .== yval)/length(xval)
    push!(trainAccs,trainaccuracy)
    push!(valAccs,valaccuracy)
    println("accuracies: $trainaccuracy - $valaccuracy")
end

m2  = Chain(LSTM(nV, 3), Dense(3, 1))
ps2 = params(m2)
opt = ADAM(1e-3)
Flux.train!(loss, ps2, trainxy, opt)
ŷtrain = predictSentiment.(Ref(m2),xtrain)
ŷval   = predictSentiment.(Ref(m2),xval)

trainaccuracy =  sum(ŷtrain .== ytrain)/length(xtrain)
valaccuracy   =  sum(ŷval   .== yval)/length(xval)


