# ## Recursive neural networks
# ### Setting the environment…

cd(@__DIR__)    
using Pkg      
Pkg.activate(".")  
## If using a Julia version different than 1.7 please uncomment and run the following line (the guarantee of reproducibility will however be lost)
## Pkg.resolve()   
## Pkg.instantiate()
using Random
Random.seed!(123)

using LinearAlgebra, Plots, Flux

# ### Generating simulated data
# The idea is to have a sequence that depend on the first 5 values. So the first 5 values are random, but the rest of the sequence depend deterministically to these first 5 values and the objective it to recreate this second part of the sequence knowing the first 5 parts.


nSeeds    = 2
seqLength = 5
nTrains   = 1000  
nVal      = 100
nTot      = nTrains+nVal
makeSeeds(nSeeds) = 2 .* (rand(1:10,nSeeds) .- 5) # [-1,+1]
function makeSequence(seeds,seqLength)
  seq = Vector{Float32}(undef,seqLength+nSeeds) # Flux Works with Float32 for performance reasons
  [seq[i] = seeds[i] for i in 1:nSeeds]
  for i in nSeeds+1:(seqLength+nSeeds)
     #seq[i] = seq[i-1] + (seeds[4]*0.5) # the only seed that matters is the 4th
     #seq[i] = seq[i-1] + seeds[1]*0.1*seq[i-1] +seeds[2]*seeds[3]*seq[i-1]*0.4+seeds[4]*seeds[5]*(seq[i-3]-seq[i-4])
     seq[i] = seq[i-1] + 0.2*seq[i-2] 
  end
  return seq
  return seq[nSeeds+1:end]
end

x0   = [makeSeeds(nSeeds) for i in 1:nTot]
seqs = makeSequence.(x0,seqLength)
seqs_vectors = [[[e] for e in seq] for seq in seqs]
y    = [s[2:end] for s in seqs_vectors] # y here is the value of the sequence itself at next step

xtrain = seqs_vectors[1:nTrains]
xval   = seqs_vectors[nTrains+1:end]
ytrain = y[1:nTrains]
yval   = y[nTrains+1:end]

# Flux wants a vector of sequences of individual items, when these in turns are vectors
allData   = xtrain;
aSequence = allData[1]
anElement = aSequence[1]

# ### Utility functions
function predictSequence(m,seeds,seqLength)
    seq = Vector{Vector{Float32}}(undef,seqLength+length(seeds)-1)
    Flux.reset!(m) # Reset the state (not the weigtht!)
    [seq[i] = [convert(Float32, seeds[i])] for i in 1:nSeeds]
    [seq[i] = m(seq[i-1]) for i in nSeeds+1:nSeeds+seqLength-1]
    [s[1] for s in seq]
end
function myloss(x, y)
    Flux.reset!(m)                 # Reset the state (not the weigtht!)
    [m(x[i]) for i in 1:nSeeds-1]  # Ignores the output but updates the hidden states
    ## y_i is x_(i+1), i.e. next element
    sum(Flux.mse(m(xi), yi) for (xi, yi) in zip(x[nSeeds:(end-1)], y[nSeeds:end]))
end
"""
   batchSequences(x,batchSize)

Transform a vector of sequences of individual elements represented as feature vectors to a vector of sequences of elements represented as features ×  batched record matrices
"""
function batchSequences(x,batchSize)
    x = copy(xtrain)
    batchSize = 3
    nRecords  = length(x)
    nItems    = length(x[1])
    nDims     = size(x[1][1],1) 
    nBatches  = Int(floor(nRecords/batchSize))

    emptyBatchedElement = Matrix{Float32}(undef,nDims,batchSize)
    emptySeq = [similar(emptyBatchedElement) for i in 1:nItems]
    outx = [similar(emptySeq) for i in 1:nBatches]
    for b in 1:nBatches
        xmin = (b-1)*batchSize + 1
        xmax = b*batchSize
        for e in 1:nItems
            outx[b][e] = hcat([x[i][e][:,1] for i in xmin:xmax]... )
        end
    end  
    return outx
end

# ### Defining the model
m         = Chain(Dense(1,30,σ),LSTM(30, 30), Dense(30, 30,relu),Dense(30,1))
m         = Chain(Dense(1,3),LSTM(3, 3), Dense(3, 3,relu),Dense(3,1))
ps        = params(m)
opt       = Flux.ADAM()
epochs    = 20 
batchSize = 1

# ### Plotting a random sequence and its prediction from untrained model…
seq1True = makeSequence(x0[1],seqLength)
seq1Est0 = predictSequence(m,x0[1],seqLength)
plot(seq1True)
plot!(seq1Est0)

# ### Model training
trainMSE  = Float64[]
valMSE    = Float64[]
for e in 1:epochs
    print("Epoch $e ")
    ## Shuffling at each epoch
    ids = shuffle(1:length(xtrain))
    x0e      = x0[ids]
    xtraine  = xtrain[ids]
    ytraine  = ytrain[ids]

    xtraine =batchSequences(xtraine,batchSize)
    ytraine =batchSequences(ytraine,batchSize)
    trainxy = zip(xtraine,ytraine)

    ## Actual training
    Flux.train!(myloss, ps, trainxy, opt)
    ## Making prediction on the trained model and computing accuracies
    global trainMSE, valMSE
    ŷtrain  = [predictSequence(m,x0[i],seqLength) for i in 1:nTrains]
    ŷval    = [predictSequence(m,x0[i],seqLength) for i in (nTrains+1):nTot]
    ytrain  = [makeSequence(x0[i],seqLength) for i in  1:nTrains]
    yval    = [makeSequence(x0[i],seqLength) for i in  (nTrains+1):nTot]

    trainmse =  sum(norm(ŷtrain[i][nSeeds+1:end] - ytrain[i][nSeeds+1:end-1])^2 for i in 1:nTrains)/nTrains
    valmse   =  sum(norm(ŷval[i][nSeeds+1:end] - yval[i][nSeeds+1:end-1])^2 for i in 1:nVal)/nVal
    push!(trainMSE,trainmse)
    push!(valMSE,valmse)
    println("MEan Sq Error: $trainmse - $valmse")
end

### Plotting the errors
plot(trainMSE,label="Train MSE")
plot!(valMSE,label="Validation MSE")

### Plotting some random sequences
for i = rand(1:nTot,5)
    trueseq = makeSequence(x0[i],seqLength)
    estseq  = predictSequence(m,x0[i],seqLength)
    seqPlot = plot(trueseq[1:end-1],label="true", title = "Seq $i")
    plot!(seqPlot, estseq, label="est")
    display(seqPlot)
end








