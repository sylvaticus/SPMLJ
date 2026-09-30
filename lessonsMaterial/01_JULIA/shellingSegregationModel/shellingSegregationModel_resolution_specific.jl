
# Shelling Segregation model
# Thomas C. Schelling (1971) Dynamic models of segregation, The Journal of Mathematical Sociology, 1:2, 143-186, DOI: 10.1080/0022250X.1971.9989794 

# A classical agent-based model in the social sciences, introducing the concepts of emerging macrobehaviours and tipping points.
# 2005 Nobel price in Economics
# One of the very first results obtained from running simulations



cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)

using Plots

mutable struct Env
    nR::Int64
    nC::Int64
    similarityThreeshold::Float64
    neighborhood::Int64
    nSteps::Int64
    cells::Vector{Int64}
    nBlues::Int64
    nReds::Int64
end

mypal   = [:white,:red,:blue]

xyToId(x,y,nR,nC) =  nR*(x-1)+y
iDToXY(id,nR,nC)  =  Int(floor((id-1)/nR)+1), (id-1)%(nR)+1

function getNeighbours(x,y,type,env)
    board  = reshape(env.cells,env.nR,env.nC)
    region = board[max(1,y-env.neighborhood):min(nR,y+env.neighborhood),max(1,x-env.neighborhood):min(nC,x+env.neighborhood)]
    sum(region .== type)
end


function isHappy(x,y,type,env)
    nBlues = getNeighbours(x,y,1,env)
    nReds  = getNeighbours(x,y,2,env)
    type == 1 ? (return nBlues/(nReds+nBlues) > env.similarityThreeshold) : (return nReds/(nReds+nBlues) > env.similarityThreeshold)
end

function reallocatePoints!(env)
    happyCount = 0
    for (i,t) in enumerate(env.cells)
        if t == 0 continue; end
        (x,y) = iDToXY(i,env.nR,env.nC)
        if isHappy(x,y,t,env)
            happyCount += 1
        else
            candIds = shuffle(1:env.nR*env.nC)
            for cId in candIds
                if env.cells[cId] != 0 continue; end
                (xc,yc) = iDToXY(cId,env.nR,env.nC)
                if isHappy(xc,yc,t,env)
                    env.cells[cId] = t
                    env.cells[i]   = 0
                    break
                end
            end
        end
    end
    return happyCount/(env.nBlues+env.nReds)
end


function run!(env)
    outplot = heatmap(reshape(env.cells,env.nR,env.nC), legend=nothing, title="START", color=mypal,aspect_ratio=env.nR/env.nC, size=(600,600*env.nR/env.nC))
    nHappyCount = Float64[]
    display(outplot)
    for i in 1:env.nSteps
        println("Running iteration $i...")
        nHappy = reallocatePoints!(env)
        push!(nHappyCount,nHappy)
        outplot = heatmap(reshape(env.cells,env.nR,env.nC), legend=nothing, title="Iteration $i", color=mypal,aspect_ratio=env.nR/env.nC, size=(600,600*env.nR/env.nC))
        display(outplot)
    end
    happyCountPlot = plot(nHappyCount,title="Share of happy agents by iteration")
    display(happyCountPlot)
end

# ------------------------------------------------------------------------------

# Parameters...
nR = 200
nC = 200
blueShare = 0.4
redShare  = 0.4 
nSteps = 20
similarityThreeshold = 0.4 # Agent is happy if at least 40% similar
neighborhood = 5           # Defining how far looking for similar agents

# Computation
nCells = nR*nC
nBlues = Int(ceil(nCells*blueShare))  # "1"
nReds  = Int(ceil(nCells*redShare))   # "2"

cells                           = fill(0,nCells)
cells[1:nBlues]                .= 1
cells[nBlues+1 : nBlues+nReds] .= 2
shuffle!(cells)
env = Env(nR,nC,similarityThreeshold,neighborhood,nSteps,cells,nBlues,nReds)
heatmap(reshape(env.cells,env.nR,env.nC), legend=nothing, title="Iteration 0", color=mypal,aspect_ratio=env.nR/env.nC, size=(600,600*env.nR/env.nC))
run!(env)