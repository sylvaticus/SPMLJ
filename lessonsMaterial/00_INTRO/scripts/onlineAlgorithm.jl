cd(@__DIR__)

function normalMean(x)
    N    = length(x)
    sumx = 0.0
    for i in 1:N
       sumx += x[i]
    end
    return sumx/N
end


function onlineMean(oldMean,oldN,newX)
     oldSum  = oldMean * oldN  
     newN    = oldN+1
     newMean = (oldSum + newX) / newN
     return (newMean, newN)
end

X = [1,2,3,4,5]

nMean = normalMean(X)


function computeMeanInLargeFile(filename)
    currentMean = 0.0
    currentN    = 0.0
    open(filename,"r") do f
        for ln in eachline(f)
            if ln != ""
                (currentMean, currentN) =  onlineMean(currentMean,currentN,parse(Float64,ln))
                println("Current mean: $currentMean")
            end
        end
    end
   
    return currentMean
end

olMean = computeMeanInLargeFile("x.txt")