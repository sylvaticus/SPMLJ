
cd(@__DIR__)         
using Pkg             
Pkg.activate(".")  
# If using a Julia version different than 1.10 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
Pkg.instantiate()
using Random
Random.seed!(123)

using DelimitedFiles, JuMP, GLPK, DataFrames, CSV, Pipe, HTTP

urlActivities = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/activities.csv"
urlResources = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/resources.csv"
urlCoefficients = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/coefficients.csv"


activities = @pipe HTTP.get(urlActivities).body   |> CSV.File(_) |> DataFrame
resources  = @pipe HTTP.get(urlResources).body    |> CSV.File(_) |> DataFrame
coef       = @pipe HTTP.get(urlCoefficients).body |> readdlm(_,';')

(nA, nR) = (size(activities,1), size(resources,1)) 


# #### Optimisation model definition

profitModel = Model(GLPK.Optimizer)
set_optimizer_attribute(profitModel, "msg_lev", GLPK.GLP_MSG_ALL)

# #### Model's endogenous variables definition


@variables profitModel begin
    x[1:nA] >= 0
end

for a in 1:nA
    if activities.integer[a] == 1
        set_integer(x[a])
    end
end



@constraints profitModel begin
    resLimit[r in 1:nR], # observe resources limits
        sum(coef[r,a]*x[a] for a in 1:nA) <= resources.initial[r]
end

@objective profitModel Max begin
    sum(activities.gm[a] * x[a] for a in 1:nA)
end

print(profitModel)

# #### Model resolution
optimize!(profitModel)
status = termination_status(profitModel)

if (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.TIME_LIMIT) && has_values(profitModel)
    println("#################################################################")
    if (status == MOI.OPTIMAL)
        println("** Problem solved correctly **")
    else
        println("** Problem returned a (possibly suboptimal) solution **")
    end
    println("- Objective value (total costs): ", objective_value(profitModel))
    println("- Optimal Activities:\n")
    optValues = value.(x)
    for a in 1:nA
      println("* $(activities.label[a]):\t $(optValues[a])")
    end
    if JuMP.has_duals(profitModel)
        println("\n\n- Shadow prices of the resources:\n")
        for r in 1:nR
            println("* $(resources.label[r]):\t $(dual(resLimit[r]))")
        end
    end
else
    println("The model was not solved correctly.")
    println(status)
end

# ####  Updating the model to consider a larger company

set_normalized_rhs.(resLimit, resources.initial2)
normalized_rhs.(resLimit)
optimize!(profitModel)
status = termination_status(profitModel)
if (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.TIME_LIMIT) && has_values(profitModel)
    println("#################################################################")
    if (status == MOI.OPTIMAL)
        println("** Problem solved correctly **")
    else
        println("** Problem returned a (possibly suboptimal) solution **")
    end
    println("- Objective value (total costs): ", objective_value(profitModel))
    println("- Optimal Activities:\n")
    optValues = value.(x)
    for a in 1:nA
      println("* $(activities.label[a]):\t $(optValues[a])")
    end
    if JuMP.has_duals(profitModel)
        println("\n\n- Shadow prices of the resources:\n")
        for r in 1:nR
            println("* $(resources.label[r]):\t $(dual(resLimit[r]))")
        end
    end
else
    println("The model was not solved correctly.")
    println(status)
end