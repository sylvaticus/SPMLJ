################################################################################
#    loggingOptimisation Problem
#
# The objective of this problem is to find the optimal level of activities to perform to maximise the company profit, given (a) the profitability (gross margin) of each activity, (b) the resources available to the company and (c) the technical coefficients that links each activity to the resource needed (positive coefficient) or provided (negative coefficient).
#
# The problem is depicted in the SpreadSheet file "Optimal production mix.ods" and can be solved there by using the "Solver tool" as indicated in that file. You can use it to check your results.
#
#

# ### Environment set-up and data loading

# Start by setting the working directory to the directory of this file and activate it. If you have the `Manifest.toml` and `ProjecT.toml` files in the directory, run `instantiate()`, otherwise manually add the packages JuMP, GLPK, DataFrames, CSV, Pipe, HTTP.

# Load the packages using DelimitedFiles, JuMP, GLPK, DataFrames, CSV, Pipe, HTTP

# Load from internet or from local files the following data:

urlActivities = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/activities.csv"
urlResources = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/resources.csv"
urlCoefficients = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/coefficients.csv"

# The result must be:
# - `activities`, a dataframe with the columns `label`, `gm` and `integer`
# - `resources`, a dataframe with the columns `label`, `initial` and `initial2`
# - `coeff`, a 10x12 Matrix of Float64

# For example to download from internet and import the matrix you can use something like:
coef  = @pipe HTTP.get(urlCoefficients).body |> readdlm(_,';')

# Determine `nA` and `nR` as the number of activities and the number of resources (use the `size` function)


# ### Optimisation model definition

# Define `profitModel` as a model to be optimised using the `GLPK.Optimizer`
# Optional: set `GLPK.GLP_MSG_ALL` as the `msg_lev` of GLPK

# ### Model's endogenous variables definition

# Define the non-negative model `x` variable, indexed by the positions between 1 and `nA` (i.e. `x[1:nA] >= 0`) (you could use the @variables macro)

# Set the variables for which the corresponding `integer` column in the `activity` dataframe is equal to 1 as a integer variable.
# To set the specific vaciable `x[a]` as integer use  `set_integer(x[a])`

# ### Model's constraint definition

# Define the `resLimit[r in 1:nR]` contraint(s), such that when you sum `coef[r,a]*x[a]` for all the `1:nA` activities you have a value lower than `resources.initial[r]`

# ### Objective definition

# Define the objective as the maximisation of the profit given by summing for each of the `1:nA` activities `activities.gm[a] * x[a]`

# ### Model resolution
# (optionally) print the model to check it and optimize if. Also, check with the function `status = termination_status(profitModel)` that the status is `OPTIMAL`.

# ### Print optimal level of activities

# Run the following code to print the results (optimal level of activities)

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


# ### OPTIONAL: Observe the emergence of scale effects

# Optionally re-run the model with `initial2` initial resources instead of `initial` and notice how this bigger company can afford differnt kind of activities (logging high forest instead of coppices in this example) obtaining a better profitability per unit of resource emploied.
# Can you guess which are the aspects of the optimisation model that allow for the emergence of these scale effects ?
