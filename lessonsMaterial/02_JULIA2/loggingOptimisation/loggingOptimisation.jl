################################################################################
#    loggingOptimisation Problem
#
# You are the decision-maker of a production unit (here a logging company) and have a set of possible activities to undergo.
# How do you choose the activities that lead to a maximisation of the unit's profit? 
# The objective of this problem is hence to find the optimal level of activities to maximise the company profit, given (a) the profitability (gross margin) of each activity, (b) the resources available to the company and (c) the matrix of technical coefficients that link each activity to the resources required (positive coefficient) or provided (negative coefficient) by that specific activity.

# The problem is the same as those in the SpreadSheet file "Optimal production mix.ods" (available in the `data` folder) and can be solved in LibreOffice by using the "Solver tool" as indicated in that file. You can use it to check your results.
#
#

# ### Environment set-up and data loading

# 1) Start by setting the working directory to the directory of this file and activate it. If you have the `Manifest.toml` file in the directory, just run `Pkg.instantiate()`, otherwise manually add the packages JuMP, GLPK, DataFrames, CSV, Pipe, HTTP.

# 2) Load the packages DelimitedFiles, JuMP, GLPK, DataFrames, CSV, Pipe, HTTP

# 3) Load from internet or from local files the following data:

urlActivities   = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/activities.csv"
urlResources    = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/resources.csv"
urlCoefficients = "https://raw.githubusercontent.com/sylvaticus/IntroSPMLJuliaCourse/main/lessonsMaterial/02_JULIA2/loggingOptimisation/data/coefficients.csv"

# The result must be:
# - `activities`, a dataframe with the columns `label`, `gm` and `integer`
# - `resources`, a dataframe with the columns `label`, `initial` and `initial2`
# - `coeff`, a 10x12 Matrix of Float64

# For example to download from internet and import the matrix you can use something like:
coef  = @pipe HTTP.get(urlCoefficients).body |> readdlm(_,';')

# 4) Determine `nA` and `nR` as the number of activities and the number of resources (use the `size` function)


# ### Optimisation model definition

# 5) Define `profitModel` as a model to be optimised using the `GLPK.Optimizer`
# 6) [OPTIONAL] set `GLPK.GLP_MSG_ALL` as the `msg_lev` of GLPK

# ### Model's endogenous variables definition

# 7) Define the non-negative model `x` variable, indexed by the positions between 1 and `nA` (i.e. `x[1:nA] >= 0`) (you could use the @variables macro). Don't set the `x` variable to be integer at this step, as some variables are continuous, just set them to be non-negative.

# 8) Set the variables for which the corresponding `integer` column in the `activity` dataframe is equal to 1 as a integer variable.
# To set the specific vaciable `x[a]` as integer use  `set_integer(x[a])`

# ### Model's constraint definition

# 9) Define the `resLimit[r in 1:nR]` family of contraints, such that when you sum `coef[r,a]*x[a]` for all the `1:nA` activities you must have a value not greater than `resources.initial[r]`

# ### Objective definition

# 10) Define the objective as the maximisation of the profit given by summing for each of the `1:nA` activities `activities.gm[a] * x[a]`

# ### Model resolution
# 11) [OPTIONAL] Print the model to check it
# 12) Optimize the model
# 13) Check with the function `status = termination_status(profitModel)` that the status is `OPTIMAL` (it should be!)

# ### Print optimal level of activities

# 14) Run the following code to print the results (optimal level of activities)

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


# 15) [OPTIONAL] Observe the emergence of scale effects
# Optionally re-run the model with `initial2` initial resources (instead of `initial`) and notice how this larger company can afford to perform different types of activities (logging high forest instead of coppices in this example) and obtain a better profitability per unit of resource employed.
# Can you guess which are the aspects of the optimisation model that allow for the emergence of these scale effects?
