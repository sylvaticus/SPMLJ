cd(@__DIR__)         
using Pkg             
Pkg.activate(".") 
using Agents # bring package into scope
using Random
using Plots

# make an agent type appropriate to this space and with the
# properties we want based on the ABM we will simulate
@agent struct SchelAgent(GridAgent{2}) # inherit all properties of `GridAgent{2}`
    mood::Bool = false # all agents are sad by default :'(
    group::Int # the group does not have a default value!
end

# define the evolution rule: a function that acts once per step on
# all activated agents (acts in-place on the given agent)
function schelling_step!(agent, model)
    # Here we access a model-level property `min_to_be_happy`
    # This will have an assigned value once we create the model
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
    # For each neighbor, get group and compare to current agent's group
    # and increment `count_neighbors_same_group` as appropriately.
    # Here `nearby_agents` (with default arguments) will provide an iterator
    # over the nearby agents one grid cell away, which are at most 8.
    for neighbor in nearby_agents(agent, model, 5)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    # After counting the neighbors, decide whether or not to move the agent.
    # If `count_neighbors_same_group` is at least min_to_be_happy, set the
    # mood to true. Otherwise, move the agent to a random position, and set
    # mood to false.
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end

function initialize(; total_agents = 32000, gridsize = (200, 200), min_to_be_happy = 40, seed = 125)
    space = GridSpaceSingle(gridsize; periodic = false)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Xoshiro(seed)
    model = StandardABM(
        SchelAgent, space;
        agent_step! = schelling_step!, properties, rng,
        container = Vector, # agents are not removed, so we us this
        scheduler = Schedulers.Randomly() # all agents are activated once at random
    )
    # populate the model with agents, adding equal amount of the two types of agents
    # at random positions in the model. At the start all agents are unhappy.
    for n in 1:total_agents
        add_agent_single!(model; mood = false, group = n < total_agents / 2 ? 1 : 2)
    end
    return model
end

schelling = initialize()
adata = [:pos, :mood, :group]
mypal = [:white,:red,:blue]
happys_by_step = zeros(21)


# Print initial state
Tdf, mdf = run!(schelling, 0; adata)  # run for 0 step just to get initial data
printable_grid = zeros(Int64, 200,200)
tdf = Tdf[Tdf.time .== 0, :]
for r in eachrow(tdf)
    printable_grid[r.pos[1],r.pos[2]] = r.group
end
happys_by_step[1] = sum(tdf.mood)/length(tdf.mood)
heatmap(printable_grid, legend=nothing, title="START", color=mypal,aspect_ratio=1, size=(600,600))

# Iterate and print each iteration 
for t in 1:20
    Tdf, mdf = run!(schelling, 1; adata)  # run for 1 step
    tdf = Tdf[Tdf.time .== t, :]
    printable_grid = zeros(Int64, 200,200)
    for r in eachrow(tdf)
        printable_grid[r.pos[1],r.pos[2]] = r.group
    end
    happys_by_step[t+1] = sum(tdf.mood)/length(tdf.mood)
    plot = heatmap(printable_grid, legend=nothing, title="Iteration $t", color=mypal,aspect_ratio=1, size=(600,600))
    display(plot)
end

plot(happys_by_step,title="Share of happy agents by iteration")


