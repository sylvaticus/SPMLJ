################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################


# # 0202 - Further Topics 

# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")   
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will hower be lost)
# Pkg.resolve()   
# Pkg.instantiate() # run this if you didn't in Segment 02.01
using Random
Random.seed!(123)


# ## Plotting

# Within the many possible package to plot in Julia we use here the `Plots` package
# Plotting in Julia is a topic where a single package has not yet become a de-facto standard. There are really _many_ implementations  