################################################################################
###  Introduction to Scientific Programming and Machine Learning with Julia  ###
###                                                                          ###
### Run each script on a new clean Julia session                             ###
### GitHub: https://github.com/sylvaticus/IntroSPMLJuliaCourse               ###
### Licence (apply to all material of the course: scripts, videos, quizes,..)###
### Creative Commons By Attribution (CC BY 4.0), Antonello Lobianco          ###
################################################################################

# # 0106 Further Topics


# ## Some stuff to set-up the environment..

cd(@__DIR__)         
using Pkg             
Pkg.activate(".")     
# If using a Julia version different than 1.7 please uncomment and run the following line (reproductibility guarantee will however be lost)
# Pkg.resolve()   
# Pkg.instantiate()
using Random
Random.seed!(123)
using InteractiveUtils # loaded automatically when working... interactively

# ## Metaprogramming

# ## Interfacing with other languages

# ## Some performance tips

