
# To build the documentation:
#    - julia --project="." make.jl
#    - empty!(ARGS); include("make.jl")
# To build the documentation without running the tests:
#    - julia --project="." make.jl preview
#    - push!(ARGS,"preview"); include("make.jl")

# !!! note "An optional title"
#    4 spaces idented
# note, tip, warning, danger, compat



# Format notes:


# # A markdown H1 title
# A non-code markdown normal line

## A comment within the code chunk

#src: line exclusive to the source code and thus filtered out unconditionally

using Pkg
cd(@__DIR__)
Pkg.activate(".")

#Pkg.resolve()
#Pkg.instantiate()
#Pkg.add(["Documenter", "Literate"])

using Documenter, Literate, Test


#push!(LOAD_PATH,"./lessonsSources/")


const LESSONS_ROOTDIR = joinpath(@__DIR__, "lessonsSources")
# Important: If some lesson is removed but the md file is left, this may still be used by Documenter


LESSONS_SUBDIR = Dict(
  "KOM - Kick off meeting"                     => "00_-_KOM_-_Kickoff_meeting",
  "JULIA1 - Basic Julia Programming"           => "01_-_JULIA1_-_Basic_Julia_programming",
  "JULIA2_-_Scientific_programming_with_Julia" => "02_-_JULIA2_-_Scientific_programming_with_Julia",
  #"ML1 - Introduction to Machine Learning_1"   => "03_-_ML1_-_Introduction_to_Machine_Learning_1",
  #"ML2_-_Introduction_to_Machine_Learning_2"   => "04_-_ML2_-_Introduction_to_Machine_Learning_2",
  #"NN_-_Neural_Networks"                       => "05_-_NN_-_Neural_Networks",
  #"RF&CL_-_Random_Forests_and_Clustering"      => "06_-_RF&CL_-_Random_Forests_and_Clustering"
  )



# Utility functions.....

function link_example(content)
    edit_url = match(r"EditURL = \"(.+?)\"", content)[1]
    footer = match(r"^(---\n\n\*This page was generated using)"m, content)[1]
    content = replace(
        content, footer => "[View this file on Github]($(edit_url)).\n\n" * footer
    )
    return content
end


"""
    include_sandbox(filename)
Include the `filename` in a temporary module that acts as a sandbox. (Ensuring
no constants or functions leak into other files.)
"""
function include_sandbox(filename)
    mod = @eval module $(gensym()) end
    return Base.include(mod, filename)
end

function makeList(rootDir,subDirList)
    outArray = []
    for l in sort(collect(subDirList), by=x->x[2])
      #println(l)
      lessonName = l[1]
      lessonDir  = l[2]
      lessonName  = replace(lessonName,"_"=>" ")
      dirArray =[]
      for file in filter(file -> endswith(file, ".md"), sort(readdir(joinpath(rootDir,lessonDir))))
        displayFilename = replace(file,".md"=>"","_"=>" ")
        push!(dirArray,displayFilename=>joinpath(lessonDir,file))
      end
      push!(outArray,lessonName=>dirArray)
    end
    return outArray
end

function literate_directory(dir)
    # Removing old compiled md files...
    for filename in filter(file -> endswith(file, ".md"), readdir(dir))
        rm(joinpath(dir,filename))
    end

    for filename in filter(file -> endswith(file, ".jl"), readdir(dir))
        filenameNoPath = filename
        filename = joinpath(dir,filename)
        # `include` the file to test it before `#src` lines are removed. It is
        # in a testset to isolate local variables between files.
        if ! ("preview" in ARGS)
            @testset "$(filename)" begin
                println(filename)
                include_sandbox(filename)
             end
             Literate.markdown(
                 filename,
                 dir;
                 documenter = true,
                 postprocess = link_example,
                 # default is @example -> evaluated by documenter at the end of the block
                 codefence =  "```@repl $filenameNoPath" => "```" 
             )
        else
            Literate.markdown(
                filename,
                dir;
                documenter = true,
                postprocess = link_example,
                codefence =  "```text" => "```"
            )
        end
    end
    return nothing
end

println("Starting literating tutorials (.jl --> .md)...")
literate_directory.(map(lsubdir->joinpath(LESSONS_ROOTDIR ,lsubdir),values(LESSONS_SUBDIR)))

println("Starting making the documentation...")
makedocs(sitename="Introduction to Scientific Programming and Machine Learning with Julia",
         authors = "Antonello Lobianco",
         pages = [
            "Index" => "index.md",
            "Lessons" => makeList(LESSONS_ROOTDIR,LESSONS_SUBDIR),
         ],
         format = Documenter.HTML(prettyurls = false),
         #strict = true,
         #doctest = false
         analytics = "G-JYKX8QY5JW",
         source  = "lessonsSources",
         build   = "buildedDoc",
)


println("Starting deploying the documentation...")
deploydocs(
    repo = "github.com/sylvaticus/IntroSPMLJuliaCourse.git",
    devbranch = "main"
)
