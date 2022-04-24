
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
Pkg.instantiate()
#Pkg.add(["Documenter", "Literate", "Glob", "DataFrames", "OdsIO"])

using Documenter, Literate, Test, Glob, DataFrames, OdsIO


const LESSONS_ROOTDIR = joinpath(@__DIR__, "lessonsSources")
# Important: If some lesson is removed but the md file is left, this may still be used by Documenter

const LESSONS_ROOTDIR_TMP = joinpath(@__DIR__, "lessonsSources_tmp")
# Where to save the lessons before they are preprocessed


LESSONS_SUBDIR = Dict(
  "INTRO - Introduction to the course, Julia and ML"  => "00_-_INTRO_-_Introduction_julia_ml",
  #"JULIA1 - Basic Julia programming"           => "01_-_JULIA1_-_Basic_Julia_programming",
  #"JULIA2 - Scientific programming with Julia" => #"02_-_JULIA2_-_Scientific_programming_with_Julia",
  #"ML1 - Introduction to Machine Learning"     => "03_-_ML1_-_Introduction_to_Machine_Learning",
  #"NN - Neural Networks"                      => "04_-_NN_-_Neural_Networks",
  #"DT - Decision trees based algorithms"     => "05_-_DT_-_Decision_trees_based_algorithms"
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
        displayFilename = replace(file,".md"=>"")
        displayFilename = replace(displayFilename,"_"=>" ")
        push!(dirArray,displayFilename=>joinpath(lessonDir,file))
      end
      push!(outArray,lessonName=>dirArray)
    end
    return outArray
end

"""
    rdir(string,match)

Return a vector of all files (full paths) of a given directory recursivelly taht matches `match`, recursivelly.

# example
filenames = getindex.(splitdir.(rdir(LESSONS_ROOTDIR,"*.jl")),2) #get all md filenames
"""
function rdir(dir::AbstractString, pat::Glob.FilenameMatch)
    result = String[]
    for (root, dirs, files) in walkdir(dir)
        append!(result, filter!(f -> occursin(pat, f), joinpath.(root, files)))
    end
    return result
end
rdir(dir::AbstractString, pat::AbstractString) = rdir(dir, Glob.FilenameMatch(pat))


function literate_directory(dir)
    # Removing old compiled md files...
    #for filename in filter(file -> endswith(file, ".md"), readdir(dir))
    #    rm(joinpath(dir,filename))
    #end

    for filename in filter(file -> endswith(file, ".jl"), readdir(dir))
        filenameNoPath = filename
        filename = joinpath(dir,filename)
        # if the md file exist, let's delete it first...
        filenameMD = replace(filename,".jl" => ".md")
        if isfile(filenameMD)
            rm(filenameMD)
        end

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

function preprocess(rootDir)
    cd(@__DIR__)
    Pkg.activate(".")
    #rootDir = LESSONS_ROOTDIR 
    files  = rdir(rootDir,"*.md")
    videos = ods_read(joinpath(@__DIR__,"videosList.ods");sheetName="videos",retType="DataFrame")
    for file in files
        #file = files[4]
        origContent = read(file,String)
        outContent = ""
        filename = splitdir(file)[2]
        segmentVideos = videos[videos.host_filename .== filename,:]
        if size(segmentVideos,1) > 0
            outContent *= """
                        ```@raw html
                        <div id="ytb-videos">
                        <span style=font-weight:bold;>Videos related to this segment (click the title to watch)</span>
                        """
            for video in eachrow(segmentVideos)
                #video = segmentVideos[1,:]
                outContent *= """
                    <details><summary>$(video.lesson_short_name) - $(video.segment_id)$(video.part_id): $(video.part_name) ($(video.minutes):$(video.seconds))</summary>
                    <div class="container ytb-container">
                        <div class="embed-responsive embed-responsive-16by9">
                            <iframe class="embed-responsive-item" src="https://www.youtube.com/embed/$(video.vid)" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen="" frameborder="0"></iframe>
                        </div>
                    </div>
                    </details>
                    """
            end # end of each video
            outContent *= """
                </div>
                ```
                ------
                """
        end # end of if there are videos
        outContent *= origContent
        if (filename != "index.md")
            commentCode = """
                ```@raw html
                <script src="https://utteranc.es/client.js"
                        repo="sylvaticus/SPMLJ"
                        issue-term="title"
                        label="💬 website_comment"
                        theme="github-dark"
                        crossorigin="anonymous"
                        async>
                </script>
                ```
                """
            addThisCode1 = """
                ```@raw html
                <div class="addthis_inline_share_toolbox"></div>
                ```
                """
            addThisCode2 = """
                ```@raw html
                <!-- Go to www.addthis.com/dashboard to customize your tools -->
                <script type="text/javascript" src="//s7.addthis.com/js/300/addthis_widget.js#pubid=ra-6256c971c4f745bc"></script>
                ```
                """
            # https://crowdsignal.com/support/rating-widget/
            ratingCode1 = """
                ```@raw html
                <div id="pd_rating_holder_8962705"></div>
                <script type="text/javascript">
                PDRTJS_settings_8962705 = {
                "id" : "8962705",
                "unique_id" : "$(file)",
                "title" : "$(filename)",
                "permalink" : ""
                };
                </script>
                ```
                """
            ratingCode2 = """
                ```@raw html
                <script type="text/javascript" charset="utf-8" src="https://polldaddy.com/js/rating/rating.js"></script>
                ```
                """
            outContent *= "\n---------\n"
            outContent *= ratingCode1
            outContent *= addThisCode1
            outContent *= "\n---------\n"
            outContent *= commentCode
            outContent *= ratingCode2
            outContent *= addThisCode2
        end
        write(file,outContent)
    end # end for each file
end #end preprocess function

# ------------------------------------------------------------------------------
# Saving the unmodified source to a temp directory
cp(LESSONS_ROOTDIR, LESSONS_ROOTDIR_TMP; force=true)

println("Starting literating tutorials (.jl --> .md)...")
literate_directory.(map(lsubdir->joinpath(LESSONS_ROOTDIR ,lsubdir),values(LESSONS_SUBDIR)))

println("Starting preprocessing markdown pages...")
preprocess(LESSONS_ROOTDIR)

println("Starting making the documentation...")
makedocs(sitename="SPMLJ",
         authors = "Antonello Lobianco",
         pages = [
            "Index" => "index.md",
            "Lessons" => makeList(LESSONS_ROOTDIR,LESSONS_SUBDIR),
         ],
         format = Documenter.HTML(
             prettyurls = false,
             analytics = "G-Q39LHCRBB6",
             assets = ["assets/custom.css"],
             ),
         #strict = true,
         #doctest = false
         source  = "lessonsSources", # Attention here !!!!!!!!!!!
         build   = "buildedDoc",
         #preprocess = preprocess
)

# Copying back the unmodified source
cp(LESSONS_ROOTDIR_TMP, LESSONS_ROOTDIR; force=true)

println("Starting deploying the documentation...")
deploydocs(
    repo = "github.com/sylvaticus/SPMLJ.git",
    devbranch = "main",
    target = "buildedDoc"
)
