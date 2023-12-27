using Pkg
cd(@__DIR__)
Pkg.activate(".")
#Pkg.upgrade_manifest()
Pkg.resolve()
Pkg.instantiate()

#Pkg.add("Flux")
#Pkg.add("MLDatasets")
#Pkg.add("BetaML")
#Pkg.add("Images")
#Pkg.add("FileIO")
#Pkg.add("ImageTransformations")
#Pkg.add("MLDatasets")
#Pkg.instantiate()
#Pkg.update()
using Random
Random.seed!(123);
using DelimitedFiles
using Statistics
using Flux
using Flux: Data.DataLoader
using Flux: onehotbatch, onecold, crossentropy
using MLDatasets # For loading the training data
using Images, FileIO, ImageTransformations # For loading the actual images

################################################################################
# Helper functions

function cleanImg!(img,threshold=0.3,radius=0)
    (R,C) = size(img)
    for c in 1:C
        for r in 1:R
            if img[r,c] <= threshold
                allneighmoursunderthreshold = true
                for c2 in max(1,c-radius):min(C,c+radius)
                    for r2 in max(1,r-radius):min(R,r+radius)
                        if img[r2,c2] > threshold
                            allneighmoursunderthreshold = false
                            break
                        end
                    end
                end
                if allneighmoursunderthreshold
                    img[r,c] = Gray(0.0)
                end
            end
        end
    end
    return img
end

accuracy(y,ŷ) =  (mean(onecold(ŷ) .== onecold(y)))
loss(x, y)     = Flux.crossentropy(model(x), y)

################################################################################
# Definition and training of the model
# (including extra images to MNIST db)
x_train, y_train = MLDatasets.MNIST(split=:train)[:]
x_train          = permutedims(x_train,(2,1,3)) # For correct img axis
y_train_add      = convert(Array{Int64,1},dropdims(readdlm("./data/additionalTestingImgs/img_labels.txt"),dims=2))
x_train_add_path = ["./data/additionalTestingImgs/test$(i).png" for i in 1:64]
x_train_add_imgs = load.(x_train_add_path)
x_train_add_imgs = [Gray.(i) for i in x_train_add_imgs]
x_train_add_imgs = [imresize(i, (28,28)) for i in x_train_add_imgs]
x_train_add_imgs = [1.0 .- i for i in x_train_add_imgs]
x_train_add_imgs = cleanImg!.(x_train_add_imgs, 0.3,1)
x_train_add_imgs = convert.(Array{Float32,2},x_train_add_imgs)

resample = 30 # 15
x_train_add_imgs2 = Array{Float32,2}[]
[append!(x_train_add_imgs2,x_train_add_imgs) for i in 1:resample]
x_train = cat(x_train, reshape(reduce(hcat, x_train_add_imgs2), 28, 28, :), dims=3)


y_train_add2 = Int64[]
[append!(y_train_add2,y_train_add) for i in 1:resample]
append!(y_train,y_train_add2)

#x_train_imgs     = convert(Array{Gray{N0f8},3},deepcopy(x_train))
x_train          = reshape(x_train,(28,28,1,60000+64*resample))

y_train          = onehotbatch(y_train, 0:9)
train_data       = DataLoader((x_train, y_train), batchsize=128)
x_test, y_test   = MLDatasets.MNIST(split=:test)[:]
x_test           = permutedims(x_test,(2,1,3)) # For correct img axis
x_test           = reshape(x_test,(28,28,1,10000))
y_test           = onehotbatch(y_test, 0:9)
#train_data       = DataLoader((x_test, y_test), batchsize=128)

model = Chain(
    # 28x28 => 14x14
    Conv((5, 5), 1=>8, pad=2, stride=2, relu),
    # 14x14 => 7x7
    Conv((3, 3), 8=>16, pad=1, stride=2, relu),
    # 7x7 => 4x4
    Conv((3, 3), 16=>32, pad=1, stride=2, relu),
    # 4x4 => 2x2
    Conv((3, 3), 32=>32, pad=1, stride=2, relu),
    # Average pooling on each width x height feature map
    GlobalMeanPool(),
    Flux.flatten,
    Dense(32, 10),
    softmax
)

#=
# alternative model, but slower and less accurate
alt_model = Chain(
  Conv((2,2), 1=>16, relu),
  x -> maxpool(x, (2,2)),
  Conv((2,2), 16=>8, relu),
  x -> maxpool(x, (2,2)),
  x -> reshape(x, :, size(x, 4)),
  Dense(288, 10), softmax
)
=#

opt = Flux.ADAM()
ps  = Flux.params(model)
number_epochs = 5

[(println(e); Flux.train!(loss, ps, train_data, opt)) for e in 1:number_epochs]

accuracy(y_train,model(x_train)) # 0.95
accuracy(y_test, model(x_test) ) # 0.95

################################################################################
# Loading imgs

# Imgs obtained with photo -> Gimp: Colour -> auto white levels then: Levels -> Input levels -> Reduce the value on the left
folder = "class"
imgs_y = convert(Array{Int64,1},dropdims(readdlm("./data/$(folder)/img_labels.txt"),dims=2))
imgs_path = ["./data/$(folder)/img$(i).png" for i in 1:20]
imgs = load.(imgs_path)
imgs = [Gray.(i) for i in imgs]
imgs = [imresize(i, (28,28)) for i in imgs]
imgs = [1.0 .- i for i in imgs]
imgs = cleanImg!.(imgs, 0.3,1)
imgs = cat(imgs...,dims=3)
imgs = reshape(imgs,(28,28,1,size(imgs,3)))


################################################################################
# Doing the actual classification and printing results...

imgs_est = model(imgs)
imgs_ŷ = onecold(imgs_est, 0:9)
probs = maximum(imgs_est,dims=1)

nImgs = length(imgs_y)
println("*** Classification report")
println("Overall accuracy: $(mean(imgs_ŷ .== imgs_y))")
println("")
println("#id \t succ  \t true \t est \t prob")
for i in 1:nImgs
   resultSymbol = imgs_ŷ[i] == imgs_y[i] ? "✔" : "❌"
   println("$i \t $(resultSymbol) \t $(imgs_y[i])    \t $(imgs_ŷ[i])   \t $(probs[i])" )
end
