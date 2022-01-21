println("I am in includedfoo.jl")

module foo

println("I am in module foo in includedfoo.jl")
export x

const x=1
const y=2
z() = println("I am in a function of module foo")

end

module foo2

println("I am in module foo2 in includedfoo.jl")
export a

const a=1
const b=2
c() = println("I am in a function of module foo2")

end

