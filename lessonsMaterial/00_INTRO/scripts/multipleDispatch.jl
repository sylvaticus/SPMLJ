
X = [1 2 3; 4 5 6]
Y = [[1,2,3] [4,5,6]]

# Low level function:
mydivide(x,y) = x/y
mydivide(x::Int64,y::Int64) = x%y
mydivide(x,y::Matrix) = x * y^(-1)

8%3 

# High level function: doesn't need to care about types, as long the low level functions work for them
function foo(x,y)
     z = x*y
     mydivide(5,z)
end

foo(2,3)
foo(3,1.5)
foo(X,Y)

# When introducing a new type I can care only to the (small) low level aspects, not to rebuild the whole API


