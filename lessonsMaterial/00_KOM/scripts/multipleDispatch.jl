

X = [1 2 3; 4 5 6]
Y =  [[1,2,3] [4,5,6]]

mydivide(x,y) = x/y
mydivide(x::Int64,y::Int64) = x%y
mydivide(x,y::Matrix) = x * y^(-1)

8%3 
function foo(x,y)
     z = x*y
     mydivide(5,z)
end

foo(2,3)
foo(3,1.5)
foo(X,Y)

mydivide([1 2; 3 4], [5 6; 7 8])
mydivide(2, [5 6; 7 8])

