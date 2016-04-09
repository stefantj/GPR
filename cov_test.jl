# Load module
include("GPR.jl")
using GPR

# Make covariance
k_SE = GPR.SquaredExponential(1.0)

println("Test with scalar float");
x1 = 1.0
x2 = 0.0
c=GPR.covar(k_SE, x1,x2)
println(c)

println("Test with scalar int")
x1 = 1
x2 = 0
c=GPR.covar(k_SE, x1,x2)
println(c)

println("Test with scalar char")
x1='a'
x2='b'
c=GPR.covar(k_SE, x1,x2)
println(c)

println("Test with vector float")
x1 = [1.0,1.0,1.0]
x2 = [0.0,0.0,0.0]
c=GPR.covar(k_SE, x1,x2)
println(c)

println("Test with vector int")
x1 = [1,1,1]
x2 = [0,0,0]
c=GPR.covar(k_SE, x1,x2)
println(c)

println("Test with vector char")
x1 = ['a','b','c']
x2 = ['d','e','f']
c=GPR.covar(k_SE, x1,x2)
println(c)

println("Test combinatoric covariance")
x1 = [ [1.0 1.0], [1.0 1.0], [3.0 1.0]];
x2=x1;
c=GPR.covarComb(k_SE,x1,x2);
println(c)


# Test prediction:




