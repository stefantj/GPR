#!/usr/bin/julia

include("GPR.jl")
using GPR

# Model
noisevar = 0.01
bandwidth = 0.1
k_SE = GPR.SquaredExponential(bandwidth)

model = GPR.GaussianProcess(noisevar, k_SE)

T = 500

estimator = GPR.GaussianProcessEstimate(model, 1)
err1 = zeros(T);

println("Testing scalar case")
for iter = 1:T
	x = 0*rand(1)
	y = x + sqrt(noisevar)*randn()
        t2 = GPR.predict(estimator, x)
	err1[iter] = GPR.update(estimator, x, y[1])
end
println("Done.\n Testing vector case");

estimator2 = GPR.GaussianProcessEstimate(model, 2);
err2 = zeros(T)

for iter = 1:500
    x = rand(2);
    y = x[1] + sqrt(noisevar)*randn()
    y_hat = GPR.predict(estimator2, x);
    GPR.update(estimator2, x,y[1])
    err2[iter] = norm(x[1] - y_hat[1])
end

println("Done.\n Testing sample function:")
x = zeros(2,10000)
iter =0;
for i = 1:100
    for j = 1:100
        iter += 1
        x[1,iter] = i;
        x[2,iter] = j;
    end
end

@time GPR.sample_n(estimator2, x)
