# covariance.jl
#
# Contains functions for calculating the similarity of data
# Default functions convert everything to float64
#
# Written by Stefan Jorgensen 09/10/15
# Loosely based on github.com/pschulam-attic/GP.jl

## Covariance implementations ##

abstract CovarianceKernel

# Add: Matern, some more from Srinivas
type SquaredExponential <: CovarianceKernel
    bandwidth::Float64
end

# Squared Exponential,
# returns the covariance between x1 and x2
function covar(kernel::SquaredExponential,
                x1,
                x2)
    r=norm(float(x1)-float(x2));
    return exp(-r^2/(2*(kernel.bandwidth^2)))
end

# Evaluates covariance elementwise on an array.
function covar(kernel::SquaredExponential, X::Matrix{Float64})
    sigma = similar(X)
    iter = 0
    [ sigma[iter+=1] = exp(-x^2/(2*(kernel.bandwidth^2))) for x in X]
    return sigma
end

## Condensed covariance function calls ##

# Fills a covariance matrix between collections of objects
function covarMatrix(kernel::CovarianceKernel,
                      v1,
                      v2)			
    n1 = size(v1,1)
    n2 = size(v2,1)
    cv = zeros(n1,n2)
    for i1 =1:n1
        for i2=1:n2
            cv[i1,i2] = covar(kernel, v1[i1],v2[i2]);
        end
    end
    return cv
end

# Combinatoric covariance:
# returns sum of covariances between objects
function covarComb(kernel::CovarianceKernel,
                    v1::Array,
                    v2::Array)
    if(size(v1,1) == size(v2,1))
        c = 0;
        for i = 1:size(v1,1)
            c += covar(kernel, v1[i],v2[i]);
        end
        return c;
    else
        println("Can't compare objects of different size");
    end
end
