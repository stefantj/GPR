# predict.jl
#
# Predicts the value of the Gaussian process using Gaussian process regression.
# The basic algorithm is presented in Principe, Liu "Kernel Adaptive Filtering"
# We implement a square-root version of the algorithm which is better conditioned.
#
# The algorithm: 
# Variables: 
#	x		Measurement location
#	y		Measurement value
#	σ^2		Noise variance
#	κ(x1,x2)	Kernel (covariance) function
#	Q_sqrt		Square-root of the kalman gain matrix
#
# Init(x0,y0)
#	Q_sqrt = 1/sqrt(σ^2 + κ(x0,x0))
#	a = Q_sqrt^2*y0
#
# Update(x_i, y_i)
#	h_i = [κ(x_i,x_0), … κ(x_i, x_{i-1})]^T
#	q_i = Q_sqrt^T*h_i
#	r_i = sqrt(σ^2 + κ(x_i,x_i) - normsq(q_i))
#	Q_sqrt_i = [ Q_sqrt_{i-1}	-Q_sqrt_{i-1}q_i/r_i
#		 	0		     1/r_i	]
#	e_i = d_i - h_i^Ta_{i-1}
#	a_i = [a_{i-1} - Q_sqrt_{i-1}q_ie_i/(r_i^2)
#		e_i/(r_i^2)			]
#
#
# Predict(x)
#	h_x = [κ(x, x_0), …, κ(x,x_i)]
#	μ_x = h_x^Ta_i
#	σ^2_x = σ^2 + κ(x,x) - h_x^T Q_sqrt_i^T Q_sqrt_i h_x
#

# Prior
type GaussianProcess
    noise::Float64
    kernel::CovarianceKernel
end

# Estimator
type GaussianProcessEstimate
    prior::GaussianProcess    # Gaussian Process prior
    weights::Vector{Float64}  # weights for centers (a(i) in Liu et al.)
    centers::Array{Float64,2} # locations of data points
    Q_sqrt::Matrix{Float64}   # sqrt of kalman gain
    numcenters                # Number of data samples in estimator

    # Constructor takes model and data dimension as arguments.
    GaussianProcessEstimate(gp::GaussianProcess, L::Integer) = new(gp, zeros(1), zeros(L,1), zeros(1,1), 0);
end

# Predict at point x
function predict(gp::GaussianProcessEstimate,
                 x)

    if(gp.numcenters==0)
        return 0.,NaN;
    else
        x = float(x)
        h = zeros(gp.numcenters,1)
        for i=1:gp.numcenters
            h[i] = covar(gp.prior.kernel, x, gp.centers[:,i])
        end
        z = (gp.Q_sqrt*h)
        var = gp.prior.noise + covar(gp.prior.kernel, x, x) - z'*z
        mean = h'*gp.weights
        if(var[1] < 0)
            println("Error: Negative variance!")
        end
        return (mean[1], var[1])
    end
end

# Sample n points from posterior distribution
function sample_n(gp::GaussianProcessEstimate,
                       X::Matrix{Float64})
    n_X = size(X,2); # Number of entries. If scalars, X must be a row vector.
    μ = zeros(n_X);
    σ = zeros(n_X);
    for i=1:n_X
        μ[n_X],σ[n_X] = predict(gp,X[i]);
    end

    # Is this correct?
    f = μ + sqrt(σ).*rand(n_X);
    return f
end



# Update recursively using square root
# (suitable for on-line use)
# x is the location
# y is the measurement
function update(gp::GaussianProcessEstimate,
                x,
                y::Float64)
    numcenters = gp.numcenters
    error = 0.0
    # In case the kernel isn't normalized
    kernel_norm = covar(gp.prior.kernel,x,x);
    x = float(x)

    # Initialize if needed
    if(numcenters == 0)
        tmp = gp.prior.noise + kernel_norm;
        gp.Q_sqrt[1,1] = 1.0/sqrt(tmp)
        gp.weights[1] = tmp*y
        gp.centers[:,1] = x
        gp.numcenters += 1
    else

        # Compute similarities, note any repeats
        h=zeros(numcenters,1)
        repeat = -1
        for i =1:numcenters
            h[i] = covar(gp.prior.kernel,gp.centers[:,i],x)
            if(norm(x-gp.centers[:,i]) < 1e-9)
                repeat = i
            end
        end

        # Helpers for easy notation
        q_i = gp.Q_sqrt'*h
        z_i = gp.Q_sqrt*q_i

        # Compute error
        error = y - gp.weights'*h
        
        gp.centers = [gp.centers x]
        gp.numcenters += 1
        numcenters += 1

        # r_i = 1/r
        r_i = 1./(gp.prior.noise + kernel_norm - q_i'*q_i)
	    r_i = r_i[1] #Convert back to scalar

        # Room for improvement here
        # The stronger you type the data, the faster you'll go.
        gp.Q_sqrt = [ gp.Q_sqrt         z_i/(-1*sqrt(r_i));
                       zeros(1,numcenters-1) 1/sqrt(r_i) ]

        # Update weights
	    gp.weights -= vec(z_i.*(error./r_i))
	    gp.weights = [gp.weights; vec(error./r_i)]

        # If there was a repeat, collapse Q, a to their equivalent n-1 size
        if(false)
            # Update weights
            gp.weights[repeat] += gp.weights[numcenters];
            splice!(gp.weights, numcenters);

            # Update Q_matrix
            gp.Q_sqrt[:,repeat] += gp.Q_sqrt[:,numcenters]
            gp.Q_sqrt[repeat,:] += gp.Q_sqrt[numcenters,:]
            gp.Q_sqrt = gp.Q_sqrt[1:numcenters-1,1:numcenters-1]

            # Update centers
            splice!(gp.centers,numcenters)
            gp.numcenters -= 1
        end
    end
    return error[1]
end
