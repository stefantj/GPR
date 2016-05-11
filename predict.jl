# predict.jl
#
# Predicts the value of the Gaussian process using Gaussian process regression.

# Fix this to be just the 

# Prior
type GaussianProcess
    noise::Float64
    kernel::CovarianceKernel
end

# Estimator
type GaussianProcessEstimate
    prior::GaussianProcess
    weights::Vector{Float64}
    centers::Array{Float64,2}
    Q_matrix::Matrix{Float64}
    numcenters

    # Constructor takes model and data size as arguments.
    GaussianProcessEstimate(gp::GaussianProcess, L::Integer) = new(gp, zeros(1), zeros(L,1), zeros(1,1), 0);
end

# Predict at point x
function predict(gp::GaussianProcessEstimate,
                 x)

    if(gp.numcenters==0)
        return 0.,covar(gp.prior.kernel, x,x)
    else
        x = float(x)
        h = zeros(gp.numcenters,1)
        for i=1:gp.numcenters
            h[i] = covar(gp.prior.kernel, x, gp.centers[:,i])
        end
        z = gp.Q_matrix*h
        var = gp.prior.noise + covar(gp.prior.kernel, x, x) - z'*h
        mean = h'*gp.weights
        if(var[1] < 0)
            println("Error: Negative variance!")
        end
        return (mean[1], var[1])
    end
end

# Forms upper confidence bound at specified points
function form_ucb(gp::GaussianProcessEstimate, x::Array{Float64,2}, beta::Float64)
    num_points = size(x,2)
    if(gp.numcenters==0)
        return sqrt(beta)*ones(num_points)
    else
        x = float(x)
        ucb = zeros(num_points)
        srb = sqrt(beta)
        sigma_0 = gp.prior.noise + 1 # Assuming that kernel is normalized. Otherwise replace 1 with covar(gp.prior.kernel, x[:,k],x[:,k])
        # This is devectorized for speed. The computation is z = h'Qh where h is the kernel vector
        for k = 1:num_points
            h_k = zeros(gp.numcenters)
            for i=1:gp.numcenters
                h_k[i] = covar(gp.prior.kernel, x[:,k],gp.centers[:,i])
            end
            var_k = ((h_k'*gp.Q_matrix)*h_k)[1]
            mean_k = (h_k'*gp.weights)[1]
            ucb[k] = mean_k  + srb*sqrt(sigma_0 - var_k)
        end
        return ucb
    end
end

# Predict at point x
function predict_var(gp::GaussianProcessEstimate,
                 x::Vector{Float64})

    if(gp.numcenters==0)
        return 0.
    else
        x = float(x)
        h = zeros(gp.numcenters,1)
        for i=1:gp.numcenters
            h[i] = covar(gp.prior.kernel, x, gp.centers[:,i])
        end
        z = gp.Q_matrix*h
        var = gp.prior.noise + covar(gp.prior.kernel, x, x) - z'*h
        if(var[1] < 0)
            println("Error: Negative variance!")
        end
        return (var[1])
    end
end

# Predict at vector of points x
function predict_mean(gp::GaussianProcessEstimate,
                 x::Array{Float64,2})

    # Assume that it's stored with points as columns
    if(gp.numcenters==0)
        return zeros(size(x,2))
    else
        num_points = size(x, 2);
        
        x = float(x)
        H = zeros(gp.numcenters,num_points)
        for k = 1:num_points
            for i=1:gp.numcenters
                H[i,k] = covar(gp.prior.kernel, x[:,k], gp.centers[:,i])
            end
        end
        mean = H'*gp.weights
        return mean
    end
end

# Predict at point x
function predict_mean(gp::GaussianProcessEstimate,
                 x::Vector{Float64})
    if(gp.numcenters==0)
        return 0.
    else
        x = float(x)
        h = zeros(gp.numcenters,1)
        for i=1:gp.numcenters
            h[i] = covar(gp.prior.kernel, x, gp.centers[:,i])
        end
        mean = h'*gp.weights
        return mean[1]
    end
end



# Sample n points from posterior distribution
function sample_n(gp::GaussianProcessEstimate,
                  X::Vector{Vector{Float64}})
    X_mat = zeros(length(X),2)
    indx = 0;
    for x in X
      indx+=1;
      X_mat[indx, 1] = x[1]
      X_mat[indx, 2] = x[2]
    end
    return sample_n(gp, X_mat')
end

function sample_n(gp::GaussianProcessEstimate,
                       X::Matrix{Float64})
    n_X = size(X,2); # Number of entries. If scalars, X must be a row vector.
    mean = zeros(n_X)
    cov_mat = eye(n_X)

    if(gp.numcenters != 0) # generate based on data
        H_x = zeros(n_X, gp.numcenters);
        K_x = zeros(n_X, n_X)

        for i=1:n_X
            for j = 1:gp.numcenters
                H_x[i, j] = covar(gp.prior.kernel, X[:,i], gp.centers[:,j])
            end
            for j = i:n_X
                K_x[i,j] = covar(gp.prior.kernel, X[:,i], X[:,j])
            end
        end

    # statistics
        mean = H_x*gp.weights
        cov_mat = K_x - H_x*(gp.Q_matrix*(H_x'));
    
    else # dealing with the prior
        for i = 1:n_X
            for j = i:n_X
                c_ij = covar(gp.prior.kernel, X[:,i], X[:,j])
                cov_mat[i,j] = c_ij
                cov_mat[j,i] = c_ij
            end
        end

    end

#    println(cov_mat+ 1e-3*eye(n_X))

        f = mean + chol(cov_mat)*randn(n_X,1);


        if(norm(real(f)) < norm(abs(f)) - 0.01)
            warn("Negative covariance matrix gave imaginary prediction $f");
            warn("mean = $mean");
            warn("Cov = $cov_mat")
        end

        return real(f)
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
        gp.Q_matrix[1,1] = 1.0/(gp.prior.noise + kernel_norm)
        gp.weights[1] = gp.Q_matrix[1,1]*y
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

        # Compute error
        error = y - gp.weights'*h
        z = gp.Q_matrix*h
        
        gp.centers = [gp.centers x]
        gp.numcenters += 1
        numcenters += 1

        # r_i = 1/r
        r_i = 1./(gp.prior.noise + kernel_norm - z'*h)
	    r_i = r_i[1] #Convert back to scalar

        # Rank-1 update of Q_i:
        #  Q_i = [ Q_{i-1}+zz'/r  -z/r
        #             z'/r         1/r]
        #
        # Q_i is numcenters X numcenters
        # Q_{i-1} is (numcenters-1) X (numcenters-1)
        #
        
        # Room for improvement here
        # The stronger you type the data, the faster you'll go.
        # Symmetric rank 1 update: Q_{i-1} += zz'/r
        gp.Q_matrix += z*z'*r_i
        # Concatenate
        gp.Q_matrix =[gp.Q_matrix -z*r_i;
                       -z'*r_i     r_i]

        # Update weights
	    gp.weights -= vec(z.*(error*r_i))
	    gp.weights = [gp.weights; vec(error*r_i)]

        # If there was a repeat, collapse Q, a to their equivalent n-1 size
        if(true && repeat > 0)
            # Update weights
            gp.weights[repeat] += gp.weights[numcenters];
            splice!(gp.weights, numcenters);

            # Update Q_matrix
            gp.Q_matrix[:,repeat] += gp.Q_matrix[:,numcenters]
            gp.Q_matrix[repeat,:] += gp.Q_matrix[numcenters,:]
            gp.Q_matrix = gp.Q_matrix[1:numcenters-1,1:numcenters-1]

            # Update centers
            gp.centers=gp.centers[:,1:numcenters-1]
            gp.numcenters -= 1
        end
    end
    return error[1]
end
