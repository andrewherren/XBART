###############################################################################
# Simple demo for XBCF discrete
###############################################################################

set.seed(4321)

# Load libraries
library(XBART)
library(dbarts)

# Data size parameters
n <- 500
p <- 6
snr <- 0.25
ate_true <- 0.5
estimated_propensities <- 0
n_prop_submodels <- 5

# Covariates
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.5)
x3 <- sample(1:3, n, replace = TRUE, prob = c(0.3, 0.3, 0.3))
x4 <- rnorm(n)
x5 <- rbinom(n, 1, 0.5)

if (p > 5){
    x_rest <- matrix(rnorm(n*(p-5)), ncol = p-5)
    colnames(x_rest) <- paste0("x", 6:p)
    x <- cbind(x1, x2, x3, x4, x5, x_rest)
} else {
    x <- cbind(x1, x2, x3, x4, x5)
}

# Prognostic function
mu <- function(x) {
    lev <- c(-4, 4, 0)
    result <- 1 + 2. * x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + lev[x[,3]]
    return(result)
}

# Treatment effect
tau <- ate_true + 0.5 * x[, 1] + 0.5 * (2 * x[, 2] - 1)

# Propensity score
pi <- pnorm(-mean(mu(x)) + mu(x) - 2. * (2*x[, 5] - 1) + 2. * x[, 4], 0, 3)

# Propensity submodels
# 1. Oracle instrument-free propensity
grid_size <- 1000
pi_no_instr <- rep(0, nrow(x))
lev <- c(-4, 4, 0)
for (k in 1:grid_size){
    temp_x <- x
    temp_x[,4] <- rnorm(n)
    temp_x[,5] <- rbinom(n,1,0.5)
    pi_no_instr <- pi_no_instr + pnorm(-mean(mu(temp_x)) + mu(temp_x) - 2. * (2*temp_x[, 5] - 1) + 2. * temp_x[, 4], 0, 3)
}
pi_no_instr <- pi_no_instr/grid_size

# 2. Randomly-chosen submodels
pi.subset <- matrix(NA, nrow = n, ncol = n_prop_submodels)
for (j in 1:n_prop_submodels){
    # Select a subset of covariates
    # First, choose the number of covariates to draw
    num_covariates_subset <- sample(1:(ncol(x)-1), size = 1)
    # Then, choose the covariates
    covariates_sampled <- sort(sample(1:ncol(x), size = num_covariates_subset, replace = F))
    # Integrate out all variables not selected
    pi_temp <- rep(0, nrow(x))
    lev <- c(-4, 4, 0)
    for (k in 1:grid_size){
        temp_x <- x
        if (!(1 %in% covariates_sampled)){
            temp_x[,1] <- rnorm(n)
        }
        if (!(2 %in% covariates_sampled)){
            temp_x[,2] <- rbinom(n, 1, 0.5)
        }
        if (!(3 %in% covariates_sampled)){
            temp_x[,3] <- sample(1:3, n, replace = TRUE, prob = c(0.3, 0.3, 0.3))
        }
        if (!(4 %in% covariates_sampled)){
            temp_x[,4] <- rnorm(n)
        }
        if (!(5 %in% covariates_sampled)){
            temp_x[,5] <- rbinom(n, 1, 0.5)
        }
        pi_temp <- pi_temp + pnorm(-mean(mu(temp_x)) + mu(temp_x) - 2. * (2*temp_x[, 5] - 1) + 2. * temp_x[, 4], 0, 3)
    }
    pi_temp <- pi_temp/grid_size
    pi.subset[,j] <- pi_temp
}

# Arrange covariates according to the way XBCF distinguishes continuous and categorical features
p_categorical_x <- 5
x_orig <- x
x_transformed <- data.frame(x)
x_transformed[, 3] <- as.factor(x_transformed[, 3])
x_transformed <- makeModelMatrixFromDataFrame(data.frame(x_transformed))
cat_inds <- c(2, 3, 4, 5, 7)
x_transformed <- cbind(x_transformed[, -cat_inds], x_transformed[, cat_inds])
categorical_index <- c(rep(0, ncol(x_transformed)-length(cat_inds)), rep(1, length(cat_inds)))


# Treatment assignment
# hist(pi,100)
z <- rbinom(n, 1, pi)

# Outcome variable
mu_x <- mu(x)
Ey <- mu_x + tau * z
sig <- snr * sd(Ey)
y <- Ey + sig * rnorm(n)

# Define pi(X) covariates for prognostic and treatment models (the same in this case)
# pi_x_con = cbind(pi, pi.subset)
# pi_x_mod <- cbind(pi, pi.subset)
# pi_in_use <- pi
pi_in_use_xbcf <- pi

# Don't residualize y
y_xbcf <- y

# Define covariates for prognostic and treatment models (the same in this case)
x_con <- x_transformed
x_mod <- x_transformed

# Run `num_sweeps` of the algorithm
xbcf.fit <- XBART::XBCF.discrete(
    y = y_xbcf, Z = z, X_con = x_con, X_mod = x_mod, pihat = pi_in_use_xbcf,
    p_categorical_con = p_categorical_x, p_categorical_mod = p_categorical_x,
    num_sweeps = 60, burnin = 30, parallel = F, num_cutpoints = n, 
    include_pi = "both", verbose = F, num_trees_con = 30, num_trees_mod = 10, 
    alpha_con = 0.95, beta_con = 1.25, alpha_mod = 0.25, beta_mod = 3, 
    a_scaling = T, b_scaling = T, Nmin = 1, random_seed = 4321
)

# Compute tauhat(X)
pred_xbcf <- predict(xbcf.fit, X_con = x_con, X_mod = x_mod, Z = z,
                     pihat = pi_in_use_xbcf, burnin = 30,
                     include_pi = "both")
tauhats_xbcf <- pred_xbcf$tau.adj.mean

# # Plot
# plot(tau, tauhats_xbcf); abline(0, 1)
# plot(mu_x, rowMeans(pred_xbcf$mu.adj)); abline(0, 1)
