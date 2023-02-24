###############################################################################
# Debugging script for XBCF with trees that depend only on sets of pi(X)
###############################################################################

# Load Libraries
library(XBART)
library(dbarts)

#### 1. DATA GENERATING PROCESS
n <- 500

# Covariates
x1 <- rnorm(n)
x2 <- rbinom(n, 1, 0.2)
x3 <- sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
x4 <- rnorm(n)
x5 <- rbinom(n, 1, 0.7)
x <- cbind(x1, x2, x3, x4, x5)

# Treatment effect
tau <- 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)

# Prognostic function
mu <- function(x) {
    lev <- c(-0.5, 0.75, 0)
    result <- 1 + x[, 1] * (2 * x[, 2] - 2 * (1 - x[, 2])) + lev[x3]
    return(result)
}

# Propensity scores and treatment assignment
pi <- pnorm(-0.5 + mu(x) - x[, 2] + 0. * x[, 4], 0, 3)
# hist(pi,100)
z <- rbinom(n, 1, pi)

# Outcome variable
mu_x <- mu(x)
Ey <- mu_x + tau * z
sig <- 0.25 * sd(Ey)
y <- Ey + sig * rnorm(n)

# Arrange covariates according to the way XBCF distinguishes continuous and categorical features
x_orig <- x
x <- data.frame(x)
x[, 3] <- as.factor(x[, 3])
x <- makeModelMatrixFromDataFrame(data.frame(x))
x <- cbind(x[, 1], x[, 6], x[, -c(1, 6)])

# Fit the full "pihat" model
prop_model_full <- XBART::XBART.multinomial(z, 2, x)
prop_model_full_predictions <- predict(prop_model_full, X = x)
pihat.full <- prop_model_full_predictions$prob[,2]

# Fit a series of propensity models with a sample of the covariates in x
n_prop_submodels <- 1
pihat.subset <- matrix(NA, nrow = n, ncol = n_prop_submodels)
for (i in 1:n_prop_submodels){
    # Select a subset of covariates
    # First, choose the number of covariates to draw
    num_covariates_subset <- sample(1:(ncol(x)-1), size = 1)
    # Then, choose the covariates
    covariates_sampled <- sample(1:ncol(x), size = num_covariates_subset, replace = F)
    # Subset X to these variables
    x_subset <- x[,covariates_sampled]
    # Fit the XBART multinomial model as above
    prop_model_subset <- XBART::XBART.multinomial(z, 2, x)
    prop_model_subset_predictions <- predict(prop_model_subset, X = x)
    pihat.subset[,i] <- prop_model_subset_predictions$prob[,2]
}

# Define pi(X) covariates for prognostic and treatment models (the same in this case)
pi_x_con = cbind(pihat.full, pihat.subset)
pi_x_mod <- cbind(pihat.full, pihat.subset)

# Define covariates for prognostic and treatment models (the same in this case)
x_con = x
x_mod <- x

#### 2. Model Fitting and Estimation

# Run XBCF with propensity shrinkage
t1 = proc.time()
xbcf.fit.xb <- XBART::XBCF.discrete.propensity.shrinkage(
    y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat.full,
    pi_X_con = pi_x_con, pi_X_mod = pi_x_mod,
    p_categorical_con = 5, p_categorical_mod = 5,
    num_sweeps = 60, burnin = 30, 
)
# xbcf.fit.xb <- XBART::XBCF.discrete(
#     y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pi,
#     p_categorical_con = 5, p_categorical_mod = 5,
#     num_sweeps = 60, burnin = 30
# )
t1 = proc.time() - t1

# Compute tauhat(X)
pred <- predict(xbcf.fit.xb, X_con = x_con, X_mod = x_mod, Z = z, pi_X_con = pi_x_con, pi_X_mod = pi_x_mod, pihat = pihat.full, burnin = 30)
# pred <- predict(xbcf.fit.xb, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat.full, burnin = 30)
tauhats <- pred$tau.adj.mean

# Evaluate RMSE and runtime
print(paste0("XBCF Propensity Shrinkage RMSE: ", sqrt(mean((tauhats - tau)^2))))
print(paste0("XBCF Propensity Shrinkage Runtime: ", round(as.list(t1)$elapsed, 2), " seconds"))

# Plot results
plot(tau, tauhats)
abline(a=0, b=1)
