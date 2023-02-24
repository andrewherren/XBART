###############################################################################
# Simulation script comparing the default XBCF with a variant that 
# also includes propensity-only trees (idea is that separately regularizing 
# trees that use enough propensity scores and trees that use all covariates 
# might improve estimator RMSE since we can regularize the "necessary 
# deconfounding" propensity trees less than the covariate-based trees).
# 
# NOTE: for debugging purposes, have commented out the calls to XBCF.discrete 
# and predict() since memory access errors appear to be accruing only 
# in XBCF.discrete.propensity.shrinkage
###############################################################################

# Load Libraries
library(XBART)
library(xgboost)
library(dbarts)

#### 1. DATA GENERATING PROCESS
n <- 500
n_sim <- 20
sim_results <- matrix(NA, nrow = n_sim, ncol = 6)
print_during_sim <- F
plot_during_sim <- F
set.seed(1234)

for (i in 1:n_sim){
    # Covariates
    x1 <- rnorm(n)
    x2 <- rbinom(n, 1, 0.2)
    x3 <- sample(1:3, n, replace = TRUE, prob = c(0.1, 0.6, 0.3))
    x4 <- rnorm(n)
    x5 <- rbinom(n, 1, 0.7)
    x <- cbind(x1, x2, x3, x4, x5)
    
    # Treatment effect
    # tau <- 2 + 0.5 * x[, 4] * (2 * x[, 5] - 1)
    tau <- 2 + 0.25 * x[, 4]
    
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
    categorical_index <- c(0,0,1,1,1,1,1)
    
    # Fit the full "pihat" model
    prop_model_full <- xgboost(
        data = x, label = z, max_depth = 3, eta = 1, verbose = F, 
        nthread = 10, nrounds = 10, objective = "binary:logistic")
    pihat.full <- predict(prop_model_full, x)
    # prop_model_full <- XBART::XBART.multinomial(z, 2, x)
    # prop_model_full_predictions <- predict(prop_model_full, X = x)
    # pihat.full <- prop_model_full_predictions$prob[,2]
    
    # Fit a series of propensity models with a sample of the covariates in x
    n_prop_submodels <- 4
    pihat.subset <- matrix(NA, nrow = n, ncol = n_prop_submodels)
    for (j in 1:n_prop_submodels){
        # Select a subset of covariates
        # First, choose the number of covariates to draw
        num_covariates_subset <- sample(1:(ncol(x)-1), size = 1)
        # Then, choose the covariates
        covariates_sampled <- sort(sample(1:ncol(x), size = num_covariates_subset, replace = F))
        num_categorical_sampled <- sum(categorical_index[covariates_sampled])
        # Subset X to these variables
        x_subset <- x[,covariates_sampled,drop=F]
        # Fit the XBART multinomial model as above
        prop_model_subset <- xgboost(
            data = x_subset, label = z, max_depth = 3, eta = 1, verbose = F, 
            nthread = 10, nrounds = 10, objective = "binary:logistic")
        pred <- predict(prop_model_subset, x_subset)
        pihat.subset[,j] <- pred
        # prop_model_subset <- XBART::XBART.multinomial(z, 2, x_subset, p_categorical = num_categorical_sampled, tau_a = 3, tau_b = 3.5)
        # prop_model_subset_predictions <- predict(prop_model_subset, X = x_subset)
        # pihat.subset[,j] <- prop_model_subset_predictions$prob[,2]
    }
    
    # Define pi(X) covariates for prognostic and treatment models (the same in this case)
    # pi_x_con = cbind(pihat.full, pihat.subset)
    # pi_x_mod <- cbind(pihat.full, pihat.subset)
    
    # Define covariates for prognostic and treatment models (the same in this case)
    x_con = x
    x_mod <- x
    pi_x_con <- x
    pi_x_mod <- x
    
    #### 2. Model Fitting and Estimation
    
    #### 2.a. Default XBCF

    # # Run `num_sweeps` of the algorithm
    # t1 = proc.time()
    # xbcf.fit <- XBART::XBCF.discrete(
    #     y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat.full,
    #     p_categorical_con = 5, p_categorical_mod = 5,
    #     num_sweeps = 60, burnin = 30, parallel = F, random_seed = 1234
    # )
    # t1 = proc.time() - t1

    # # Compute tauhat(X)
    # pred_xbcf <- predict(xbcf.fit, X_con = x_con, X_mod = x_mod, Z = z, pihat = pihat.full, burnin = 30)
    # tauhats_xbcf <- pred_xbcf$tau.adj.mean
    # 
    # # Evaluate RMSE and runtime
    # ate_xbcf <- mean(tauhats_xbcf)
    # rmse_xbcf <- sqrt(mean((tauhats_xbcf - tau)^2))
    # runtime_xbcf <- round(as.list(t1)$elapsed, 2)
    # if (print_during_sim){
    #     print(paste0("XBCF RMSE: ", rmse_xbcf))
    #     print(paste0("XBCF Runtime: ", runtime_xbcf, " seconds"))
    # }
    # 
    # sim_results[i,1] <- rmse_xbcf
    # sim_results[i,3] <- ate_xbcf
    # sim_results[i,5] <- runtime_xbcf
    # 
    # # Plot results
    # if (plot_during_sim){
    #     plot(tau, tauhats_xbcf, main = "XBCF", xlab = "Tau", ylab = "Tauhat")
    #     abline(a=0, b=1)
    # }
    
    #### 2.b. XBCF with propensity shrinkage

    # Run `num_sweeps` of the algorithm
    t1 = proc.time()
    xbcf.fit.prop.shrinkage <- XBART::XBCF.discrete.propensity.shrinkage(
        y = y, Z = z, X_con = x_con, X_mod = x_mod, pihat = pihat.full,
        pi_X_con = pi_x_con, pi_X_mod = pi_x_mod,
        p_categorical_con = 5, p_categorical_mod = 5,
        num_sweeps = 60, burnin = 30, parallel = F,
        random_seed = 1234, verbose = T
    )
    t1 = proc.time() - t1

    # # Compute tauhat(X)
    # pred_xbcf_prop_shrinkage <- predict(xbcf.fit.prop.shrinkage, X_con = x_con, X_mod = x_mod, pi_X_con = x_con, pi_X_mod = x_mod, Z = z, burnin = 30)
    # tauhats_xbcf_prop_shrinkage <- pred_xbcf_prop_shrinkage$tau.adj.mean
    # 
    # # Evaluate RMSE and runtime
    # ate_xbcf_prop_shrinkage <- mean(tauhats_xbcf_prop_shrinkage)
    # rmse_xbcf_prop_shrinkage <- sqrt(mean((tauhats_xbcf_prop_shrinkage - tau)^2))
    # runtime_xbcf_prop_shrinkage <- round(as.list(t1)$elapsed, 2)
    # if (print_during_sim){
    #     print(paste0("XBCF Propensity Shrinkage RMSE: ", rmse_xbcf_prop_shrinkage))
    #     print(paste0("XBCF Propensity Shrinkage Runtime: ", runtime_xbcf_prop_shrinkage, " seconds"))
    # }
    # 
    # sim_results[i,2] <- rmse_xbcf_prop_shrinkage
    # sim_results[i,4] <- ate_xbcf_prop_shrinkage
    # sim_results[i,6] <- runtime_xbcf_prop_shrinkage
    # 
    # # Plot results
    # if (plot_during_sim){
    #     plot(tau, tauhats_xbcf_prop_shrinkage, main = "XBCF with Propensity Shrinkage", xlab = "Tau", ylab = "Tauhat")
    #     abline(a=0, b=1)
    # }
}

# (rmse_mean <- apply(sim_results[,c(1,2),drop=F], 2, mean))
# (ate_mean <- apply(sim_results[,c(3,4),drop=F], 2, mean))
# (ate_sd <- apply(sim_results[,c(3,4),drop=F], 2, sd))
# (ate_bias <- (apply(sim_results[,c(3,4),drop=F], 2, mean)-2))
# (ate_rmse <- sqrt(ate_bias^2 + ate_sd^2))
