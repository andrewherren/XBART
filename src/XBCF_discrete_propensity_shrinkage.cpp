#include <ctime>
// #include <RcppArmadillo.h>
#include "tree.h"
#include <chrono>
#include "mcmc_loop.h"
#include "X_struct.h"
#include "utility_rcpp.h"
#include "json_io.h"

using namespace std;
using namespace chrono;
using namespace arma;

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::export]]
Rcpp::List XBCF_discrete_propensity_shrinkage_cpp(
    arma::mat y, arma::mat Z, 
    arma::mat X_con, arma::mat X_mod, size_t num_trees_con, size_t num_trees_mod, 
    arma::mat pi_X_con, arma::mat pi_X_mod, size_t num_trees_con_pi, size_t num_trees_mod_pi, 
    size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, 
    double alpha_con, double beta_con, double alpha_mod, double beta_mod, double tau_con, double tau_mod, 
    double alpha_con_pi, double beta_con_pi, double alpha_mod_pi, double beta_mod_pi, double tau_con_pi, double tau_mod_pi, 
    double no_split_penalty, size_t burnin = 1, size_t mtry_con = 0, size_t mtry_mod = 0, size_t mtry_con_pi = 0, size_t mtry_mod_pi = 0, 
    size_t p_categorical_con = 0, size_t p_categorical_mod = 0, size_t p_categorical_con_pi = 0, size_t p_categorical_mod_pi = 0, 
    double kap = 16, double s = 4, double tau_con_kap = 3, double tau_con_s = 0.5, double tau_mod_kap = 3, double tau_mod_s = 0.5, 
    double tau_con_pi_kap = 3, double tau_con_pi_s = 0.5, double tau_mod_pi_kap = 3, double tau_mod_pi_s = 0.5, 
    bool pr_scale = false, bool trt_scale = false, bool a_scaling = true, bool b_scaling = true, 
    bool verbose = false, bool sampling_tau = true, bool parallel = true, bool set_random_seed = false, 
    size_t random_seed = 0, bool sample_weights = true, double nthread = 0
){
    if (parallel)
    {
        thread_pool.start(nthread);
        COUT << "Running in parallel with " << nthread << " threads." << endl;
    }
    else
    {
        COUT << "Running with single thread." << endl;
    }

    size_t N = X_con.n_rows;

    // number of total variables
    size_t p_con = X_con.n_cols;
    size_t p_mod = X_mod.n_cols;

    COUT << "size of X_con " << X_con.n_rows << " " << X_con.n_cols << endl;
    COUT << "size of X_mod " << X_mod.n_rows << " " << X_mod.n_cols << endl;

    // number of total propensity score columns
    size_t p_con_pi = pi_X_con.n_cols;
    size_t p_mod_pi = pi_X_mod.n_cols;
    
    COUT << "size of pi_X_con " << pi_X_con.n_rows << " " << pi_X_con.n_cols << endl;
    COUT << "size of pi_X_mod " << pi_X_mod.n_rows << " " << pi_X_mod.n_cols << endl;
    
    // number of basis functions (1 in the case of the OG bcf)
    size_t p_z = Z.n_cols;

    // number of continuous variables
    size_t p_continuous_con = p_con - p_categorical_con;
    size_t p_continuous_mod = p_mod - p_categorical_mod;
    
    // number of continuous pi variables
    size_t p_continuous_con_pi = p_con_pi - p_categorical_con_pi;
    size_t p_continuous_mod_pi = p_mod_pi - p_categorical_mod_pi;

    // suppose first p_continuous variables are continuous, then categorical
    assert(mtry_con <= p_con);

    assert(mtry_mod <= p_mod);
    
    assert(mtry_con_pi <= p_con_pi);
    
    assert(mtry_mod_pi <= p_mod_pi);

    assert(burnin <= num_sweeps);

    if (mtry_con == 0)
    {
        mtry_con = p_con;
    }

    if (mtry_mod == 0)
    {
        mtry_mod = p_mod;
    }

    if (mtry_con != p_con)
    {
        COUT << "Sample " << mtry_con << " out of " << p_con << " variables when grow each prognostic tree." << endl;
    }

    if (mtry_mod != p_mod)
    {
        COUT << "Sample " << mtry_mod << " out of " << p_mod << " variables when grow each treatment tree." << endl;
    }

    if (mtry_con_pi == 0)
    {
        mtry_con_pi = p_con_pi;
    }
    
    if (mtry_mod_pi == 0)
    {
        mtry_mod_pi = p_mod_pi;
    }
    
    if (mtry_con_pi != p_con_pi)
    {
        COUT << "Sample " << mtry_con_pi << " out of " << p_con_pi << " variables when grow each propensity prognostic tree." << endl;
    }
    
    if (mtry_mod_pi != p_mod_pi)
    {
        COUT << "Sample " << mtry_mod_pi << " out of " << p_mod_pi << " variables when grow each propensity treatment tree." << endl;
    }
    
    arma::umat Xorder_con(X_con.n_rows, X_con.n_cols);
    matrix<size_t> Xorder_std_con;
    ini_matrix(Xorder_std_con, N, p_con);

    arma::umat Xorder_mod(X_mod.n_rows, X_mod.n_cols);
    matrix<size_t> Xorder_std_mod;
    ini_matrix(Xorder_std_mod, N, p_mod);

    cout << "size of Xorder con and mode " << Xorder_std_con.size() << " " << Xorder_std_con[0].size() << " " << Xorder_std_mod.size() << " " << Xorder_std_mod[0].size() << endl;

    arma::umat pi_Xorder_con(pi_X_con.n_rows, pi_X_con.n_cols);
    matrix<size_t> pi_Xorder_std_con;
    ini_matrix(pi_Xorder_std_con, N, p_con_pi);
    
    arma::umat pi_Xorder_mod(pi_X_mod.n_rows, pi_X_mod.n_cols);
    matrix<size_t> pi_Xorder_std_mod;
    ini_matrix(pi_Xorder_std_mod, N, p_mod_pi);
    
    cout << "size of pi_Xorder con and mode " << pi_Xorder_std_con.size() << " " << pi_Xorder_std_con[0].size() << " " << pi_Xorder_std_mod.size() << " " << pi_Xorder_std_mod[0].size() << endl;
    
    std::vector<double> y_std(N);

    double y_mean = 0.0;

    size_t N_trt = 0;  // number of treated in training
    size_t N_ctrl = 0; // number of control in training

    for (size_t i = 0; i < N; i++)
    {
        y_mean += y[i];

        // count number of treated and control data
        if (Z[i] == 1)
        {
            N_trt++;
        }
        else
        {
            N_ctrl++;
        }
    }
    y_mean = y_mean / N;

    Rcpp::NumericMatrix X_std_con(N, p_con);
    Rcpp::NumericMatrix X_std_mod(N, p_mod);

    Rcpp::NumericMatrix pi_X_std_con(N, p_con_pi);
    Rcpp::NumericMatrix pi_X_std_mod(N, p_mod_pi);
    
    matrix<double> Z_std;
    ini_matrix(Z_std, N, p_z);

    rcpp_to_std2(y, Z, X_con, X_mod, y_std, y_mean, Z_std, X_std_con, X_std_mod, Xorder_std_con, Xorder_std_mod);
    
    rcpp_to_std2(pi_X_con, pi_X_mod, pi_X_std_con, pi_X_std_mod, pi_Xorder_std_con, pi_Xorder_std_mod);

    ///////////////////////////////////////////////////////////////////

    double *Xpointer_con = &X_std_con[0];
    double *Xpointer_mod = &X_std_mod[0];
    
    double *pi_Xpointer_con = &pi_X_std_con[0];
    double *pi_Xpointer_mod = &pi_X_std_mod[0];

    matrix<double> sigma0_draw_xinfo;
    ini_matrix(sigma0_draw_xinfo, num_trees_con + num_trees_mod, num_sweeps);

    matrix<double> sigma1_draw_xinfo;
    ini_matrix(sigma1_draw_xinfo, num_trees_con + num_trees_mod, num_sweeps);

    matrix<double> a_xinfo;
    ini_matrix(a_xinfo, num_sweeps, 1);

    matrix<double> b_xinfo;
    ini_matrix(b_xinfo, num_sweeps, 2);

    matrix<double> tau_con_xinfo;
    ini_matrix(tau_con_xinfo, num_sweeps, 1);

    matrix<double> tau_mod_xinfo;
    ini_matrix(tau_mod_xinfo, num_sweeps, 1);
    
    matrix<double> a_pi_xinfo;
    ini_matrix(a_pi_xinfo, num_sweeps, 1);
    
    matrix<double> b_pi_xinfo;
    ini_matrix(b_pi_xinfo, num_sweeps, 2);
    
    matrix<double> tau_con_pi_xinfo;
    ini_matrix(tau_con_pi_xinfo, num_sweeps, 1);
    
    matrix<double> tau_mod_pi_xinfo;
    ini_matrix(tau_mod_pi_xinfo, num_sweeps, 1);

    // create trees
    vector<vector<tree>> trees_con(num_sweeps);
    vector<vector<tree>> trees_mod(num_sweeps);
    vector<vector<tree>> trees_con_pi(num_sweeps);
    vector<vector<tree>> trees_mod_pi(num_sweeps);

    for (size_t i = 0; i < num_sweeps; i++)
    {
        trees_con[i].resize(num_trees_con);
        trees_mod[i].resize(num_trees_mod);
        trees_con_pi[i].resize(num_trees_con_pi);
        trees_mod_pi[i].resize(num_trees_mod_pi);
    }

    // define model
    XBCFDiscretePropensityShrinkageModel *model = new XBCFDiscretePropensityShrinkageModel(
        kap, s, tau_con, tau_mod, tau_con_pi, tau_mod_pi, alpha_con, beta_con, alpha_mod, beta_mod, 
        alpha_con_pi, beta_con_pi, alpha_mod_pi, beta_mod_pi, sampling_tau, 
        tau_con_kap, tau_con_s, tau_mod_kap, tau_mod_s, tau_con_pi_kap, 
        tau_con_pi_s, tau_mod_pi_kap, tau_mod_pi_s
    );
    model->setNoSplitPenalty(no_split_penalty);

    // State settings
    XBCFDiscretePropensityShrinkageState state(
            &Z_std, Xpointer_con, Xpointer_mod, Xorder_std_con, Xorder_std_mod, 
            pi_Xpointer_con, pi_Xpointer_mod, pi_Xorder_std_con, pi_Xorder_std_mod, 
            N, p_con, p_mod, num_trees_con, num_trees_mod, 
            p_con_pi, p_mod_pi, num_trees_con_pi, num_trees_mod_pi, 
            p_categorical_con, p_categorical_mod, p_continuous_con, p_continuous_mod, 
            p_categorical_con_pi, p_categorical_mod_pi, p_continuous_con_pi, p_continuous_mod_pi, 
            set_random_seed, random_seed, n_min, num_cutpoints, 
            mtry_con, mtry_mod, mtry_con_pi, mtry_mod_pi, num_sweeps, sample_weights, 
            &y_std, 1.0, max_depth, y_mean, burnin, model->dim_residual, nthread, parallel, a_scaling, b_scaling, N_trt, N_ctrl);

    // initialize X_struct
    std::vector<double> initial_theta_con(1, 0);
    X_struct x_struct_con(Xpointer_con, &y_std, N, Xorder_std_con, p_categorical_con, p_continuous_con, &initial_theta_con, num_trees_con);

    std::vector<double> initial_theta_mod(1, y_mean / (double)num_trees_mod);
    X_struct x_struct_mod(Xpointer_mod, &y_std, N, Xorder_std_mod, p_categorical_mod, p_continuous_mod, &initial_theta_mod, num_trees_mod);

    std::vector<double> initial_theta_con_pi(1, 0);
    X_struct x_struct_con_pi(pi_Xpointer_con, &y_std, N, pi_Xorder_std_con, p_categorical_con_pi, p_continuous_con_pi, &initial_theta_con_pi, num_trees_con_pi);
    
    std::vector<double> initial_theta_mod_pi(1, y_mean / (double)num_trees_mod_pi);
    X_struct x_struct_mod_pi(pi_Xpointer_mod, &y_std, N, pi_Xorder_std_mod, p_categorical_mod_pi, p_continuous_mod_pi, &initial_theta_mod_pi, num_trees_mod_pi);
    
    ////////////////////////////////////////////////////////////////
    mcmc_loop_xbcf_discrete_propensity_shrinkage(
        Xorder_std_con, Xorder_std_mod, pi_Xorder_std_con, pi_Xorder_std_mod, verbose, 
        sigma0_draw_xinfo, sigma1_draw_xinfo, a_xinfo, b_xinfo, a_pi_xinfo, 
        b_pi_xinfo, tau_con_xinfo, tau_mod_xinfo, tau_con_pi_xinfo, tau_mod_pi_xinfo, 
        trees_con, trees_mod, trees_con_pi, trees_mod_pi, no_split_penalty, 
        state, model, x_struct_con, x_struct_mod, x_struct_con_pi, x_struct_mod_pi
    );

    // R Objects to Return
    Rcpp::NumericMatrix sigma0_draw(num_trees_con + num_trees_mod + num_trees_con_pi + num_trees_mod_pi, num_sweeps); // save predictions of each tree

    Rcpp::NumericMatrix sigma1_draw(num_trees_con + num_trees_mod + num_trees_con_pi + num_trees_mod_pi, num_sweeps); // save predictions of each tree

    Rcpp::NumericMatrix a_draw(num_sweeps, 1);

    Rcpp::NumericMatrix b_draw(num_sweeps, 2);

    Rcpp::NumericMatrix tau_con_draw(num_sweeps, 1);

    Rcpp::NumericMatrix tau_mod_draw(num_sweeps, 1);

    Rcpp::NumericVector split_count_sum_con(p_con, 0);

    Rcpp::NumericVector split_count_sum_mod(p_mod, 0);

    Rcpp::NumericMatrix a_pi_draw(num_sweeps, 1);
    
    Rcpp::NumericMatrix b_pi_draw(num_sweeps, 2);
    
    Rcpp::NumericMatrix tau_con_pi_draw(num_sweeps, 1);
    
    Rcpp::NumericMatrix tau_mod_pi_draw(num_sweeps, 1);
    
    Rcpp::NumericVector split_count_sum_con_pi(p_con_pi, 0);
    
    Rcpp::NumericVector split_count_sum_mod_pi(p_mod_pi, 0);
    
    // copy from std vector to Rcpp Numeric Matrix objects
    Matrix_to_NumericMatrix(sigma0_draw_xinfo, sigma0_draw);
    Matrix_to_NumericMatrix(sigma0_draw_xinfo, sigma0_draw);
    Matrix_to_NumericMatrix(a_xinfo, a_draw);
    Matrix_to_NumericMatrix(tau_con_xinfo, tau_con_draw);
    Matrix_to_NumericMatrix(tau_mod_xinfo, tau_mod_draw);
    Matrix_to_NumericMatrix(b_xinfo, b_draw);
    Matrix_to_NumericMatrix(a_pi_xinfo, a_pi_draw);
    Matrix_to_NumericMatrix(tau_con_pi_xinfo, tau_con_pi_draw);
    Matrix_to_NumericMatrix(tau_mod_pi_xinfo, tau_mod_pi_draw);
    Matrix_to_NumericMatrix(b_pi_xinfo, b_pi_draw);

    for (size_t i = 0; i < p_con; i++)
    {
        split_count_sum_con(i) = (int)(*state.split_count_all_con)[i];
    }

    for (size_t i = 0; i < p_mod; i++)
    {
        split_count_sum_mod(i) = (int)(*state.split_count_all_mod)[i];
    }
    for (size_t i = 0; i < p_con_pi; i++)
    {
        split_count_sum_con_pi(i) = (int)(*state.split_count_all_con_pi)[i];
    }
    
    for (size_t i = 0; i < p_mod_pi; i++)
    {
        split_count_sum_mod_pi(i) = (int)(*state.split_count_all_mod_pi)[i];
    }

    // print out tree structure, for usage of BART warm-start
    Rcpp::StringVector output_tree_con(num_sweeps);
    Rcpp::StringVector output_tree_mod(num_sweeps);

    tree_to_string(trees_mod, output_tree_mod, num_sweeps, num_trees_mod, p_mod);
    tree_to_string(trees_con, output_tree_con, num_sweeps, num_trees_con, p_con);

    Rcpp::StringVector tree_json_mod(1);
    Rcpp::StringVector tree_json_con(1);
    json j = get_forest_json(trees_mod, y_mean);
    json j2 = get_forest_json(trees_con, y_mean);
    tree_json_mod[0] = j.dump(4);
    tree_json_con[0] = j2.dump(4);

    Rcpp::StringVector output_tree_con_pi(num_sweeps);
    Rcpp::StringVector output_tree_mod_pi(num_sweeps);
    
    tree_to_string(trees_mod_pi, output_tree_mod_pi, num_sweeps, num_trees_mod_pi, p_mod_pi);
    tree_to_string(trees_con_pi, output_tree_con_pi, num_sweeps, num_trees_con_pi, p_con_pi);
    
    Rcpp::StringVector tree_json_mod_pi(1);
    Rcpp::StringVector tree_json_con_pi(1);
    json j3 = get_forest_json(trees_mod_pi, y_mean);
    json j4 = get_forest_json(trees_con_pi, y_mean);
    tree_json_mod_pi[0] = j3.dump(4);
    tree_json_con_pi[0] = j4.dump(4);
    
    thread_pool.stop();

    return Rcpp::List::create(
        Rcpp::Named("sigma0") = sigma0_draw,
        Rcpp::Named("sigma1") = sigma1_draw,
        Rcpp::Named("a") = a_draw,
        Rcpp::Named("b") = b_draw,
        Rcpp::Named("tau_con") = tau_con_draw,
        Rcpp::Named("tau_mod") = tau_mod_draw,
        Rcpp::Named("a_pi") = a_pi_draw,
        Rcpp::Named("b_pi") = b_pi_draw,
        Rcpp::Named("tau_con_pi") = tau_con_pi_draw,
        Rcpp::Named("tau_mod_pi") = tau_mod_pi_draw,
        Rcpp::Named("importance_list") = Rcpp::List::create(Rcpp::Named("importance_prognostic") = split_count_sum_con, Rcpp::Named("importance_treatment") = split_count_sum_mod, Rcpp::Named("importance_prognostic_pi_x") = split_count_sum_con_pi, Rcpp::Named("importance_treatment_pi_x") = split_count_sum_mod_pi),
        Rcpp::Named("model_list") = Rcpp::List::create(Rcpp::Named("y_mean") = y_mean, Rcpp::Named("p_con") = p_con, Rcpp::Named("p_mod") = p_mod),
        Rcpp::Named("tree_json_mod") = tree_json_mod,
        Rcpp::Named("tree_json_con") = tree_json_con,
        Rcpp::Named("tree_string_mod") = output_tree_mod,
        Rcpp::Named("tree_string_con") = output_tree_con,
        Rcpp::Named("tree_json_mod_pi") = tree_json_mod_pi,
        Rcpp::Named("tree_json_con_pi") = tree_json_con_pi,
        Rcpp::Named("tree_string_mod_pi") = output_tree_mod_pi,
        Rcpp::Named("tree_string_con_pi") = output_tree_con_pi);
}
