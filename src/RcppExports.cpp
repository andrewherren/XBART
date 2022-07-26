// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include "XBART_types.h"
#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// XBART_cpp
Rcpp::List XBART_cpp(mat y, mat X, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin, size_t mtry, size_t p_categorical, double kap, double s, double tau_kap, double tau_s, bool verbose, bool sampling_tau, bool parallel, bool set_random_seed, size_t random_seed, bool sample_weights, double nthread);
RcppExport SEXP _XBART_XBART_cpp(SEXP ySEXP, SEXP XSEXP, SEXP num_treesSEXP, SEXP num_sweepsSEXP, SEXP max_depthSEXP, SEXP n_minSEXP, SEXP num_cutpointsSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP tauSEXP, SEXP no_split_penalitySEXP, SEXP burninSEXP, SEXP mtrySEXP, SEXP p_categoricalSEXP, SEXP kapSEXP, SEXP sSEXP, SEXP tau_kapSEXP, SEXP tau_sSEXP, SEXP verboseSEXP, SEXP sampling_tauSEXP, SEXP parallelSEXP, SEXP set_random_seedSEXP, SEXP random_seedSEXP, SEXP sample_weightsSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_sweeps(num_sweepsSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_depth(max_depthSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_min(n_minSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_cutpoints(num_cutpointsSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type no_split_penality(no_split_penalitySEXP);
    Rcpp::traits::input_parameter< size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< size_t >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< size_t >::type p_categorical(p_categoricalSEXP);
    Rcpp::traits::input_parameter< double >::type kap(kapSEXP);
    Rcpp::traits::input_parameter< double >::type s(sSEXP);
    Rcpp::traits::input_parameter< double >::type tau_kap(tau_kapSEXP);
    Rcpp::traits::input_parameter< double >::type tau_s(tau_sSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type sampling_tau(sampling_tauSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    Rcpp::traits::input_parameter< bool >::type set_random_seed(set_random_seedSEXP);
    Rcpp::traits::input_parameter< size_t >::type random_seed(random_seedSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_weights(sample_weightsSEXP);
    Rcpp::traits::input_parameter< double >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(XBART_cpp(y, X, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, mtry, p_categorical, kap, s, tau_kap, tau_s, verbose, sampling_tau, parallel, set_random_seed, random_seed, sample_weights, nthread));
    return rcpp_result_gen;
END_RCPP
}
// XBART_multinomial_cpp
Rcpp::List XBART_multinomial_cpp(Rcpp::IntegerVector y, size_t num_class, mat X, size_t num_trees, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau_a, double tau_b, double no_split_penality, size_t burnin, size_t mtry, size_t p_categorical, bool verbose, bool parallel, bool set_random_seed, size_t random_seed, bool sample_weights, bool separate_tree, double weight, bool update_weight, bool update_tau, double nthread, double hmult, double heps);
RcppExport SEXP _XBART_XBART_multinomial_cpp(SEXP ySEXP, SEXP num_classSEXP, SEXP XSEXP, SEXP num_treesSEXP, SEXP num_sweepsSEXP, SEXP max_depthSEXP, SEXP n_minSEXP, SEXP num_cutpointsSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP tau_aSEXP, SEXP tau_bSEXP, SEXP no_split_penalitySEXP, SEXP burninSEXP, SEXP mtrySEXP, SEXP p_categoricalSEXP, SEXP verboseSEXP, SEXP parallelSEXP, SEXP set_random_seedSEXP, SEXP random_seedSEXP, SEXP sample_weightsSEXP, SEXP separate_treeSEXP, SEXP weightSEXP, SEXP update_weightSEXP, SEXP update_tauSEXP, SEXP nthreadSEXP, SEXP hmultSEXP, SEXP hepsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< size_t >::type num_class(num_classSEXP);
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_trees(num_treesSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_sweeps(num_sweepsSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_depth(max_depthSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_min(n_minSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_cutpoints(num_cutpointsSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type tau_a(tau_aSEXP);
    Rcpp::traits::input_parameter< double >::type tau_b(tau_bSEXP);
    Rcpp::traits::input_parameter< double >::type no_split_penality(no_split_penalitySEXP);
    Rcpp::traits::input_parameter< size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< size_t >::type mtry(mtrySEXP);
    Rcpp::traits::input_parameter< size_t >::type p_categorical(p_categoricalSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    Rcpp::traits::input_parameter< bool >::type set_random_seed(set_random_seedSEXP);
    Rcpp::traits::input_parameter< size_t >::type random_seed(random_seedSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_weights(sample_weightsSEXP);
    Rcpp::traits::input_parameter< bool >::type separate_tree(separate_treeSEXP);
    Rcpp::traits::input_parameter< double >::type weight(weightSEXP);
    Rcpp::traits::input_parameter< bool >::type update_weight(update_weightSEXP);
    Rcpp::traits::input_parameter< bool >::type update_tau(update_tauSEXP);
    Rcpp::traits::input_parameter< double >::type nthread(nthreadSEXP);
    Rcpp::traits::input_parameter< double >::type hmult(hmultSEXP);
    Rcpp::traits::input_parameter< double >::type heps(hepsSEXP);
    rcpp_result_gen = Rcpp::wrap(XBART_multinomial_cpp(y, num_class, X, num_trees, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau_a, tau_b, no_split_penality, burnin, mtry, p_categorical, verbose, parallel, set_random_seed, random_seed, sample_weights, separate_tree, weight, update_weight, update_tau, nthread, hmult, heps));
    return rcpp_result_gen;
END_RCPP
}
// XBCF_continuous_cpp
Rcpp::List XBCF_continuous_cpp(arma::mat y, arma::mat Z, arma::mat X_ps, arma::mat X_trt, size_t num_trees_ps, size_t num_trees_trt, size_t num_sweeps, size_t max_depth, size_t n_min, size_t num_cutpoints, double alpha, double beta, double tau, double no_split_penality, size_t burnin, size_t mtry_ps, size_t mtry_trt, size_t p_categorical_ps, size_t p_categorical_trt, double kap, double s, double tau_kap, double tau_s, bool verbose, bool sampling_tau, bool parallel, bool set_random_seed, size_t random_seed, bool sample_weights, double nthread);
RcppExport SEXP _XBART_XBCF_continuous_cpp(SEXP ySEXP, SEXP ZSEXP, SEXP X_psSEXP, SEXP X_trtSEXP, SEXP num_trees_psSEXP, SEXP num_trees_trtSEXP, SEXP num_sweepsSEXP, SEXP max_depthSEXP, SEXP n_minSEXP, SEXP num_cutpointsSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP tauSEXP, SEXP no_split_penalitySEXP, SEXP burninSEXP, SEXP mtry_psSEXP, SEXP mtry_trtSEXP, SEXP p_categorical_psSEXP, SEXP p_categorical_trtSEXP, SEXP kapSEXP, SEXP sSEXP, SEXP tau_kapSEXP, SEXP tau_sSEXP, SEXP verboseSEXP, SEXP sampling_tauSEXP, SEXP parallelSEXP, SEXP set_random_seedSEXP, SEXP random_seedSEXP, SEXP sample_weightsSEXP, SEXP nthreadSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X_ps(X_psSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X_trt(X_trtSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_trees_ps(num_trees_psSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_trees_trt(num_trees_trtSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_sweeps(num_sweepsSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_depth(max_depthSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_min(n_minSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_cutpoints(num_cutpointsSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type beta(betaSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< double >::type no_split_penality(no_split_penalitySEXP);
    Rcpp::traits::input_parameter< size_t >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< size_t >::type mtry_ps(mtry_psSEXP);
    Rcpp::traits::input_parameter< size_t >::type mtry_trt(mtry_trtSEXP);
    Rcpp::traits::input_parameter< size_t >::type p_categorical_ps(p_categorical_psSEXP);
    Rcpp::traits::input_parameter< size_t >::type p_categorical_trt(p_categorical_trtSEXP);
    Rcpp::traits::input_parameter< double >::type kap(kapSEXP);
    Rcpp::traits::input_parameter< double >::type s(sSEXP);
    Rcpp::traits::input_parameter< double >::type tau_kap(tau_kapSEXP);
    Rcpp::traits::input_parameter< double >::type tau_s(tau_sSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< bool >::type sampling_tau(sampling_tauSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel(parallelSEXP);
    Rcpp::traits::input_parameter< bool >::type set_random_seed(set_random_seedSEXP);
    Rcpp::traits::input_parameter< size_t >::type random_seed(random_seedSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_weights(sample_weightsSEXP);
    Rcpp::traits::input_parameter< double >::type nthread(nthreadSEXP);
    rcpp_result_gen = Rcpp::wrap(XBCF_continuous_cpp(y, Z, X_ps, X_trt, num_trees_ps, num_trees_trt, num_sweeps, max_depth, n_min, num_cutpoints, alpha, beta, tau, no_split_penality, burnin, mtry_ps, mtry_trt, p_categorical_ps, p_categorical_trt, kap, s, tau_kap, tau_s, verbose, sampling_tau, parallel, set_random_seed, random_seed, sample_weights, nthread));
    return rcpp_result_gen;
END_RCPP
}
// xbart_predict
Rcpp::List xbart_predict(mat X, double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt);
RcppExport SEXP _XBART_xbart_predict(SEXP XSEXP, SEXP y_meanSEXP, SEXP tree_pntSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type y_mean(y_meanSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_pnt(tree_pntSEXP);
    rcpp_result_gen = Rcpp::wrap(xbart_predict(X, y_mean, tree_pnt));
    return rcpp_result_gen;
END_RCPP
}
// xbcf_predict
Rcpp::List xbcf_predict(mat X_ps, mat X_trt, mat Z, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_ps, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_trt);
RcppExport SEXP _XBART_xbcf_predict(SEXP X_psSEXP, SEXP X_trtSEXP, SEXP ZSEXP, SEXP tree_psSEXP, SEXP tree_trtSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type X_ps(X_psSEXP);
    Rcpp::traits::input_parameter< mat >::type X_trt(X_trtSEXP);
    Rcpp::traits::input_parameter< mat >::type Z(ZSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_ps(tree_psSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_trt(tree_trtSEXP);
    rcpp_result_gen = Rcpp::wrap(xbcf_predict(X_ps, X_trt, Z, tree_ps, tree_trt));
    return rcpp_result_gen;
END_RCPP
}
// xbart_predict_full
Rcpp::List xbart_predict_full(mat X, double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt);
RcppExport SEXP _XBART_xbart_predict_full(SEXP XSEXP, SEXP y_meanSEXP, SEXP tree_pntSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type y_mean(y_meanSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_pnt(tree_pntSEXP);
    rcpp_result_gen = Rcpp::wrap(xbart_predict_full(X, y_mean, tree_pnt));
    return rcpp_result_gen;
END_RCPP
}
// gp_predict
Rcpp::List gp_predict(mat y, mat X, mat Xtest, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt, Rcpp::NumericVector resid, mat sigma, double theta, double tau, size_t p_categorical);
RcppExport SEXP _XBART_gp_predict(SEXP ySEXP, SEXP XSEXP, SEXP XtestSEXP, SEXP tree_pntSEXP, SEXP residSEXP, SEXP sigmaSEXP, SEXP thetaSEXP, SEXP tauSEXP, SEXP p_categoricalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< mat >::type Xtest(XtestSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_pnt(tree_pntSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type resid(residSEXP);
    Rcpp::traits::input_parameter< mat >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< size_t >::type p_categorical(p_categoricalSEXP);
    rcpp_result_gen = Rcpp::wrap(gp_predict(y, X, Xtest, tree_pnt, resid, sigma, theta, tau, p_categorical));
    return rcpp_result_gen;
END_RCPP
}
// xbart_multinomial_predict
Rcpp::List xbart_multinomial_predict(mat X, double y_mean, size_t num_class, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt);
RcppExport SEXP _XBART_xbart_multinomial_predict(SEXP XSEXP, SEXP y_meanSEXP, SEXP num_classSEXP, SEXP tree_pntSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type y_mean(y_meanSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_class(num_classSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_pnt(tree_pntSEXP);
    rcpp_result_gen = Rcpp::wrap(xbart_multinomial_predict(X, y_mean, num_class, tree_pnt));
    return rcpp_result_gen;
END_RCPP
}
// xbart_multinomial_predict_separatetrees
Rcpp::List xbart_multinomial_predict_separatetrees(mat X, double y_mean, size_t num_class, Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt);
RcppExport SEXP _XBART_xbart_multinomial_predict_separatetrees(SEXP XSEXP, SEXP y_meanSEXP, SEXP num_classSEXP, SEXP tree_pntSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type y_mean(y_meanSEXP);
    Rcpp::traits::input_parameter< size_t >::type num_class(num_classSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> >::type tree_pnt(tree_pntSEXP);
    rcpp_result_gen = Rcpp::wrap(xbart_multinomial_predict_separatetrees(X, y_mean, num_class, tree_pnt));
    return rcpp_result_gen;
END_RCPP
}
// r_to_json
Rcpp::StringVector r_to_json(double y_mean, Rcpp::XPtr<std::vector<std::vector<tree>>> tree_pnt);
RcppExport SEXP _XBART_r_to_json(SEXP y_meanSEXP, SEXP tree_pntSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type y_mean(y_meanSEXP);
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<tree>>> >::type tree_pnt(tree_pntSEXP);
    rcpp_result_gen = Rcpp::wrap(r_to_json(y_mean, tree_pnt));
    return rcpp_result_gen;
END_RCPP
}
// json_to_r
Rcpp::List json_to_r(Rcpp::StringVector json_string_r);
RcppExport SEXP _XBART_json_to_r(SEXP json_string_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type json_string_r(json_string_rSEXP);
    rcpp_result_gen = Rcpp::wrap(json_to_r(json_string_r));
    return rcpp_result_gen;
END_RCPP
}
// r_to_json_3D
Rcpp::StringVector r_to_json_3D(Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> tree_pnt);
RcppExport SEXP _XBART_r_to_json_3D(SEXP tree_pntSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::XPtr<std::vector<std::vector<std::vector<tree>>>> >::type tree_pnt(tree_pntSEXP);
    rcpp_result_gen = Rcpp::wrap(r_to_json_3D(tree_pnt));
    return rcpp_result_gen;
END_RCPP
}
// json_to_r_3D
Rcpp::List json_to_r_3D(Rcpp::StringVector json_string_r);
RcppExport SEXP _XBART_json_to_r_3D(SEXP json_string_rSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::StringVector >::type json_string_r(json_string_rSEXP);
    rcpp_result_gen = Rcpp::wrap(json_to_r_3D(json_string_r));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_XBART_XBART_cpp", (DL_FUNC) &_XBART_XBART_cpp, 25},
    {"_XBART_XBART_multinomial_cpp", (DL_FUNC) &_XBART_XBART_multinomial_cpp, 28},
    {"_XBART_XBCF_continuous_cpp", (DL_FUNC) &_XBART_XBCF_continuous_cpp, 30},
    {"_XBART_xbart_predict", (DL_FUNC) &_XBART_xbart_predict, 3},
    {"_XBART_xbcf_predict", (DL_FUNC) &_XBART_xbcf_predict, 5},
    {"_XBART_xbart_predict_full", (DL_FUNC) &_XBART_xbart_predict_full, 3},
    {"_XBART_gp_predict", (DL_FUNC) &_XBART_gp_predict, 9},
    {"_XBART_xbart_multinomial_predict", (DL_FUNC) &_XBART_xbart_multinomial_predict, 4},
    {"_XBART_xbart_multinomial_predict_separatetrees", (DL_FUNC) &_XBART_xbart_multinomial_predict_separatetrees, 4},
    {"_XBART_r_to_json", (DL_FUNC) &_XBART_r_to_json, 2},
    {"_XBART_json_to_r", (DL_FUNC) &_XBART_json_to_r, 1},
    {"_XBART_r_to_json_3D", (DL_FUNC) &_XBART_r_to_json_3D, 1},
    {"_XBART_json_to_r_3D", (DL_FUNC) &_XBART_json_to_r_3D, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_XBART(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
