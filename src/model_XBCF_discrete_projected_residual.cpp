#include "tree.h"
#include "model.h"
#include <cfenv>

//////////////////////////////////////////////////////////////////////////////////////
//
//
//  Normal Linear Model for binary treatment XBCF
//
//
//////////////////////////////////////////////////////////////////////////////////////

void XBCFDiscreteProjectedResidualModel::incSuffStat(
        State &state, size_t index_next_obs, std::vector<double> &suffstats
){
    if (state.treatment_flag & !state.projection_flag)
    {
        // treatment forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[1] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[1] * (*state.tau_pi_fit)[index_next_obs]) / state.b_vec[1];
            suffstats[3] += 1;
        }
        else
        {
            // if control group
            suffstats[0] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[0] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[0] * (*state.tau_pi_fit)[index_next_obs]) / state.b_vec[0];
            suffstats[2] += 1;
        }
    } 
    else if (state.treatment_flag & state.projection_flag)
    {
        // treatment forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[1] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[1] * (*state.tau_pi_fit)[index_next_obs]) / state.b_vec_pi[1];
            suffstats[3] += 1;
        }
        else
        {
            // if control group
            suffstats[0] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[0] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[0] * (*state.tau_pi_fit)[index_next_obs]) / state.b_vec_pi[0];
            suffstats[2] += 1;
        }
    }
    else if (!state.treatment_flag & !state.projection_flag)
    {
        // treatment forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[1] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[1] * (*state.tau_pi_fit)[index_next_obs]) / state.a;
            suffstats[3] += 1;
        }
        else
        {
            // if control group
            suffstats[0] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[0] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[0] * (*state.tau_pi_fit)[index_next_obs]) / state.a;
            suffstats[2] += 1;
        }
    }
    else
    {
        // prognostic forest
        if ((*state.Z_std)[0][index_next_obs] == 1)
        {
            // if treated
            suffstats[1] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[1] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[1] * (*state.tau_pi_fit)[index_next_obs]) / state.a_pi;
            suffstats[3] += 1;
        }
        else
        {
            // if control group
            suffstats[0] += ((*state.y_std)[index_next_obs] - state.a * (*state.mu_fit)[index_next_obs] - state.a_pi * (*state.mu_pi_fit)[index_next_obs] - state.b_vec[0] * (*state.tau_fit)[index_next_obs] - state.b_vec_pi[0] * (*state.tau_pi_fit)[index_next_obs]) / state.a_pi;
            suffstats[2] += 1;
        }
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::samplePars(
        State &state, std::vector<double> &suff_stat, 
        std::vector<double> &theta_vector, double &prob_leaf
){
    std::normal_distribution<double> normal_samp(0.0, 1.0);

    double tau_use;

    if (state.treatment_flag & state.projection_flag)
    {
        tau_use = tau_mod_pi;
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        tau_use = tau_mod;
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        tau_use = tau_con_pi;
    }
    else
    {
        tau_use = tau_con;
    }

    double s0 = 0;
    double s1 = 0;

    if (state.treatment_flag & state.projection_flag)
    {
        s0 = state.sigma_vec[0] / fabs(state.b_vec_pi[0]);
        s1 = state.sigma_vec[1] / fabs(state.b_vec_pi[1]);
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        s0 = state.sigma_vec[0] / fabs(state.b_vec[0]);
        s1 = state.sigma_vec[1] / fabs(state.b_vec[1]);
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        s0 = state.sigma_vec[0] / fabs(state.a_pi);
        s1 = state.sigma_vec[1] / fabs(state.a_pi);
    }
    else
    {
        s0 = state.sigma_vec[0] / fabs(state.a);
        s1 = state.sigma_vec[1] / fabs(state.a);
    }

    // step 1 (control group)
    double denominator0 = 1.0 / tau_use + suff_stat[2] / pow(s0, 2);
    double m0 = (suff_stat[0] / pow(s0, 2)) / denominator0;
    double v0 = 1.0 / denominator0;

    // step 2 (treatment group)
    double denominator1 = (1.0 / v0 + suff_stat[3] / pow(s1, 2));
    double m1 = (1.0 / v0) * m0 / denominator1 + suff_stat[1] / pow(s1, 2) / denominator1;
    double v1 = 1.0 / denominator1;

    // sample leaf parameter
    theta_vector[0] = m1 + sqrt(v1) * normal_samp(state.gen);

    // also update probability of leaf parameters
    prob_leaf = 1.0;

    return;
}

void XBCFDiscreteProjectedResidualModel::update_state(
        State &state, size_t tree_ind, X_struct &x_struct, size_t ind
){
    // Draw Sigma
    std::vector<double> full_residual_trt(state.N_trt);
    std::vector<double> full_residual_ctrl(state.N_ctrl);

    // index
    size_t index_trt = 0;
    size_t index_ctrl = 0;

    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            full_residual_trt[index_trt] = (*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - state.b_vec[1] * (*state.tau_fit)[i] - state.b_vec_pi[1] * (*state.tau_pi_fit)[i];
            index_trt++;
        }
        else
        {
            // if control group
            full_residual_ctrl[index_ctrl] = (*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - state.b_vec[0] * (*state.tau_fit)[i] - state.b_vec_pi[0] * (*state.tau_pi_fit)[i];
            index_ctrl++;
        }
    }

    std::gamma_distribution<double> gamma_samp1((state.N_trt + kap) / 2.0, 2.0 / (sum_squared(full_residual_trt) + s));

    std::gamma_distribution<double> gamma_samp0((state.N_ctrl + kap) / 2.0, 2.0 / (sum_squared(full_residual_ctrl) + s));

    double sigma;

    if (ind == 0)
    {
        sigma = 1.0 / sqrt(gamma_samp0(state.gen));
    }
    else
    {
        sigma = 1.0 / sqrt(gamma_samp1(state.gen));
    }

    state.update_sigma(sigma, ind);

    return;
}

void XBCFDiscreteProjectedResidualModel::update_tau(
        State &state, size_t tree_ind, size_t sweeps, vector<vector<tree>> &trees
){
    std::vector<tree *> leaf_nodes;
    trees[sweeps][tree_ind].getbots(leaf_nodes);
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    }

    double kap = (state.treatment_flag) ? this->tau_mod_kap : this->tau_con_kap;

    double s = (state.treatment_flag) ? this->tau_mod_s * this->tau_mod_mean : this->tau_con_s * this->tau_con_mean;

    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));

    double tau_sample = 1.0 / gamma_samp(state.gen);

    if (state.treatment_flag & state.projection_flag)
    {
        this->tau_mod_pi = tau_sample;
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        this->tau_mod = tau_sample;
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        this->tau_con_pi = tau_sample;
    }
    else
    {
        this->tau_con = tau_sample;
    }

    return;
};

void XBCFDiscreteProjectedResidualModel::update_tau_per_forest(
        State &state, size_t sweeps, vector<vector<tree>> &trees
){
    std::vector<tree *> leaf_nodes;
    for (size_t tree_ind = 0; tree_ind < state.num_trees; tree_ind++)
    {
        trees[sweeps][tree_ind].getbots(leaf_nodes);
    }
    double sum_squared = 0.0;
    for (size_t i = 0; i < leaf_nodes.size(); i++)
    {
        sum_squared = sum_squared + pow(leaf_nodes[i]->theta_vector[0], 2);
    };
    
    double kap = (state.treatment_flag) ? this->tau_mod_kap : this->tau_con_kap;

    double s = (state.treatment_flag) ? this->tau_mod_s * this->tau_mod_mean : this->tau_con_s * this->tau_con_mean;

    std::gamma_distribution<double> gamma_samp((leaf_nodes.size() + kap) / 2.0, 2.0 / (sum_squared + s));
    double tau_sample = 1.0 / gamma_samp(state.gen);

    if (state.treatment_flag & state.projection_flag)
    {
        this->tau_mod_pi = tau_sample;
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        this->tau_mod = tau_sample;
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        this->tau_con_pi = tau_sample;
    }
    else
    {
        this->tau_con = tau_sample;
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::initialize_root_suffstat(
        State &state, std::vector<double> &suff_stat
){
    suff_stat.resize(4);
    std::fill(suff_stat.begin(), suff_stat.end(), 0.0);
    for (size_t i = 0; i < state.n_y; i++)
    {
        incSuffStat(state, i, suff_stat);
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::updateNodeSuffStat(
        State &state, std::vector<double> &suff_stat, matrix<size_t> &Xorder_std, size_t &split_var, size_t row_ind
){

    incSuffStat(state, Xorder_std[split_var][row_ind], suff_stat);

    return;
}

void XBCFDiscreteProjectedResidualModel::calculateOtherSideSuffStat(
        std::vector<double> &parent_suff_stat, std::vector<double> &lchild_suff_stat, 
        std::vector<double> &rchild_suff_stat, size_t &N_parent, size_t &N_left, 
        size_t &N_right, bool &compute_left_side
){
    // in function split_xorder_std_categorical, for efficiency, the function only calculates suff stat of ONE child
    // this function calculate the other side based on parent and the other child
    if (compute_left_side)
    {
        rchild_suff_stat = parent_suff_stat - lchild_suff_stat;
    }
    else
    {
        lchild_suff_stat = parent_suff_stat - rchild_suff_stat;
    }
    return;
}

double XBCFDiscreteProjectedResidualModel::likelihood(
        std::vector<double> &temp_suff_stat, std::vector<double> &suff_stat_all, 
        size_t N_left, bool left_side, bool no_split, 
        State &state
) const {
    // likelihood equation for XBCF with discrete binary treatment variable Z

    double tau_use;

    if (state.treatment_flag & state.projection_flag)
    {
        tau_use = tau_mod_pi;
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        tau_use = tau_mod;
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        tau_use = tau_con_pi;
    }
    else
    {
        tau_use = tau_con;
    }

    double s0 = 0;
    double s1 = 0;
    double denominator;   // the denominator (1 + tau * precision_squared) is the same for both terms
    double s_psi_squared; // (residual * precision_squared)^2

    if (state.treatment_flag & state.projection_flag)
    {
        // if this is treatment forest
        s0 = state.sigma_vec[0] / fabs(state.b_vec_pi[0]);
        s1 = state.sigma_vec[1] / fabs(state.b_vec_pi[1]);
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        // if this is treatment forest
        s0 = state.sigma_vec[0] / fabs(state.b_vec[0]);
        s1 = state.sigma_vec[1] / fabs(state.b_vec[1]);
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        s0 = state.sigma_vec[0] / fabs(state.a_pi);
        s1 = state.sigma_vec[1] / fabs(state.a_pi);
    }
    else
    {
        s0 = state.sigma_vec[0] / fabs(state.a);
        s1 = state.sigma_vec[1] / fabs(state.a);
    }

    if (no_split)
    {
        denominator = 1 + (suff_stat_all[2] / pow(s0, 2) + suff_stat_all[3] / pow(s1, 2)) * tau_use;
        s_psi_squared = suff_stat_all[0] / pow(s0, 2) + suff_stat_all[1] / pow(s1, 2);
    }
    else
    {
        if (left_side)
        {
            denominator = 1 + (temp_suff_stat[2] / pow(s0, 2) + temp_suff_stat[3] / pow(s1, 2)) * tau_use;
            s_psi_squared = temp_suff_stat[0] / pow(s0, 2) + temp_suff_stat[1] / pow(s1, 2);
        }
        else
        {
            denominator = 1 + ((suff_stat_all[2] - temp_suff_stat[2]) / pow(s0, 2) + (suff_stat_all[3] - temp_suff_stat[3]) / pow(s1, 2)) * tau_use;
            s_psi_squared = (suff_stat_all[0] - temp_suff_stat[0]) / pow(s0, 2) + (suff_stat_all[1] - temp_suff_stat[1]) / pow(s1, 2);
        }
    }
    return 0.5 * log(1 / denominator) + 0.5 * pow(s_psi_squared, 2) * tau_use / denominator;
}

void XBCFDiscreteProjectedResidualModel::ini_residual_std(State &state)
{
    // initialize the vector of full residuals
    double b_value;
    double b_value_pi;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        b_value = ((*state.Z_std)[0][i] == 1) ? state.b_vec[1] : state.b_vec[0];
        b_value_pi = ((*state.Z_std)[0][i] == 1) ? state.b_vec_pi[1] : state.b_vec_pi[0];

        (*state.residual_std)[0][i] = (*state.y_std)[i] - (state.a) * (*state.mu_fit)[i] - (state.a_pi) * (*state.mu_pi_fit)[i] - b_value * (*state.tau_fit)[i] - b_value_pi * (*state.tau_pi_fit)[i];
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::predict_std(
        matrix<double> &Ztestpointer, const double *Xtestpointer_con, const double *Xtestpointer_mod, 
        double *pi_Xtestpointer_con, double *pi_Xtestpointer_mod, 
        size_t N_test, size_t p_con, size_t p_mod, size_t num_trees_con, size_t num_trees_mod, 
        size_t num_sweeps, matrix<double> &yhats_test_xinfo, 
        matrix<double> &prognostic_xinfo, matrix<double> &treatment_xinfo, 
        matrix<double> &prognostic_proj_xinfo, matrix<double> &treatment_proj_xinfo, 
        vector<vector<tree>> &trees_con, vector<vector<tree>> &trees_mod, 
        vector<vector<tree>> &trees_con_pi, vector<vector<tree>> &trees_mod_pi
){
    // predict the output as a matrix
    // row : dimension of theta, column : number of trees
    matrix<double> output_mod;
    ini_matrix(output_mod, this->dim_theta, trees_mod[0].size());

    matrix<double> output_con;
    ini_matrix(output_con, this->dim_theta, trees_con[0].size());
    
    matrix<double> output_mod_pi;
    // ini_matrix(output_mod_pi, this->dim_theta, trees_mod_pi[0].size());
    ini_matrix(output_mod_pi, this->dim_theta, 1);
    
    matrix<double> output_con_pi;
    // ini_matrix(output_con_pi, this->dim_theta, trees_con_pi[0].size());
    ini_matrix(output_con_pi, this->dim_theta, 1);
    
    // Vector of pihat_con and pihat_mod for each tree and sample
    // to be overwritten in each sweep
    vector<vector<double>> pihat_cons(num_trees_con);
    vector<vector<double>> pihat_mods(num_trees_mod);
    for (size_t i = 0; i < num_trees_con; i++)
    {
        pihat_cons[i].resize(N_test);
    }
    for (size_t i = 0; i < num_trees_mod; i++)
    {
        pihat_mods[i].resize(N_test);
    }
    tree::tree_p bn; // pointer to bottom node
    
    for (size_t sweeps = 0; sweeps < num_sweeps; sweeps++)
    {
        // Extract pihat_cons for sweep
        for (size_t i = 0; i < trees_con[sweeps].size(); i++)
        {
            // loop over trees
            // tree search
            // d = 0; // max distance of outliers
            trees_con[sweeps][i].compute_pi_hat(Xtestpointer_con, &Ztestpointer, pihat_cons[i], N_test, 1);
        }
        
        // Extract pihat_mods for sweep
        for (size_t i = 0; i < trees_mod[sweeps].size(); i++)
        {
            // loop over trees
            // tree search
            // d = 0; // max distance of outliers
            trees_mod[sweeps][i].compute_pi_hat(Xtestpointer_con, &Ztestpointer, pihat_mods[i], N_test, 1);
        }
        
        for (size_t data_ind = 0; data_ind < N_test; data_ind++)
        {
            getThetaForObs_Outsample(output_mod, trees_mod[sweeps], data_ind, Xtestpointer_mod, N_test, p_mod);

            getThetaForObs_Outsample(output_con, trees_con[sweeps], data_ind, Xtestpointer_con, N_test, p_con);

            // take sum of predictions of each tree, as final prediction
            for (size_t i = 0; i < trees_mod[0].size(); i++)
            {
                getThetaForObs_Outsample(output_mod_pi, trees_mod_pi[sweeps][i], data_ind, &pihat_mods[i][0], N_test, p_mod);
                
                treatment_xinfo[sweeps][data_ind] += output_mod[i][0];
                treatment_proj_xinfo[sweeps][data_ind] += output_mod_pi[0][0];
            }

            for (size_t i = 0; i < trees_con[0].size(); i++)
            {
                getThetaForObs_Outsample(output_con_pi, trees_con_pi[sweeps][i], data_ind, &pihat_cons[i][0], N_test, p_con);
                
                prognostic_xinfo[sweeps][data_ind] += output_con[i][0];
                prognostic_proj_xinfo[sweeps][data_ind] += output_mod_pi[0][0];
            }

            if (Ztestpointer[0][data_ind] == 1)
            {
                // yhats_test_xinfo[sweeps][data_ind] = (state.a) * prognostic_xinfo[sweeps][data_ind] + (state.b_vec[1]) * treatment_xinfo[sweeps][data_ind];
            }
            else
            {
                // yhats_test_xinfo[sweeps][data_ind] = (state.a) * prognostic_xinfo[sweeps][data_ind] + (state.b_vec[0]) * treatment_xinfo[sweeps][data_ind];
            }
            yhats_test_xinfo[sweeps][data_ind] = prognostic_xinfo[sweeps][data_ind] + treatment_xinfo[sweeps][data_ind];
        }
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::ini_tau_mu_fit(State &state)
{
    double value = state.ini_var_yhat;
    for (size_t i = 0; i < (*state.residual_std)[0].size(); i++)
    {
        (*state.mu_fit)[i] = 0;
        (*state.tau_fit)[i] = value;
        (*state.mu_pi_fit)[i] = 0;
        (*state.tau_pi_fit)[i] = 0;
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::set_treatmentflag(State &state, bool value)
{
    state.treatment_flag = value;
    if (value)
    {
        // if treatment forest
        state.p = state.p_mod;
        state.p_categorical = state.p_categorical_mod;
        state.p_continuous = state.p_continuous_mod;
        state.Xorder_std = state.Xorder_std_mod;
        state.mtry = state.mtry_mod;
        state.num_trees = state.num_trees_mod;
        state.X_std = state.X_std_mod;
        this->alpha = this->alpha_mod;
        this->beta = this->beta_mod;
    }
    else
    {
        state.p = state.p_con;
        state.p_categorical = state.p_categorical_con;
        state.p_continuous = state.p_continuous_con;
        state.Xorder_std = state.Xorder_std_con;
        state.mtry = state.mtry_con;
        state.num_trees = state.num_trees_con;
        state.X_std = state.X_std_con;
        this->alpha = this->alpha_con;
        this->beta = this->beta_con;
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::set_treatment_proj_flag(State &state, bool treatment, bool proj)
{
    state.treatment_flag = treatment;
    state.projection_flag = proj;
    if (treatment & !proj)
    {
        // if treatment forest
        state.p = state.p_mod;
        state.p_categorical = state.p_categorical_mod;
        state.p_continuous = state.p_continuous_mod;
        state.Xorder_std = state.Xorder_std_mod;
        state.mtry = state.mtry_mod;
        state.num_trees = state.num_trees_mod;
        state.X_std = state.X_std_mod;
        this->alpha = this->alpha_mod;
        this->beta = this->beta_mod;
    }
    else if (!treatment & !proj)
    {
        state.p = state.p_con;
        state.p_categorical = state.p_categorical_con;
        state.p_continuous = state.p_continuous_con;
        state.Xorder_std = state.Xorder_std_con;
        state.mtry = state.mtry_con;
        state.num_trees = state.num_trees_con;
        state.X_std = state.X_std_con;
        this->alpha = this->alpha_con;
        this->beta = this->beta_con;
    }
    else if (treatment & proj)
    {
        state.p = state.p_mod_pi;
        state.p_categorical = state.p_categorical_mod_pi;
        state.p_continuous = state.p_continuous_mod_pi;
        state.Xorder_std = state.pi_Xorder_std_mod;
        state.mtry = state.mtry_mod_pi;
        state.num_trees = state.num_trees_mod_pi;
        state.X_std = state.pi_X_std_mod;
        this->alpha = this->alpha_mod_pi;
        this->beta = this->beta_mod_pi;
    } 
    else if (!treatment & proj)
    {
        state.p = state.p_con_pi;
        state.p_categorical = state.p_categorical_con_pi;
        state.p_continuous = state.p_continuous_con_pi;
        state.Xorder_std = state.pi_Xorder_std_con;
        state.mtry = state.mtry_con_pi;
        state.num_trees = state.num_trees_con_pi;
        state.X_std = state.pi_X_std_con;
        this->alpha = this->alpha_con_pi;
        this->beta = this->beta_con_pi;
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::subtract_old_tree_fit(
        size_t tree_ind, State &state, X_struct &x_struct
){
    if (state.treatment_flag & state.projection_flag)
    {
        for (size_t i = 0; i < (*state.tau_pi_fit).size(); i++)
        {
            (*state.tau_pi_fit)[i] -= (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            (*state.tau_fit)[i] -= (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        for (size_t i = 0; i < (*state.mu_pi_fit).size(); i++)
        {
            (*state.mu_pi_fit)[i] -= (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state.mu_fit).size(); i++)
        {
            (*state.mu_fit)[i] -= (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::add_new_tree_fit(
        size_t tree_ind, State &state, X_struct &x_struct
){

    if (state.treatment_flag & state.projection_flag)
    {
        for (size_t i = 0; i < (*state.tau_pi_fit).size(); i++)
        {
            (*state.tau_pi_fit)[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else if (state.treatment_flag & ~state.projection_flag)
    {
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            (*state.tau_fit)[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        for (size_t i = 0; i < (*state.mu_pi_fit).size(); i++)
        {
            (*state.mu_pi_fit)[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    else
    {
        for (size_t i = 0; i < (*state.mu_fit).size(); i++)
        {
            (*state.mu_fit)[i] += (*(x_struct.data_pointers[tree_ind][i]))[0];
        }
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::update_partial_residuals(
        size_t tree_ind, State &state, X_struct &x_struct
){
    if (state.treatment_flag & state.projection_flag)
    {
        // treatment forest
        // (y - a * mu - b * tau) / b
        for (size_t i = 0; i < (*state.tau_pi_fit).size(); i++)
        {
            if ((*state.Z_std)[0][i] == 1)
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.b_vec_pi[1]);
            }
            else
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.b_vec_pi[0]);
            }
        }
    }
    else if (state.treatment_flag & !state.projection_flag)
    {
        // treatment forest
        // (y - a * mu - b * tau) / b
        for (size_t i = 0; i < (*state.tau_fit).size(); i++)
        {
            if ((*state.Z_std)[0][i] == 1)
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.b_vec[1]);
            }
            else
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.b_vec[0]);
            }
        }
    }
    else if (!state.treatment_flag & state.projection_flag)
    {
        // prognostic forest
        // (y - a * mu - b * tau) / a
        for (size_t i = 0; i < (*state.mu_pi_fit).size(); i++)
        {
            if ((*state.Z_std)[0][i] == 1)
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.a_pi);
            }
            else
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.a_pi);
            }
        }
    }
    else
    {
        // prognostic forest
        // (y - a * mu - b * tau) / a
        for (size_t i = 0; i < (*state.mu_fit).size(); i++)
        {
            if ((*state.Z_std)[0][i] == 1)
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.a);
            }
            else
            {
                ((*state.residual_std))[0][i] = ((*state.y_std)[i] - state.a * (*state.mu_fit)[i] - state.a_pi * (*state.mu_pi_fit)[i] - (state.b_vec[1]) * (*state.tau_fit)[i] - (state.b_vec_pi[1]) * (*state.tau_pi_fit)[i]) / (state.a);
            }
        }
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::update_split_counts(
        State &state, size_t tree_ind
){
    if (state.treatment_flag)
    {
        (*state.mtry_weight_current_tree_mod) = (*state.mtry_weight_current_tree_mod) + (*state.split_count_current_tree);
        (*state.split_count_all_tree_mod)[tree_ind] = (*state.split_count_current_tree);
    }
    else
    {
        (*state.mtry_weight_current_tree_con) = (*state.mtry_weight_current_tree_con) + (*state.split_count_current_tree);
        (*state.split_count_all_tree_con)[tree_ind] = (*state.split_count_current_tree);
    }
    return;
}

void XBCFDiscreteProjectedResidualModel::update_a(State &state)
{
    // update parameter a, y = a * mu + b_z * tau

    std::normal_distribution<double> normal_samp(0.0, 1.0);

    double mu2sum_ctrl = 0;
    double mu2sum_trt = 0;
    double muressum_ctrl = 0;
    double muressum_trt = 0;

    // compute the residual y - b * tau(x)

    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            (*state.residual_std)[0][i] = (*state.y_std)[i] - (*state.mu_pi_fit)[i] * state.a_pi - (*state.tau_fit)[i] * state.b_vec[1] - (*state.tau_pi_fit)[i] * state.b_vec_pi[1];
        }
        else
        {
            (*state.residual_std)[0][i] = (*state.y_std)[i] - (*state.mu_pi_fit)[i] * state.a_pi - (*state.tau_fit)[i] * state.b_vec[0] - (*state.tau_pi_fit)[i] * state.b_vec_pi[0];
        }
    }
    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            mu2sum_trt += pow((*state.mu_fit)[i], 2);
            muressum_trt += (*state.mu_fit)[i] * (*state.residual_std)[0][i];
        }
        else
        {
            mu2sum_ctrl += pow((*state.mu_fit)[i], 2);
            muressum_ctrl += (*state.mu_fit)[i] * (*state.residual_std)[0][i];
        }
    }
    // update parameters
    double v0 = 1.0 / (1.0 + mu2sum_ctrl / pow(state.sigma_vec[0], 2));
    double m0 = v0 * (muressum_ctrl) / pow(state.sigma_vec[0], 2);
    double v1 = 1 / (1.0 / v0 + mu2sum_trt / pow(state.sigma_vec[1], 2));
    double m1 = v1 * (m0 / v0 + (muressum_trt) / pow(state.sigma_vec[1], 2));

    state.a = m1 + sqrt(v1) * normal_samp(state.gen);

    return;
}

void XBCFDiscreteProjectedResidualModel::update_b(State &state)
{
    // update b0 and b1 for XBCF discrete treatment

    std::normal_distribution<double> normal_samp(0.0, 1.0);

    double tau2sum_ctrl = 0;
    double tau2sum_trt = 0;
    double tauressum_ctrl = 0;
    double tauressum_trt = 0;

    // compute the residual y-a*mu(x) using state's objects y_std, mu_fit and a
    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            (*state.residual_std)[0][i] = (*state.y_std)[i] - state.a * (*state.mu_fit)[i] - (*state.mu_pi_fit)[i] * state.a_pi - (*state.tau_pi_fit)[i] * state.b_vec_pi[1];
        }
        else
        {
            (*state.residual_std)[0][i] = (*state.y_std)[i] - state.a * (*state.mu_fit)[i] - (*state.mu_pi_fit)[i] * state.a_pi - (*state.tau_pi_fit)[i] * state.b_vec_pi[0];
        }
    }

    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            tau2sum_trt += pow((*state.tau_fit)[i], 2);
            tauressum_trt += (*state.tau_fit)[i] * (*state.residual_std)[0][i];
        }
        else
        {
            tau2sum_ctrl += pow((*state.tau_fit)[i], 2);
            tauressum_ctrl += (*state.tau_fit)[i] * (*state.residual_std)[0][i];
        }
    }

    // update parameters
    double v0 = 1.0 / (2.0 + tau2sum_ctrl / pow(state.sigma_vec[0], 2));
    double v1 = 1.0 / (2.0 + tau2sum_trt / pow(state.sigma_vec[1], 2));

    double m0 = v0 * (tauressum_ctrl) / pow(state.sigma_vec[0], 2);
    double m1 = v1 * (tauressum_trt) / pow(state.sigma_vec[1], 2);

    // sample b0, b1
    double b0 = m0 + sqrt(v0) * normal_samp(state.gen);
    double b1 = m1 + sqrt(v1) * normal_samp(state.gen);

    state.b_vec[1] = b1;
    state.b_vec[0] = b0;

    return;
}

void XBCFDiscreteProjectedResidualModel::update_a_pi(State &state)
{
    // update parameter a, y = a * mu + b_z * tau
    
    std::normal_distribution<double> normal_samp(0.0, 1.0);
    
    double mu2sum_ctrl = 0;
    double mu2sum_trt = 0;
    double muressum_ctrl = 0;
    double muressum_trt = 0;
    
    // compute the residual y - b * tau(x)
    
    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            (*state.residual_std)[0][i] = (*state.y_std)[i] - (*state.mu_fit)[i] * state.a - (*state.tau_fit)[i] * state.b_vec[1] - (*state.tau_pi_fit)[i] * state.b_vec_pi[1];
        }
        else
        {
            (*state.residual_std)[0][i] = (*state.y_std)[i] - (*state.mu_fit)[i] * state.a - (*state.tau_fit)[i] * state.b_vec[0] - (*state.tau_pi_fit)[i] * state.b_vec_pi[0];
        }
    }
    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            mu2sum_trt += pow((*state.mu_pi_fit)[i], 2);
            muressum_trt += (*state.mu_pi_fit)[i] * (*state.residual_std)[0][i];
        }
        else
        {
            mu2sum_ctrl += pow((*state.mu_pi_fit)[i], 2);
            muressum_ctrl += (*state.mu_pi_fit)[i] * (*state.residual_std)[0][i];
        }
    }
    // update parameters
    double v0 = 1.0 / (1.0 + mu2sum_ctrl / pow(state.sigma_vec[0], 2));
    double m0 = v0 * (muressum_ctrl) / pow(state.sigma_vec[0], 2);
    double v1 = 1 / (1.0 / v0 + mu2sum_trt / pow(state.sigma_vec[1], 2));
    double m1 = v1 * (m0 / v0 + (muressum_trt) / pow(state.sigma_vec[1], 2));
    
    state.a = m1 + sqrt(v1) * normal_samp(state.gen);
    
    return;
}

void XBCFDiscreteProjectedResidualModel::update_b_pi(State &state)
{
    // update b0 and b1 for XBCF discrete treatment
    
    std::normal_distribution<double> normal_samp(0.0, 1.0);
    
    double tau2sum_ctrl = 0;
    double tau2sum_trt = 0;
    double tauressum_ctrl = 0;
    double tauressum_trt = 0;
    
    // compute the residual y-a*mu(x) using state's objects y_std, mu_fit and a
    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            // if treated
            (*state.residual_std)[0][i] = (*state.y_std)[i] - state.a * (*state.mu_fit)[i] - (*state.mu_pi_fit)[i] * state.a_pi - (*state.tau_fit)[i] * state.b_vec[1];
        }
        else
        {
            (*state.residual_std)[0][i] = (*state.y_std)[i] - state.a * (*state.mu_fit)[i] - (*state.mu_pi_fit)[i] * state.a_pi - (*state.tau_fit)[i] * state.b_vec[0];
        }
    }
    
    for (size_t i = 0; i < state.n_y; i++)
    {
        if ((*state.Z_std)[0][i] == 1)
        {
            tau2sum_trt += pow((*state.tau_pi_fit)[i], 2);
            tauressum_trt += (*state.tau_pi_fit)[i] * (*state.residual_std)[0][i];
        }
        else
        {
            tau2sum_ctrl += pow((*state.tau_pi_fit)[i], 2);
            tauressum_ctrl += (*state.tau_pi_fit)[i] * (*state.residual_std)[0][i];
        }
    }
    
    // update parameters
    double v0 = 1.0 / (2.0 + tau2sum_ctrl / pow(state.sigma_vec[0], 2));
    double v1 = 1.0 / (2.0 + tau2sum_trt / pow(state.sigma_vec[1], 2));
    
    double m0 = v0 * (tauressum_ctrl) / pow(state.sigma_vec[0], 2);
    double m1 = v1 * (tauressum_trt) / pow(state.sigma_vec[1], 2);
    
    // sample b0, b1
    double b0 = m0 + sqrt(v0) * normal_samp(state.gen);
    double b1 = m1 + sqrt(v1) * normal_samp(state.gen);
    
    state.b_vec[1] = b1;
    state.b_vec[0] = b0;
    
    return;
}
