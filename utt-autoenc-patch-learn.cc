#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include "nn/nn.h"
#include "nn/tensor-tree.h"
#include "nn/autoenc-fc.h"
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>

struct learning_env {

    speech::scp frame_scp;

    std::shared_ptr<tensor_tree::vertex> param;

    double input_dropout;
    double hidden_dropout;

    int batch_size;
    int rand_patches_per_utt;

    int patch_time;
    int patch_freq;

    std::string output_param;
    std::string output_opt_data;

    std::shared_ptr<tensor_tree::optimizer> opt;

    double step_size;
    double decay;
    double clip;

    int seed;
    std::default_random_engine gen;

    int save_every;
    std::string save_every_prefix;

    std::vector<int> indices;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "utt-autoenc-patch-learn",
        "Train a patch autoencoder",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"input-dropout", "", false},
            {"hidden-dropout", "", false},
            {"patch-time", "", false},
            {"patch-freq", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,adagrad,rmsprop,adam", true},
            {"step-size", "", true},
            {"momentum", "", false},
            {"clip", "", false},
            {"decay", "", false},
            {"save-every", "", false},
            {"save-every-prefix", "", false},
            {"loss-mode", "", false},
        }
    };

    if (argc == 1) {
        ebt::usage(spec);
        exit(1);
    }

    auto args = ebt::parse_args(argc, argv, spec);

    for (int i = 0; i < argc; ++i) {
        std::cout << argv[i] << " ";
    }
    std::cout << std::endl;

    learning_env env { args };

    env.run();

    return 0;
}

learning_env::learning_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_scp.open(args.at("frame-scp"));

    std::ifstream param_ifs { args.at("param") };
    param = autoenc::make_symmetric_ae_tensor_tree();
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    step_size = std::stod(args.at("step-size"));

    if (ebt::in(std::string("decay"), args)) {
        decay = std::stod(args.at("decay"));
    }

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    output_opt_data = "opt-data-last";
    if (ebt::in(std::string("output-opt-data"), args)) {
        output_opt_data = args.at("output-opt-data");
    }

    clip = std::numeric_limits<double>::infinity();
    if (ebt::in(std::string("clip"), args)) {
        clip = std::stod(args.at("clip"));
    }

    batch_size = 1;
    if (ebt::in(std::string("batch-size"), args)) {
        batch_size = std::stoi(args.at("batch-size"));
    }

    patch_time = 5;
    if (ebt::in(std::string("patch-time"), args)) {
        patch_time = std::stoi(args.at("patch-time"));
    }

    patch_freq = 5;
    if (ebt::in(std::string("patch-freq"), args)) {
        patch_freq = std::stoi(args.at("patch-freq"));
    }

    rand_patches_per_utt = 50;
    if (ebt::in(std::string("rand-patches-per-utt"), args)) {
        rand_patches_per_utt = std::stoi(args.at("rand-patches-per-utt"));
    }

    input_dropout = 0;
    if (ebt::in(std::string("input-dropout"), args)) {
        input_dropout = std::stod(args.at("input-dropout"));
    }

    hidden_dropout = 0;
    if (ebt::in(std::string("hidden-dropout"), args)) {
        hidden_dropout = std::stod(args.at("hidden-dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    save_every = -1;
    if (ebt::in(std::string("save-every"), args)) {
        save_every = std::stoi(args.at("save-every"));
    }

    if (args.at("opt") == "rmsprop") {
        double decay = std::stod(args.at("decay"));
        opt = std::make_shared<tensor_tree::rmsprop_opt>(
            tensor_tree::rmsprop_opt(param, step_size, decay));
    } else if (args.at("opt") == "const-step") {
        opt = std::make_shared<tensor_tree::const_step_opt>(
            tensor_tree::const_step_opt(param, step_size));
    } else if (args.at("opt") == "const-step-momentum") {
        double momentum = std::stod(args.at("momentum"));
        opt = std::make_shared<tensor_tree::const_step_momentum_opt>(
            tensor_tree::const_step_momentum_opt(param, step_size, momentum));
    } else if (args.at("opt") == "adagrad") {
        opt = std::make_shared<tensor_tree::adagrad_opt>(
            tensor_tree::adagrad_opt(param, step_size));
    } else {
        std::cout << "unknown optimizer " << args.at("opt") << std::endl;
        exit(1);
    }

    std::ifstream opt_data_ifs { args.at("opt-data") };
    opt->load_opt_data(opt_data_ifs);
    opt_data_ifs.close();

    gen = std::default_random_engine { seed };

    for (int i = 0; i < frame_scp.entries.size(); ++i) {
        indices.push_back(i);
    }

    if (ebt::in(std::string("shuffle"), args)) {
        std::shuffle(indices.begin(), indices.end(), gen);
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < indices.size()) {
        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(indices.at(nsample)));

        int input_dim = frames.front().size();

        std::vector<double> input_vec;

        for (int i = 0; i < frames.size(); ++i) {
            for (int j = 0; j < input_dim; ++j) {
                input_vec.push_back(frames[i][j]);
            }
        }

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_vec),
            { 1, (unsigned int) frames.size(), (unsigned int) input_dim, 1 }};

        la::cpu::tensor<double> gold_corr_lin;
        gold_corr_lin.resize({(unsigned int) frames.size(), (unsigned int) input_dim, (unsigned int) patch_time * patch_freq});

        la::cpu::corr_linearize(gold_corr_lin, input_tensor, patch_time, patch_freq);

        auto input_var = graph.var(input_tensor);

        if (input_dropout != 0.0) {
            auto mask = autodiff::dropout_mask(input_var, input_dropout, gen);
            input_var = autodiff::emul(input_var, mask);
        }

        auto& input_t = autodiff::get_output<la::cpu::tensor_like<double>>(input_var);

        la::cpu::tensor<double> input_corr_lin;
        input_corr_lin.resize({(unsigned int) frames.size(), (unsigned int) input_dim, (unsigned int) patch_time * patch_freq});
        la::cpu::corr_linearize(input_corr_lin, input_t, patch_time, patch_freq);

        auto input = graph.var(input_corr_lin);

        std::shared_ptr<autodiff::op_t> pred = autoenc::make_symmetric_ae(input, var_tree, 0.0, hidden_dropout, gen);

        auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

        nn::l2_loss loss { gold_corr_lin, pred_t };

        pred->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell << std::endl;
        std::cout << "E: " << ell / frames.size() << std::endl;

        if (ebt::in(std::string("loss-mode"), args)) {
            ++nsample;
            continue;
        }

        auto topo_order = autodiff::natural_topo_order(graph);
        autodiff::guarded_grad(topo_order, autodiff::grad_funcs);

        std::shared_ptr<tensor_tree::vertex> grad = autoenc::make_symmetric_ae_tensor_tree();
        tensor_tree::copy_grad(grad, var_tree);

#if 0
        {
            std::shared_ptr<tensor_tree::vertex> param2 = tensor_tree::deep_copy(param);
            la::cpu::tensor<double>& t = tensor_tree::get_tensor(param2->children[0]->children[0]);
            t({0, 0, 0, 0}) += 1e-8;

            autodiff::computation_graph graph2;
            auto var_tree2 = tensor_tree::make_var_tree(graph2, param2);
            std::shared_ptr<autodiff::op_t> input2 = graph2.var(input_tensor);

            std::shared_ptr<cnn::transcriber> trans2 = cnn::make_transcriber(cnn_config, dropout, &gen);
            std::shared_ptr<autodiff::op_t> logprob2 = (*trans)(var_tree2, input2);

            auto& pred2 = autodiff::get_output<la::cpu::tensor_like<double>>(logprob2);
            nn::log_loss loss2 { gold, pred2 };

            double ell2 = loss2.loss();

            std::cout << "numeric grad: " << (ell2 - ell) / 1e-8 << std::endl;

            la::cpu::tensor<double>& grad_t = tensor_tree::get_tensor(grad->children[0]->children[0]);
            std::cout << "analytic grad: " << grad_t({0, 0, 0, 0}) << std::endl;
        }
#endif

        double n = tensor_tree::norm(grad);

        std::cout << "grad norm: " << n << std::endl;

        if (ebt::in(std::string("clip"), args)) {
            if (n > clip) {
                tensor_tree::imul(grad, clip / n);
                std::cout << "gradient clipped" << std::endl;
            }
        }

        std::vector<std::shared_ptr<tensor_tree::vertex>> vars
            = tensor_tree::leaves_pre_order(param);

        la::cpu::tensor<double> const& v = tensor_tree::get_tensor(vars.front());

        double v1 = v.data()[0];

        opt->update(grad);

        double v2 = v.data()[0];

        std::cout << vars.front()->name << " weight: " << v1
            << " update: " << v2 - v1 << " rate: " << (v2 - v1) / v1 << std::endl;

        std::cout << std::endl;

        if (save_every > 0 && nsample % save_every == 0) {
            std::ofstream param_ofs { args.at("save-every-prefix") + "-" + std::to_string(nsample) };
            tensor_tree::save_tensor(param, param_ofs);
            param_ofs.close();
        }

        ++nsample;

    }

    if (ebt::in(std::string("loss-mode"), args)) {
        return;
    }

    std::ofstream param_ofs { output_param };
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

