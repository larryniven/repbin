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

void block_noise(la::cpu::tensor<double>& input, int win_size, int input_dim,
    int time, int freq, std::default_random_engine& gen)
{
    for (int b = 0; b < input.size(0); ++b) {
        std::uniform_int_distribution<> time_dist {0, win_size - time};
        std::uniform_int_distribution<> freq_dist {0, input_dim - freq};

        int t = time_dist(gen);
        int f = freq_dist(gen);

        for (int i = t; i < time; ++i) {
            for (int j = f; j < freq; ++j) {
                input({b, i * input_dim + j}) = 0;
            }
        }
    }
}

struct learning_env {

    speech::scp frame_scp;

    std::shared_ptr<tensor_tree::vertex> param;

    unsigned int win_size;

    double input_dropout;
    double hidden_dropout;

    int block_noise_time;
    int block_noise_freq;

    std::string output_param;
    std::string output_opt_data;

    std::shared_ptr<tensor_tree::optimizer> opt;

    double step_size;
    double decay;
    double clip;
    unsigned int batch_size;

    int seed;
    std::default_random_engine gen;

    int save_every;
    std::string save_every_prefix;

    std::vector<std::pair<int, int>> indices;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "frame-autoenc-fc-win-learn",
        "Train an FC frame autoencoder",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"opt-data", "", true},
            {"output-param", "", false},
            {"output-opt-data", "", false},
            {"win-size", "", true},
            {"input-dropout", "", false},
            {"hidden-dropout", "", false},
            {"block-noise-time", "", false},
            {"block-noise-freq", "", false},
            {"batch-size", "", false},
            {"seed", "", false},
            {"shuffle", "", false},
            {"opt", "const-step,adagrad,rmsprop,adam", true},
            {"step-size", "", true},
            {"momentum", "", false},
            {"clip", "", false},
            {"decay", "", false},
            {"save-every", "", false},
            {"save-every-prefix", "", false},
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

    win_size = std::stoi(args.at("win-size"));

    block_noise_time = -1;
    if (ebt::in(std::string("block-noise-time"), args)) {
        block_noise_time = std::stoi(args.at("block-noise-time"));
    }

    block_noise_freq = -1;
    if (ebt::in(std::string("block-noise-freq"), args)) {
        block_noise_freq = std::stoi(args.at("block-noise-freq"));
    }

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
        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_scp.at(i));

        for (int t = 0; t < frames.size(); ++t) {
            indices.push_back(std::make_pair(i, t));
        }
    }
    std::cout << std::endl;

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

        std::vector<double> input_tensor_vec;
        std::vector<double> gold_vec;

        unsigned int input_dim = 0;
        unsigned int loaded_samples = 0;

        while (nsample < indices.size() && loaded_samples < batch_size) {
            std::vector<std::vector<double>> frames = speech::load_frame_batch(
                frame_scp.at(indices.at(nsample).first));

            input_dim = frames.front().size();

            int t = indices.at(nsample).second;

            for (int i = 0; i < win_size; ++i) {
                if (0 <= t + i - (int) win_size / 2 && t + i - (int) win_size / 2 < frames.size()) {
                    for (int j = 0; j < input_dim; ++j) {
                        input_tensor_vec.push_back(frames[t + i - (int) win_size / 2][j]);
                        gold_vec.push_back(frames[t + i - (int) win_size / 2][j]);
                    }
                } else {
                    for (int j = 0; j < input_dim; ++j) {
                        input_tensor_vec.push_back(0);
                        gold_vec.push_back(0);
                    }
                }
            }

            ++nsample;
            ++loaded_samples;
        }

        std::cout << "loaded samples: " << loaded_samples << std::endl;

        assert(input_tensor_vec.size() == win_size * input_dim * loaded_samples);

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_tensor_vec),
            std::vector<unsigned int> { loaded_samples, win_size * input_dim }};

        if (block_noise_time != -1 && block_noise_freq != -1) {
            block_noise(input_tensor, win_size, input_dim, block_noise_time, block_noise_freq, gen);
        }

        std::shared_ptr<autodiff::op_t> input = graph.var(input_tensor);

        std::shared_ptr<autodiff::op_t> pred = autoenc::make_symmetric_ae(input, var_tree, input_dropout, hidden_dropout, gen);

        auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

        la::cpu::tensor<double> gold_t { la::cpu::vector<double>(gold_vec),
            { loaded_samples, win_size * input_dim } };

        nn::l2_loss loss { gold_t, pred_t };

        pred->grad = std::make_shared<la::cpu::tensor<double>>(loss.grad());
        
        double ell = loss.loss();

        if (std::isnan(ell)) {
            std::cerr << "loss is nan" << std::endl;
            exit(1);
        }

        std::cout << "loss: " << ell / loaded_samples << std::endl;

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

        tensor_tree::imul(grad, 1.0 / loaded_samples);

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

    }

    std::ofstream param_ofs { output_param };
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();

    std::ofstream opt_data_ofs { output_opt_data };
    opt->save_opt_data(opt_data_ofs);
    opt_data_ofs.close();
}

