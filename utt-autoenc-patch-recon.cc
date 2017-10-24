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

struct prediction_env {

    speech::scp frame_scp;

    std::shared_ptr<tensor_tree::vertex> param;

    int patch_time;
    int patch_freq;

    std::unordered_map<std::string, std::string> args;

    prediction_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "utt-autoenc-patch-recon",
        "Reconstruct input with a patch autoencoder",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"patch-time", "", false},
            {"patch-freq", "", false},
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

    prediction_env env { args };

    env.run();

    return 0;
}

prediction_env::prediction_env(std::unordered_map<std::string, std::string> args)
    : args(args)
{
    frame_scp.open(args.at("frame-scp"));

    std::ifstream param_ifs { args.at("param") };
    param = autoenc::make_symmetric_ae_tensor_tree();
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    patch_time = 5;
    if (ebt::in(std::string("patch-time"), args)) {
        patch_time = std::stoi(args.at("patch-time"));
    }

    patch_freq = 5;
    if (ebt::in(std::string("patch-freq"), args)) {
        patch_freq = std::stoi(args.at("patch-freq"));
    }
}

void prediction_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    while (nsample < frame_scp.entries.size()) {
        autodiff::computation_graph graph;
        std::shared_ptr<tensor_tree::vertex> var_tree = tensor_tree::make_var_tree(graph, param);

        std::vector<std::vector<double>> frames = speech::load_frame_batch(
            frame_scp.at(nsample));

        int input_dim = frames.front().size();

        std::vector<double> input_vec;

        for (int i = 0; i < frames.size(); ++i) {
            for (int j = 0; j < input_dim; ++j) {
                input_vec.push_back(frames[i][j]);
            }
        }

        la::cpu::tensor<double> input_tensor { la::cpu::vector<double>(input_vec),
            { 1, (unsigned int) frames.size(), (unsigned int) input_dim, 1 }};

        la::cpu::tensor<double> input_corr_lin;
        input_corr_lin.resize({(unsigned int) frames.size(), (unsigned int) input_dim, (unsigned int) patch_time * patch_freq});

        la::cpu::corr_linearize(input_corr_lin, input_tensor, patch_time, patch_freq);

        auto input = graph.var(input_corr_lin);

        std::shared_ptr<autodiff::op_t> pred = autoenc::make_symmetric_ae(input, var_tree, 0.0, 0.0, nullptr);

        auto& pred_t = autodiff::get_output<la::cpu::tensor_like<double>>(pred);

        la::cpu::tensor<double> input_recon;
        input_recon.resize({ 1, (unsigned int) frames.size(), (unsigned int) input_dim, 1 });

        la::cpu::corr_delinearize(input_recon, pred_t, patch_time, patch_freq, 0, 0, 1, 1);

        std::cout << frame_scp.entries[nsample].key << std::endl;
        for (int i = 0; i < frames.size(); ++i) {
            for (int j = 0; j < input_dim; ++j) {
                std::cout << input_recon({0, i, j, 0}) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "." << std::endl;

        ++nsample;

    }
}

