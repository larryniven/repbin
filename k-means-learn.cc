#include "la/la.h"
#include "autodiff/autodiff.h"
#include "ebt/ebt.h"
#include "speech/speech.h"
#include <fstream>
#include <vector>
#include "nn/nn.h"
#include <random>
#include "nn/tensor-tree.h"
#include "nn/cnn.h"
#include "nn/cnn-frame.h"
#include <algorithm>

std::shared_ptr<tensor_tree::vertex> make_tensor_tree(int layer)
{
    assert(layer == 1);

    tensor_tree::vertex root { "nil" };

    for (int i = 0; i < layer; ++i) {
        root.children.push_back(tensor_tree::make_tensor("means"));
    }

    return std::make_shared<tensor_tree::vertex>(root);
}

struct learning_env {

    speech::scp frame_scp;

    std::shared_ptr<tensor_tree::vertex> param;

    unsigned int win_size;

    double input_dropout;
    int seed;

    std::string output_param;

    std::default_random_engine gen;

    double center_prior;
    int max_nsample;

    std::unordered_map<std::string, std::string> args;

    learning_env(std::unordered_map<std::string, std::string> args);

    void run();

};

int main(int argc, char *argv[])
{
    ebt::ArgumentSpec spec {
        "k-means-learn",
        "Find k means",
        {
            {"frame-scp", "", true},
            {"param", "", true},
            {"output-param", "", false},
            {"win-size", "", true},
            {"input-dropout", "", false},
            {"max-nsample", "", false},
            {"random-init", "", false},
            {"center-prior", "", false},
            {"seed", "", false}
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
    param = make_tensor_tree(1);
    tensor_tree::load_tensor(param, param_ifs);
    param_ifs.close();

    win_size = std::stoi(args.at("win-size"));

    output_param = "param-last";
    if (ebt::in(std::string("output-param"), args)) {
        output_param = args.at("output-param");
    }

    input_dropout = 0;
    if (ebt::in(std::string("input-dropout"), args)) {
        input_dropout = std::stod(args.at("input-dropout"));
    }

    seed = 1;
    if (ebt::in(std::string("seed"), args)) {
        seed = std::stoi(args.at("seed"));
    }

    max_nsample = -1;
    if (ebt::in(std::string("max-nsample"), args)) {
        max_nsample = std::stoi(args.at("max-nsample"));
    }

    gen = std::default_random_engine { seed };

    if (ebt::in(std::string("center-prior"), args)) {
        center_prior = std::stod(args.at("center-prior"));
        assert(0 <= center_prior && center_prior <= 1);
    }
}

void learning_env::run()
{
    ebt::Timer timer;

    int nsample = 0;

    la::cpu::tensor_like<double>& centers = tensor_tree::get_tensor(param->children[0]);

    la::cpu::tensor<double> new_centers;
    la::cpu::resize_as(new_centers, centers);
    std::vector<int> nassign;
    nassign.resize(centers.size(0));

    double loss_sum = 0;
    int nloss = 0;

    std::bernoulli_distribution dist { center_prior };

    while (ebt::in(std::string("random-init"), args) && nsample < frame_scp.entries.size()) {
        if (max_nsample != -1 && nsample >= max_nsample) {
            break;
        }

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_scp.at(nsample));

        unsigned int input_dim = frames.front().size();

        for (int t = 0; t < frames.size(); ++t) {
            if (t - (int) win_size / 2 < 0 || t + (int) win_size / 2 >= frames.size()) {
                continue;
            }

            std::vector<double> input_tensor_vec;

            for (int i = 0; i < win_size; ++i) {
                for (int j = 0; j < input_dim; ++j) {
                    input_tensor_vec.push_back(frames[t + i - win_size / 2][j]);
                }
            }

            assert(input_tensor_vec.size() == win_size * input_dim);

            la::cpu::weak_tensor<double> x_t { input_tensor_vec.data(), {win_size * input_dim} };

            int nactive = 0;
            int argmin = -1;

            if (dist(gen) == false) {
                continue;
            }

            for (int i = 0; i < nassign.size(); ++i) {
                if (nassign[i] == 0) {
                    argmin = i;
                    break;
                } else {
                    nactive += 1;
                }
            }

            if (nactive == nassign.size()) {
                break;
            }

            if (argmin == -1) {
                continue;
            }

            nassign[argmin] += 1;

            la::cpu::weak_tensor<double> z_star { new_centers.data() + argmin * win_size * input_dim,
                { win_size * input_dim }};

            la::cpu::imul(z_star, (nassign[argmin] - 1) / double(nassign[argmin]));
            la::cpu::imul(x_t, 1.0 / nassign[argmin]);
            la::cpu::iadd(z_star, x_t);
        }

        std::cout << "active: ";

        int nactive = 0;

        for (int i = 0; i < nassign.size(); ++i) {
            if (nassign[i] != 0) {
                std::cout << i << " ";
                nactive += 1;
            }
        }
        std::cout << std::endl;

        std::cout << "nactive: " << nactive << std::endl;
        std::cout << "assignment: " << nassign << std::endl;
        std::cout << "nsample: " << nsample << std::endl;
        std::cout << std::endl;

        ++nsample;
    }

    while (nsample < frame_scp.entries.size()) {

        if (max_nsample != -1 && nsample >= max_nsample) {
            break;
        }

        std::vector<std::vector<double>> frames = speech::load_frame_batch(frame_scp.at(nsample));

        unsigned int input_dim = frames.front().size();

        for (int t = 0; t < frames.size(); ++t) {
            if (t - (int) win_size / 2 < 0 || t + (int) win_size / 2 >= frames.size()) {
                continue;
            }

            std::vector<double> input_tensor_vec;

            for (int i = 0; i < win_size; ++i) {
                for (int j = 0; j < input_dim; ++j) {
                    input_tensor_vec.push_back(frames[t + i - win_size / 2][j]);
                }
            }

            assert(input_tensor_vec.size() == win_size * input_dim);

            la::cpu::weak_tensor<double> x_t { input_tensor_vec.data(), {win_size * input_dim} };

            double min = std::numeric_limits<double>::infinity();
            int argmin = -1;

            for (int c = 0; c < centers.size(0); ++c) {
                la::cpu::weak_tensor<double> z_c { centers.data() + c * win_size * input_dim,
                    { win_size * input_dim }};

                nn::l2_loss loss_func { z_c, x_t };

                double loss = loss_func.loss();

                if (loss < min) {
                    argmin = c;
                    min = loss;
                }
            }

            loss_sum += min;
            nloss += 1;

            nassign[argmin] += 1;

            la::cpu::weak_tensor<double> z_star { new_centers.data() + argmin * win_size * input_dim,
                { win_size * input_dim }};

            la::cpu::imul(z_star, (nassign[argmin] - 1) / double(nassign[argmin]));
            la::cpu::imul(x_t, 1.0 / nassign[argmin]);
            la::cpu::iadd(z_star, x_t);
        }

        std::cout << "active: ";

        int nactive = 0;

        for (int i = 0; i < nassign.size(); ++i) {
            if (nassign[i] != 0) {
                std::cout << i << " ";
                nactive += 1;
            }
        }
        std::cout << std::endl;

        std::cout << "nactive: " << nactive << std::endl;
        std::cout << "assignment: " << nassign << std::endl;
        std::cout << "nsample: " << nsample << std::endl;
        std::cout << "nloss: " << nloss << std::endl;
        std::cout << "loss: " << loss_sum / nloss << std::endl;
        std::cout << std::endl;

        ++nsample;
    }

    param->children[0]->data = std::make_shared<la::cpu::tensor<double>>(new_centers);

    std::ofstream param_ofs { output_param };
    tensor_tree::save_tensor(param, param_ofs);
    param_ofs.close();
}

