#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "commondef.h"
#include "runtime.h"
#include "sampler.h"
#include "tensor.h"

namespace {

double percentile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>((values.size() - 1) * q + 0.5);
    if (index >= values.size()) {
        index = values.size() - 1;
    }
    return values[index];
}

void print_result(const std::string & status, const std::string & error, const std::vector<double> & times_ms) {
    if (status != "ok") {
        std::cout << status << "," << error << ",,,\n";
        return;
    }
    std::cout << "ok,,"
              << percentile(times_ms, 0.10) << ","
              << percentile(times_ms, 0.50) << ","
              << percentile(times_ms, 0.90) << "\n";
}

} // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <model_file> <backend> <task> <B> <T> <warmup> <repeat>\n";
        return 2;
    }

    const char * model_path = argv[1];
    const char * backend = argv[2];
    std::string task = argv[3];
    int B = std::atoi(argv[4]);
    int T = std::atoi(argv[5]);
    int warmup = std::atoi(argv[6]);
    int repeat = std::atoi(argv[7]);

    if (task == "batch_prefill") {
        print_result("unsupported", "rwkv-mobile Runtime exposes eval_logits for one sequence and eval_logits_batch_decode for BnT1, but no true direct BnTn batch prefill entrypoint", {});
        return 0;
    }

    rwkvmobile::Runtime runtime;
    int model_id = runtime.load_model(model_path, backend, "", nullptr);
    if (model_id < 0) {
        print_result("failed", "load_model failed", {});
        return 0;
    }

    int vocab_size = runtime.get_vocab_size(model_id);
    std::mt19937 rng(static_cast<uint32_t>(B * 1000003 + T * 9176));
    std::uniform_int_distribution<int> dist(0, std::max(1, vocab_size - 1));
    rwkvmobile::Tensor1D logits;
    rwkvmobile::NucleusSampler sampler;
    sampler.set_temperature(runtime.get_temperature(model_id));
    sampler.set_top_k(runtime.get_top_k(model_id));
    sampler.set_top_p(runtime.get_top_p(model_id));
    std::vector<int> prefill_ids(std::max(1, T));
    for (int & id : prefill_ids) id = dist(rng);
    const int decode_prefix_id = dist(rng);
    const int decode_id = dist(rng);
    std::vector<int> batch_decode_ids(std::max(1, B));
    for (int & id : batch_decode_ids) id = dist(rng);

    auto prepare_once = [&]() -> int {
        runtime.clear_state(model_id);
        if (task == "decode") {
            int ret = runtime.eval_logits(model_id, decode_prefix_id, logits);
            if (ret != rwkvmobile::RWKV_SUCCESS) return ret;
        }
        return rwkvmobile::RWKV_SUCCESS;
    };

    auto run_measured = [&]() -> int {
        if (task == "decode") {
            int ret = runtime.eval_logits(model_id, decode_id, logits);
            if (ret != rwkvmobile::RWKV_SUCCESS) return ret;
            (void)sampler.sample(logits, static_cast<size_t>(vocab_size));
            return rwkvmobile::RWKV_SUCCESS;
        }
        if (task == "prefill") {
            int ret = runtime.eval_logits(model_id, prefill_ids, logits);
            if (ret != rwkvmobile::RWKV_SUCCESS) return ret;
            (void)sampler.sample(logits, static_cast<size_t>(vocab_size));
            return rwkvmobile::RWKV_SUCCESS;
        }
        if (task == "batch_decode") {
            int ret = B == 1 ? runtime.eval_logits(model_id, batch_decode_ids[0], logits) : runtime.eval_logits_batch_decode(model_id, batch_decode_ids, logits);
            if (ret != rwkvmobile::RWKV_SUCCESS) return ret;
            (void)sampler.sample_batch(logits, static_cast<size_t>(vocab_size), static_cast<size_t>(vocab_size), B);
            return rwkvmobile::RWKV_SUCCESS;
        }
        return rwkvmobile::RWKV_ERROR_INVALID_PARAMETERS;
    };

    for (int i = 0; i < warmup; ++i) {
        int ret = prepare_once();
        if (ret == rwkvmobile::RWKV_SUCCESS) {
            ret = run_measured();
        }
        if (ret != rwkvmobile::RWKV_SUCCESS) {
            print_result("failed", "warmup eval failed", {});
            runtime.release();
            return 0;
        }
    }

    std::vector<double> times_ms;
    times_ms.reserve(static_cast<size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        int ret = prepare_once();
        if (ret != rwkvmobile::RWKV_SUCCESS) {
            print_result("failed", "prepare eval failed", {});
            runtime.release();
            return 0;
        }
        auto start = std::chrono::steady_clock::now();
        ret = run_measured();
        auto end = std::chrono::steady_clock::now();
        if (ret != rwkvmobile::RWKV_SUCCESS) {
            print_result("failed", "eval failed", {});
            runtime.release();
            return 0;
        }
        times_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    print_result("ok", "", times_ms);
    runtime.release();
    return 0;
}
