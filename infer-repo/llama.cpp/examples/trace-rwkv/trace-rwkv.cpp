#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr const char * RWKV_TRACE_PROMPT =
    "User: You are a very talented expert in aime24. Solve the problem and output the final answer in \\boxed{}. Problem: Let AB​CD be a tetrahedron such that AB = CD = \\sqrt{41}, AC = BD = \\sqrt{80}, and BC = AD = \\sqrt{89}. There exists a point I inside the tetrahedron such that the distances from I to each of the faces of the tetrahedron are all equal. This distance can be written in the form \\frac{m\\sqrt{n}}{p}, where m, n, and p are positive integers, m and p are relatively prime, and n is not divisible by the square of any prime. Find m + n + p. Assistant: <think";

struct trace_context {
    std::filesystem::path case_root;
    std::map<std::string, std::string> node_to_file;
    std::set<std::string> exported;
};

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s --model MODEL.gguf [-ngl N]\n", argv0);
}

static std::string tensor_name(const std::string & filename) {
    return std::filesystem::path(filename).stem().string();
}

static void write_trace_time(const std::filesystem::path & path, const std::string & filename, long long elapsed_ns) {
    auto time_path = path;
    time_path.replace_extension("time.json");
    std::ofstream out(time_path);
    out << "{\"filename\":\"" << filename << "\",\"elapsed_ns\":" << elapsed_ns << "}";
}

static const char * ggml_dtype_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_BF16: return "BF16";
        case GGML_TYPE_I64:  return "I64";
        case GGML_TYPE_I32:  return "I32";
        case GGML_TYPE_I16:  return "I16";
        case GGML_TYPE_I8:   return "I8";
        default: throw std::runtime_error(std::string("unsupported ggml dtype for trace: ") + ggml_type_name(type));
    }
}

static std::string shape_json(const std::vector<size_t> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) {
            out += ",";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

static void write_u64_le(std::ofstream & out, uint64_t value) {
    char bytes[8];
    for (int i = 0; i < 8; ++i) {
        bytes[i] = char((value >> (8 * i)) & 0xff);
    }
    out.write(bytes, sizeof(bytes));
}

static void save_safetensor(
        const std::filesystem::path & path,
        const std::string & name,
        const std::string & dtype,
        const std::vector<size_t> & shape,
        const std::vector<uint8_t> & bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::string header = "{\"" + name + "\":{\"dtype\":\"" + dtype + "\",\"shape\":" + shape_json(shape) +
                         ",\"data_offsets\":[0," + std::to_string(bytes.size()) + "]}}";
    const size_t pad = (8 - (header.size() % 8)) % 8;
    header.append(pad, ' ');

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open trace output: " + path.string());
    }
    write_u64_le(out, header.size());
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
}

static void trace_bytes(
        const std::filesystem::path & case_root,
        const std::string & filename,
        const std::string & dtype,
        const std::vector<size_t> & shape,
        const std::vector<uint8_t> & bytes) {
    const auto path = case_root / filename;
    const auto start = std::chrono::steady_clock::now();
    save_safetensor(path, tensor_name(filename), dtype, shape, bytes);
    const auto end = std::chrono::steady_clock::now();
    write_trace_time(path, filename, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
}

static void trace_tokens(const std::filesystem::path & case_root, const std::vector<llama_token> & tokens) {
    std::vector<int64_t> tokens_i64(tokens.begin(), tokens.end());
    std::vector<uint8_t> bytes(tokens_i64.size() * sizeof(int64_t));
    memcpy(bytes.data(), tokens_i64.data(), bytes.size());
    trace_bytes(case_root, "embedding/token_ids.safetensors", "I64", {tokens.size()}, bytes);
}

static void trace_ggml(const std::filesystem::path & case_root, const std::string & filename, const ggml_tensor * tensor) {
    const bool trace_last_token_only =
        filename == "lm_head/embedded_context.safetensors" ||
        filename == "lm_head/logits.safetensors";

    std::vector<size_t> shape;
    if (trace_last_token_only && tensor->ne[1] > 1) {
        shape.push_back(static_cast<size_t>(tensor->ne[0]));
    } else {
        for (int i = ggml_n_dims(tensor) - 1; i >= 0; --i) {
            shape.push_back(static_cast<size_t>(tensor->ne[i]));
        }
    }

    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    if (tensor->type == GGML_TYPE_F32) {
        const auto * f32_all = reinterpret_cast<const float *>(bytes.data());
        const size_t offset = trace_last_token_only && tensor->ne[1] > 1 ? (tensor->ne[1] - 1) * tensor->ne[0] : 0;
        const auto * f32 = f32_all + offset;
        const size_t n = trace_last_token_only && tensor->ne[1] > 1 ? tensor->ne[0] : bytes.size() / sizeof(float);
        std::vector<ggml_fp16_t> f16(n);
        for (size_t i = 0; i < n; ++i) {
            f16[i] = ggml_fp32_to_fp16(f32[i]);
        }
        std::vector<uint8_t> f16_bytes(f16.size() * sizeof(ggml_fp16_t));
        memcpy(f16_bytes.data(), f16.data(), f16_bytes.size());
        trace_bytes(case_root, filename, "F16", shape, f16_bytes);
        return;
    }
    if (trace_last_token_only && tensor->ne[1] > 1) {
        const size_t row_size = static_cast<size_t>(tensor->ne[0]) * ggml_element_size(tensor);
        const size_t offset = static_cast<size_t>(tensor->ne[1] - 1) * row_size;
        std::vector<uint8_t> last(bytes.begin() + offset, bytes.begin() + offset + row_size);
        trace_bytes(case_root, filename, ggml_dtype_name(tensor->type), shape, last);
        return;
    }
    trace_bytes(case_root, filename, ggml_dtype_name(tensor->type), shape, bytes);
}

static bool trace_cb_eval(ggml_tensor * t, bool ask, void * user_data) {
    auto * trace = static_cast<trace_context *>(user_data);
    const auto it = trace->node_to_file.find(t->name);
    if (ask) {
        return it != trace->node_to_file.end() && !trace->exported.count(t->name);
    }
    if (it == trace->node_to_file.end() || trace->exported.count(t->name)) {
        return true;
    }
    trace_ggml(trace->case_root, it->second, t);
    trace->exported.insert(t->name);
    return true;
}

static std::map<std::string, std::string> make_node_map(int n_layer) {
    std::map<std::string, std::string> out = {
        {"rwkv.embedding", "embedding/embedded_context.safetensors"},
        {"rwkv.ln0",      "layer_norm0/embedded_context.safetensors"},
        {"rwkv.lm_embd",  "lm_head/embedded_context.safetensors"},
        {"rwkv.logits",   "lm_head/logits.safetensors"},
    };
    for (int il = 0; il < n_layer; ++il) {
        char node_prefix[32];
        char cell_prefix[64];
        snprintf(node_prefix, sizeof(node_prefix), "rwkv.%02d", il);
        snprintf(cell_prefix, sizeof(cell_prefix), "cells/cell_%04d", il);
        out[std::string(node_prefix) + ".t_ln"]       = std::string(cell_prefix) + "/pre_layer_norm_for_time_mix/embedded_context.safetensors";
        out[std::string(node_prefix) + ".v_first"]    = std::string(cell_prefix) + "/time_mixer/value_from_first_cell.safetensors";
        out[std::string(node_prefix) + ".tmix"]       = std::string(cell_prefix) + "/time_mixer/embedded_context.safetensors";
        out[std::string(node_prefix) + ".after_tmix"] = std::string(cell_prefix) + "/embedded_context_after_time_mixer.safetensors";
        out[std::string(node_prefix) + ".c_ln"]       = std::string(cell_prefix) + "/pre_layer_norm_for_channel_mix/embedded_context.safetensors";
        out[std::string(node_prefix) + ".cmix"]       = std::string(cell_prefix) + "/channel_mixer/embedded_context.safetensors";
        out[std::string(node_prefix) + ".after_cmix"] = std::string(cell_prefix) + "/embedded_context_after_channel_mixer.safetensors";
    }
    return out;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    int ngl = 99;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "--model") || !strcmp(argv[i], "-m")) && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((!strcmp(argv[i], "--n-gpu-layers") || !strcmp(argv[i], "-ngl")) && i + 1 < argc) {
            ngl = std::stoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }
    if (model_path.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    const bool trace_on = getenv("RWKV_TRACE_ONCE") && !strcmp(getenv("RWKV_TRACE_ONCE"), "1");
    const char * root_env = getenv("RWKV_TRACE_ROOT");
    const std::filesystem::path trace_root = root_env ? root_env : "test_gen";

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, RWKV_TRACE_PROMPT, strlen(RWKV_TRACE_PROMPT), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n_prompt);
    if (llama_tokenize(vocab, RWKV_TRACE_PROMPT, strlen(RWKV_TRACE_PROMPT), tokens.data(), tokens.size(), true, true) < 0) {
        fprintf(stderr, "failed to tokenize prompt\n");
        llama_model_free(model);
        return 1;
    }

    trace_context trace;
    trace.case_root = trace_root / "llama_cpp" / "fp16" / "case_000000";
    trace.node_to_file = make_node_map(llama_model_n_layer(model));

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = tokens.size();
    ctx_params.n_batch = tokens.size();
    ctx_params.no_perf = false;
    if (trace_on) {
        ctx_params.cb_eval = trace_cb_eval;
        ctx_params.cb_eval_user_data = &trace;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr, "failed to create llama_context\n");
        llama_model_free(model);
        return 1;
    }

    if (trace_on) {
        trace_tokens(trace.case_root, tokens);
    }

    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = tokens.size();

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d\n", rc);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    if (trace_on && trace.exported.size() != trace.node_to_file.size()) {
        fprintf(stderr, "trace exported %zu/%zu graph tensors\n", trace.exported.size(), trace.node_to_file.size());
        for (const auto & [node, file] : trace.node_to_file) {
            if (!trace.exported.count(node)) {
                fprintf(stderr, "missing trace node %s -> %s\n", node.c_str(), file.c_str());
            }
        }
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
