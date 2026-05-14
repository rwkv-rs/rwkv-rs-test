#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "safetensors.hh" // syoyo/safetensors-cpp
#include <torch/torch.h>

static std::string tensor_name(const std::string & filename) {
    return std::filesystem::path(filename).stem().string();
}

static bool starts_with(const std::string & text, const std::string & prefix) {
    return text.rfind(prefix, 0) == 0;
}

static bool canonical_cell_path(const std::string & filename) {
    const std::string prefix = "cells/cell_";
    if (!starts_with(filename, prefix)) {
        return false;
    }
    const std::string rest = filename.substr(prefix.size());
    const auto slash = rest.find('/');
    if (slash != 4) {
        return false;
    }
    for (size_t i = 0; i < 4; ++i) {
        if (rest[i] < '0' || rest[i] > '9') {
            return false;
        }
    }
    const std::string name = rest.substr(slash + 1);
    return name == "pre_layer_norm_for_time_mix/embedded_context.safetensors"
        || name == "time_mixer/value_from_first_cell.safetensors"
        || name == "time_mixer/embedded_context.safetensors"
        || name == "embedded_context_after_time_mixer.safetensors"
        || name == "pre_layer_norm_for_channel_mix/embedded_context.safetensors"
        || name == "channel_mixer/embedded_context.safetensors"
        || name == "embedded_context_after_channel_mixer.safetensors";
}

static bool is_canonical_path(const std::string & filename) {
    return filename == "embedding/token_ids.safetensors"
        || filename == "embedding/embedded_context.safetensors"
        || filename == "layer_norm0/embedded_context.safetensors"
        || filename == "lm_head/embedded_context.safetensors"
        || filename == "lm_head/logits.safetensors"
        || filename == "loss/l2wrap_cross_entropy.safetensors"
        || filename == "loss/l2wrap_cross_entropy/lse.safetensors"
        || filename == "loss/l2wrap_cross_entropy/max_vals.safetensors"
        || filename == "loss/l2wrap_cross_entropy/argmax.safetensors"
        || filename == "loss/head_l2wrap_cross_entropy.safetensors"
        || filename == "loss/head_l2wrap_cross_entropy/grad_hidden.safetensors"
        || filename == "loss/head_l2wrap_cross_entropy/grad_weight.safetensors"
        || canonical_cell_path(filename);
}

class TraceWriter {
public:
    bool should_write(const std::string & filename, const std::string & key) {
        const auto it = saved_by_key.find(key);
        if (it != saved_by_key.end()) {
            if (it->second == filename || !is_canonical_path(filename)) {
                return false;
            }
            throw std::runtime_error(
                "tensor already saved as " + it->second + ", cannot also save canonical " + filename
            );
        }

        if (!is_canonical_path(filename)) {
            throw std::runtime_error(filename + " is not a canonical trace path");
        }
        saved_by_key[key] = filename;
        return true;
    }

private:
    std::unordered_map<std::string, std::string> saved_by_key;
};

struct GgmlOutputSpec {
    std::string filename;
    const ggml_tensor * tensor;
};

template <int Index>
struct LibtorchOutputSpec {
    std::string filename;
};

template <typename... Specs>
struct OutputList {
    std::tuple<Specs...> specs;
};

static GgmlOutputSpec node(const std::string & filename, const ggml_tensor * tensor) {
    return {filename, tensor};
}

template <int Index>
static LibtorchOutputSpec<Index> out(const std::string & filename) {
    return {filename};
}

template <typename... Specs>
static OutputList<Specs...> outputs(Specs... specs) {
    return {std::make_tuple(std::move(specs)...)};
}

static void write_module_time(
    const std::filesystem::path & output_path,
    const std::string & module,
    long long elapsed_ns
) {
    if (elapsed_ns <= 0) {
        throw std::runtime_error(module + " timing must be a positive module forward duration");
    }
    auto time_path = output_path / "timing" / (module + ".time.json");
    std::filesystem::create_directories(time_path.parent_path());
    std::ofstream out(time_path);
    out << "{\"module\":\"" << module << "\",\"elapsed_ns\":" << elapsed_ns
        << ",\"repeat\":1,\"warmup\":0,\"samples_ns\":[" << elapsed_ns << "]}";
}

static safetensors::dtype ggml_dtype(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return safetensors::dtype::kFLOAT32;
        case GGML_TYPE_F16:  return safetensors::dtype::kFLOAT16;
        case GGML_TYPE_BF16: return safetensors::dtype::kBFLOAT16;
        case GGML_TYPE_I32:  return safetensors::dtype::kINT32;
        case GGML_TYPE_I16:  return safetensors::dtype::kINT16;
        case GGML_TYPE_I8:   return safetensors::dtype::kINT8;
        default: throw std::runtime_error("unsupported ggml dtype for safetensors trace");
    }
}

static safetensors::dtype torch_dtype(torch::ScalarType type) {
    switch (type) {
        case torch::kFloat64:  return safetensors::dtype::kFLOAT64;
        case torch::kFloat32:  return safetensors::dtype::kFLOAT32;
        case torch::kFloat16:  return safetensors::dtype::kFLOAT16;
        case torch::kBFloat16: return safetensors::dtype::kBFLOAT16;
        case torch::kInt64:    return safetensors::dtype::kINT64;
        case torch::kInt32:    return safetensors::dtype::kINT32;
        case torch::kInt16:    return safetensors::dtype::kINT16;
        case torch::kInt8:     return safetensors::dtype::kINT8;
        case torch::kUInt8:    return safetensors::dtype::kUINT8;
        case torch::kBool:     return safetensors::dtype::kBOOL;
        default: throw std::runtime_error("unsupported torch dtype for safetensors trace");
    }
}

static void save_safetensor(
    const std::filesystem::path & path,
    const std::string & name,
    safetensors::dtype dtype,
    const std::vector<size_t> & shape,
    const std::vector<uint8_t> & bytes
) {
    std::filesystem::create_directories(path.parent_path());
    safetensors::safetensors_t st;
    safetensors::tensor_t t;
    t.dtype = dtype;
    t.shape = shape;
    t.data_offsets = {0, bytes.size()};
    st.tensors[name] = t;
    st.storage = bytes;

    std::string warn;
    std::string err;
    const bool ok = safetensors::save_to_file(st, path.string(), &warn, &err);
    if (!ok) {
        throw std::runtime_error(err);
    }
}

static std::string ggml_tensor_key(const ggml_tensor * tensor) {
    std::string key = "ggml:";
    key += std::to_string(reinterpret_cast<uintptr_t>(tensor->buffer));
    key += ":" + std::to_string(reinterpret_cast<uintptr_t>(tensor->data));
    key += ":" + std::to_string(reinterpret_cast<uintptr_t>(tensor->view_src));
    key += ":" + std::to_string(tensor->view_offs);
    key += ":" + std::to_string(static_cast<int>(tensor->type));
    key += ":" + std::to_string(ggml_nbytes(tensor));
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        key += ":" + std::to_string(tensor->ne[i]);
        key += ":" + std::to_string(tensor->nb[i]);
    }
    return key;
}

static std::string torch_tensor_key(const torch::Tensor & tensor) {
    std::string key = "libtorch:";
    key += tensor.device().str();
    key += ":" + std::to_string(static_cast<int>(tensor.scalar_type()));
    key += ":" + std::to_string(reinterpret_cast<uintptr_t>(tensor.storage().data_ptr().get()));
    key += ":" + std::to_string(tensor.storage_offset());
    for (const auto dim : tensor.sizes()) {
        key += ":" + std::to_string(dim);
    }
    key += "|";
    for (const auto stride : tensor.strides()) {
        key += ":" + std::to_string(stride);
    }
    return key;
}

static void sync_libtorch_value(const torch::Tensor & tensor) {
    if (tensor.defined() && tensor.is_cuda()) {
        torch::cuda::synchronize(tensor.device().index());
    }
}

template <typename T>
static void sync_libtorch_value(const T &) {}

template <typename... Args>
static void sync_libtorch_values(const Args & ... args) {
    (sync_libtorch_value(args), ...);
}

template <typename... Items>
static void sync_libtorch_value(const std::tuple<Items...> & values) {
    std::apply(
        [](const auto & ... items) {
            sync_libtorch_values(items...);
        },
        values
    );
}

static void activation_ggml(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const std::string & filename,
    const ggml_tensor * tensor
) {
    if (!writer.should_write(filename, ggml_tensor_key(tensor))) {
        return;
    }

    const auto path = output_path / filename;

    std::vector<size_t> shape;
    for (int i = ggml_n_dims(tensor) - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    save_safetensor(path, tensor_name(filename), ggml_dtype(tensor->type), shape, bytes);
}

template <typename... Specs>
static void save_ggml_outputs(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const OutputList<Specs...> & output_specs
) {
    std::apply(
        [&](const auto & ... specs) {
            (activation_ggml(writer, output_path, specs.filename, specs.tensor), ...);
        },
        output_specs.specs
    );
}

// for llama.cpp / ggml. The scheduler supplies elapsed_ns from the ask=true/ask=false boundary.
template <typename OutputSpecs>
static void trace_ggml(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const std::string & module,
    long long elapsed_ns,
    const OutputSpecs & output_specs
) {
    save_ggml_outputs(writer, output_path, output_specs);
    write_module_time(output_path, module, elapsed_ns);
}

// for libtorch
static void activation_libtorch(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const std::string & filename,
    const torch::Tensor & tensor
) {
    if (!writer.should_write(filename, torch_tensor_key(tensor))) {
        return;
    }

    const auto path = output_path / filename;

    TORCH_CHECK(tensor.is_contiguous(), "trace_libtorch requires a contiguous tensor");
    const torch::Tensor host = tensor.device().is_cpu() ? tensor : tensor.cpu();

    std::vector<size_t> shape;
    for (const auto dim : host.sizes()) {
        shape.push_back(static_cast<size_t>(dim));
    }

    const auto nbytes = static_cast<size_t>(host.nbytes());
    const auto * ptr = static_cast<const uint8_t *>(host.const_data_ptr());
    std::vector<uint8_t> bytes(ptr, ptr + nbytes);
    save_safetensor(path, tensor_name(filename), torch_dtype(host.scalar_type()), shape, bytes);
}

template <int Index, typename Result>
static const torch::Tensor & libtorch_output_at(const Result & result) {
    if constexpr (std::is_same_v<std::decay_t<Result>, torch::Tensor>) {
        static_assert(Index == 0, "single-tensor trace_libtorch result only supports out<0>()");
        return result;
    } else {
        return std::get<Index>(result);
    }
}

template <int Index, typename Result>
static void save_libtorch_output(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const LibtorchOutputSpec<Index> & spec,
    const Result & result
) {
    activation_libtorch(writer, output_path, spec.filename, libtorch_output_at<Index>(result));
}

template <typename Result, typename... Specs>
static void save_libtorch_outputs(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const OutputList<Specs...> & output_specs,
    const Result & result
) {
    std::apply(
        [&](const auto & ... specs) {
            (save_libtorch_output(writer, output_path, specs, result), ...);
        },
        output_specs.specs
    );
}

template <typename Fn, typename OutputSpecs, typename... Args>
static auto trace_libtorch(
    TraceWriter & writer,
    const std::filesystem::path & output_path,
    const std::string & module,
    Fn && target,
    const OutputSpecs & output_specs,
    const Args & ... sync_inputs
) {
    sync_libtorch_values(sync_inputs...);
    const auto start = std::chrono::steady_clock::now();
    auto result = std::invoke(std::forward<Fn>(target));
    sync_libtorch_value(result);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    save_libtorch_outputs(writer, output_path, output_specs, result);
    write_module_time(output_path, module, elapsed_ns);
    return result;
}
