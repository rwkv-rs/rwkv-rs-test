use std::{
    collections::HashMap,
    fs::{create_dir_all, write},
    path::Path,
    time::Instant,
};

use safetensors::{serialize_to_file, Dtype as SafeDtype, TensorView};
use web_rwkv::{
    num::Scalar,
    tensor::{kind::Kind, TensorGpu, TensorShape},
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TensorKey {
    backend: &'static str,
    id: String,
    dtype: &'static str,
    shape: Vec<usize>,
}

struct TraceWriter {
    saved_by_key: HashMap<TensorKey, String>,
}

impl TraceWriter {
    fn new() -> Self {
        Self {
            saved_by_key: HashMap::new(),
        }
    }

    fn should_write(&mut self, filename: &str, key: TensorKey) -> bool {
        if let Some(saved_filename) = self.saved_by_key.get(&key) {
            if saved_filename == filename || !is_canonical_path(filename) {
                return false;
            }
            panic!(
                "tensor already saved as {}, cannot also save canonical {}",
                saved_filename, filename
            );
        }
        if !is_canonical_path(filename) {
            panic!("{} is not a canonical trace path", filename);
        }
        self.saved_by_key.insert(key, filename.to_owned());
        true
    }
}

fn is_canonical_path(filename: &str) -> bool {
    matches!(
        filename,
        "embedding/token_ids.safetensors"
            | "embedding/embedded_context.safetensors"
            | "layer_norm0/embedded_context.safetensors"
            | "lm_head/embedded_context.safetensors"
            | "lm_head/logits.safetensors"
            | "loss/l2wrap_cross_entropy.safetensors"
            | "loss/l2wrap_cross_entropy/lse.safetensors"
            | "loss/l2wrap_cross_entropy/max_vals.safetensors"
            | "loss/l2wrap_cross_entropy/argmax.safetensors"
            | "loss/head_l2wrap_cross_entropy.safetensors"
            | "loss/head_l2wrap_cross_entropy/grad_hidden.safetensors"
            | "loss/head_l2wrap_cross_entropy/grad_weight.safetensors"
    ) || canonical_cell_path(filename)
}

fn canonical_cell_path(filename: &str) -> bool {
    let Some(rest) = filename.strip_prefix("cells/cell_") else {
        return false;
    };
    let Some((cell, name)) = rest.split_once('/') else {
        return false;
    };
    cell.len() == 4
        && cell.bytes().all(|b| b.is_ascii_digit())
        && matches!(
            name,
            "time_mixer/value_from_first_cell.safetensors"
                | "time_mixer/embedded_context.safetensors"
                | "embedded_context_after_time_mixer.safetensors"
                | "channel_mixer/embedded_context.safetensors"
                | "embedded_context_after_channel_mixer.safetensors"
        )
}

fn tensor_name(filename: &str) -> String {
    Path::new(filename)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned()
}

fn write_module_time(output_path: &Path, module: &str, elapsed_ns: u128) {
    if elapsed_ns == 0 {
        panic!(
            "{} timing must be a positive module forward duration",
            module
        );
    }
    let time_path = output_path
        .join("timing")
        .join(format!("{module}.time.json"));
    create_dir_all(time_path.parent().unwrap()).unwrap();
    write(
        time_path,
        format!(
            r#"{{"module":"{}","elapsed_ns":{},"repeat":1,"warmup":0,"samples_ns":[{}]}}"#,
            module, elapsed_ns, elapsed_ns
        ),
    )
    .unwrap();
}

fn write_safetensor(path: &Path, name: String, dtype: SafeDtype, shape: Vec<usize>, bytes: &[u8]) {
    create_dir_all(path.parent().unwrap()).unwrap();
    let view = TensorView::new(dtype, shape, bytes).unwrap();
    serialize_to_file([(name, view)], None, path).unwrap();
}

// for webgpu based by web-rwkv
fn activation_webgpu<T, K>(
    writer: &mut TraceWriter,
    output_path: &Path,
    filename: &str,
    tensor: TensorGpu<T, K>,
) where
    T: Scalar,
    K: Kind,
{
    let shape: [usize; 4] = tensor.shape().into();
    let key = TensorKey {
        backend: "web-rwkv",
        id: format!("{:?}", tensor.id()),
        dtype: std::any::type_name::<T>(),
        shape: shape.to_vec(),
    };
    if !writer.should_write(filename, key) {
        return;
    }

    let path = output_path.join(filename);
    let cpu = tensor.back_in_place();
    let bytes = bytemuck::cast_slice(cpu.data().as_ref());
    write_safetensor(
        &path,
        tensor_name(filename),
        T::DATA_TYPE,
        shape.to_vec(),
        bytes,
    );
}

macro_rules! outputs {
    (value => $filename:expr $(,)?) => {
        |writer, output_path, result| {
            activation_webgpu(writer, output_path, $filename, (*result).clone());
        }
    };
    ($($index:tt => $filename:expr),+ $(,)?) => {
        |writer, output_path, result| {
            $(
                activation_webgpu(writer, output_path, $filename, result.$index.clone());
            )+
        }
    };
}

fn trace_webgpu<R, F, O>(
    writer: &mut TraceWriter,
    output_path: &Path,
    module: &str,
    target: F,
    output_specs: O,
) -> R
where
    F: FnOnce() -> R,
    O: FnOnce(&mut TraceWriter, &Path, &R),
{
    let start = Instant::now();
    let result = target();
    let elapsed_ns = start.elapsed().as_nanos();
    output_specs(writer, output_path, &result);
    write_module_time(output_path, module, elapsed_ns);
    result
}
