use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{anyhow, Result};
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use safetensors::{serialize_to_file, tensor::TensorView, Dtype as SafeDtype, SafeTensors};
use tokio::{
    fs::File,
    io::{AsyncReadExt, BufReader},
};
use web_rwkv::{
    context::{Context, ContextBuilder},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        v7, Runtime, TokioRuntime,
    },
    tensor::{
        kind::ReadWrite,
        ops::TensorOp,
        TensorCpu, TensorGpu, TensorShape,
    },
    tokenizer::Tokenizer,
};

const PROMPT: &str = r"User: You are a very talented expert in aime24. Solve the problem and output the final answer in \boxed{}. Problem: Let AB​CD be a tetrahedron such that AB = CD = \sqrt{41}, AC = BD = \sqrt{80}, and BC = AD = \sqrt{89}. There exists a point I inside the tetrahedron such that the distances from I to each of the faces of the tetrahedron are all equal. This distance can be written in the form \frac{m\sqrt{n}}{p}, where m, n, and p are positive integers, m and p are relatively prime, and n is not divisible by the square of any prime. Find m + n + p. Assistant: <think";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
}

enum TraceSnapshot {
    F16(String, TensorGpu<f16, ReadWrite>),
    F32(String, TensorGpu<f32, ReadWrite>),
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let required_binding_size = info.max_non_head_buffer_size().max(info.head_buffer_size()) as u64;
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    for adapter in &adapters {
        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        println!(
            "web-rwkv adapter candidate: {:?}, max_storage_buffer_binding_size={}",
            adapter_info, limits.max_storage_buffer_binding_size
        );
    }
    let adapter = adapters
        .into_iter()
        .filter(|adapter| adapter.limits().max_storage_buffer_binding_size >= required_binding_size)
        .max_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 3,
            wgpu::DeviceType::IntegratedGpu => 2,
            wgpu::DeviceType::VirtualGpu => 1,
            _ => 0,
        })
        .ok_or_else(|| {
            anyhow!(
                "no WebGPU adapter has max_storage_buffer_binding_size >= {required_binding_size}"
            )
        })?;
    println!("web-rwkv adapter: {:?}", adapter.get_info());
    println!("web-rwkv adapter limits: {:?}", adapter.limits());
    println!("web-rwkv adapter features: {:?}", adapter.features());
    Ok(ContextBuilder::new(adapter).auto_limits(info).build().await?)
}

async fn load_tokenizer() -> Result<Tokenizer> {
    let file = File::open("assets/vocab/rwkv_vocab_v20230424.json").await?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents).await?;
    Ok(Tokenizer::new(&contents)?)
}

fn case_root() -> Result<PathBuf> {
    let root = std::env::var("RWKV_TRACE_ROOT")
        .map_err(|_| anyhow!("RWKV_TRACE_ROOT must be set when RWKV_TRACE_ONCE=1"))?;
    Ok(Path::new(&root)
        .join("web_rwkv")
        .join("fp16")
        .join("case_000000"))
}

fn write_time(path: &Path, filename: &str, elapsed_ns: u128) -> Result<()> {
    fs::write(
        path.with_extension("time.json"),
        serde_json::json!({"filename": filename, "elapsed_ns": elapsed_ns}).to_string(),
    )?;
    Ok(())
}

fn shape_vec<T>(tensor: &TensorCpu<T>) -> Vec<usize>
where
    T: web_rwkv::num::Scalar,
{
    tensor.shape().iter().collect()
}

fn write_bytes(
    root: &Path,
    filename: &str,
    dtype: SafeDtype,
    shape: Vec<usize>,
    bytes: &[u8],
) -> Result<()> {
    let path = root.join(filename);
    fs::create_dir_all(path.parent().unwrap())?;

    let start = Instant::now();
    let view = TensorView::new(dtype, shape, bytes)?;
    serialize_to_file([(path.file_stem().unwrap().to_string_lossy(), view)], None, &path)?;
    write_time(&path, filename, start.elapsed().as_nanos())
}

fn write_cpu_f16(root: &Path, filename: &str, tensor: TensorCpu<f16>) -> Result<()> {
    write_bytes(
        root,
        filename,
        SafeDtype::F16,
        shape_vec(&tensor),
        bytemuck::cast_slice(tensor.data().as_ref()),
    )
}

fn write_cpu_f32(root: &Path, filename: &str, tensor: TensorCpu<f32>) -> Result<()> {
    write_bytes(
        root,
        filename,
        SafeDtype::F32,
        shape_vec(&tensor),
        bytemuck::cast_slice(tensor.data().as_ref()),
    )
}

fn write_token_ids(root: &Path, tokens: &[u32]) -> Result<()> {
    let ids: Vec<i64> = tokens.iter().map(|&token| i64::from(token)).collect();
    write_bytes(
        root,
        "embedding/token_ids.safetensors",
        SafeDtype::I64,
        vec![ids.len()],
        bytemuck::cast_slice(&ids),
    )
}

fn snapshot_f16(
    context: &Context,
    snapshots: &mut Vec<TraceSnapshot>,
    filename: impl Into<String>,
    shape: impl Into<web_rwkv::tensor::shape::Shape>,
) -> TensorGpu<f16, ReadWrite> {
    let filename = filename.into();
    let tensor: TensorGpu<f16, ReadWrite> = context.tensor_init(shape);
    snapshots.push(TraceSnapshot::F16(filename, tensor.clone()));
    tensor
}

fn snapshot_f32(
    context: &Context,
    snapshots: &mut Vec<TraceSnapshot>,
    filename: impl Into<String>,
    shape: impl Into<web_rwkv::tensor::shape::Shape>,
) -> TensorGpu<f32, ReadWrite> {
    let filename = filename.into();
    let tensor: TensorGpu<f32, ReadWrite> = context.tensor_init(shape);
    snapshots.push(TraceSnapshot::F32(filename, tensor.clone()));
    tensor
}

fn build_hooks(
    context: &Context,
    info: &ModelInfo,
    num_token: usize,
) -> (v7::HookMap<f16>, Vec<TraceSnapshot>) {
    let mut hooks: v7::HookMap<f16> = HashMap::new();
    let mut snapshots = vec![];
    let hidden_shape = [info.num_emb, num_token, 1, 1];
    let logits_shape = [info.num_vocab_padded(), 1, 1, 1];

    let target = snapshot_f16(
        context,
        &mut snapshots,
        "embedding/embedded_context.safetensors",
        hidden_shape,
    );
    hooks.insert(
        v7::Hook::PostEmbedLoaded,
        Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.input, &target)?)),
    );

    let target = snapshot_f16(
        context,
        &mut snapshots,
        "layer_norm0/embedded_context.safetensors",
        hidden_shape,
    );
    hooks.insert(
        v7::Hook::PostEmbedLayerNorm,
        Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.x, &target)?)),
    );

    for layer in 0..info.num_layer {
        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/pre_layer_norm_for_time_mix/embedded_context.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAttLayerNorm(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.att_x, &target)?)),
        );

        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/time_mixer/value_from_first_cell.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAttValueResidual(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.att_v0, &target)?)),
        );

        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/time_mixer/embedded_context.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAttOut(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.att_o, &target)?)),
        );

        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/embedded_context_after_time_mixer.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAtt(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.x, &target)?)),
        );

        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/pre_layer_norm_for_channel_mix/embedded_context.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostFfnLayerNorm(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.ffn_x, &target)?)),
        );

        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/channel_mixer/embedded_context.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostFfnChannelMix(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.ffn_x, &target)?)),
        );

        let target = snapshot_f16(
            context,
            &mut snapshots,
            format!("cells/cell_{layer:04}/embedded_context_after_channel_mixer.safetensors"),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostFfn(layer),
            Box::new(move |frame| Ok(TensorOp::blit(&frame.buffer.x, &target)?)),
        );
    }

    let target = snapshot_f16(
        context,
        &mut snapshots,
        "lm_head/embedded_context.safetensors",
        [info.num_emb, 1, 1, 1],
    );
    hooks.insert(
        v7::Hook::PostHeadLayerNorm,
        Box::new(move |frame| Ok(TensorOp::blit(&frame.header.head_x, &target)?)),
    );

    let target = snapshot_f32(
        context,
        &mut snapshots,
        "lm_head/logits.safetensors",
        logits_shape,
    );
    hooks.insert(
        v7::Hook::PostHead,
        Box::new(move |frame| Ok(TensorOp::blit(&frame.header.head_o, &target)?)),
    );

    (hooks, snapshots)
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("RWKV_TRACE_ONCE").as_deref() != Ok("1") {
        return Err(anyhow!("set RWKV_TRACE_ONCE=1 to run trace export"));
    }

    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()?;

    let cli = Cli::parse();
    let root = case_root()?;
    let tokenizer = load_tokenizer().await?;
    let tokens = tokenizer.encode(PROMPT.as_bytes())?;
    println!("web-rwkv trace prefill tokens={}", tokens.len());

    let file = File::open(cli.model).await?;
    let data = unsafe { Mmap::map(&file)? };
    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;
    if info.version != ModelVersion::V7 {
        return Err(anyhow!("trace_infer only supports RWKV v7 models"));
    }

    let context = create_context(&info).await?;
    let builder = ModelBuilder::new(&context, model);
    let model = builder.build_v7().await?;
    let (hooks, snapshots) = build_hooks(&context, &info, tokens.len());
    let bundle = v7::Bundle::<f16>::new_with_hooks(model, 1, hooks);
    let runtime: Box<dyn Runtime<Rnn>> = Box::new(TokioRuntime::<Rnn>::new(bundle).await);

    let token_chunk_size = tokens.len();
    let prompt = RnnInputBatch::new(tokens.clone(), RnnOption::Last);
    let prompt = RnnInput::new(vec![prompt], token_chunk_size);
    let (remaining, output) = runtime.infer(prompt).await?;
    if remaining.num_token() != 0 {
        return Err(anyhow!("trace prefill did not consume the full prompt"));
    }
    println!(
        "web-rwkv trace prefill complete logits_shape={:?}",
        output[0].shape()
    );

    write_token_ids(&root, &tokens)?;
    for snapshot in snapshots {
        match snapshot {
            TraceSnapshot::F16(filename, tensor) => write_cpu_f16(&root, &filename, tensor.back().await)?,
            TraceSnapshot::F32(filename, tensor) => write_cpu_f32(&root, &filename, tensor.back().await)?,
        }
    }

    Ok(())
}
