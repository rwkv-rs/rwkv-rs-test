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
        infer::{RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion},
        v7, Dispatcher, Job, JobInput,
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
    elapsed_ns: u128,
) -> Result<()> {
    let path = root.join(filename);
    fs::create_dir_all(path.parent().unwrap())?;

    let view = TensorView::new(dtype, shape, bytes)?;
    serialize_to_file([(path.file_stem().unwrap().to_string_lossy(), view)], None, &path)?;
    write_time(&path, filename, elapsed_ns)
}

fn write_cpu_f16(
    root: &Path,
    filename: &str,
    tensor: TensorCpu<f16>,
    elapsed_ns: u128,
) -> Result<()> {
    write_bytes(
        root,
        filename,
        SafeDtype::F16,
        shape_vec(&tensor),
        bytemuck::cast_slice(tensor.data().as_ref()),
        elapsed_ns,
    )
}

fn write_cpu_f32(
    root: &Path,
    filename: &str,
    tensor: TensorCpu<f32>,
    elapsed_ns: u128,
) -> Result<()> {
    write_bytes(
        root,
        filename,
        SafeDtype::F32,
        shape_vec(&tensor),
        bytemuck::cast_slice(tensor.data().as_ref()),
        elapsed_ns,
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
        0,
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

fn traced_blit_f16(
    filename: &str,
    input: &TensorGpu<f16, ReadWrite>,
    output: &TensorGpu<f16, ReadWrite>,
) -> std::result::Result<TensorOp, web_rwkv::tensor::TensorError> {
    Ok(TensorOp::List(vec![
        TensorOp::trace(filename),
        TensorOp::blit(input, output)?,
        TensorOp::Sep,
    ]))
}

fn traced_blit_f32(
    filename: &str,
    input: &TensorGpu<f32, ReadWrite>,
    output: &TensorGpu<f32, ReadWrite>,
) -> std::result::Result<TensorOp, web_rwkv::tensor::TensorError> {
    Ok(TensorOp::List(vec![
        TensorOp::trace(filename),
        TensorOp::blit(input, output)?,
        TensorOp::Sep,
    ]))
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

    let filename = "embedding/embedded_context.safetensors".to_owned();
    let target = snapshot_f16(
        context,
        &mut snapshots,
        filename.clone(),
        hidden_shape,
    );
    hooks.insert(
        v7::Hook::PostEmbedLoaded,
        Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.input, &target)),
    );

    let filename = "layer_norm0/embedded_context.safetensors".to_owned();
    let target = snapshot_f16(
        context,
        &mut snapshots,
        filename.clone(),
        hidden_shape,
    );
    hooks.insert(
        v7::Hook::PostEmbedLayerNorm,
        Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.x, &target)),
    );

    for layer in 0..info.num_layer {
        let filename =
            format!("cells/cell_{layer:04}/time_mixer/value_from_first_cell.safetensors");
        let target = snapshot_f16(
            context,
            &mut snapshots,
            filename.clone(),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAttValueResidual(layer),
            Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.att_v0, &target)),
        );

        let filename = format!("cells/cell_{layer:04}/time_mixer/embedded_context.safetensors");
        let target = snapshot_f16(
            context,
            &mut snapshots,
            filename.clone(),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAttOut(layer),
            Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.att_o, &target)),
        );

        let filename =
            format!("cells/cell_{layer:04}/embedded_context_after_time_mixer.safetensors");
        let target = snapshot_f16(
            context,
            &mut snapshots,
            filename.clone(),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostAtt(layer),
            Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.x, &target)),
        );

        let filename = format!("cells/cell_{layer:04}/channel_mixer/embedded_context.safetensors");
        let target = snapshot_f16(
            context,
            &mut snapshots,
            filename.clone(),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostFfnChannelMix(layer),
            Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.ffn_x, &target)),
        );

        let filename =
            format!("cells/cell_{layer:04}/embedded_context_after_channel_mixer.safetensors");
        let target = snapshot_f16(
            context,
            &mut snapshots,
            filename.clone(),
            hidden_shape,
        );
        hooks.insert(
            v7::Hook::PostFfn(layer),
            Box::new(move |frame| traced_blit_f16(&filename, &frame.buffer.x, &target)),
        );
    }

    let filename = "lm_head/embedded_context.safetensors".to_owned();
    let target = snapshot_f16(
        context,
        &mut snapshots,
        filename.clone(),
        [info.num_emb, 1, 1, 1],
    );
    hooks.insert(
        v7::Hook::PostHeadLayerNorm,
        Box::new(move |frame| traced_blit_f16(&filename, &frame.header.head_x, &target)),
    );

    let filename = "lm_head/logits.safetensors".to_owned();
    let target = snapshot_f32(
        context,
        &mut snapshots,
        filename.clone(),
        logits_shape,
    );
    hooks.insert(
        v7::Hook::PostHead,
        Box::new(move |frame| traced_blit_f32(&filename, &frame.header.head_o, &target)),
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

    let token_chunk_size = tokens.len();
    let prompt = RnnInputBatch::new(tokens.clone(), RnnOption::Last);
    let prompt = RnnInput::new(vec![prompt], token_chunk_size);
    let info = prompt
        .iter()
        .next()
        .ok_or_else(|| anyhow!("trace input produced no inference job"))?;
    let chunk = prompt.chunk();
    if chunk.num_token() != tokens.len() {
        return Err(anyhow!("trace prefill did not consume the full prompt"));
    }
    let mut job = bundle.dispatch(info)?;

    let start = Instant::now();
    job.load(&chunk)?;
    let input_elapsed_ns = start.elapsed().as_nanos();

    let mut timings = job.submit_timed();
    timings.insert(
        "embedding/embedded_context.safetensors".to_owned(),
        input_elapsed_ns,
    );

    let output = job.back().await?;
    println!(
        "web-rwkv trace prefill complete logits_shape={:?}",
        output[0].shape()
    );

    write_token_ids(&root, &tokens)?;
    for snapshot in snapshots {
        match snapshot {
            TraceSnapshot::F16(filename, tensor) => {
                let elapsed_ns = timings.get(&filename).copied().unwrap_or(0);
                write_cpu_f16(&root, &filename, tensor.back().await, elapsed_ns)?
            }
            TraceSnapshot::F32(filename, tensor) => {
                let elapsed_ns = timings.get(&filename).copied().unwrap_or(0);
                write_cpu_f32(&root, &filename, tensor.back().await, elapsed_ns)?
            }
        }
    }

    Ok(())
}
