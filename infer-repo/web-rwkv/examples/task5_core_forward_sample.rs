use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context as _, Result};
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnIter, RnnOption, RnnOutput},
        loader::Loader,
        model::{
            AsAny as _, Bundle as _, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion,
            State as _,
        },
        v7, SimpleRuntime,
    },
};

const BENCHMARK_KIND: &str = "core_forward_sample_throughput";
const CSV_HEADER: &str = "run_id,repo,backend,runner,benchmark_kind,task,model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,B,T,warmup,repeat,seed,status,error,input_tokens,measured_tokens,total_time_s,forward_time_s,sample_time_s,p10_ms,p50_ms,p90_ms,forward_sample_tps,entrypoint,measurement_boundary,command,binary_path,binary_build_id,model_bytes,model_sha256,started_at,ended_at";
const MEASUREMENT_BOUNDARY: &str = "forward+sampler; no tokenizer decode; no scheduler; no server";

#[derive(Debug)]
struct Args {
    model: PathBuf,
    output: PathBuf,
    tasks: Vec<String>,
    prefill_t: Vec<u64>,
    batch_decode_b: Vec<u64>,
    batch_prefill_pairs: Vec<(u64, u64)>,
    warmup: u64,
    repeat: u64,
    seed: u64,
    token_chunk_size: usize,
}

struct Runner {
    runtime: SimpleRuntime<v7::Bundle<f16>, Rnn, v7::RnnJob>,
    state: v7::State,
    info: ModelInfo,
    token_chunk_size: usize,
}

enum SetupFailure {
    Unsupported(String),
    Failed(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = parse_args()?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{CSV_HEADER}")?;

    let started_at = timestamp();
    let run_id = format!("task5-core-web-rwkv-{}-{}", started_at, std::process::id());
    let model_bytes = fs::metadata(&args.model)
        .map(|metadata| metadata.len().to_string())
        .unwrap_or_default();
    let command = env::args().collect::<Vec<_>>().join(" ");
    let binary_path = env::current_exe()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let (gpu_name, gpu_uuid) = query_gpu_info();
    let model_path = args.model.display().to_string();
    let model_format = args
        .model
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("unknown")
        .to_owned();
    let model_size = infer_model_size(&args.model);
    let quantization = infer_quantization(&args.model);

    for (task, b, t) in cases(&args) {
        let setup = Runner::new(&args.model, b as usize, args.token_chunk_size).await;
        match setup {
            Ok((runner, device)) => {
                let now = timestamp();
                let row = match runner
                    .run_case(
                        &task,
                        b as usize,
                        t as usize,
                        args.warmup as usize,
                        args.repeat as usize,
                        args.seed,
                    )
                    .await
                {
                    Ok((p10_ms, p50_ms, p90_ms)) => ok_row(
                        &run_id,
                        &task,
                        b,
                        t,
                        &model_size,
                        &model_path,
                        &model_format,
                        &device,
                        &gpu_name,
                        &gpu_uuid,
                        &quantization,
                        args.warmup,
                        args.repeat,
                        args.seed,
                        p10_ms,
                        p50_ms,
                        p90_ms,
                        &command,
                        &binary_path,
                        &model_bytes,
                        &now,
                    ),
                    Err(error) => status_row(
                        "failed",
                        error.to_string(),
                        &run_id,
                        &task,
                        b,
                        t,
                        &model_size,
                        &model_path,
                        &model_format,
                        &device,
                        &gpu_name,
                        &gpu_uuid,
                        &quantization,
                        args.warmup,
                        args.repeat,
                        args.seed,
                        &command,
                        &binary_path,
                        &model_bytes,
                        &now,
                    ),
                };
                writeln!(writer, "{row}")?;
            }
            Err(failure) => {
                let (setup_status, error) = match failure {
                    SetupFailure::Unsupported(error) => ("unsupported", error),
                    SetupFailure::Failed(error) => ("failed", error),
                };
                let now = timestamp();
                let status = if setup_status == "unsupported"
                    && matches!(task.as_str(), "decode" | "prefill")
                {
                    "failed"
                } else {
                    setup_status
                };
                let row = status_row(
                    status,
                    error.clone(),
                    &run_id,
                    &task,
                    b,
                    t,
                    &model_size,
                    &model_path,
                    &model_format,
                    "unknown",
                    &gpu_name,
                    &gpu_uuid,
                    &quantization,
                    args.warmup,
                    args.repeat,
                    args.seed,
                    &command,
                    &binary_path,
                    &model_bytes,
                    &now,
                );
                writeln!(writer, "{row}")?;
            }
        }
    }
    Ok(())
}

impl Runner {
    async fn new(
        model_path: &Path,
        num_batch: usize,
        token_chunk_size: usize,
    ) -> std::result::Result<(Self, String), SetupFailure> {
        let file = tokio::fs::File::open(model_path)
            .await
            .map_err(|error| SetupFailure::Failed(error.to_string()))?;
        let data =
            unsafe { Mmap::map(&file).map_err(|error| SetupFailure::Failed(error.to_string()))? };
        let model = SafeTensors::deserialize(&data)
            .map_err(|error| SetupFailure::Failed(error.to_string()))?;
        let info = Loader::info(&model).map_err(|error| SetupFailure::Failed(error.to_string()))?;
        if info.version != ModelVersion::V7 {
            return Err(SetupFailure::Unsupported(format!(
                "task5_core_forward_sample only supports RWKV v7 models, found {:?}",
                info.version
            )));
        }
        let (context, device) = create_context(&info, num_batch).await?;
        let model = ModelBuilder::new(&context, model)
            .build_v7()
            .await
            .map_err(|error| SetupFailure::Failed(error.to_string()))?;
        let bundle = v7::Bundle::<f16>::new(model, num_batch);
        let state_handle = bundle.state();
        let state = state_handle
            .as_any()
            .downcast_ref::<v7::State>()
            .ok_or_else(|| {
                SetupFailure::Failed("web-rwkv v7 bundle returned non-v7 state".to_owned())
            })?
            .clone();
        let runtime = SimpleRuntime::new::<web_rwkv::runtime::infer::RnnInfo, RnnIter>(bundle);
        Ok((
            Self {
                runtime,
                state,
                info,
                token_chunk_size,
            },
            device,
        ))
    }

    async fn run_case(
        &self,
        task: &str,
        b: usize,
        t: usize,
        warmup: usize,
        repeat: usize,
        seed: u64,
    ) -> Result<(f64, f64, f64)> {
        for index in 0..warmup {
            self.reset_state(b)?;
            let _ = self.run_once(task, b, t, seed + index as u64).await?;
        }

        let mut times = Vec::with_capacity(repeat);
        for index in 0..repeat {
            let case_seed = seed + warmup as u64 + index as u64;
            self.reset_state(b)?;
            let prepared = if matches!(task, "decode" | "batch_decode") {
                Some(self.infer_prefill(b, 1, case_seed).await?)
            } else {
                None
            };
            let start = Instant::now();
            if let Some(tokens) = prepared {
                let _ = self.decode_from_prepared(tokens).await?;
            } else {
                let _ = self.infer_prefill(b, t, case_seed).await?;
            }
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        Ok((
            percentile(times.clone(), 0.10),
            percentile(times.clone(), 0.50),
            percentile(times, 0.90),
        ))
    }

    fn reset_state(&self, b: usize) -> Result<()> {
        for batch in 0..b {
            self.state.load(self.state.init(), batch)?;
        }
        Ok(())
    }

    async fn run_once(&self, task: &str, b: usize, t: usize, seed: u64) -> Result<Vec<u32>> {
        if matches!(task, "decode" | "batch_decode") {
            let prepared = self.infer_prefill(b, 1, seed).await?;
            let input = RnnInput::new(
                prepared
                    .into_iter()
                    .map(|token| RnnInputBatch::new(vec![token], RnnOption::Last))
                    .collect(),
                self.token_chunk_size,
            );
            return self.infer_until_sample(input).await;
        }
        if matches!(task, "prefill" | "batch_prefill") {
            return self.infer_prefill(b, t, seed).await;
        }
        Err(anyhow!("unknown task: {task}"))
    }

    async fn decode_from_prepared(&self, prepared: Vec<u32>) -> Result<Vec<u32>> {
        let input = RnnInput::new(
            prepared
                .into_iter()
                .map(|token| RnnInputBatch::new(vec![token], RnnOption::Last))
                .collect(),
            self.token_chunk_size,
        );
        self.infer_until_sample(input).await
    }

    async fn infer_prefill(&self, b: usize, t: usize, seed: u64) -> Result<Vec<u32>> {
        self.infer_until_sample(self.input(b, t, seed)).await
    }

    fn input(&self, b: usize, t: usize, seed: u64) -> RnnInput {
        let mut rng = fastrand::Rng::with_seed(seed);
        let max_token = self.info.num_vocab.saturating_sub(1).min(u32::MAX as usize) as u32;
        let batches = (0..b)
            .map(|_| {
                let tokens = (0..t)
                    .map(|_| rng.u32(0..max_token.max(1)))
                    .collect::<Vec<_>>();
                RnnInputBatch::new(tokens, RnnOption::Last)
            })
            .collect();
        RnnInput::new(batches, self.token_chunk_size)
    }

    async fn infer_until_sample(&self, mut input: RnnInput) -> Result<Vec<u32>> {
        let expected = input.batches.len();
        let mut tokens = vec![None; expected];
        loop {
            let (next, output) = self.runtime.infer(input).await?;
            input = next;
            fill_sampled_tokens(&output, &mut tokens)?;
            if tokens.iter().all(Option::is_some) {
                break;
            }
            if input.num_token() == 0 {
                return Err(anyhow!("runtime produced no output logits"));
            }
        }
        tokens
            .into_iter()
            .enumerate()
            .map(|(index, token)| {
                token.with_context(|| format!("missing sampled token for batch {index}"))
            })
            .collect()
    }
}

async fn create_context(
    info: &ModelInfo,
    num_batch: usize,
) -> std::result::Result<(Context, String), SetupFailure> {
    let instance = wgpu::Instance::default();
    let required_buffer_size = required_buffer_size(info, num_batch);
    let adapters = instance.enumerate_adapters(wgpu::Backends::all()).await;
    let adapter = adapters
        .into_iter()
        .filter(|adapter| {
            let limits = adapter.limits();
            limits.max_buffer_size >= required_buffer_size
                && limits.max_storage_buffer_binding_size >= required_buffer_size
        })
        .max_by_key(|adapter| match adapter.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 3,
            wgpu::DeviceType::IntegratedGpu => 2,
            wgpu::DeviceType::VirtualGpu => 1,
            _ => 0,
        })
        .ok_or_else(|| {
            SetupFailure::Unsupported(format!(
                "no WebGPU adapter has max_buffer_size and max_storage_buffer_binding_size >= {required_buffer_size}"
            ))
        })?;
    let adapter_info = adapter.get_info();
    let device = format!("{} ({:?})", adapter_info.name, adapter_info.backend);
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .update_limits(|limits| {
            limits.max_buffer_size = limits.max_buffer_size.max(required_buffer_size);
            limits.max_storage_buffer_binding_size = limits
                .max_storage_buffer_binding_size
                .max(required_buffer_size);
        })
        .build()
        .await
        .map_err(|error| SetupFailure::Failed(error.to_string()))?;
    Ok((context, device))
}

fn required_buffer_size(info: &ModelInfo, num_batch: usize) -> u64 {
    info.max_non_head_buffer_size()
        .max(info.head_buffer_size())
        .max(v7_state_buffer_size(info, num_batch)) as u64
}

fn v7_state_buffer_size(info: &ModelInfo, num_batch: usize) -> usize {
    let head_size = info.num_emb / info.num_head;
    info.num_emb * (head_size + 2) * num_batch * std::mem::size_of::<f32>()
}

fn fill_sampled_tokens(output: &RnnOutput, tokens: &mut [Option<u32>]) -> Result<()> {
    if output.len() != tokens.len() {
        return Err(anyhow!(
            "output batch count {} does not match expected {}",
            output.len(),
            tokens.len()
        ));
    }
    for (batch, token) in output.iter().zip(tokens.iter_mut()) {
        if batch.0.size() == 0 {
            continue;
        }
        let logits = batch.0.data();
        *token = logits
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index as u32);
    }
    Ok(())
}

fn ok_row(
    run_id: &str,
    task: &str,
    b: u64,
    t: u64,
    model_size: &str,
    model_path: &str,
    model_format: &str,
    device: &str,
    gpu_name: &str,
    gpu_uuid: &str,
    quantization: &str,
    warmup: u64,
    repeat: u64,
    seed: u64,
    p10_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    command: &str,
    binary_path: &str,
    model_bytes: &str,
    started_at: &str,
) -> String {
    let measured_tokens = if matches!(task, "decode" | "batch_decode") {
        b
    } else {
        b * t
    };
    let total_time_s = p50_ms / 1000.0;
    let tps = measured_tokens as f64 / total_time_s;
    row(vec![
        run_id.to_owned(),
        "web-rwkv".to_owned(),
        "web-rwkv-direct-runtime".to_owned(),
        "task5_core_forward_sample".to_owned(),
        BENCHMARK_KIND.to_owned(),
        task.to_owned(),
        model_size.to_owned(),
        model_path.to_owned(),
        model_format.to_owned(),
        device.to_owned(),
        gpu_name.to_owned(),
        gpu_uuid.to_owned(),
        quantization.to_ascii_lowercase(),
        quantization.to_owned(),
        b.to_string(),
        t.to_string(),
        warmup.to_string(),
        repeat.to_string(),
        seed.to_string(),
        "ok".to_owned(),
        String::new(),
        (b * t).to_string(),
        measured_tokens.to_string(),
        total_time_s.to_string(),
        String::new(),
        String::new(),
        p10_ms.to_string(),
        p50_ms.to_string(),
        p90_ms.to_string(),
        tps.to_string(),
        entrypoint(task).to_owned(),
        MEASUREMENT_BOUNDARY.to_owned(),
        command.to_owned(),
        binary_path.to_owned(),
        env!("CARGO_PKG_VERSION").to_owned(),
        model_bytes.to_owned(),
        String::new(),
        started_at.to_owned(),
        timestamp(),
    ])
}

#[allow(clippy::too_many_arguments)]
fn status_row(
    status: &str,
    error: String,
    run_id: &str,
    task: &str,
    b: u64,
    t: u64,
    model_size: &str,
    model_path: &str,
    model_format: &str,
    device: &str,
    gpu_name: &str,
    gpu_uuid: &str,
    quantization: &str,
    warmup: u64,
    repeat: u64,
    seed: u64,
    command: &str,
    binary_path: &str,
    model_bytes: &str,
    started_at: &str,
) -> String {
    row(vec![
        run_id.to_owned(),
        "web-rwkv".to_owned(),
        "web-rwkv-direct-runtime".to_owned(),
        "task5_core_forward_sample".to_owned(),
        BENCHMARK_KIND.to_owned(),
        task.to_owned(),
        model_size.to_owned(),
        model_path.to_owned(),
        model_format.to_owned(),
        device.to_owned(),
        gpu_name.to_owned(),
        gpu_uuid.to_owned(),
        quantization.to_ascii_lowercase(),
        quantization.to_owned(),
        b.to_string(),
        t.to_string(),
        warmup.to_string(),
        repeat.to_string(),
        seed.to_string(),
        status.to_owned(),
        error,
        (b * t).to_string(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        String::new(),
        entrypoint(task).to_owned(),
        MEASUREMENT_BOUNDARY.to_owned(),
        command.to_owned(),
        binary_path.to_owned(),
        env!("CARGO_PKG_VERSION").to_owned(),
        model_bytes.to_owned(),
        String::new(),
        started_at.to_owned(),
        timestamp(),
    ])
}

fn parse_args() -> Result<Args> {
    let mut model: Option<PathBuf> = None;
    let mut output = PathBuf::from("infer-repo/web-rwkv/results/task5_core_forward_sample.csv");
    let mut tasks = split_list("decode,prefill,batch_decode,batch_prefill");
    let mut prefill_t = parse_ints("16,64,256,1024,4096")?;
    let mut batch_decode_b = parse_ints("2,4,8,16,32,64,128,256,512,960,1024")?;
    let mut batch_prefill_pairs = parse_pairs("2x2,4x4,8x8,16x16,32x32")?;
    let mut warmup = 3;
    let mut repeat = 10;
    let mut seed = 0;
    let mut token_chunk_size = 128;

    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        let value = iter
            .next()
            .with_context(|| format!("{arg} requires a value"))?;
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--output" => output = PathBuf::from(value),
            "--tasks" => tasks = split_list(&value),
            "--prefill-t" => prefill_t = parse_ints(&value)?,
            "--batch-decode-b" => batch_decode_b = parse_ints(&value)?,
            "--batch-prefill-pairs" => batch_prefill_pairs = parse_pairs(&value)?,
            "--warmup" => warmup = value.parse().context("invalid --warmup")?,
            "--repeat" => repeat = value.parse().context("invalid --repeat")?,
            "--seed" => seed = value.parse().context("invalid --seed")?,
            "--token-chunk-size" => {
                token_chunk_size = value.parse().context("invalid --token-chunk-size")?
            }
            other => return Err(anyhow!("unknown argument: {other}")),
        }
    }

    Ok(Args {
        model: model.context("--model is required")?,
        output,
        tasks,
        prefill_t,
        batch_decode_b,
        batch_prefill_pairs,
        warmup,
        repeat,
        seed,
        token_chunk_size,
    })
}

fn cases(args: &Args) -> Vec<(String, u64, u64)> {
    let mut cases = Vec::new();
    if args.tasks.iter().any(|task| task == "decode") {
        cases.push(("decode".to_owned(), 1, 1));
    }
    if args.tasks.iter().any(|task| task == "prefill") {
        cases.extend(args.prefill_t.iter().map(|&t| ("prefill".to_owned(), 1, t)));
    }
    if args.tasks.iter().any(|task| task == "batch_decode") {
        cases.extend(
            args.batch_decode_b
                .iter()
                .map(|&b| ("batch_decode".to_owned(), b, 1)),
        );
    }
    if args.tasks.iter().any(|task| task == "batch_prefill") {
        cases.extend(
            args.batch_prefill_pairs
                .iter()
                .map(|&(b, t)| ("batch_prefill".to_owned(), b, t)),
        );
    }
    cases
}

fn parse_ints(value: &str) -> Result<Vec<u64>> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| part.trim().parse().context("invalid integer list"))
        .collect()
}

fn parse_pairs(value: &str) -> Result<Vec<(u64, u64)>> {
    value
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            let (left, right) = part
                .trim()
                .split_once('x')
                .with_context(|| format!("invalid BxT pair: {part}"))?;
            Ok((left.parse()?, right.parse()?))
        })
        .collect()
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .filter_map(|part| match part.trim() {
            "" => None,
            item => Some(item.to_owned()),
        })
        .collect()
}

fn entrypoint(task: &str) -> &'static str {
    match task {
        "decode" => "web-rwkv Rnn runtime B1T1 + argmax sampler",
        "prefill" => "web-rwkv Rnn runtime B1Tn + argmax sampler",
        "batch_decode" => "web-rwkv Rnn runtime BnT1 + argmax sampler",
        "batch_prefill" => "web-rwkv Rnn runtime BnTn + argmax sampler",
        _ => "web-rwkv Rnn runtime",
    }
}

fn percentile(mut values: Vec<f64>, q: f64) -> f64 {
    values.sort_by(f64::total_cmp);
    let index = ((values.len().saturating_sub(1)) as f64 * q).round() as usize;
    values[index.min(values.len().saturating_sub(1))]
}

fn row(fields: Vec<String>) -> String {
    fields
        .into_iter()
        .map(csv_escape)
        .collect::<Vec<_>>()
        .join(",")
}

fn csv_escape(value: String) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value
    }
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_owned())
}

fn infer_model_size(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let re = regex::Regex::new(r"([0-9]+(?:\.[0-9]+)?)b").expect("valid regex");
    re.captures(&stem)
        .and_then(|captures| captures.get(1))
        .map(|value| format!("{}B", value.as_str()))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn infer_quantization(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or_default()
        .to_ascii_uppercase();
    ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]
        .into_iter()
        .find(|name| stem.contains(name))
        .unwrap_or("FP16")
        .to_owned()
}

fn query_gpu_info() -> (String, String) {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,uuid", "--format=csv,noheader,nounits"])
        .output();
    let Ok(output) = output else {
        return (String::new(), String::new());
    };
    if !output.status.success() {
        return (String::new(), String::new());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts = stdout
        .lines()
        .next()
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .collect::<Vec<_>>();
    (
        parts.first().copied().unwrap_or_default().to_owned(),
        parts.get(1).copied().unwrap_or_default().to_owned(),
    )
}
