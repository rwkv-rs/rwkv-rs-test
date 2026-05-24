//! Task 5 benchmark CSV writer for the raw web-rwkv runtime path.

use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context as _, Result};
use clap::Parser;
use half::f16;
use memmap2::Mmap;
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnIter, RnnOption, RnnOutput},
        loader::Loader,
        model::{ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant},
        v7, SimpleRuntime,
    },
};

const CSV_HEADER: &str = "run_id,repo,backend,runner,benchmark_kind,model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,bsz,prompt_len,decode_len,warmup,repeat,seed,status,error,prompt_source,prompt_count,prompt_tokens,prefill_tokens,output_tokens,prefill_time_s,ttft_s,ttft_p95_s,e2el_s,e2el_p95_s,token_generation_time_s,prefill_tps,decode_tps,e2e_tps,time_per_output_token_ms,requests_per_s,itl_mean_ms,itl_p50_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,command,binary_path,binary_build_id,model_bytes,model_sha256,started_at,ended_at";

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, value_name = "FILE")]
    model: PathBuf,
    #[arg(
        short,
        long,
        value_name = "FILE",
        default_value = "results/task5_web_rwkv.csv"
    )]
    output: PathBuf,
    #[arg(long, value_name = "FILE", default_value = "results/gpu_telemetry.csv")]
    telemetry_output: PathBuf,
    #[arg(long, value_name = "FILE", default_value = "results/manifest.jsonl")]
    manifest_output: PathBuf,
    #[arg(
        long,
        value_name = "LIST",
        default_value = "1,16,64,128,256,320,512,960,1024"
    )]
    bsz: String,
    #[arg(long, value_name = "LIST", default_value = "16")]
    prompt_len: String,
    #[arg(long, default_value_t = 16)]
    decode_len: usize,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    repeat: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 128)]
    token_chunk_size: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_nf4: usize,
    #[arg(long, value_name = "LAYERS", default_value_t = 0)]
    quant_sf4: usize,
}

#[derive(Debug, Clone, Copy)]
struct BenchmarkCase {
    bsz: usize,
    prompt_len: usize,
    decode_len: usize,
    warmup: usize,
    repeat: usize,
    seed: u64,
}

#[derive(Debug, Clone)]
struct RunMeasurement {
    prefill_time_s: f64,
    ttft_s: f64,
    e2el_s: f64,
    token_generation_time_s: f64,
    itl_ms: Vec<f64>,
}

#[derive(Debug, Default, Clone)]
struct BenchmarkMetrics {
    prefill_time_s: Option<f64>,
    ttft_s: Option<f64>,
    e2el_s: Option<f64>,
    token_generation_time_s: Option<f64>,
    prefill_tps: Option<f64>,
    decode_tps: Option<f64>,
    e2e_tps: Option<f64>,
    time_per_output_token_ms: Option<f64>,
    itl_mean_ms: Option<f64>,
    itl_p50_ms: Option<f64>,
    itl_p90_ms: Option<f64>,
    itl_p95_ms: Option<f64>,
    itl_p99_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct BenchmarkRow {
    run_id: String,
    repo: String,
    backend: String,
    runner: String,
    benchmark_kind: String,
    model_size: String,
    model_path: String,
    model_format: String,
    device: String,
    gpu_name: String,
    gpu_uuid: String,
    dtype: String,
    quantization: String,
    case: BenchmarkCase,
    status: String,
    error: String,
    metrics: BenchmarkMetrics,
    prompt_source: String,
    command: String,
    binary_path: String,
    binary_build_id: String,
    model_bytes: String,
    model_sha256: String,
    started_at: String,
    ended_at: String,
}

#[derive(Debug, Clone, Copy)]
struct QuantConfig {
    int8: usize,
    nf4: usize,
    sf4: usize,
}

#[derive(Debug, Clone)]
struct GpuInfo {
    name: String,
    uuid: String,
    driver_version: String,
}

#[derive(Debug, Clone)]
struct RunMetadata {
    model_size: String,
    gpu_name: String,
    gpu_uuid: String,
    command: String,
    binary_path: String,
    binary_build_id: String,
    model_bytes: String,
    model_sha256: String,
    prompt_source: String,
}

struct Runner {
    runtime: SimpleRuntime<v7::Bundle<f16>, Rnn, v7::RnnJob>,
    info: ModelInfo,
    token_chunk_size: usize,
}

enum SetupFailure {
    Unsupported(String),
    Failed(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .init()
        .ok();

    let cli = Cli::parse();
    let bsz = parse_list(&cli.bsz, "bsz")?;
    let prompt_lens = parse_list(&cli.prompt_len, "prompt-len")?;
    let cases = benchmark_cases(&bsz, &prompt_lens, &cli);
    let quant = QuantConfig {
        int8: cli.quant,
        nf4: cli.quant_nf4,
        sf4: cli.quant_sf4,
    };

    let model_path = cli.model.display().to_string();
    let model_format = model_format(&cli.model);
    let quantization = quantization_label(quant);
    let gpu_info = query_gpu_info().unwrap_or_else(|error| GpuInfo {
        name: "unknown".to_owned(),
        uuid: String::new(),
        driver_version: format!("nvidia-smi error: {error}"),
    });
    let metadata = RunMetadata {
        model_size: infer_model_size(&cli.model),
        gpu_name: gpu_info.name.clone(),
        gpu_uuid: gpu_info.uuid.clone(),
        command: env::args().collect::<Vec<_>>().join(" "),
        binary_path: env::current_exe()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| "task5_benchmark".to_owned()),
        binary_build_id: format!(
            "web-rwkv={} driver={}",
            env!("CARGO_PKG_VERSION"),
            gpu_info.driver_version
        ),
        model_bytes: cli
            .model
            .metadata()
            .map(|metadata| metadata.len().to_string())
            .unwrap_or_default(),
        model_sha256: sha256sum(&cli.model).unwrap_or_default(),
        prompt_source: "synthetic_rng".to_owned(),
    };
    append_manifest(
        &cli.manifest_output,
        &metadata,
        &cli,
        &model_format,
        &quantization,
    )?;
    let setup = Runner::new(&cli.model, quant, max_bsz(&cases), cli.token_chunk_size).await;

    let mut rows = vec![];
    match setup {
        Ok((runner, device)) => {
            for case in cases {
                let run_id = make_run_id(case);
                let started_at = timestamp();
                let row = match runner.run_case(case).await {
                    Ok(metrics) => BenchmarkRow::ok(
                        case,
                        run_id,
                        &model_path,
                        &model_format,
                        &device,
                        &quantization,
                        &metadata,
                        started_at,
                        timestamp(),
                        metrics,
                    ),
                    Err(error) => BenchmarkRow::failed(
                        case,
                        run_id,
                        &model_path,
                        &model_format,
                        &device,
                        &quantization,
                        &metadata,
                        started_at,
                        timestamp(),
                        error.to_string(),
                    ),
                };
                append_gpu_telemetry(&cli.telemetry_output, &row.run_id, &metadata.gpu_uuid)?;
                rows.push(row);
            }
        }
        Err(failure) => {
            let (status, error) = match failure {
                SetupFailure::Unsupported(error) => ("unsupported", error),
                SetupFailure::Failed(error) => ("failed", error),
            };
            for case in cases {
                let run_id = make_run_id(case);
                let now = timestamp();
                rows.push(BenchmarkRow::with_status(
                    case,
                    run_id.clone(),
                    &model_path,
                    &model_format,
                    "unknown",
                    &quantization,
                    &metadata,
                    now.clone(),
                    now,
                    status,
                    error.clone(),
                    BenchmarkMetrics::default(),
                ));
                append_gpu_telemetry(&cli.telemetry_output, &run_id, &metadata.gpu_uuid)?;
            }
        }
    }

    write_csv(&cli.output, &rows)?;
    println!("wrote {}", cli.output.display());
    Ok(())
}

impl Runner {
    async fn new(
        model_path: &Path,
        quant: QuantConfig,
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
                "task5_benchmark only supports RWKV v7 models, found {:?}",
                info.version
            )));
        }

        let (context, device) = create_context(&info, num_batch).await?;
        let quant = (0..quant.int8)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..quant.nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..quant.sf4).map(|layer| (layer, Quant::SF4)))
            .collect();
        let model = ModelBuilder::new(&context, model)
            .quant(quant)
            .build_v7()
            .await
            .map_err(|error| SetupFailure::Failed(error.to_string()))?;
        let bundle = v7::Bundle::<f16>::new(model, num_batch);
        let runtime = SimpleRuntime::new::<web_rwkv::runtime::infer::RnnInfo, RnnIter>(bundle);

        Ok((
            Self {
                runtime,
                info,
                token_chunk_size,
            },
            device,
        ))
    }

    async fn run_case(&self, case: BenchmarkCase) -> Result<BenchmarkMetrics> {
        if case.bsz == 0 {
            return Err(anyhow!("bsz must be greater than zero"));
        }
        if case.prompt_len == 0 {
            return Err(anyhow!("prompt_len must be greater than zero"));
        }
        if case.decode_len == 0 {
            return Err(anyhow!("decode_len must be greater than zero"));
        }
        if case.repeat == 0 {
            return Err(anyhow!("repeat must be greater than zero"));
        }

        for index in 0..case.warmup {
            let _ = self.run_once(case, index as u64).await?;
        }

        let mut runs = Vec::with_capacity(case.repeat);
        for index in 0..case.repeat {
            runs.push(self.run_once(case, (case.warmup + index) as u64).await?);
        }

        Ok(BenchmarkMetrics::from_runs(&runs, case))
    }

    async fn run_once(&self, case: BenchmarkCase, offset: u64) -> Result<RunMeasurement> {
        let mut rng = fastrand::Rng::with_seed(case.seed.wrapping_add(offset));
        let prompt = make_prompt(&self.info, case, &mut rng);
        let mut input = RnnInput::new(prompt, self.token_chunk_size);

        let start = Instant::now();
        let mut next_tokens = vec![None; case.bsz];
        loop {
            let (next, output) = self.runtime.infer(input).await?;
            input = next;
            fill_sampled_tokens(&output, &mut next_tokens)?;
            if next_tokens.iter().all(Option::is_some) {
                break;
            }
            if input.num_token() == 0 {
                return Err(anyhow!("prefill produced no output logits"));
            }
        }
        let prefill = start.elapsed();
        let mut next_tokens = collect_tokens(next_tokens, "prefill")?;

        let mut itl_ms = Vec::with_capacity(case.decode_len.saturating_sub(1));
        let mut token_generation = Duration::ZERO;
        for _ in 1..case.decode_len {
            let mut input = RnnInput::new(
                next_tokens
                    .iter()
                    .map(|&token| RnnInputBatch::new(vec![token], RnnOption::Last))
                    .collect(),
                self.token_chunk_size,
            );

            let step_start = Instant::now();
            let mut decoded_tokens = vec![None; case.bsz];
            loop {
                let (next, output) = self.runtime.infer(input).await?;
                input = next;
                fill_sampled_tokens(&output, &mut decoded_tokens)?;
                if decoded_tokens.iter().all(Option::is_some) {
                    break;
                }
                if input.num_token() == 0 {
                    return Err(anyhow!("decode produced no output logits"));
                }
            }
            let elapsed = step_start.elapsed();
            token_generation += elapsed;
            itl_ms.push(elapsed.as_secs_f64() * 1000.0);
            next_tokens = collect_tokens(decoded_tokens, "decode")?;
        }

        let e2e = start.elapsed();
        Ok(RunMeasurement {
            prefill_time_s: prefill.as_secs_f64(),
            ttft_s: prefill.as_secs_f64(),
            e2el_s: e2e.as_secs_f64(),
            token_generation_time_s: token_generation.as_secs_f64(),
            itl_ms,
        })
    }
}

impl BenchmarkMetrics {
    fn from_runs(runs: &[RunMeasurement], case: BenchmarkCase) -> Self {
        let prefill_time_s = mean(runs.iter().map(|run| run.prefill_time_s));
        let ttft_s = mean(runs.iter().map(|run| run.ttft_s));
        let e2el_s = mean(runs.iter().map(|run| run.e2el_s));
        let token_generation_time_s = mean(runs.iter().map(|run| run.token_generation_time_s));
        let mut itl_ms = runs
            .iter()
            .flat_map(|run| run.itl_ms.iter().copied())
            .collect::<Vec<_>>();

        let prefill_tokens = (case.bsz * case.prompt_len) as f64;
        let output_tokens = (case.bsz * case.decode_len) as f64;
        let decode_tokens = (case.bsz * case.decode_len.saturating_sub(1)) as f64;

        Self {
            prefill_time_s: Some(prefill_time_s),
            ttft_s: Some(ttft_s),
            e2el_s: Some(e2el_s),
            token_generation_time_s: Some(token_generation_time_s),
            prefill_tps: rate(prefill_tokens, prefill_time_s),
            decode_tps: rate(decode_tokens, token_generation_time_s),
            e2e_tps: rate(output_tokens, e2el_s),
            time_per_output_token_ms: match case.decode_len {
                0 | 1 => None,
                decode_len => Some(token_generation_time_s * 1000.0 / (decode_len - 1) as f64),
            },
            itl_mean_ms: match itl_ms.is_empty() {
                true => None,
                false => Some(mean(itl_ms.iter().copied())),
            },
            itl_p50_ms: percentile_option(&mut itl_ms, 0.50),
            itl_p90_ms: percentile_option(&mut itl_ms, 0.90),
            itl_p95_ms: percentile_option(&mut itl_ms, 0.95),
            itl_p99_ms: percentile_option(&mut itl_ms, 0.99),
        }
    }
}

impl BenchmarkRow {
    fn ok(
        case: BenchmarkCase,
        run_id: String,
        model_path: &str,
        model_format: &str,
        device: &str,
        quantization: &str,
        metadata: &RunMetadata,
        started_at: String,
        ended_at: String,
        metrics: BenchmarkMetrics,
    ) -> Self {
        Self::with_status(
            case,
            run_id,
            model_path,
            model_format,
            device,
            quantization,
            metadata,
            started_at,
            ended_at,
            "ok",
            String::new(),
            metrics,
        )
    }

    fn failed(
        case: BenchmarkCase,
        run_id: String,
        model_path: &str,
        model_format: &str,
        device: &str,
        quantization: &str,
        metadata: &RunMetadata,
        started_at: String,
        ended_at: String,
        error: String,
    ) -> Self {
        Self::with_status(
            case,
            run_id,
            model_path,
            model_format,
            device,
            quantization,
            metadata,
            started_at,
            ended_at,
            "failed",
            error,
            BenchmarkMetrics::default(),
        )
    }

    #[cfg(test)]
    fn unsupported(case: BenchmarkCase, error: impl Into<String>) -> Self {
        let metadata = RunMetadata::for_test();
        Self::with_status(
            case,
            "task5-web-rwkv-test".to_owned(),
            "",
            "",
            "unknown",
            "none",
            &metadata,
            "0".to_owned(),
            "0".to_owned(),
            "unsupported",
            error.into(),
            BenchmarkMetrics::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn with_status(
        case: BenchmarkCase,
        run_id: String,
        model_path: &str,
        model_format: &str,
        device: &str,
        quantization: &str,
        metadata: &RunMetadata,
        started_at: String,
        ended_at: String,
        status: &str,
        error: String,
        metrics: BenchmarkMetrics,
    ) -> Self {
        Self {
            run_id,
            repo: "web-rwkv".to_owned(),
            backend: "webgpu-simple-runtime".to_owned(),
            runner: "webgpu-simple-runtime".to_owned(),
            benchmark_kind: "synthetic_throughput".to_owned(),
            model_size: metadata.model_size.clone(),
            model_path: model_path.to_owned(),
            model_format: model_format.to_owned(),
            device: device.to_owned(),
            gpu_name: metadata.gpu_name.clone(),
            gpu_uuid: metadata.gpu_uuid.clone(),
            dtype: "fp16".to_owned(),
            quantization: quantization.to_owned(),
            case,
            status: status.to_owned(),
            error,
            metrics,
            prompt_source: metadata.prompt_source.clone(),
            command: metadata.command.clone(),
            binary_path: metadata.binary_path.clone(),
            binary_build_id: metadata.binary_build_id.clone(),
            model_bytes: metadata.model_bytes.clone(),
            model_sha256: metadata.model_sha256.clone(),
            started_at,
            ended_at,
        }
    }

    fn to_csv_line(&self) -> String {
        [
            self.run_id.clone(),
            self.repo.clone(),
            self.backend.clone(),
            self.runner.clone(),
            self.benchmark_kind.clone(),
            self.model_size.clone(),
            self.model_path.clone(),
            self.model_format.clone(),
            self.device.clone(),
            self.gpu_name.clone(),
            self.gpu_uuid.clone(),
            self.dtype.clone(),
            self.quantization.clone(),
            self.case.bsz.to_string(),
            self.case.prompt_len.to_string(),
            self.case.decode_len.to_string(),
            self.case.warmup.to_string(),
            self.case.repeat.to_string(),
            self.case.seed.to_string(),
            self.status.clone(),
            self.error.clone(),
            self.prompt_source.clone(),
            self.case.bsz.to_string(),
            (self.case.bsz * self.case.prompt_len).to_string(),
            (self.case.bsz * self.case.prompt_len).to_string(),
            (self.case.bsz * self.case.decode_len).to_string(),
            format_opt(self.metrics.prefill_time_s),
            format_opt(self.metrics.ttft_s),
            format_opt(self.metrics.ttft_s),
            format_opt(self.metrics.e2el_s),
            format_opt(self.metrics.e2el_s),
            format_opt(self.metrics.token_generation_time_s),
            format_opt(self.metrics.prefill_tps),
            format_opt(self.metrics.decode_tps),
            format_opt(self.metrics.e2e_tps),
            format_opt(self.metrics.time_per_output_token_ms),
            String::new(),
            format_opt(self.metrics.itl_mean_ms),
            format_opt(self.metrics.itl_p50_ms),
            format_opt(self.metrics.itl_p90_ms),
            format_opt(self.metrics.itl_p95_ms),
            format_opt(self.metrics.itl_p99_ms),
            self.command.clone(),
            self.binary_path.clone(),
            self.binary_build_id.clone(),
            self.model_bytes.clone(),
            self.model_sha256.clone(),
            self.started_at.clone(),
            self.ended_at.clone(),
        ]
        .into_iter()
        .map(csv_escape)
        .collect::<Vec<_>>()
        .join(",")
    }
}

impl RunMetadata {
    #[cfg(test)]
    fn for_test() -> Self {
        Self {
            model_size: "unknown".to_owned(),
            gpu_name: "NVIDIA GeForce RTX 5090".to_owned(),
            gpu_uuid: "GPU-test".to_owned(),
            command: "task5_benchmark".to_owned(),
            binary_path: "task5_benchmark".to_owned(),
            binary_build_id: "test".to_owned(),
            model_bytes: String::new(),
            model_sha256: String::new(),
            prompt_source: "synthetic_rng".to_owned(),
        }
    }
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_owned())
}

fn make_run_id(case: BenchmarkCase) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!(
        "task5-web-rwkv-throughput-{nanos}-bsz{}-pl{}",
        case.bsz, case.prompt_len
    )
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

fn query_gpu_info() -> Result<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("failed to run nvidia-smi")?;
    if !output.status.success() {
        return Err(anyhow!(
            "{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let first = stdout
        .lines()
        .next()
        .context("nvidia-smi returned no GPU")?;
    let parts = first.split(',').map(str::trim).collect::<Vec<_>>();
    Ok(GpuInfo {
        name: parts.first().copied().unwrap_or_default().to_owned(),
        uuid: parts.get(1).copied().unwrap_or_default().to_owned(),
        driver_version: parts.get(2).copied().unwrap_or_default().to_owned(),
    })
}

fn sha256sum(path: &Path) -> Result<String> {
    let output = Command::new("sha256sum")
        .arg(path)
        .output()
        .with_context(|| format!("failed to run sha256sum for {}", path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "{}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .unwrap_or_default()
        .to_owned())
}

fn append_manifest(
    path: &Path,
    metadata: &RunMetadata,
    cli: &Cli,
    model_format: &str,
    quantization: &str,
) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let record = serde_json::json!({
        "run_id": format!("task5-web-rwkv-preflight-{}", timestamp()),
        "repo": "web-rwkv",
        "gpu_name": metadata.gpu_name,
        "gpu_uuid": metadata.gpu_uuid,
        "model_path": cli.model.display().to_string(),
        "model_format": model_format,
        "model_size": metadata.model_size,
        "quantization": quantization,
        "model_bytes": metadata.model_bytes,
        "model_sha256": metadata.model_sha256,
        "command": metadata.command,
        "binary_path": metadata.binary_path,
        "binary_build_id": metadata.binary_build_id,
        "created_at": timestamp(),
    });
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{}", record)?;
    Ok(())
}

fn append_gpu_telemetry(path: &Path, run_id: &str, gpu_uuid: &str) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let exists = path.exists();
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    if !exists {
        writeln!(
            file,
            "timestamp,run_id,gpu_uuid,gpu_util,mem_used,mem_total,power_w,sm_clock,mem_clock,pstate,process_name"
        )?;
    }
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,pstate",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let values = match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .map(|line| {
                line.split(',')
                    .map(|part| part.trim().to_owned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    };
    let field = |index: usize| values.get(index).cloned().unwrap_or_default();
    let row = [
        timestamp(),
        run_id.to_owned(),
        if field(1).is_empty() {
            gpu_uuid.to_owned()
        } else {
            field(1)
        },
        field(2),
        field(3),
        field(4),
        field(5),
        field(6),
        field(7),
        field(8),
        "web-rwkv".to_owned(),
    ]
    .into_iter()
    .map(csv_escape)
    .collect::<Vec<_>>()
    .join(",");
    writeln!(file, "{row}")?;
    Ok(())
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
    let normalized = device.to_ascii_lowercase();
    if normalized.contains("llvmpipe")
        || normalized.contains("software")
        || normalized.contains("unknown")
    {
        return Err(SetupFailure::Unsupported(format!(
            "web-rwkv adapter is not a traceable RTX 5090 device: {device}"
        )));
    }
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

fn make_prompt(
    info: &ModelInfo,
    case: BenchmarkCase,
    rng: &mut fastrand::Rng,
) -> Vec<RnnInputBatch> {
    let max_token = info.num_vocab.saturating_sub(1).min(u32::MAX as usize) as u32;
    (0..case.bsz)
        .map(|_| {
            let tokens = (0..case.prompt_len)
                .map(|_| rng.u32(0..max_token.max(1)))
                .collect::<Vec<_>>();
            RnnInputBatch::new(tokens, RnnOption::Last)
        })
        .collect()
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
        *token = Some({
            let logits = batch.0.data();
            logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as u32)
                .context("empty logits")
        }?);
    }
    Ok(())
}

fn collect_tokens(tokens: Vec<Option<u32>>, stage: &str) -> Result<Vec<u32>> {
    tokens
        .into_iter()
        .enumerate()
        .map(|(index, token)| {
            token.with_context(|| format!("{stage} produced no output logits for batch {index}"))
        })
        .collect()
}

fn csv_header() -> &'static str {
    CSV_HEADER
}

fn write_csv(path: &Path, rows: &[BenchmarkRow]) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut output = String::new();
    output.push_str(csv_header());
    output.push('\n');
    for row in rows {
        output.push_str(&row.to_csv_line());
        output.push('\n');
    }
    fs::write(path, output)?;
    Ok(())
}

fn benchmark_cases(bsz: &[usize], prompt_lens: &[usize], cli: &Cli) -> Vec<BenchmarkCase> {
    bsz.iter()
        .flat_map(|&bsz| {
            prompt_lens.iter().map(move |&prompt_len| BenchmarkCase {
                bsz,
                prompt_len,
                decode_len: cli.decode_len,
                warmup: cli.warmup,
                repeat: cli.repeat,
                seed: cli.seed,
            })
        })
        .collect()
}

fn parse_list(value: &str, name: &str) -> Result<Vec<usize>> {
    value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<usize>()
                .with_context(|| format!("invalid {name} value: {part}"))
        })
        .collect()
}

fn max_bsz(cases: &[BenchmarkCase]) -> usize {
    cases.iter().map(|case| case.bsz).max().unwrap_or(1)
}

fn model_format(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("unknown")
        .to_owned()
}

fn quantization_label(quant: QuantConfig) -> String {
    let mut parts = vec![];
    if quant.int8 > 0 {
        parts.push(format!("int8:{}", quant.int8));
    }
    if quant.nf4 > 0 {
        parts.push(format!("nf4:{}", quant.nf4));
    }
    if quant.sf4 > 0 {
        parts.push(format!("sf4:{}", quant.sf4));
    }
    match parts.is_empty() {
        true => "none".to_owned(),
        false => parts.join("+"),
    }
}

fn format_opt(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.9}"))
        .unwrap_or_default()
}

fn csv_escape(value: String) -> String {
    match value.contains([',', '"', '\n', '\r']) {
        true => format!("\"{}\"", value.replace('"', "\"\"")),
        false => value,
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    sum / count as f64
}

fn rate(tokens: f64, seconds: f64) -> Option<f64> {
    (seconds > 0.0).then_some(tokens / seconds)
}

fn percentile_option(values: &mut [f64], quantile: f64) -> Option<f64> {
    match values.is_empty() {
        true => None,
        false => Some(percentile_ms(values, quantile)),
    }
}

fn percentile_ms(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_header_matches_readme_schema() {
        assert_eq!(
            csv_header(),
            "run_id,repo,backend,runner,benchmark_kind,model_size,model_path,model_format,device,gpu_name,gpu_uuid,dtype,quantization,bsz,prompt_len,decode_len,warmup,repeat,seed,status,error,prompt_source,prompt_count,prompt_tokens,prefill_tokens,output_tokens,prefill_time_s,ttft_s,ttft_p95_s,e2el_s,e2el_p95_s,token_generation_time_s,prefill_tps,decode_tps,e2e_tps,time_per_output_token_ms,requests_per_s,itl_mean_ms,itl_p50_ms,itl_p90_ms,itl_p95_ms,itl_p99_ms,command,binary_path,binary_build_id,model_bytes,model_sha256,started_at,ended_at"
        );
    }

    #[test]
    fn unsupported_row_is_written_instead_of_skipped() {
        let row = BenchmarkRow::unsupported(
            BenchmarkCase {
                bsz: 64,
                prompt_len: 4096,
                decode_len: 16,
                warmup: 1,
                repeat: 1,
                seed: 42,
            },
            "no compatible adapter",
        );

        let line = row.to_csv_line();

        assert!(line.contains(",64,4096,16,1,1,42,unsupported,"));
        assert!(line.contains("no compatible adapter"));
    }

    #[test]
    fn percentile_uses_nearest_rank_for_recorded_itls() {
        let values = [10.0, 20.0, 30.0, 40.0];

        assert_eq!(percentile_ms(&values, 0.50), 20.0);
        assert_eq!(percentile_ms(&values, 0.90), 40.0);
        assert_eq!(percentile_ms(&values, 0.95), 40.0);
        assert_eq!(percentile_ms(&values, 0.99), 40.0);
    }
}
