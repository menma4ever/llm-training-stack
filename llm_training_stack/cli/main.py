"""Typer CLI interface for LLM Training Stack."""

from pathlib import Path
from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table

from llm_training_stack.config.loader import ConfigLoader
from llm_training_stack.config.schema import TaskType
from llm_training_stack.preflight.hardware import HardwareInspector
from llm_training_stack.preflight.probe import EmpiricalMemoryProbe
from llm_training_stack.preflight.policy_contract import PreflightPolicyContract
from llm_training_stack.checkpoints.resume import ResumeManager
from llm_training_stack.eval.comparator import RunComparator
from llm_training_stack.lifecycle.manager import LifecycleManager, JobStatus

app = typer.Typer(
    name="train-stack",
    help="Professional open-source LLM Training Stack for CPT, SFT, LoRA/PEFT, and DPO.",
    add_completion=False,
)
console = Console()


@app.command()
def inspect():
    """Inspect local host hardware, CUDA availability, and compute specs."""
    info = HardwareInspector.inspect()
    table = Table(title="LLM Training Stack — Host Hardware Inspection")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Platform", str(info["platform"]))
    table.add_row("Python Version", str(info["python_version"]))
    table.add_row("CPU Physical / Logical", f"{info['cpu_count_physical']} / {info['cpu_count_logical']}")
    table.add_row("System RAM Total / Available", f"{info['system_ram_total_gb']} GB / {info['system_ram_available_gb']} GB")
    table.add_row("PyTorch Version", str(info["torch_version"]))
    table.add_row("CUDA Available", "[green]True[/green]" if info["cuda_available"] else "[yellow]False (CPU Mode)[/yellow]")
    table.add_row("GPU Device Count", str(info["cuda_device_count"]))
    table.add_row("Recommended Device", str(info["recommended_device"]))

    console.print(table)


@app.command()
def preflight(config_path: Path = typer.Option(..., "--config", "-c", help="Path to YAML/JSON config file")):
    """Run analytical memory preflight calculation and bounded empirical probe."""
    config = ConfigLoader.load_from_file(config_path)
    console.print(f"[bold blue]Loading config for task:[/] {config.task_type.value.upper()} on model [italic]{config.model.model_name_or_path}[/]")

    report = PreflightPolicyContract.run_isolated_probe(
        model_path=config.model.model_name_or_path,
        config=config,
    )
    if not report.get("probe_successful", False):
        err = report.get("error", "Empirical probe failed.")
        console.print(f"[bold red]Probe failed with error:[/] {err}")
        raise typer.Exit(code=1)

    table = Table(title="Preflight Memory & Probe Verdict")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green" if report.get("probe_successful") else "red")

    table.add_row("Probe Verdict", str(report.get("verdict", "N/A")))
    table.add_row("Device Target", str(report.get("device", "N/A")))
    table.add_row("Measured Peak Memory", f"{report.get('measured_peak_allocated_mb', 0)} MB")
    table.add_row("Micro-Batch Delta", f"{report.get('measured_step_delta_mb', 0)} MB")
    table.add_row("Available Memory", f"{report.get('available_device_memory_mb', 0)} MB")
    table.add_row("Memory Headroom", f"{report.get('memory_headroom_percent', 0)}%")

    console.print(table)


@app.command()
def train(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to YAML/JSON config file"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Checkpoint directory to resume from"),
):
    """Launch validated training pipeline (CPT, SFT, LoRA, DPO) with durable state tracking."""
    config = ConfigLoader.load_from_file(config_path)
    console.print(f"[bold green]Starting Training Job:[/] Task={config.task_type.value.upper()} Output={config.logging.output_dir}")

    record = LifecycleManager.submit_job(config, resume_from=str(resume) if resume else None, run_in_background=False, isolate_job_dir=False)
    if record.status == JobStatus.COMPLETED:
        console.print(f"[bold green]Job Finished Successfully![/]")
        res = record.metrics or {}
        console.print(f"Total Steps: {res.get('total_steps', 0)}, Final Loss: {res.get('final_loss', 'N/A')}")
        console.print(f"Checkpoint saved to: {res.get('final_checkpoint', 'N/A')}")
    else:
        console.print(f"[bold red]Job Finished with Status: {record.status.value}[/]")
        if record.error:
            console.print(f"Error: {record.error}")
        raise typer.Exit(code=1)


@app.command()
def submit(
    config_path: Path = typer.Option(..., "--config", "-c", help="Path to YAML/JSON config file"),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Checkpoint directory to resume from"),
    background: bool = typer.Option(True, "--background/--foreground", help="Run job in background thread"),
):
    """Durable asynchronous job submission returning tracking handle."""
    config = ConfigLoader.load_from_file(config_path)
    record = LifecycleManager.submit_job(config, resume_from=str(resume) if resume else None, run_in_background=background, isolate_job_dir=True)
    console.print(f"[bold green]Job Submitted:[/] ID={record.job_id} Status={record.status.value}")
    console.print(f"Run Directory: {record.run_dir}")


@app.command(name="run-worker", hidden=True)
def run_worker(
    run_dir: Path = typer.Option(..., "--run-dir", help="Path to run directory"),
):
    """Internal entry point executed by background worker processes."""
    LifecycleManager.run_worker(run_dir)


@app.command()
def status(job_dir_or_id: str = typer.Argument(..., help="Path to run directory or Job ID")):
    """Check status of a submitted training job."""
    try:
        record = LifecycleManager.get_job_status(job_dir_or_id)
    except Exception as e:
        console.print(f"[bold red]Failed to get status:[/] {e}")
        raise typer.Exit(code=1)

    table = Table(title=f"Training Job Status: {record.job_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green" if record.status == JobStatus.COMPLETED else "yellow")

    table.add_row("Status", record.status.value)
    table.add_row("Task Type", record.task_type.value)
    table.add_row("Created At", record.created_at)
    table.add_row("Started At", record.started_at or "N/A")
    table.add_row("Finished At", record.finished_at or "N/A")
    table.add_row("Current Step", f"{record.current_step} / {record.total_steps}")
    table.add_row("Latest Loss", str(record.latest_loss) if record.latest_loss is not None else "N/A")
    if record.error:
        table.add_row("Error", f"[red]{record.error}[/]")

    console.print(table)


@app.command()
def logs(
    job_dir_or_id: str = typer.Argument(..., help="Path to run directory or Job ID"),
    tail: int = typer.Option(20, "--tail", "-n", help="Number of recent log events to show"),
):
    """Inspect structured logs and recent events of a training job."""
    events = LifecycleManager.get_job_logs(job_dir_or_id, tail_lines=tail)
    if not events:
        console.print("[yellow]No logs found in run directory.[/yellow]")
        return

    table = Table(title=f"Recent Job Logs ({job_dir_or_id})")
    table.add_column("Step", style="cyan")
    table.add_column("Loss", style="green")
    table.add_column("Tokens/sec", style="magenta")
    table.add_column("Timestamp", style="white")

    for ev in events:
        if "raw" in ev:
            console.print(ev["raw"])
        else:
            table.add_row(
                str(ev.get("step", "N/A")),
                f"{ev.get('loss', 0.0):.4f}" if "loss" in ev else "N/A",
                f"{ev.get('tokens_per_second', 0.0):.1f}" if "tokens_per_second" in ev else "N/A",
                str(ev.get("timestamp", "")),
            )
    console.print(table)


@app.command()
def cancel(job_dir_or_id: str = typer.Argument(..., help="Path to run directory or Job ID")):
    """Cancel an active or running training job."""
    try:
        record = LifecycleManager.cancel_job(job_dir_or_id)
        if record.status == JobStatus.COMPLETED:
            console.print(f"[bold yellow]Job {record.job_id} had already completed prior to cancellation. Status: {record.status.value}[/]")
        elif record.status == JobStatus.FAILED:
            console.print(f"[bold yellow]Job {record.job_id} had already failed prior to cancellation. Status: {record.status.value}[/]")
        elif record.status == JobStatus.CANCELLED:
            console.print(f"[bold yellow]Job {record.job_id} cancelled. Status: {record.status.value}[/]")
        else:
            console.print(f"[bold yellow]Job {record.job_id} cancellation requested. Status: {record.status.value}[/]")
    except Exception as e:
        console.print(f"[bold red]Failed to cancel job:[/] {e}")
        raise typer.Exit(code=1)


@app.command()
def resume(checkpoint_dir: Path = typer.Option(..., "--checkpoint", help="Checkpoint folder")):
    """Inspect and verify checkpoint integrity for resumption."""
    info = ResumeManager.inspect_checkpoint(checkpoint_dir)
    console.print(f"[bold blue]Checkpoint Inspection:[/] {info['path']}")
    console.print(f"Step: {info['step']}, Epoch: {info['epoch']}, Loss: {info['loss']}")
    console.print(f"Is Complete Bundle: {info['is_complete_bundle']}")


@app.command()
def compare(run_dirs: List[Path] = typer.Argument(..., help="List of run directories to compare")):
    """Compare multiple training runs by parsing manifests, held-out evaluation reports, and metrics."""
    result = RunComparator.compare_runs(run_dirs)
    console.print(result["markdown_report"])


@app.command()
def eval(
    model: str = typer.Option(..., "--model", "-m", help="Model name, path, or checkpoint directory"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to evaluation dataset file or dataset identifier"),
    max_seq_length: int = typer.Option(512, "--max-seq-length", help="Maximum sequence length for evaluation"),
    split: str = typer.Option("test", "--split", help="Dataset split to evaluate (if HF dataset)"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", help="Maximum number of samples to evaluate"),
    output_report: Optional[Path] = typer.Option(None, "--output-report", "-o", help="Optional path to save evaluation report JSON"),
):
    """Evaluate model loss and perplexity on held-out dataset."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llm_training_stack.pipelines.base import load_strict_dataset
    from llm_training_stack.eval.evaluator import Evaluator

    console.print(f"[bold blue]Evaluating model:[/] {model} on dataset [italic]{dataset}[/]")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        eval_model = AutoModelForCausalLM.from_pretrained(model)
        ds = load_strict_dataset(dataset, split=split)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize evaluation:[/] {e}")
        raise typer.Exit(code=1)

    eval_samples = []
    if isinstance(ds, list):
        for item in ds:
            if isinstance(item, str):
                eval_samples.append(item)
            elif isinstance(item, dict):
                eval_samples.append(item)
    else:
        for item in ds:
            eval_samples.append(item)

    if sample_limit:
        eval_samples = eval_samples[:sample_limit]

    try:
        metrics = Evaluator.evaluate(
            eval_model,
            tokenizer,
            eval_samples,
            max_seq_length=max_seq_length,
            model_name_or_path=model,
        )
    except Exception as e:
        console.print(f"[bold red]Evaluation failed:[/] {e}")
        raise typer.Exit(code=1)

    if output_report:
        Evaluator.save_report(metrics, output_report)
        console.print(f"[green]Saved evaluation report to {output_report}[/green]")

    table = Table(title="LLM Training Stack — Evaluation Results")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", str(model))
    table.add_row("Evaluated Samples", str(metrics["evaluated_samples"]))
    table.add_row("Total Valid Tokens", str(metrics["total_tokens"]))
    table.add_row("Cross-Entropy Loss", f"{metrics['eval_loss']:.4f}")
    table.add_row("Perplexity", f"{metrics['perplexity']:.3f}")
    table.add_row("Masking Scheme", metrics.get("masking_scheme", "all_tokens"))
    table.add_row("Dataset Fingerprint", metrics.get("dataset_fingerprint", "N/A"))

    console.print(table)


@app.command()
def mcp():
    """Launch the safe Model Context Protocol server."""
    try:
        from llm_training_stack.mcp.server import run_server
    except ImportError as e:
        console.print("[bold red]Error:[/] MCP dependencies are not installed.")
        console.print("Please install with: [bold green]pip install \"llm-training-stack[mcp]\"[/]")
        raise typer.Exit(code=1)
    run_server()


if __name__ == "__main__":
    app()
