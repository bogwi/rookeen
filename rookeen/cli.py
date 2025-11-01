from __future__ import annotations

import sys

import click

from rookeen.cli_func import (
    _get_version,
    analyze_file,
    analyze_url,
    batch_analyze,
)
from rookeen.config import load_settings
from rookeen.errors import USAGE, emit_and_exit


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=_get_version(), prog_name="rookeen")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str),
    default=None,
    help="Path to TOML config file with defaults.",
)
@click.option(
    "--errors-json",
    is_flag=True,
    default=False,
    help="Force machine-readable JSON error output for automation and scripting.",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, errors_json: bool) -> None:
    """Rookeen CLI: analyze web pages or files using spaCy-only analyzers."""
    # Load settings once and pass via Click context
    settings = load_settings(config_path)
    ctx.ensure_object(dict)
    ctx.obj = ctx.obj or {}
    ctx.obj["settings"] = settings
    ctx.obj["errors_json"] = errors_json



@cli.command("analyze", short_help="Analyze web pages or text with 20+ linguistic analyzers (embeddings, sentiment, POS, NER, readability, etc.)")
@click.argument("url", type=str, nargs=-1)
@click.option(
    "--output",
    "output_base",
    "-o",
    type=str,
    default=None,
    help="Base output path (JSON always saved).",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(["json", "md", "html", "all"], case_sensitive=False),
    default=None,
    help="Output format(s). JSON is always written.",
)
@click.option(
    "--lang",
    "lang_override",
    type=str,
    default=None,
    help="Override language detection (e.g., en,de,es,fr).",
)
@click.option(
    "--languages",
    "preload_languages_csv",
    type=str,
    default=None,
    help="Comma-separated languages to preload models (e.g., en,de,es,fr).",
)
@click.option(
    "--models-auto-download/--no-models-auto-download",
    default=None,
    help="Attempt to auto-download missing spaCy models.",
)
@click.option(
    "--export-spacy-json/--no-export-spacy-json",
    default=False,
    help="Also write token-level spaCy JSON to <base>.spacy.json",
)
@click.option(
    "--export-docbin/--no-export-docbin",
    default=False,
    help="Also write DocBin snapshot to <base>.docbin",
)
@click.option(
    "--export-conllu/--no-export-conllu",
    default=False,
    help="Also write CoNLL-U format to <base>.conllu",
)
@click.option(
    "--conllu-engine",
    type=click.Choice(["auto", "stanza", "basic"], case_sensitive=False),
    default="auto",
    help="Engine for CoNLL-U export: 'stanza' (UD-valid) or 'basic' (heuristic).",
)
@click.option(
    "--ud-auto-download/--no-ud-auto-download",
    default=True,
    help="Auto-download Stanza models when using --export-conllu with stanza engine.",
)
@click.option(
    "--allow-non-ud-conllu",
    is_flag=True,
    default=False,
    help="Allow heuristic basic exporter when UD engine is unavailable.",
)
@click.option(
    "--stdin",
    is_flag=True,
    default=False,
    help="Read text from stdin instead of URL.",
)
@click.option(
    "--stdout",
    is_flag=True,
    default=False,
    help="Stream JSON to stdout (no files) for pipeline composition.",
)
@click.option(
    "--export-parquet/--no-export-parquet",
    default=False,
    help="Also write analyzer summary table to <base>.parquet",
)
@click.option(
    "--rate-limit",
    "rate_limit",
    type=float,
    default=0.5,
    help="Rate limit in requests per second (default: 0.5)",
)
@click.option(
    "--robots",
    "robots_policy",
    type=click.Choice(["respect", "ignore"], case_sensitive=False),
    default="respect",
    help="Robots.txt policy: 'respect' or 'ignore' (default: respect)",
)
@click.option("--trace-id", "trace_id", type=str, default=None, help="Attach trace ID to logs.")
@click.option("--verbose", "verbose", is_flag=True, default=False, help="Verbose logs.")
@click.option(
    "--enable-embeddings",
    is_flag=True,
    default=False,
    help="Enable sentence embeddings analysis (requires 'rookeen[embeddings]')",
)
@click.option(
    "--enable-sentiment",
    is_flag=True,
    default=False,
    help="Enable sentiment analysis (requires 'rookeen[sentiment]')",
)
@click.option(
    "--embeddings-backend",
    type=click.Choice(["miniLM", "bge-m3", "openai-te3"], case_sensitive=False),
    default=None,
    help="Embeddings backend to use when --enable-embeddings is set.",
)
@click.option(
    "--embeddings-model",
    type=str,
    default=None,
    help="Model identifier for the selected backend (e.g., BAAI/bge-m3, text-embedding-3-small).",
)
@click.option(
    "--openai-api-key",
    type=str,
    default=None,
    help="OpenAI API key for openai-te3 backend (falls back to env).",
)
@click.option(
    "--embeddings-preload/--no-embeddings-preload",
    is_flag=True,
    default=False,
    help="Preload embeddings backend/model at startup to avoid first-call latency.",
)
@click.option(
    "--enable",
    "enabled_analyzers",
    multiple=True,
    help="Enable specific analyzers by name (can be used multiple times)",
)
@click.option(
    "--disable",
    "disabled_analyzers",
    multiple=True,
    help="Disable specific analyzers by name (can be used multiple times)",
)
@click.pass_context
def cmd_analyze(
    ctx: click.Context,
    url: tuple[str, ...],
    output_base: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    stdin: bool,
    stdout: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    rate_limit: float,
    robots_policy: str,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: tuple[str, ...],
    disabled_analyzers: tuple[str, ...],
) -> None:
    """Analyze a single URL or text from stdin."""
    settings = ctx.obj.get("settings")

    # Validate input: either URL provided or --stdin flag
    if stdin and url:
        raise click.UsageError("Cannot specify both URL and --stdin")
    if not stdin and not url:
        raise click.UsageError("Must specify either URL or --stdin")

    if stdin:
        from rookeen.cli_func import analyze_stdin

        analyze_stdin(
            output_base=output_base,
            format_=format_,
            lang_override=lang_override,
            preload_languages_csv=preload_languages_csv,
            models_auto_download=models_auto_download,
            export_spacy_json=export_spacy_json,
            export_docbin=export_docbin,
            export_conllu=export_conllu,
            conllu_engine=conllu_engine,
            ud_auto_download=ud_auto_download,
            allow_non_ud_conllu=allow_non_ud_conllu,
            stdout=stdout,
            trace_id=trace_id,
            verbose=verbose,
            export_parquet=export_parquet,
            enable_embeddings=enable_embeddings,
            enable_sentiment=enable_sentiment,
            embeddings_backend=embeddings_backend,
            embeddings_model=embeddings_model,
            openai_api_key=openai_api_key,
            embeddings_preload=embeddings_preload,
            enabled_analyzers=list(enabled_analyzers),
            disabled_analyzers=list(disabled_analyzers),
            settings=settings,
        )
    else:
        # URL mode - take first URL from tuple
        url_str = url[0] if isinstance(url, tuple) else url
        analyze_url(
            url=url_str,
            output_base=output_base,
            format_=format_,
            lang_override=lang_override,
            preload_languages_csv=preload_languages_csv,
            models_auto_download=models_auto_download,
            export_spacy_json=export_spacy_json,
            export_docbin=export_docbin,
            export_conllu=export_conllu,
            conllu_engine=conllu_engine,
            ud_auto_download=ud_auto_download,
            allow_non_ud_conllu=allow_non_ud_conllu,
            stdout=stdout,
            trace_id=trace_id,
            verbose=verbose,
            export_parquet=export_parquet,
            rate_limit=rate_limit,
            robots_policy=robots_policy,
            enable_embeddings=enable_embeddings,
            enable_sentiment=enable_sentiment,
            embeddings_backend=embeddings_backend,
            embeddings_model=embeddings_model,
            openai_api_key=openai_api_key,
            embeddings_preload=embeddings_preload,
            enabled_analyzers=list(enabled_analyzers),
            disabled_analyzers=list(disabled_analyzers),
            settings=settings,
        )


@cli.command("analyze-file", short_help="Analyze local text files with 20+ linguistic analyzers (embeddings, sentiment, POS, NER, readability, etc.)")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str))
@click.option(
    "--output",
    "output_base",
    "-o",
    type=str,
    default=None,
    help="Base output path (JSON always saved).",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(["json", "md", "html", "all"], case_sensitive=False),
    default=None,
    help="Output format(s). JSON is always written.",
)
@click.option(
    "--lang",
    "lang_override",
    type=str,
    default=None,
    help="Override language detection (e.g., en,de,es,fr).",
)
@click.option(
    "--languages",
    "preload_languages_csv",
    type=str,
    default=None,
    help="Comma-separated languages to preload models (e.g., en,de,es,fr).",
)
@click.option(
    "--models-auto-download/--no-models-auto-download",
    default=None,
    help="Attempt to auto-download missing spaCy models.",
)
@click.option(
    "--export-spacy-json/--no-export-spacy-json",
    default=False,
    help="Also write token-level spaCy JSON to <base>.spacy.json",
)
@click.option(
    "--export-docbin/--no-export-docbin",
    default=False,
    help="Also write DocBin snapshot to <base>.docbin",
)
@click.option(
    "--export-conllu/--no-export-conllu",
    default=False,
    help="Also write CoNLL-U format to <base>.conllu",
)
@click.option(
    "--conllu-engine",
    type=click.Choice(["auto", "stanza", "basic"], case_sensitive=False),
    default="auto",
    help="Engine for CoNLL-U export: 'stanza' (UD-valid) or 'basic' (heuristic).",
)
@click.option(
    "--ud-auto-download/--no-ud-auto-download",
    default=True,
    help="Auto-download Stanza models when using --export-conllu with stanza engine.",
)
@click.option(
    "--allow-non-ud-conllu",
    is_flag=True,
    default=False,
    help="Allow heuristic basic exporter when UD engine is unavailable.",
)
@click.option(
    "--stdout",
    is_flag=True,
    default=False,
    help="Stream JSON to stdout (no files) for pipeline composition.",
)
@click.option(
    "--export-parquet/--no-export-parquet",
    default=False,
    help="Also write analyzer summary table to <base>.parquet",
)
@click.option("--trace-id", "trace_id", type=str, default=None, help="Attach trace ID to logs.")
@click.option("--verbose", "verbose", is_flag=True, default=False, help="Verbose logs.")
@click.option(
    "--enable-embeddings",
    is_flag=True,
    default=False,
    help="Enable sentence embeddings analysis (requires 'rookeen[embeddings]')",
)
@click.option(
    "--enable-sentiment",
    is_flag=True,
    default=False,
    help="Enable sentiment analysis (requires 'rookeen[sentiment]')",
)
@click.option(
    "--embeddings-backend",
    type=click.Choice(["miniLM", "bge-m3", "openai-te3"], case_sensitive=False),
    default=None,
    help="Embeddings backend to use when --enable-embeddings is set.",
)
@click.option(
    "--embeddings-model",
    type=str,
    default=None,
    help="Model identifier for the selected backend (e.g., BAAI/bge-m3, text-embedding-3-small).",
)
@click.option(
    "--openai-api-key",
    type=str,
    default=None,
    help="OpenAI API key for openai-te3 backend (falls back to env).",
)
@click.option(
    "--embeddings-preload/--no-embeddings-preload",
    is_flag=True,
    default=False,
    help="Preload embeddings backend/model at startup to avoid first-call latency.",
)
@click.option(
    "--enable",
    "enabled_analyzers",
    multiple=True,
    help="Enable specific analyzers by name (can be used multiple times)",
)
@click.option(
    "--disable",
    "disabled_analyzers",
    multiple=True,
    help="Disable specific analyzers by name (can be used multiple times)",
)
@click.pass_context
def cmd_analyze_file(
    ctx: click.Context,
    path: str,
    output_base: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    stdout: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: tuple[str, ...],
    disabled_analyzers: tuple[str, ...],
) -> None:
    """Analyze a local text file."""
    settings = ctx.obj.get("settings")
    analyze_file(
        path=path,
        output_base=output_base,
        format_=format_,
        lang_override=lang_override,
        preload_languages_csv=preload_languages_csv,
        models_auto_download=models_auto_download,
        export_spacy_json=export_spacy_json,
        export_docbin=export_docbin,
        export_conllu=export_conllu,
        conllu_engine=conllu_engine,
        ud_auto_download=ud_auto_download,
        allow_non_ud_conllu=allow_non_ud_conllu,
        stdout=stdout,
        trace_id=trace_id,
        verbose=verbose,
        export_parquet=export_parquet,
        enable_embeddings=enable_embeddings,
        enable_sentiment=enable_sentiment,
        embeddings_backend=embeddings_backend,
        embeddings_model=embeddings_model,
        openai_api_key=openai_api_key,
            embeddings_preload=embeddings_preload,
        enabled_analyzers=list(enabled_analyzers),
        disabled_analyzers=list(disabled_analyzers),
        settings=settings,
    )


@cli.command("batch", short_help="Batch analyze multiple URLs from file with full analyzer suite (rate limiting, robots.txt support)")
@click.argument(
    "url_list_file", type=click.Path(exists=True, dir_okay=False, readable=True, path_type=str)
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory to write outputs.",
)
@click.option(
    "--format",
    "format_",
    type=click.Choice(["json", "md", "html", "all"], case_sensitive=False),
    default=None,
)
@click.option("--lang", "lang_override", type=str, default=None)
@click.option("--languages", "preload_languages_csv", type=str, default=None)
@click.option("--models-auto-download/--no-models-auto-download", default=None)
@click.option(
    "--export-spacy-json/--no-export-spacy-json",
    default=False,
    help="Also write token-level spaCy JSON to <base>.spacy.json",
)
@click.option(
    "--export-docbin/--no-export-docbin",
    default=False,
    help="Also write DocBin snapshot to <base>.docbin",
)
@click.option(
    "--export-conllu/--no-export-conllu",
    default=False,
    help="Also write CoNLL-U format to <base>.conllu",
)
@click.option(
    "--conllu-engine",
    type=click.Choice(["auto", "stanza", "basic"], case_sensitive=False),
    default="auto",
    help="Engine for CoNLL-U export: 'stanza' (UD-valid) or 'basic' (heuristic).",
)
@click.option(
    "--ud-auto-download/--no-ud-auto-download",
    default=True,
    help="Auto-download Stanza models when using --export-conllu with stanza engine.",
)
@click.option(
    "--allow-non-ud-conllu",
    is_flag=True,
    default=False,
    help="Allow heuristic basic exporter when UD engine is unavailable.",
)
@click.option(
    "--export-parquet/--no-export-parquet",
    default=False,
    help="Also write analyzer summary table to <base>.parquet",
)
@click.option(
    "--rate-limit",
    "rate_limit",
    type=float,
    default=0.5,
    help="Rate limit in requests per second (default: 0.5)",
)
@click.option(
    "--robots",
    "robots_policy",
    type=click.Choice(["respect", "ignore"], case_sensitive=False),
    default="respect",
    help="Robots.txt policy: 'respect' or 'ignore' (default: respect)",
)
@click.option("--trace-id", "trace_id", type=str, default=None, help="Attach trace ID to logs.")
@click.option("--verbose", "verbose", is_flag=True, default=False)
@click.option(
    "--enable-embeddings",
    is_flag=True,
    default=False,
    help="Enable sentence embeddings analysis (requires 'rookeen[embeddings]')",
)
@click.option(
    "--enable-sentiment",
    is_flag=True,
    default=False,
    help="Enable sentiment analysis (requires 'rookeen[sentiment]')",
)
@click.option(
    "--embeddings-backend",
    type=click.Choice(["miniLM", "bge-m3", "openai-te3"], case_sensitive=False),
    default=None,
    help="Embeddings backend to use when --enable-embeddings is set.",
)
@click.option(
    "--embeddings-model",
    type=str,
    default=None,
    help="Model identifier for the selected backend (e.g., BAAI/bge-m3, text-embedding-3-small).",
)
@click.option(
    "--openai-api-key",
    type=str,
    default=None,
    help="OpenAI API key for openai-te3 backend (falls back to env).",
)
@click.option(
    "--embeddings-preload/--no-embeddings-preload",
    is_flag=True,
    default=False,
    help="Preload embeddings backend/model at startup to avoid first-call latency.",
)
@click.option(
    "--enable",
    "enabled_analyzers",
    multiple=True,
    help="Enable specific analyzers by name (can be used multiple times)",
)
@click.option(
    "--disable",
    "disabled_analyzers",
    multiple=True,
    help="Disable specific analyzers by name (can be used multiple times)",
)
@click.pass_context
def cmd_batch(
    ctx: click.Context,
    url_list_file: str,
    output_dir: str | None,
    format_: str | None,
    lang_override: str | None,
    preload_languages_csv: str | None,
    models_auto_download: bool | None,
    export_spacy_json: bool,
    export_docbin: bool,
    export_conllu: bool,
    conllu_engine: str,
    ud_auto_download: bool,
    allow_non_ud_conllu: bool,
    trace_id: str | None,
    verbose: bool,
    export_parquet: bool,
    rate_limit: float,
    robots_policy: str,
    enable_embeddings: bool,
    enable_sentiment: bool,
    embeddings_backend: str | None,
    embeddings_model: str | None,
    openai_api_key: str | None,
    embeddings_preload: bool,
    enabled_analyzers: tuple[str, ...],
    disabled_analyzers: tuple[str, ...],
) -> None:
    """Analyze a list of URLs from a file (one per line; '#' comments allowed)."""
    settings = ctx.obj.get("settings")
    batch_analyze(
        url_list_file=url_list_file,
        output_dir=output_dir,
        format_=format_,
        lang_override=lang_override,
        preload_languages_csv=preload_languages_csv,
        models_auto_download=models_auto_download,
        export_spacy_json=export_spacy_json,
        export_docbin=export_docbin,
        export_conllu=export_conllu,
        conllu_engine=conllu_engine,
        ud_auto_download=ud_auto_download,
        allow_non_ud_conllu=allow_non_ud_conllu,
        trace_id=trace_id,
        verbose=verbose,
        export_parquet=export_parquet,
        rate_limit=rate_limit,
        robots_policy=robots_policy,
        enable_embeddings=enable_embeddings,
        enable_sentiment=enable_sentiment,
        embeddings_backend=embeddings_backend,
        embeddings_model=embeddings_model,
        openai_api_key=openai_api_key,
        embeddings_preload=embeddings_preload,
        enabled_analyzers=list(enabled_analyzers),
        disabled_analyzers=list(disabled_analyzers),
        settings=settings,
    )



def main() -> None:
    """Main entry point with proper error handling."""
    # Use Click's main function with custom exception handling
    try:
        cli.main(standalone_mode=False)
    except click.ClickException as e:
        # Handle Click's built-in exceptions (usage errors, etc.)
        errors_json = "--errors-json" in sys.argv
        if errors_json:
            emit_and_exit(USAGE)
        else:
            # Let Click handle it normally
            e.show()
            sys.exit(e.exit_code)
    except click.Abort:
        # Handle Ctrl+C
        errors_json = "--errors-json" in sys.argv
        if errors_json:
            emit_and_exit(USAGE)
        else:
            raise
    except SystemExit:
        # Re-raise SystemExit (from successful commands)
        raise
    except Exception:
        # Handle any other unexpected exceptions
        errors_json = "--errors-json" in sys.argv
        if errors_json:
            from rookeen.errors import GENERIC

            emit_and_exit(GENERIC)
        else:
            raise


if __name__ == "__main__":
    import sys

    main()
