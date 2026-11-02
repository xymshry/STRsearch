"""Exact STR panel search and probability evaluation.

The program follows the supplied STR matching specifications:
- full-panel unordered autosomal genotype matching;
- single-allele matching for loci named DYS*/YGATA*;
- Hardy-Weinberg genotype probabilities and a product rule;
- log10 RMP, complete-match log10 LR, and database-wide accidental-match risk.

The database is scanned once while full-panel query keys are looked up. This
keeps memory bounded for a million-row text database while preserving exact
matching semantics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DEFAULT_DATABASE_DIR = BASE_DIR / "STR_database_20loci"
DEFAULT_SAMPLES_DIR = BASE_DIR / "STR_test_samples_10_split"
DEFAULT_OUTPUT_DIR = BASE_DIR / "STR_processing_results"
DEFAULT_DISTRIBUTION = BASE_DIR / "STRDistribution.txt"
DEFAULT_ALPHA = None
LOG10_E = math.log10(math.e)


def is_sex_chromosome_locus(name: str) -> bool:
    upper = name.upper()
    return (
        upper.startswith(("DYS", "YGATA"))
        or upper in {"X", "Y", "AMEL", "AMELX", "AMELY", "SRY"}
        or upper.endswith("Y")
    )


def canonical_allele(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty allele label")
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        raise ValueError(f"Invalid numeric allele label: {value}")
    return format(number, ".12g")


def allele_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def canonical_panel_key(values: list[str], locus_names: list[str]) -> tuple[str, ...]:
    if len(values) != len(locus_names):
        raise ValueError(f"Expected {len(locus_names)} loci, got {len(values)}")

    key: list[str] = []
    for name, raw_value in zip(locus_names, values):
        if is_sex_chromosome_locus(name):
            if "," in raw_value:
                raise ValueError(f"Sex-chromosome locus {name} must have one allele")
            key.append(canonical_allele(raw_value))
            continue

        alleles = [canonical_allele(part) for part in raw_value.split(",")]
        if len(alleles) != 2:
            raise ValueError(f"Autosomal locus {name} must have two comma-separated alleles")
        alleles.sort(key=allele_sort_key)
        key.append(",".join(alleles))
    return tuple(key)


def parse_distribution(path: Path) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                raise ValueError(f"Invalid distribution line {line_number}: {raw_line.rstrip()}")

            name = parts[0]
            if name in models:
                raise ValueError(f"Duplicate locus in distribution: {name}")
            raw_probabilities: dict[str, float] = {}
            for item in parts[1:]:
                label, probability = item.rsplit(":", 1)
                allele = canonical_allele(label)
                if allele in raw_probabilities:
                    raise ValueError(f"Duplicate allele {label} at locus {name}")
                raw_probabilities[allele] = float(probability)

            raw_sum = sum(raw_probabilities.values())
            if not math.isfinite(raw_sum) or raw_sum <= 0:
                raise ValueError(f"Invalid probability sum for locus {name}: {raw_sum}")
            probabilities = {allele: value / raw_sum for allele, value in raw_probabilities.items()}
            models[name] = {
                "name": name,
                "sex_chromosome": is_sex_chromosome_locus(name),
                "raw_probabilities": raw_probabilities,
                "probabilities": probabilities,
                "raw_sum": raw_sum,
            }
    if not models:
        raise ValueError(f"No loci found in {path}")
    return models


def read_header(handle: Any) -> list[str]:
    line = handle.readline()
    if not line:
        raise ValueError("Input file is empty")
    header = line.rstrip("\r\n").split("\t")
    if not header or header[0] != "SampleID":
        raise ValueError("The first column must be SampleID")
    return header


def find_database_file(database_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates = sorted(database_dir.glob("*.txt"))
    if len(candidates) == 1:
        return candidates[0]
    preferred = [p for p in candidates if "20loci" in p.name.lower()]
    if len(preferred) == 1:
        return preferred[0]
    names = ", ".join(p.name for p in candidates) or "none"
    raise ValueError(f"Could not choose one database TXT from {database_dir}: {names}")


def load_queries(sample_dir: Path, locus_names: list[str]) -> tuple[list[dict[str, Any]], dict[tuple[str, ...], list[int]]]:
    sample_files = sorted(sample_dir.glob("*.txt"))
    if not sample_files:
        raise ValueError(f"No sample TXT files found in {sample_dir}")

    queries: list[dict[str, Any]] = []
    query_keys: dict[tuple[str, ...], list[int]] = defaultdict(list)
    expected_header = ["SampleID", *locus_names]
    for sample_file in sample_files:
        with sample_file.open("r", encoding="utf-8-sig") as handle:
            header = read_header(handle)
            if header != expected_header:
                raise ValueError(f"Header mismatch in {sample_file}")
            for row_number, raw_line in enumerate(handle, 2):
                row = raw_line.rstrip("\r\n").split("\t")
                if len(row) != len(expected_header):
                    raise ValueError(f"Column count mismatch in {sample_file}:{row_number}")
                sample_id = row[0].strip()
                if not sample_id:
                    raise ValueError(f"Empty SampleID in {sample_file}:{row_number}")
                key = canonical_panel_key(row[1:], locus_names)
                index = len(queries)
                queries.append(
                    {
                        "query_file": str(sample_file.resolve()),
                        "query_file_name": sample_file.name,
                        "query_row": row_number,
                        "query_id": sample_id,
                        "fields": row[1:],
                        "key": key,
                    }
                )
                query_keys[key].append(index)
    return queries, query_keys


def scan_database(
    database_file: Path,
    locus_names: list[str],
    query_keys: dict[tuple[str, ...], list[int]],
) -> tuple[int, list[list[str]], list[dict[str, int]], int]:
    expected_header = ["SampleID", *locus_names]
    matches: list[list[str]] = [[] for _ in range(sum(len(v) for v in query_keys.values()))]
    allele_counts: list[dict[str, int]] = [defaultdict(int) for _ in locus_names]
    seen_ids: set[str] = set()
    row_count = 0
    duplicate_ids = 0

    with database_file.open("r", encoding="utf-8-sig") as handle:
        header = read_header(handle)
        if header != expected_header:
            raise ValueError(f"Database header mismatch in {database_file}")
        for row_number, raw_line in enumerate(handle, 2):
            row = raw_line.rstrip("\r\n").split("\t")
            if len(row) != len(expected_header):
                raise ValueError(f"Column count mismatch in {database_file}:{row_number}")
            sample_id = row[0].strip()
            if not sample_id:
                raise ValueError(f"Empty SampleID in {database_file}:{row_number}")
            if sample_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(sample_id)
            row_count += 1

            key = canonical_panel_key(row[1:], locus_names)
            for query_index in query_keys.get(key, []):
                matches[query_index].append(sample_id)

            for i, (name, value) in enumerate(zip(locus_names, row[1:])):
                if is_sex_chromosome_locus(name):
                    allele_counts[i][canonical_allele(value)] += 1
                else:
                    for allele in value.split(","):
                        allele_counts[i][canonical_allele(allele)] += 1

    if duplicate_ids:
        raise ValueError(f"Database contains {duplicate_ids} duplicate SampleID values")
    return row_count, matches, allele_counts, len(seen_ids)


def genotype_probability(
    locus: dict[str, Any],
    canonical_value: str,
) -> tuple[float | None, list[str]]:
    probabilities = locus["probabilities"]
    if locus["sex_chromosome"]:
        allele = canonical_value
        probability = probabilities.get(allele)
        return probability, [] if probability is not None else [allele]

    left, right = canonical_value.split(",")
    missing = [allele for allele in (left, right) if allele not in probabilities]
    if missing:
        return None, sorted(set(missing))
    probability = probabilities[left] * probabilities[right]
    if left != right:
        probability *= 2.0
    return probability, []


def stable_risk(log10_rmp: float, database_size: int) -> dict[str, float | None]:
    log10_lambda = math.log10(database_size) + log10_rmp
    lambda_value = 10.0**log10_lambda if log10_lambda > -323 else 0.0
    if lambda_value == 0.0:
        p_any = 0.0
    else:
        rmp = 10.0**log10_rmp
        if rmp >= 1.0:
            p_any = 1.0
        else:
            p_any = -math.expm1(database_size * math.log1p(-rmp))
    log10_p_any = math.log10(p_any) if p_any > 0 else None
    return {
        "log10_expected_accidental_matches": log10_lambda,
        "expected_accidental_matches": lambda_value,
        "p_any_independent": p_any,
        "log10_p_any_independent": log10_p_any,
        "p_any_union_bound": min(1.0, lambda_value),
        "log10_p_any_union_bound": min(0.0, log10_lambda),
    }


def evaluate_query(
    query: dict[str, Any],
    matches: list[str],
    locus_names: list[str],
    frequency_models: dict[str, dict[str, Any]],
    database_size: int,
    alpha: float | None,
    frequency_source: str,
) -> dict[str, Any]:
    key = query["key"]
    locus_results: list[dict[str, Any]] = []
    log10_rmp: float | None = 0.0
    unknown_alleles: list[dict[str, Any]] = []

    for name, canonical_value in zip(locus_names, key):
        model = frequency_models[name]
        probability, missing = genotype_probability(model, canonical_value)
        item: dict[str, Any] = {
            "locus": name,
            "genotype": canonical_value,
            "model_probability": probability,
            "log10_model_probability": math.log10(probability) if probability and probability > 0 else None,
        }
        if missing:
            unknown_alleles.append({"locus": name, "alleles": missing})
            log10_rmp = None
        elif log10_rmp is not None:
            log10_rmp += math.log10(probability)
        locus_results.append(item)

    h = len(matches)
    if h == 0:
        match_status = "no_exact_match"
    elif h == 1:
        match_status = "single_exact_match"
    else:
        match_status = "multiple_exact_matches"

    stats: dict[str, Any] = {
        "computable": log10_rmp is not None,
        "unknown_alleles": unknown_alleles,
        "log10_rmp": log10_rmp,
        "rmp": 10.0**log10_rmp if log10_rmp is not None and log10_rmp > -323 else None,
        "log10_lr_for_complete_match": -log10_rmp if log10_rmp is not None else None,
        "log10_lr_for_returned_matches": -log10_rmp if h > 0 and log10_rmp is not None else None,
    }
    if log10_rmp is not None:
        stats["database_risk"] = stable_risk(log10_rmp, database_size)
        if alpha is not None:
            stats["risk_target"] = {
                "alpha": alpha,
                "criterion": "M * RMP <= alpha",
                "met": stats["database_risk"]["expected_accidental_matches"] <= alpha,
                "log10_alpha": math.log10(alpha),
            }
        else:
            stats["risk_target"] = None
    else:
        stats["database_risk"] = None
        stats["risk_target"] = None

    return {
        "query_file": query["query_file"],
        "query_file_name": query["query_file_name"],
        "query_row": query["query_row"],
        "query_id": query["query_id"],
        "panel": {
            "loci": locus_names,
            "locus_count": len(locus_names),
        },
        "database_search": {
            "database_size_M": database_size,
            "exact_match_count_H": h,
            "match_status": match_status,
            "matched_person_ids": matches,
        },
        "statistics": stats,
        "locus_probabilities": locus_results,
        "model": {
            "frequency_source": frequency_source,
            "probability_normalization": (
                "Each STRDistribution locus row is normalized by its raw sum because source values are rounded to six decimal places."
                if frequency_source == "STRDistribution.txt"
                else "Frequencies are derived from database allele copy counts with denominator 2N for autosomal loci and N for single-allele loci."
            ),
            "autosomal_model": "homozygote p^2; heterozygote 2pq",
            "sex_chromosome_model": "single allele frequency",
            "hardy_weinberg": True,
            "product_rule": True,
            "theta": 0,
            "single_source_no_error": True,
        },
    }


def build_frequency_validation(
    locus_names: list[str],
    models: dict[str, dict[str, Any]],
    counts: list[dict[str, int]],
    database_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, count_map in zip(locus_names, counts):
        model = models[name]
        denominator = database_size if model["sex_chromosome"] else 2 * database_size
        observed = {allele: count / denominator for allele, count in count_map.items()}
        target = model["probabilities"]
        all_alleles = sorted(set(observed) | set(target), key=allele_sort_key)
        max_error = max(abs(observed.get(allele, 0.0) - target.get(allele, 0.0)) for allele in all_alleles)
        output.append(
            {
                "locus": name,
                "chromosome_type": "sex_chromosome_single_allele" if model["sex_chromosome"] else "autosomal_two_alleles",
                "database_copy_count": sum(count_map.values()),
                "expected_copy_count": denominator,
                "source_probability_sum": model["raw_sum"],
                "max_abs_frequency_difference": max_error,
            }
        )
    return output


def build_database_frequency_models(
    locus_names: list[str],
    counts: list[dict[str, int]],
    database_size: int,
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for name, count_map in zip(locus_names, counts):
        denominator = database_size if is_sex_chromosome_locus(name) else 2 * database_size
        probabilities = {allele: count / denominator for allele, count in count_map.items()}
        models[name] = {
            "name": name,
            "sex_chromosome": is_sex_chromosome_locus(name),
            "raw_probabilities": probabilities,
            "probabilities": probabilities,
            "raw_sum": sum(probabilities.values()),
        }
    return models


def write_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results, 1):
        stem = Path(result["query_file_name"]).stem
        output_path = output_dir / f"{index:03d}_{stem}.json"
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "query_file_name",
            "query_id",
            "database_size_M",
            "exact_match_count_H",
            "match_status",
            "matched_person_ids",
            "log10_rmp",
            "log10_lr_for_complete_match",
            "log10_expected_accidental_matches",
            "expected_accidental_matches",
            "p_any_independent",
            "p_any_union_bound",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            stats = result["statistics"]
            risk = stats.get("database_risk") or {}
            search = result["database_search"]
            writer.writerow(
                {
                    "query_file_name": result["query_file_name"],
                    "query_id": result["query_id"],
                    "database_size_M": search["database_size_M"],
                    "exact_match_count_H": search["exact_match_count_H"],
                    "match_status": search["match_status"],
                    "matched_person_ids": ";".join(search["matched_person_ids"]),
                    "log10_rmp": stats.get("log10_rmp"),
                    "log10_lr_for_complete_match": stats.get("log10_lr_for_complete_match"),
                    "log10_expected_accidental_matches": risk.get("log10_expected_accidental_matches"),
                    "expected_accidental_matches": risk.get("expected_accidental_matches"),
                    "p_any_independent": risk.get("p_any_independent"),
                    "p_any_union_bound": risk.get("p_any_union_bound"),
                }
            )

    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    database_dir = args.database_dir.resolve()
    samples_dir = args.samples_dir.resolve()
    distribution_file = args.distribution.resolve()
    database_file = find_database_file(database_dir, args.database_file.resolve() if args.database_file else None)
    distribution_models = parse_distribution(distribution_file)

    with database_file.open("r", encoding="utf-8-sig") as handle:
        header = read_header(handle)
    locus_names = header[1:]
    missing_loci = [name for name in locus_names if name not in distribution_models]
    if missing_loci:
        raise ValueError(f"Loci missing from STRDistribution: {missing_loci}")

    queries, query_keys = load_queries(samples_dir, locus_names)
    database_size, matches, database_counts, unique_person_count = scan_database(database_file, locus_names, query_keys)
    if database_size != unique_person_count:
        raise ValueError("Database row count and unique person count differ")

    database_frequency_models = build_database_frequency_models(locus_names, database_counts, database_size)
    if args.frequency_source == "database":
        frequency_models = database_frequency_models
        frequency_source_label = "database-derived frequency table"
    else:
        frequency_models = distribution_models
        frequency_source_label = "STRDistribution.txt"

    results = [
        evaluate_query(
            query=query,
            matches=matches[index],
            locus_names=locus_names,
            frequency_models=frequency_models,
            database_size=database_size,
            alpha=args.alpha,
            frequency_source=frequency_source_label,
        )
        for index, query in enumerate(queries)
    ]
    validation = build_frequency_validation(locus_names, distribution_models, database_counts, database_size)

    validation_json = database_dir / "STR_database_1000000_20loci.validation.json"
    generator_metadata = None
    if validation_json.exists():
        generator_metadata = json.loads(validation_json.read_text(encoding="utf-8"))

    summary: dict[str, Any] = {
        "program": "str_exact_match.py",
        "database_file": str(database_file),
        "database_size_M": database_size,
        "sample_directory": str(samples_dir),
        "query_count": len(results),
        "distribution_file": str(distribution_file),
        "panel": {
            "loci": locus_names,
            "locus_count": len(locus_names),
            "autosomal_locus_count": sum(not is_sex_chromosome_locus(name) for name in locus_names),
            "sex_chromosome_locus_count": sum(is_sex_chromosome_locus(name) for name in locus_names),
        },
        "frequency_model": {
            "source": frequency_source_label,
            "normalized_source_rows": frequency_source_label == "STRDistribution.txt",
            "database_derived": frequency_source_label == "database-derived frequency table",
            "hardy_weinberg": True,
            "product_rule": True,
            "theta": 0,
            "single_source_no_error": True,
        },
        "database_frequency_validation": validation,
        "generator_metadata": generator_metadata,
        "results": [
            {
                "query_file_name": result["query_file_name"],
                "query_id": result["query_id"],
                "H": result["database_search"]["exact_match_count_H"],
                "status": result["database_search"]["match_status"],
                "log10_rmp": result["statistics"]["log10_rmp"],
            }
            for result in results
        ],
    }
    write_outputs(args.output_dir.resolve(), results, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact STR search, log10 RMP/LR, and database risk calculation")
    parser.add_argument("--database-dir", type=Path, default=DEFAULT_DATABASE_DIR)
    parser.add_argument("--database-file", type=Path, default=None)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES_DIR)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--frequency-source",
        choices=("database", "distribution"),
        default="database",
        help="Use the database-derived frequency table by default, or STRDistribution.txt directly.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Optional risk target for M * RMP <= alpha")
    args = parser.parse_args()

    if args.alpha is not None and not (0 < args.alpha <= 1):
        parser.error("--alpha must be between 0 and 1")

    summary = run(args)
    print(
        json.dumps(
            {
                "database_size_M": summary["database_size_M"],
                "query_count": summary["query_count"],
                "output_dir": str(args.output_dir.resolve()),
                "results": summary["results"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
