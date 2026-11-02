from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


DEFAULT_DISTRIBUTION = Path("D:/\u684c\u9762\u5f52\u7f6e/STR/STRDistribution.txt")
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("STR_database_1000000_20loci.txt")
DEFAULT_REPORT = Path(__file__).resolve().with_name("STR_database_1000000_20loci.validation.json")
DEFAULT_SEED = 20261015
DEFAULT_LOCUS_COUNT = 20


def is_sex_chromosome_locus(name: str) -> bool:
    """Classify common Y/X marker naming conventions as single-copy loci."""
    upper = name.upper()
    return (
        upper.startswith(("DYS", "YGATA"))
        or upper in {"X", "Y", "AMEL", "AMELX", "AMELY", "SRY"}
        or upper.endswith("Y")
    )


def parse_distribution(path: Path) -> list[dict]:
    loci: list[dict] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 2:
            raise ValueError(f"Invalid distribution line {line_number}: {raw_line!r}")

        name = parts[0]
        alleles: list[str] = []
        probabilities: list[float] = []
        for item in parts[1:]:
            label, probability = item.rsplit(":", 1)
            alleles.append(label)
            probabilities.append(float(probability))

        raw_sum = float(sum(probabilities))
        if not math.isfinite(raw_sum) or raw_sum <= 0:
            raise ValueError(f"Invalid probability sum for {name}: {raw_sum}")
        normalized = np.asarray(probabilities, dtype=np.float64) / raw_sum
        if not np.isclose(normalized.sum(), 1.0, atol=1e-12):
            raise ValueError(f"Could not normalize probabilities for {name}")

        numeric_alleles = np.asarray([float(a) for a in alleles], dtype=np.float64)
        loci.append(
            {
                "name": name,
                "alleles": np.asarray(alleles, dtype=str),
                "numeric_alleles": numeric_alleles,
                "probabilities": normalized,
                "raw_probability_sum": raw_sum,
                "sex_chromosome": is_sex_chromosome_locus(name),
            }
        )

    if not loci:
        raise ValueError(f"No loci found in {path}")
    return loci


def generate_database(
    distribution_path: Path,
    output_path: Path,
    report_path: Path,
    count: int,
    seed: int,
    chunk_size: int,
    locus_count: int,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if locus_count <= 0:
        raise ValueError("locus_count must be positive")

    loci = parse_distribution(distribution_path)
    if locus_count > len(loci):
        raise ValueError(f"locus_count {locus_count} exceeds available loci {len(loci)}")
    loci = loci[:locus_count]
    rng = np.random.default_rng(seed)
    total_counts = [np.zeros(len(locus["alleles"]), dtype=np.int64) for locus in loci]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(["SampleID"] + [locus["name"] for locus in loci]) + "\n")
        for start in range(0, count, chunk_size):
            rows = min(chunk_size, count - start)
            identifiers = np.asarray([f"P{i:06d}" for i in range(start + 1, start + rows + 1)], dtype=str)
            fields: list[np.ndarray] = []

            for locus_index, locus in enumerate(loci):
                allele_count = len(locus["alleles"])
                if locus["sex_chromosome"]:
                    draws = rng.choice(allele_count, size=rows, p=locus["probabilities"])
                    total_counts[locus_index] += np.bincount(draws, minlength=allele_count)
                    fields.append(locus["alleles"][draws])
                    continue

                draws = rng.choice(allele_count, size=(rows, 2), p=locus["probabilities"])
                total_counts[locus_index] += np.bincount(draws.ravel(), minlength=allele_count)
                left_values = locus["numeric_alleles"][draws[:, 0]]
                right_values = locus["numeric_alleles"][draws[:, 1]]
                left_first = left_values <= right_values
                left = np.where(left_first, locus["alleles"][draws[:, 0]], locus["alleles"][draws[:, 1]])
                right = np.where(left_first, locus["alleles"][draws[:, 1]], locus["alleles"][draws[:, 0]])
                fields.append(np.char.add(np.char.add(left, ","), right))

            handle.write("\n".join("\t".join(row) for row in zip(identifiers, *fields)))
            handle.write("\n")

    observed: list[dict] = []
    for locus, counts in zip(loci, total_counts):
        denominator = count if locus["sex_chromosome"] else 2 * count
        observed_frequencies = counts / denominator
        target = locus["probabilities"]
        observed.append(
            {
                "locus": locus["name"],
                "chromosome_type": "sex_chromosome_single_allele" if locus["sex_chromosome"] else "autosomal_two_alleles",
                "sampled_copies": int(counts.sum()),
                "expected_copies": int(denominator),
                "max_absolute_frequency_error": float(np.max(np.abs(observed_frequencies - target))),
                "raw_probability_sum": locus["raw_probability_sum"],
                "normalized_probability_sum": float(locus["probabilities"].sum()),
            }
        )

    report = {
        "output": str(output_path.resolve()),
        "distribution_source": str(distribution_path.resolve()),
        "records": count,
        "seed": seed,
        "chunk_size": chunk_size,
        "locus_selection": "first loci in STRDistribution.txt order",
        "locus_count": len(loci),
        "autosomal_locus_count": sum(not locus["sex_chromosome"] for locus in loci),
        "sex_chromosome_locus_count": sum(locus["sex_chromosome"] for locus in loci),
        "format": {
            "encoding": "UTF-8",
            "delimiter": "TAB",
            "header": "SampleID followed by loci in STRDistribution.txt order",
            "autosomal_value": "allele1,allele2; numeric ascending order; two independent draws",
            "sex_chromosome_value": "single allele; one draw",
        },
        "probability_handling": "Each source locus probability row was normalized by its raw sum because the source values are rounded to six decimal places.",
        "loci": observed,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic STR database from allele frequency rows.")
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--locus-count", type=int, default=DEFAULT_LOCUS_COUNT)
    args = parser.parse_args()

    report = generate_database(
        distribution_path=args.distribution,
        output_path=args.output,
        report_path=args.report,
        count=args.count,
        seed=args.seed,
        chunk_size=args.chunk_size,
        locus_count=args.locus_count,
    )
    print(json.dumps({k: report[k] for k in ("output", "records", "seed", "locus_count", "autosomal_locus_count", "sex_chromosome_locus_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
