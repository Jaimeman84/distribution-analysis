# Technical Documentation - Carrier Distribution Analyzer

This document provides detailed technical explanations of the calculations, statistical methods, and data flow used in the Carrier Distribution Analyzer application.

## Table of Contents

1. [Statistical Concepts](#statistical-concepts)
2. [Calculation Flow](#calculation-flow)
3. [Table Calculations](#table-calculations)
4. [Processing Time Calculations](#processing-time-calculations)
5. [Scenario Analysis](#scenario-analysis)

---

## Statistical Concepts

### P50 (Median / 50th Percentile)

**P50** is the median value where 50% of simulated values fall below and 50% fall above. It's used throughout this application as the primary measure of central tendency.

**Why P50 instead of Mean?**

The median (P50) is more robust to outliers than the mean. In Monte Carlo simulations with Poisson distributions, occasional extreme values can skew the average, so P50 gives a more representative "typical" value.

**Example:**
```
Simulation results: [45, 47, 48, 49, 50, 50, 51, 52, 53, 200]
- Mean = 64.5 (pulled up by the outlier 200)
- P50 (Median) = 50 (unaffected by the outlier)
```

### Other Percentiles Used

| Percentile | Description | Use Case |
|------------|-------------|----------|
| P10 | 10% of values fall below | Lower bound estimate |
| P50 | 50% of values fall below (median) | Typical/expected value |
| P90 | 90% of values fall below | Upper bound / planning buffer |
| P95 | 95% of values fall below | Conservative / worst-case estimate |

### Poisson Distribution

Used to model the number of PDFs per case. The Poisson distribution is appropriate for count data where events occur independently at a constant average rate.

**Parameter:**
- λ (lambda) = Total PDFs / Total Cases from Carrier A benchmark

**Formula:**
```
Expected PDFs for Carrier X = Carrier X Cases × λ
```

### Binomial Distribution

Used to determine how many PDFs are scanned vs. machine-readable.

**Parameters:**
- n = Total number of PDFs (from Poisson simulation)
- p = Probability of a PDF being scanned (from Carrier A: Scanned PDFs / Total PDFs)

---

## Calculation Flow

### Step 1: Benchmark Metrics (from Carrier A)

```python
lambda_pdfs_per_case = carrierA_total_pdfs / carrierA_cases
p_scanned = carrierA_scanned_pdfs / carrierA_total_pdfs
p_machine = 1 - p_scanned
```

**Example:**
- Carrier A: 984 cases, 197 total PDFs, 30 scanned, 167 machine-readable
- λ = 197 / 984 = 0.20 PDFs per case
- p_scanned = 30 / 197 = 15.2%
- p_machine = 167 / 197 = 84.8%

### Step 2: Monte Carlo Simulation (per carrier)

For each carrier, the simulation runs N times (default: 1,000 for multi-carrier):

```python
# 1. Sample total PDFs using Poisson distribution
expected_pdfs = carrier_cases × lambda_pdfs_per_case
total_pdfs = np.random.poisson(lam=expected_pdfs, size=simulations)

# 2. Sample scanned PDFs using Binomial distribution
scanned_pdfs = np.random.binomial(n=total_pdfs, p=p_scanned)

# 3. Calculate machine PDFs
machine_pdfs = total_pdfs - scanned_pdfs
```

### Step 3: Calculate Percentiles

After simulation, percentiles are calculated:

```python
total_pdfs_p50 = np.percentile(sim_results["total_pdfs"], 50)
machine_pdfs_p50 = np.percentile(sim_results["machine_pdfs"], 50)
scanned_pdfs_p50 = np.percentile(sim_results["scanned_pdfs"], 50)
```

### Step 4: Calculate Processing Times

Processing times are derived from the median PDF counts:

```python
machine_time_seconds_p50 = machine_pdfs_p50 × avg_time_machine_seconds
scanned_time_seconds_p50 = scanned_pdfs_p50 × avg_time_scanned_seconds
```

---

## Table Calculations

### PDF Statistics by Case Bucket

This table shows Min/Max/Avg statistics for carriers grouped by case ranges.

| Column | Calculation | Description |
|--------|-------------|-------------|
| Case Range | Bucket boundaries | e.g., "1-50", "51-100" |
| Carriers | `len(bucket_carriers)` | Count of carriers in bucket |
| Min PDFs | `total_pdfs_p50.min()` | Lowest median PDFs among carriers in bucket |
| Max PDFs | `total_pdfs_p50.max()` | Highest median PDFs among carriers in bucket |
| Avg PDFs | `total_pdfs_p50.mean()` | Average of median PDFs across carriers in bucket |
| Min Processing Time | `total_time_per_carrier.min()` | Shortest processing time among carriers |
| Max Processing Time | `total_time_per_carrier.max()` | Longest processing time among carriers |
| Avg Processing Time | `total_time_per_carrier.mean()` | Average processing time across carriers |
| Total Processing Time | `total_time_per_carrier.sum()` | **Sum of all carrier processing times in bucket** |

### Important Note: Min/Max Independence

The Min and Max values for different columns come from **different carriers** and are calculated independently:

```
Bucket "1-50 cases" with 3 carriers:
- Carrier A: 10 PDFs, 5 machine, 5 scanned
- Carrier B: 15 PDFs, 12 machine, 3 scanned
- Carrier C: 8 PDFs, 6 machine, 2 scanned

Min PDFs = 8 (from Carrier C)
Max PDFs = 15 (from Carrier B)
Min Machine PDFs = 5 (from Carrier A)
Max Machine PDFs = 12 (from Carrier B)
Min Scanned PDFs = 2 (from Carrier C)
Max Scanned PDFs = 5 (from Carrier A)
```

This means: **Machine PDF Range + Scanned PDF Range ≠ PDF Range**

This is expected behavior because the min/max values for each column are independent.

---

## Processing Time Calculations

### Per-Carrier Processing Time

Each carrier's total processing time is:

```python
total_time_per_carrier = machine_time_p50 + scanned_time_p50
```

Where:
- `machine_time_p50 = machine_pdfs_p50 × avg_time_machine_seconds`
- `scanned_time_p50 = scanned_pdfs_p50 × avg_time_scanned_seconds`

### Bucket-Level Statistics

| Metric | Formula | Description |
|--------|---------|-------------|
| Min Processing Time | `min(all carrier times in bucket)` | Fastest carrier |
| Max Processing Time | `max(all carrier times in bucket)` | Slowest carrier |
| Avg Processing Time | `mean(all carrier times in bucket)` | Average per carrier |
| Total Processing Time | `sum(all carrier times in bucket)` | **Aggregate workload for bucket** |

### Example Calculation

**Benchmark:**
- Avg machine PDF processing: 1.45 seconds
- Avg scanned PDF processing: 20.0 seconds

**Carrier X (50 cases):**
- Median machine PDFs: 42
- Median scanned PDFs: 8
- Machine time: 42 × 1.45 = 60.9 seconds
- Scanned time: 8 × 20.0 = 160.0 seconds
- **Total time: 220.9 seconds = 3.68 minutes**

---

## Scenario Analysis

### Tier System

Scenario analysis models carrier variability by assigning carriers to volume tiers:

| Tier | Multiplier | Description |
|------|------------|-------------|
| Low | 0.5x | 50% fewer PDFs per case than benchmark |
| Same | 1.0x | Matches Carrier A benchmark |
| High | 1.5x | 50% more PDFs per case than benchmark |

### Tier Assignment

Carriers are randomly assigned based on configured percentages:

```python
# Example: 33% Low, 33% Same, 34% High
tier_assignments = assign_carriers_to_tiers(
    num_carriers=500,
    tier_low_pct=33,
    tier_same_pct=33,
    tier_high_pct=34,
    random_seed=42  # Fixed for reproducibility
)
```

### Adjusted Lambda

For scenario analysis, the λ (PDFs per case) is adjusted by the tier multiplier:

```python
adjusted_lambda = lambda_pdfs_per_case × tier_multiplier
```

**Example:**
- Base λ = 0.20 PDFs/case
- Low tier carrier: λ = 0.20 × 0.5 = 0.10 PDFs/case
- Same tier carrier: λ = 0.20 × 1.0 = 0.20 PDFs/case
- High tier carrier: λ = 0.20 × 1.5 = 0.30 PDFs/case

### Scenario vs Uniform Comparison

The application compares:

1. **Uniform Distribution**: All carriers use benchmark λ (no tier adjustments)
2. **Scenario Distribution**: Carriers use tier-adjusted λ values

Delta metrics show the percentage difference:
```python
delta_pdfs = ((scenario_total_pdfs / uniform_total_pdfs) - 1) × 100%
delta_time = ((scenario_total_time / uniform_total_time) - 1) × 100%
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CARRIER A BENCHMARK                          │
│  Cases: 984  |  Total PDFs: 197  |  Machine: 167  |  Scanned: 30│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK METRICS                            │
│  λ = 0.20 PDFs/case  |  p_scanned = 15.2%  |  p_machine = 84.8% │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   UNIFORM DISTRIBUTION  │     │  SCENARIO DISTRIBUTION  │
│   (all carriers same)   │     │  (tier multipliers)     │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  MONTE CARLO SIMULATION │     │  MONTE CARLO SIMULATION │
│  - Poisson (total PDFs) │     │  - Adjusted λ per tier  │
│  - Binomial (scanned)   │     │  - Poisson & Binomial   │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   PERCENTILE STATS      │     │   PERCENTILE STATS      │
│   P10, P50, P90         │     │   P10, P50, P90         │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   PROCESSING TIMES      │     │   PROCESSING TIMES      │
│   Machine + Scanned     │     │   Machine + Scanned     │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT TABLES & CHARTS                       │
│  - PDF Statistics (Min/Max/Avg)                                 │
│  - Processing Time Statistics                                   │
│  - Carrier Distribution by Bucket                               │
│  - Scenario vs Uniform Comparison                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Bucket Definitions

Case ranges used for grouping carriers:

| Bucket | Min Cases | Max Cases |
|--------|-----------|-----------|
| 1 | 1 | 50 |
| 2 | 51 | 100 |
| 3 | 101 | 200 |
| 4 | 201 | 500 |
| 5 | 501 | 1,000 |
| 6 | 1,001 | 2,000 |
| 7 | 2,001 | 5,000 |
| 8 | 5,001 | 10,000 |
| 9 | 10,001 | 25,000 |
| 10 | 25,001 | 50,000 |
| 11 | 50,001 | 100,000 |
| 12 | 100,001 | ∞ |

---

## Default Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Carrier A Cases | 984 | Benchmark case count |
| Carrier A Total PDFs | 197 | Benchmark PDF count |
| Carrier A Machine PDFs | 167 | Machine-readable PDFs |
| Carrier A Scanned PDFs | 30 | Scanned PDFs (OCR required) |
| Avg Machine Processing Time | 1.45 seconds | Per machine-readable PDF |
| Avg Scanned Processing Time | 20.0 seconds | Per scanned PDF |
| Single Carrier Simulations | 5,000 | Monte Carlo iterations |
| Multi-Carrier Simulations | 1,000 | Monte Carlo iterations per carrier |
| Random Seed | 42 | For reproducibility |

---

## Time Format Conversion

The `format_time_hours()` function converts seconds to human-readable format:

```python
def format_time_hours(seconds):
    hours = seconds / 3600
    if hours < 1:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    return f"{hours:.2f} hrs"
```

**Examples:**
- 120 seconds → "2.0 min"
- 3600 seconds → "1.00 hrs"
- 7200 seconds → "2.00 hrs"

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | January 2026 | Initial release |

---

*Last Updated: January 2026*
