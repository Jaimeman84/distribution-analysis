# Technical Documentation - Carrier Distribution Analyzer

This document provides detailed technical explanations of the calculations, statistical methods, and data flow used in the Carrier Distribution Analyzer application.

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Formulas](#core-formulas)
3. [Statistical Concepts](#statistical-concepts)
4. [Step-by-Step Calculation Examples](#step-by-step-calculation-examples)
5. [Excel Formulas](#excel-formulas)
6. [Table Calculations](#table-calculations)
7. [Scenario Analysis](#scenario-analysis)
8. [Data Flow Diagram](#data-flow-diagram)
9. [Reference Tables](#reference-tables)

---

## Executive Summary

This application forecasts PDF processing workload for multiple carriers using **benchmark data from one known carrier (Carrier A)**. The key insight is:

> If we know the PDF-to-case ratio from Carrier A, we can estimate PDFs for any carrier based solely on their case count.

**The Core Question**: *"If Carrier A has X PDFs per case, how many PDFs will Carrier B have with Y cases?"*

---

## Core Formulas

### The Three Key Metrics (from Carrier A Benchmark)

| Metric | Formula | What It Means |
|--------|---------|---------------|
| **λ (Lambda)** | `Total PDFs ÷ Total Cases` | Average PDFs per case |
| **Scanned Ratio** | `Scanned PDFs ÷ Total PDFs` | Percentage of PDFs requiring OCR |
| **Machine Ratio** | `Machine PDFs ÷ Total PDFs` | Percentage of machine-readable PDFs |

**Default Benchmark Values (Carrier A):**
```
Cases:        984
Total PDFs:   197
Machine PDFs: 167
Scanned PDFs: 30

λ = 197 ÷ 984 = 0.2003 PDFs per case
Scanned Ratio = 30 ÷ 197 = 15.23%
Machine Ratio = 167 ÷ 197 = 84.77%
```

### The Processing Time Formula

**For any carrier, the total processing time is:**

```
Total Processing Time = (Machine PDFs × Machine Time per PDF) + (Scanned PDFs × Scanned Time per PDF)
```

Or more specifically:

```
Total Time (seconds) = (Total PDFs × Machine Ratio × 1.45) + (Total PDFs × Scanned Ratio × 20.0)
```

**Where:**
- `1.45 seconds` = Average time to process one machine-readable PDF
- `20.0 seconds` = Average time to process one scanned PDF (OCR required)

### Simplified Single Formula

For quick estimates, you can combine everything:

```
Total Time (seconds) = Cases × λ × [(Machine Ratio × 1.45) + (Scanned Ratio × 20.0)]
```

With default values:
```
Total Time (seconds) = Cases × 0.2003 × [(0.8477 × 1.45) + (0.1523 × 20.0)]
Total Time (seconds) = Cases × 0.2003 × [1.229 + 3.046]
Total Time (seconds) = Cases × 0.2003 × 4.275
Total Time (seconds) = Cases × 0.856
```

**Quick Rule of Thumb**: Each case takes approximately **0.86 seconds** to process (with default benchmark values).

---

## Statistical Concepts

### Why We Use Monte Carlo Simulation

Instead of just multiplying `Cases × λ`, we run thousands of simulations because:

1. **Real-world variability**: Not every case has exactly the same number of PDFs
2. **Statistical distributions**: PDF counts follow a Poisson distribution (appropriate for count data)
3. **Confidence ranges**: Simulations give us P10-P90 ranges, not just single estimates

### P50 (Median / 50th Percentile)

**P50** is the median value where 50% of simulated values fall below and 50% fall above.

**Why P50 instead of Mean?**

The median (P50) is more robust to outliers than the mean:

```
Simulation results: [45, 47, 48, 49, 50, 50, 51, 52, 53, 200]
- Mean = 64.5 (pulled up by the outlier 200)
- P50 (Median) = 50 (unaffected by the outlier)
```

### Percentile Reference

| Percentile | Description | Use Case |
|------------|-------------|----------|
| P10 | 10% of values fall below | Optimistic / best-case estimate |
| P50 | 50% of values fall below (median) | **Typical / expected value** |
| P90 | 90% of values fall below | Planning buffer / conservative estimate |
| P95 | 95% of values fall below | Worst-case scenario planning |

### Poisson Distribution

Used to model total PDFs per carrier. Appropriate for count data where events occur independently.

```python
expected_pdfs = cases × λ
actual_pdfs = Poisson(λ = expected_pdfs)  # Simulated value
```

### Binomial Distribution

Used to split total PDFs into scanned vs. machine-readable:

```python
scanned_pdfs = Binomial(n = total_pdfs, p = scanned_ratio)
machine_pdfs = total_pdfs - scanned_pdfs
```

---

## Step-by-Step Calculation Examples

### Example Data from carriers_1.21.xlsx

The file contains **2,132 carriers** with a total of **1,141,039 cases**.

| Carrier ID | Cases | Description |
|------------|-------|-------------|
| 722 | 50 | Small carrier |
| 330 | 200 | Medium carrier |
| 3140 | 999 | Medium-large carrier |
| 6027 | 9,841 | Large carrier |
| 2169 | 52,631 | Very large carrier |

---

### Example 1: Small Carrier (Carrier 722 - 50 cases)

**Step 1: Estimate Total PDFs**
```
Total PDFs = Cases × λ
Total PDFs = 50 × 0.2003
Total PDFs = 10.02 ≈ 10 PDFs
```

**Step 2: Split by Type**
```
Machine PDFs = Total PDFs × Machine Ratio
Machine PDFs = 10 × 0.8477 = 8.48 ≈ 8 PDFs

Scanned PDFs = Total PDFs × Scanned Ratio
Scanned PDFs = 10 × 0.1523 = 1.52 ≈ 2 PDFs
```

**Step 3: Calculate Processing Time**
```
Machine Time = Machine PDFs × 1.45 seconds
Machine Time = 8 × 1.45 = 11.6 seconds

Scanned Time = Scanned PDFs × 20.0 seconds
Scanned Time = 2 × 20.0 = 40.0 seconds

Total Time = 11.6 + 40.0 = 51.6 seconds ≈ 0.9 minutes
```

---

### Example 2: Medium Carrier (Carrier 330 - 200 cases)

**Step 1: Estimate Total PDFs**
```
Total PDFs = 200 × 0.2003 = 40.06 ≈ 40 PDFs
```

**Step 2: Split by Type**
```
Machine PDFs = 40 × 0.8477 = 33.9 ≈ 34 PDFs
Scanned PDFs = 40 × 0.1523 = 6.1 ≈ 6 PDFs
```

**Step 3: Calculate Processing Time**
```
Machine Time = 34 × 1.45 = 49.3 seconds
Scanned Time = 6 × 20.0 = 120.0 seconds

Total Time = 49.3 + 120.0 = 169.3 seconds ≈ 2.8 minutes
```

---

### Example 3: Large Carrier (Carrier 6027 - 9,841 cases)

**Step 1: Estimate Total PDFs**
```
Total PDFs = 9,841 × 0.2003 = 1,971.2 ≈ 1,971 PDFs
```

**Step 2: Split by Type**
```
Machine PDFs = 1,971 × 0.8477 = 1,671.2 ≈ 1,671 PDFs
Scanned PDFs = 1,971 × 0.1523 = 300.2 ≈ 300 PDFs
```

**Step 3: Calculate Processing Time**
```
Machine Time = 1,671 × 1.45 = 2,422.95 seconds
Scanned Time = 300 × 20.0 = 6,000.0 seconds

Total Time = 2,422.95 + 6,000.0 = 8,422.95 seconds
Total Time = 8,422.95 ÷ 3600 = 2.34 hours
```

---

### Example 4: Very Large Carrier (Carrier 2169 - 52,631 cases)

**Step 1: Estimate Total PDFs**
```
Total PDFs = 52,631 × 0.2003 = 10,541.9 ≈ 10,542 PDFs
```

**Step 2: Split by Type**
```
Machine PDFs = 10,542 × 0.8477 = 8,936.5 ≈ 8,937 PDFs
Scanned PDFs = 10,542 × 0.1523 = 1,605.5 ≈ 1,606 PDFs
```

**Step 3: Calculate Processing Time**
```
Machine Time = 8,937 × 1.45 = 12,958.65 seconds
Scanned Time = 1,606 × 20.0 = 32,120.0 seconds

Total Time = 12,958.65 + 32,120.0 = 45,078.65 seconds
Total Time = 45,078.65 ÷ 3600 = 12.52 hours
```

---

### Example 5: Full Dataset Calculation (2,132 carriers, 1,141,039 cases)

**Step 1: Estimate Total PDFs (All Carriers)**
```
Total PDFs = 1,141,039 × 0.2003 = 228,550.1 ≈ 228,550 PDFs
```

**Step 2: Split by Type**
```
Machine PDFs = 228,550 × 0.8477 = 193,760.4 ≈ 193,760 PDFs
Scanned PDFs = 228,550 × 0.1523 = 34,808.3 ≈ 34,808 PDFs
```

**Step 3: Calculate Processing Time**
```
Machine Time = 193,760 × 1.45 = 280,952 seconds
Scanned Time = 34,808 × 20.0 = 696,160 seconds

Total Time = 280,952 + 696,160 = 977,112 seconds
Total Time = 977,112 ÷ 3600 = 271.4 hours
Total Time = 271.4 ÷ 24 = 11.3 days (continuous processing)
```

---

## Excel Formulas

Use these formulas in Excel to calculate processing times for any carrier.

### Setup: Define Constants (Reference Cells)

| Cell | Value | Description |
|------|-------|-------------|
| B1 | 0.2003 | λ (Lambda) - PDFs per case |
| B2 | 0.8477 | Machine Ratio |
| B3 | 0.1523 | Scanned Ratio |
| B4 | 1.45 | Machine Time (seconds per PDF) |
| B5 | 20.0 | Scanned Time (seconds per PDF) |

### Formulas for Carrier Data (Cases in Column A, starting row 8)

| Column | Header | Formula (Row 8) | Description |
|--------|--------|-----------------|-------------|
| A | Cases | *(input data)* | Number of cases |
| B | Est. Total PDFs | `=A8*$B$1` | Cases × λ |
| C | Est. Machine PDFs | `=B8*$B$2` | Total PDFs × Machine Ratio |
| D | Est. Scanned PDFs | `=B8*$B$3` | Total PDFs × Scanned Ratio |
| E | Machine Time (sec) | `=C8*$B$4` | Machine PDFs × 1.45 |
| F | Scanned Time (sec) | `=D8*$B$5` | Scanned PDFs × 20.0 |
| G | Total Time (sec) | `=E8+F8` | Machine Time + Scanned Time |
| H | Total Time (min) | `=G8/60` | Seconds ÷ 60 |
| I | Total Time (hrs) | `=G8/3600` | Seconds ÷ 3600 |

### Single-Cell Formula (All-in-One)

To calculate total processing time in hours from just the case count:

```excel
=A8*0.2003*((0.8477*1.45)+(0.1523*20))/3600
```

Or with cell references:
```excel
=A8*$B$1*(($B$2*$B$4)+($B$3*$B$5))/3600
```

### Excel Example Table

| Cases | Total PDFs | Machine PDFs | Scanned PDFs | Machine Time | Scanned Time | Total Time | Hours |
|-------|------------|--------------|--------------|--------------|--------------|------------|-------|
| 50 | 10 | 8 | 2 | 12 sec | 30 sec | 43 sec | 0.01 hrs |
| 200 | 40 | 34 | 6 | 49 sec | 120 sec | 169 sec | 0.05 hrs |
| 1,000 | 200 | 170 | 30 | 246 sec | 609 sec | 856 sec | 0.24 hrs |
| 10,000 | 2,003 | 1,698 | 305 | 2,462 sec | 6,095 sec | 8,557 sec | 2.38 hrs |
| 52,631 | 10,542 | 8,937 | 1,606 | 12,959 sec | 32,110 sec | 45,068 sec | 12.52 hrs |

---

## Table Calculations

### PDF Statistics by Case Bucket

This table groups carriers by case ranges and shows statistics.

| Column | Calculation | Description |
|--------|-------------|-------------|
| Case Range | Bucket boundaries | e.g., "1-50", "51-100" |
| Carriers | `count(carriers in bucket)` | Number of carriers in range |
| Min PDFs | `min(all P50 PDFs in bucket)` | Lowest median PDFs among carriers |
| Max PDFs | `max(all P50 PDFs in bucket)` | Highest median PDFs among carriers |
| Avg PDFs | `mean(all P50 PDFs in bucket)` | Average of median PDFs |
| Min Processing Time | `min(all carrier times)` | Fastest carrier in bucket |
| Max Processing Time | `max(all carrier times)` | Slowest carrier in bucket |
| Avg Processing Time | `mean(all carrier times)` | Average time per carrier |
| Total Processing Time | `sum(all carrier times)` | **Combined workload for bucket** |

### Important: Min/Max Independence

The Min and Max values for different columns come from **different carriers**:

```
Example Bucket "1-50 cases" with 3 carriers:
- Carrier A: 10 PDFs (8 machine + 2 scanned) → 52 sec
- Carrier B: 15 PDFs (12 machine + 3 scanned) → 77 sec
- Carrier C: 8 PDFs (6 machine + 2 scanned) → 49 sec

Results:
- Min PDFs = 8 (Carrier C)
- Max PDFs = 15 (Carrier B)
- Min Time = 49 sec (Carrier C)
- Max Time = 77 sec (Carrier B)
- Total Time = 52 + 77 + 49 = 178 sec (sum of all)
```

**Note**: Min Machine + Min Scanned ≠ Min Total because they may come from different carriers.

### Single-Carrier Buckets

When a bucket has only **1 carrier**, Min = Max = Avg because there's only one data point:

```
Bucket "50,001-100,000" with 1 carrier (52,631 cases):
- Min PDFs = 10,542
- Max PDFs = 10,542
- Avg PDFs = 10,542
- Processing Time = 12.52 hrs

This is expected behavior, not an error.
```

---

## Scenario Analysis

### Purpose

Scenario analysis models **carrier variability** - the fact that different carriers may have more or fewer PDFs per case than the benchmark.

### Tier System

| Tier | Multiplier | Adjusted λ | Description |
|------|------------|------------|-------------|
| Low | 0.5x | 0.1001 | 50% fewer PDFs per case |
| Same | 1.0x | 0.2003 | Matches Carrier A benchmark |
| High | 1.5x | 0.3004 | 50% more PDFs per case |

### Tier Assignment Example

With 2,132 carriers and 33%/33%/34% split:
```
Low Tier (33%):  704 carriers using λ = 0.1001
Same Tier (33%): 704 carriers using λ = 0.2003
High Tier (34%): 724 carriers using λ = 0.3004
```

### Impact on Processing Time

**Carrier with 1,000 cases:**

| Tier | λ | Total PDFs | Processing Time |
|------|---|------------|-----------------|
| Low (0.5x) | 0.1001 | 100 | 7.1 min |
| Same (1.0x) | 0.2003 | 200 | 14.3 min |
| High (1.5x) | 0.3004 | 300 | 21.4 min |

### Uniform vs Scenario Comparison

| Metric | Uniform (all Same) | Scenario (Mixed) | Delta |
|--------|-------------------|------------------|-------|
| Total PDFs | 228,550 | 228,550 | 0% |
| Processing Time | 271.4 hrs | 271.4 hrs | 0% |

*Note: With equal tier splits (33/33/34), the total averages out. Unequal splits will show deltas.*

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
│  λ = 0.2003 PDFs/case  |  Scanned = 15.23%  |  Machine = 84.77% │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   UNIFORM DISTRIBUTION  │     │  SCENARIO DISTRIBUTION  │
│   (all carriers same λ) │     │  (tier-adjusted λ)      │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│  MONTE CARLO SIMULATION │     │  MONTE CARLO SIMULATION │
│  1,000 iterations each  │     │  1,000 iterations each  │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   FOR EACH CARRIER:     │     │   FOR EACH CARRIER:     │
│   Total PDFs (Poisson)  │     │   Total PDFs (Poisson)  │
│   Scanned/Machine split │     │   with tier multiplier  │
│   Processing time calc  │     │   Processing time calc  │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT TABLES & CHARTS                       │
│  - PDF Statistics (Min/Max/Avg per bucket)                      │
│  - Processing Time Statistics                                   │
│  - Carrier Distribution charts                                  │
│  - Scenario vs Uniform comparison                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Reference Tables

### Bucket Definitions

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

### Default Values

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Carrier A Cases | 984 | Benchmark case count |
| Carrier A Total PDFs | 197 | Benchmark PDF count |
| Carrier A Machine PDFs | 167 | Machine-readable PDFs |
| Carrier A Scanned PDFs | 30 | Scanned PDFs (OCR required) |
| Avg Machine Processing Time | 1.45 seconds | Per machine-readable PDF |
| Avg Scanned Processing Time | 20.0 seconds | Per scanned PDF |
| Monte Carlo Simulations | 1,000 | Per carrier (multi-carrier mode) |

### Quick Reference: Time Conversion

| Seconds | Minutes | Hours | Days |
|---------|---------|-------|------|
| 60 | 1 | 0.017 | 0.0007 |
| 3,600 | 60 | 1 | 0.042 |
| 86,400 | 1,440 | 24 | 1 |

### Quick Reference: Processing Time by Case Count

| Cases | Est. PDFs | Est. Time |
|-------|-----------|-----------|
| 10 | 2 | 9 sec |
| 50 | 10 | 43 sec |
| 100 | 20 | 1.4 min |
| 500 | 100 | 7.1 min |
| 1,000 | 200 | 14.3 min |
| 5,000 | 1,001 | 1.2 hrs |
| 10,000 | 2,003 | 2.4 hrs |
| 50,000 | 10,015 | 11.9 hrs |
| 100,000 | 20,030 | 23.8 hrs |

---

## Human vs Automated Processing Time

### Human Processing Time Formula

The human processing time calculation models how long it would take a human to manually review cases and PDFs.

**Formula:**
```
Human Processing Time = (Cases × Human Time per Case) + (Total PDFs × Human Time per PDF)
```

**Default Values:**
| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Human Time per Case | 5 seconds | Time for a human to review one case |
| Human Time per PDF | 10 seconds | Time for a human to review one PDF (any type) |

### Human Processing Time Examples

Using data from carriers_1.21.xlsx with default human review times:

**Example 1: Carrier 722 (50 cases, ~10 PDFs)**
```
Case review:  50 × 5 sec = 250 sec
PDF review:   10 × 10 sec = 100 sec
Human Total:  350 sec = 5.8 min

Automated:    ~43 sec
Speedup:      350 ÷ 43 = 8.1x faster
```

**Example 2: Carrier 6027 (9,841 cases, ~1,971 PDFs)**
```
Case review:  9,841 × 5 sec = 49,205 sec
PDF review:   1,971 × 10 sec = 19,710 sec
Human Total:  68,915 sec = 19.1 hrs

Automated:    ~2.34 hrs
Speedup:      19.1 ÷ 2.34 = 8.2x faster
```

**Example 3: Full Dataset (1,141,039 cases, ~228,550 PDFs)**
```
Case review:  1,141,039 × 5 sec = 5,705,195 sec
PDF review:   228,550 × 10 sec = 2,285,500 sec
Human Total:  7,990,695 sec = 2,219.6 hrs = 92.5 days

Automated:    ~271.4 hrs = 11.3 days
Speedup:      2,219.6 ÷ 271.4 = 8.2x faster
Time Saved:   1,948.2 hrs = 81.2 days
```

### Comparison Table

| Cases | Est. PDFs | Automated Time | Human Time | Speedup | Time Saved |
|-------|-----------|----------------|------------|---------|------------|
| 50 | 10 | 43 sec | 5.8 min | 8.1x | 5.1 min |
| 200 | 40 | 2.8 min | 23.3 min | 8.3x | 20.5 min |
| 1,000 | 200 | 14.3 min | 1.9 hrs | 8.1x | 1.7 hrs |
| 10,000 | 2,003 | 2.4 hrs | 19.4 hrs | 8.1x | 17.0 hrs |
| 52,631 | 10,542 | 12.5 hrs | 102.3 hrs | 8.2x | 89.8 hrs |
| 1,141,039 | 228,550 | 271.4 hrs | 2,219.6 hrs | 8.2x | 1,948.2 hrs |

### Excel Formulas for Human Processing

```excel
Human Time (sec) = (Cases × 5) + (Total_PDFs × 10)
Human Time (hrs) = Human_Time_sec / 3600
Time Saved (sec) = Human_Time_sec - Automated_Time_sec
Speedup = Human_Time_sec / Automated_Time_sec
```

### Key Insight: Why Automation Matters

With default values:
- **Automated processing is ~8x faster** than manual human review
- **For the full dataset**: Automation saves **81+ days** of manual work
- **Breakeven point**: Even small carriers benefit significantly from automation

---

## Explaining to Leadership

### The Simple Story

1. **We have benchmark data** from Carrier A showing 0.2 PDFs per case
2. **We apply this ratio** to estimate PDFs for any carrier based on case count
3. **Processing time** = (Machine PDFs × 1.45 sec) + (Scanned PDFs × 20 sec)
4. **Monte Carlo simulation** gives us confidence ranges (P10-P90), not just single estimates

### Key Takeaways

- **Each case ≈ 0.86 seconds** of processing time (with default benchmark)
- **Scanned PDFs are 14x slower** than machine-readable (20 sec vs 1.45 sec)
- **15% of PDFs are scanned**, but they account for **71% of processing time**
- **The full dataset (1.1M cases)** requires approximately **271 hours** (11.3 days continuous)
- **Automation is ~8x faster** than manual human review
- **Time saved on full dataset**: ~1,948 hours (81 days) compared to manual processing

### Why This Matters

The tool helps with:
- **Capacity planning**: Know how long carrier processing will take
- **Resource allocation**: Identify which buckets need the most attention
- **What-if analysis**: Model scenarios where some carriers have higher/lower PDF ratios

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | January 2026 | Initial release |
| 1.1.0 | January 2026 | Added comprehensive formulas, Excel reference, real data examples |
| 1.2.0 | January 2026 | Added Human vs Automated processing time comparison |

---

*Last Updated: January 2026*
