"""
Carrier Scenario Analysis & Forecasting Tool
A Monte Carlo simulation-based forecasting tool for estimating PDF volumes and processing times
across different carriers using benchmark data.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
import io
import plotly.graph_objects as go


def validate_benchmark_inputs(
    total_pdfs: int,
    machine_pdfs: int,
    scanned_pdfs: int,
    cases: int
) -> Tuple[bool, str, int, int]:
    """
    Validate benchmark inputs and auto-correct PDF counts if needed.

    Returns:
        (is_valid, warning_message, corrected_machine_pdfs, corrected_scanned_pdfs)
    """
    warning = ""

    if cases == 0:
        return False, "Carrier A cases cannot be zero", machine_pdfs, scanned_pdfs

    if total_pdfs < 0 or machine_pdfs < 0 or scanned_pdfs < 0:
        return False, "All PDF counts must be non-negative", machine_pdfs, scanned_pdfs

    # Check if machine + scanned equals total
    if machine_pdfs + scanned_pdfs != total_pdfs:
        warning = f"⚠️ Warning: Machine PDFs ({machine_pdfs}) + Scanned PDFs ({scanned_pdfs}) != Total PDFs ({total_pdfs}). "
        warning += f"Auto-correcting: Using total PDFs ({total_pdfs}) and maintaining scanned ratio."

        # Recalculate based on total and scanned count
        corrected_scanned = scanned_pdfs
        corrected_machine = total_pdfs - scanned_pdfs

        # If scanned > total, adjust scanned down
        if corrected_machine < 0:
            corrected_scanned = total_pdfs
            corrected_machine = 0
            warning += f" Scanned PDFs adjusted to {corrected_scanned}."

        return True, warning, corrected_machine, corrected_scanned

    return True, warning, machine_pdfs, scanned_pdfs


def compute_benchmark_metrics(
    carrierA_cases: int,
    carrierA_total_pdfs: int,
    carrierA_machine_pdfs: int,
    carrierA_scanned_pdfs: int
) -> Dict[str, float]:
    """
    Compute key metrics from Carrier A benchmark data.

    Returns:
        Dictionary containing:
        - lambda_pdfs_per_case: Average PDFs per case
        - p_scanned: Probability a PDF is scanned
        - p_machine: Probability a PDF is machine-readable
    """
    lambda_pdfs_per_case = carrierA_total_pdfs / carrierA_cases
    p_scanned = carrierA_scanned_pdfs / carrierA_total_pdfs if carrierA_total_pdfs > 0 else 0
    p_machine = 1 - p_scanned

    return {
        "lambda_pdfs_per_case": lambda_pdfs_per_case,
        "p_scanned": p_scanned,
        "p_machine": p_machine
    }


def simulate_forecast(
    carrierX_cases: int,
    lambda_pdfs_per_case: float,
    p_scanned: float,
    avg_time_machine_seconds: float,
    avg_time_scanned_seconds: float,
    simulations: int = 5000,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Run Monte Carlo simulation for Carrier X forecasting.

    Returns:
        Dictionary containing arrays of simulated values:
        - total_pdfs: Simulated total PDF counts
        - scanned_pdfs: Simulated scanned PDF counts
        - machine_pdfs: Simulated machine-readable PDF counts
        - processing_seconds: Simulated total processing times in seconds
    """
    np.random.seed(random_seed)

    # Expected PDFs for Carrier X
    expected_pdfs = carrierX_cases * lambda_pdfs_per_case

    # Cap simulations if expected_pdfs is very large to prevent memory issues
    if expected_pdfs > 1_000_000:
        st.warning(f"⚠️ Expected PDFs is very large ({expected_pdfs:,.0f}). Results may be approximate.")

    # Monte Carlo simulation
    total_pdfs = np.random.poisson(lam=expected_pdfs, size=simulations)
    scanned_pdfs = np.array([
        np.random.binomial(n=n, p=p_scanned) if n > 0 else 0
        for n in total_pdfs
    ])
    machine_pdfs = total_pdfs - scanned_pdfs

    # Calculate processing times
    processing_seconds = (
        machine_pdfs * avg_time_machine_seconds +
        scanned_pdfs * avg_time_scanned_seconds
    )

    return {
        "total_pdfs": total_pdfs,
        "scanned_pdfs": scanned_pdfs,
        "machine_pdfs": machine_pdfs,
        "processing_seconds": processing_seconds
    }


def format_time_hours(seconds: float) -> str:
    """Convert seconds to human-readable hours format."""
    hours = seconds / 3600
    if hours < 1:
        minutes = seconds / 60
        return f"{minutes:.1f} min"
    return f"{hours:.2f} hrs"


def batch_simulate_carriers(
    carriers_df: pd.DataFrame,
    lambda_pdfs_per_case: float,
    p_scanned: float,
    avg_time_machine_seconds: float,
    avg_time_scanned_seconds: float,
    simulations: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for multiple carriers and return summary statistics.

    Args:
        carriers_df: DataFrame with columns ['Carrier ID', 'Cases']
        lambda_pdfs_per_case: PDFs per case from Carrier A benchmark
        p_scanned: Probability a PDF is scanned
        avg_time_machine_seconds: Average processing time for machine-readable PDF
        avg_time_scanned_seconds: Average processing time for scanned PDF
        simulations: Number of Monte Carlo simulations per carrier (0 = deterministic)
        random_seed: Seed for reproducibility

    Returns:
        DataFrame with carrier analysis including ranges for all metrics
    """
    results = []

    for idx, row in carriers_df.iterrows():
        carrier_id = row['Carrier ID']
        cases = int(row['Cases'])

        if cases <= 0:
            continue

        if simulations == 0:
            # Deterministic calculation
            total_pdfs = cases * lambda_pdfs_per_case
            machine_pdfs = total_pdfs * (1 - p_scanned)
            scanned_pdfs = total_pdfs * p_scanned
            machine_time_seconds = machine_pdfs * avg_time_machine_seconds
            scanned_time_seconds = scanned_pdfs * avg_time_scanned_seconds

            results.append({
                "Carrier ID": carrier_id,
                "Cases": cases,
                "Total PDFs": f"{total_pdfs:.0f}",
                "Machine PDFs": f"{machine_pdfs:.0f}",
                "Scanned PDFs": f"{scanned_pdfs:.0f}",
                "Machine Processing Time": f"{format_time_hours(machine_time_seconds)}",
                "Scanned Processing Time": f"{format_time_hours(scanned_time_seconds)}",
                # Store numeric values for sorting and aggregation
                "_cases": cases,
                "_total_pdfs_p50": total_pdfs,
                "_machine_pdfs_p50": machine_pdfs,
                "_scanned_pdfs_p50": scanned_pdfs,
                "_machine_time_p50": machine_time_seconds,
                "_scanned_time_p50": scanned_time_seconds
            })
        else:
            # Run simulation for this carrier
            sim_results = simulate_forecast(
                carrierX_cases=cases,
                lambda_pdfs_per_case=lambda_pdfs_per_case,
                p_scanned=p_scanned,
                avg_time_machine_seconds=avg_time_machine_seconds,
                avg_time_scanned_seconds=avg_time_scanned_seconds,
                simulations=simulations,
                random_seed=random_seed + idx  # Different seed per carrier
            )

            # Calculate statistics (P10, P50, P90)
            total_pdfs_p10 = np.percentile(sim_results["total_pdfs"], 10)
            total_pdfs_p50 = np.percentile(sim_results["total_pdfs"], 50)
            total_pdfs_p90 = np.percentile(sim_results["total_pdfs"], 90)

            machine_pdfs_p10 = np.percentile(sim_results["machine_pdfs"], 10)
            machine_pdfs_p50 = np.percentile(sim_results["machine_pdfs"], 50)
            machine_pdfs_p90 = np.percentile(sim_results["machine_pdfs"], 90)

            scanned_pdfs_p10 = np.percentile(sim_results["scanned_pdfs"], 10)
            scanned_pdfs_p50 = np.percentile(sim_results["scanned_pdfs"], 50)
            scanned_pdfs_p90 = np.percentile(sim_results["scanned_pdfs"], 90)

            # Calculate processing time for machine-readable and scanned separately
            machine_time_seconds_p10 = machine_pdfs_p10 * avg_time_machine_seconds
            machine_time_seconds_p50 = machine_pdfs_p50 * avg_time_machine_seconds
            machine_time_seconds_p90 = machine_pdfs_p90 * avg_time_machine_seconds

            scanned_time_seconds_p10 = scanned_pdfs_p10 * avg_time_scanned_seconds
            scanned_time_seconds_p50 = scanned_pdfs_p50 * avg_time_scanned_seconds
            scanned_time_seconds_p90 = scanned_pdfs_p90 * avg_time_scanned_seconds

            results.append({
                "Carrier ID": carrier_id,
                "Cases": cases,
                "PDF Range (P10-P90)": f"{total_pdfs_p10:.0f} - {total_pdfs_p90:.0f}",
                "PDF Median (P50)": f"{total_pdfs_p50:.0f}",
                "Machine PDF Range (P10-P90)": f"{machine_pdfs_p10:.0f} - {machine_pdfs_p90:.0f}",
                "Machine PDF Median (P50)": f"{machine_pdfs_p50:.0f}",
                "Scanned PDF Range (P10-P90)": f"{scanned_pdfs_p10:.0f} - {scanned_pdfs_p90:.0f}",
                "Scanned PDF Median (P50)": f"{scanned_pdfs_p50:.0f}",
                "Machine Processing Time Range": f"{format_time_hours(machine_time_seconds_p10)} - {format_time_hours(machine_time_seconds_p90)}",
                "Machine Processing Time Median": f"{format_time_hours(machine_time_seconds_p50)}",
                "Scanned Processing Time Range": f"{format_time_hours(scanned_time_seconds_p10)} - {format_time_hours(scanned_time_seconds_p90)}",
                "Scanned Processing Time Median": f"{format_time_hours(scanned_time_seconds_p50)}",
                # Store numeric values for sorting and aggregation
                "_cases": cases,
                "_total_pdfs_p50": total_pdfs_p50,
                "_machine_pdfs_p50": machine_pdfs_p50,
                "_scanned_pdfs_p50": scanned_pdfs_p50,
                "_machine_time_p50": machine_time_seconds_p50,
                "_scanned_time_p50": scanned_time_seconds_p50
            })

    return pd.DataFrame(results)


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute mean and key percentiles."""
    return {
        "mean": np.mean(data),
        "p50": np.percentile(data, 50),
        "p90": np.percentile(data, 90),
        "p95": np.percentile(data, 95)
    }


def assign_carriers_to_tiers(
    num_carriers: int,
    tier_low_pct: int,
    tier_same_pct: int,
    tier_high_pct: int,
    random_seed: int = 42
) -> List[str]:
    """
    Randomly assign carriers to volume tiers based on percentages.

    Args:
        num_carriers: Total number of carriers to assign
        tier_low_pct: Percentage of carriers for Low tier (0.5x)
        tier_same_pct: Percentage of carriers for Same tier (1.0x)
        tier_high_pct: Percentage of carriers for High tier (1.5x)
        random_seed: Seed for reproducibility

    Returns:
        List of tier assignments ('Low', 'Same', 'High') for each carrier
    """
    np.random.seed(random_seed)

    # Calculate number of carriers per tier
    num_low = int(num_carriers * tier_low_pct / 100)
    num_same = int(num_carriers * tier_same_pct / 100)
    num_high = num_carriers - num_low - num_same  # Remainder goes to high

    # Create tier assignments
    tiers = ['Low'] * num_low + ['Same'] * num_same + ['High'] * num_high

    # Shuffle randomly
    np.random.shuffle(tiers)

    return tiers


def batch_simulate_carriers_with_scenarios(
    carriers_df: pd.DataFrame,
    lambda_pdfs_per_case: float,
    p_scanned: float,
    avg_time_machine_seconds: float,
    avg_time_scanned_seconds: float,
    tier_assignments: List[str],
    simulations: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Run Monte Carlo simulation for multiple carriers with tier-based scenario multipliers.

    Args:
        carriers_df: DataFrame with columns ['Carrier ID', 'Cases']
        lambda_pdfs_per_case: Base PDFs per case from Carrier A benchmark
        p_scanned: Probability a PDF is scanned
        avg_time_machine_seconds: Average processing time for machine-readable PDF
        avg_time_scanned_seconds: Average processing time for scanned PDF
        tier_assignments: List of tier assignments ('Low', 'Same', 'High') for each carrier
        simulations: Number of Monte Carlo simulations per carrier (0 = deterministic)
        random_seed: Seed for reproducibility

    Returns:
        DataFrame with carrier analysis including tier assignment and adjusted metrics
    """
    # Tier multipliers
    tier_multipliers = {
        'Low': 0.5,
        'Same': 1.0,
        'High': 1.5
    }

    results = []

    for idx, row in carriers_df.iterrows():
        carrier_id = row['Carrier ID']
        cases = int(row['Cases'])
        tier = tier_assignments[idx] if idx < len(tier_assignments) else 'Same'
        multiplier = tier_multipliers[tier]

        # Apply multiplier to lambda
        adjusted_lambda = lambda_pdfs_per_case * multiplier

        if cases <= 0:
            continue

        if simulations == 0:
            # Deterministic calculation
            total_pdfs = cases * adjusted_lambda
            machine_pdfs = total_pdfs * (1 - p_scanned)
            scanned_pdfs = total_pdfs * p_scanned
            machine_time_seconds = machine_pdfs * avg_time_machine_seconds
            scanned_time_seconds = scanned_pdfs * avg_time_scanned_seconds

            results.append({
                "Carrier ID": carrier_id,
                "Cases": cases,
                "Tier": f"{tier} ({multiplier}x)",
                "Total PDFs": f"{total_pdfs:.0f}",
                "Machine PDFs": f"{machine_pdfs:.0f}",
                "Scanned PDFs": f"{scanned_pdfs:.0f}",
                "Machine Processing Time": f"{format_time_hours(machine_time_seconds)}",
                "Scanned Processing Time": f"{format_time_hours(scanned_time_seconds)}",
                # Store numeric values for sorting and aggregation
                "_cases": cases,
                "_tier": tier,
                "_multiplier": multiplier,
                "_total_pdfs_p50": total_pdfs,
                "_machine_pdfs_p50": machine_pdfs,
                "_scanned_pdfs_p50": scanned_pdfs,
                "_machine_time_p50": machine_time_seconds,
                "_scanned_time_p50": scanned_time_seconds
            })
        else:
            # Run simulation for this carrier with adjusted lambda
            sim_results = simulate_forecast(
                carrierX_cases=cases,
                lambda_pdfs_per_case=adjusted_lambda,
                p_scanned=p_scanned,
                avg_time_machine_seconds=avg_time_machine_seconds,
                avg_time_scanned_seconds=avg_time_scanned_seconds,
                simulations=simulations,
                random_seed=random_seed + idx
            )

            # Calculate statistics (P10, P50, P90)
            total_pdfs_p10 = np.percentile(sim_results["total_pdfs"], 10)
            total_pdfs_p50 = np.percentile(sim_results["total_pdfs"], 50)
            total_pdfs_p90 = np.percentile(sim_results["total_pdfs"], 90)

            machine_pdfs_p10 = np.percentile(sim_results["machine_pdfs"], 10)
            machine_pdfs_p50 = np.percentile(sim_results["machine_pdfs"], 50)
            machine_pdfs_p90 = np.percentile(sim_results["machine_pdfs"], 90)

            scanned_pdfs_p10 = np.percentile(sim_results["scanned_pdfs"], 10)
            scanned_pdfs_p50 = np.percentile(sim_results["scanned_pdfs"], 50)
            scanned_pdfs_p90 = np.percentile(sim_results["scanned_pdfs"], 90)

            machine_time_seconds_p10 = machine_pdfs_p10 * avg_time_machine_seconds
            machine_time_seconds_p50 = machine_pdfs_p50 * avg_time_machine_seconds
            machine_time_seconds_p90 = machine_pdfs_p90 * avg_time_machine_seconds

            scanned_time_seconds_p10 = scanned_pdfs_p10 * avg_time_scanned_seconds
            scanned_time_seconds_p50 = scanned_pdfs_p50 * avg_time_scanned_seconds
            scanned_time_seconds_p90 = scanned_pdfs_p90 * avg_time_scanned_seconds

            results.append({
                "Carrier ID": carrier_id,
                "Cases": cases,
                "Tier": f"{tier} ({multiplier}x)",
                "PDF Range (P10-P90)": f"{total_pdfs_p10:.0f} - {total_pdfs_p90:.0f}",
                "PDF Median (P50)": f"{total_pdfs_p50:.0f}",
                "Machine PDF Range (P10-P90)": f"{machine_pdfs_p10:.0f} - {machine_pdfs_p90:.0f}",
                "Machine PDF Median (P50)": f"{machine_pdfs_p50:.0f}",
                "Scanned PDF Range (P10-P90)": f"{scanned_pdfs_p10:.0f} - {scanned_pdfs_p90:.0f}",
                "Scanned PDF Median (P50)": f"{scanned_pdfs_p50:.0f}",
                "Machine Processing Time Range": f"{format_time_hours(machine_time_seconds_p10)} - {format_time_hours(machine_time_seconds_p90)}",
                "Machine Processing Time Median": f"{format_time_hours(machine_time_seconds_p50)}",
                "Scanned Processing Time Range": f"{format_time_hours(scanned_time_seconds_p10)} - {format_time_hours(scanned_time_seconds_p90)}",
                "Scanned Processing Time Median": f"{format_time_hours(scanned_time_seconds_p50)}",
                # Store numeric values for sorting and aggregation
                "_cases": cases,
                "_tier": tier,
                "_multiplier": multiplier,
                "_total_pdfs_p50": total_pdfs_p50,
                "_machine_pdfs_p50": machine_pdfs_p50,
                "_scanned_pdfs_p50": scanned_pdfs_p50,
                "_machine_time_p50": machine_time_seconds_p50,
                "_scanned_time_p50": scanned_time_seconds_p50
            })

    return pd.DataFrame(results)


def main():
    st.set_page_config(
        page_title="Carrier Forecasting Tool",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Carrier Scenario Analysis & Forecasting")
    st.markdown("""
    **Purpose**: Forecast PDF volumes and processing times for carriers using Carrier A as a benchmark.

    **Modes**:
    - **Single Carrier Mode**: Forecast for one specific carrier
    - **Multi-Carrier Analysis**: Upload a spreadsheet with multiple carriers for batch analysis

    **Assumptions**:
    - Carrier A is used as the benchmark for all calculations
    - PDFs per case are modeled using a Poisson distribution
    - Scanned vs. machine-readable ratio is modeled using a Binomial distribution
    - Processing times are deterministic per PDF type
    """)

    # Analysis mode selection
    analysis_mode = st.radio(
        "Select Analysis Mode",
        ["Single Carrier", "Multi-Carrier Analysis"],
        horizontal=True,
        help="Choose single carrier for detailed analysis or multi-carrier for batch processing"
    )

    # Sidebar for inputs
    st.sidebar.header("📥 Input Parameters")

    st.sidebar.subheader("Carrier A Benchmark Data")
    carrierA_cases = st.sidebar.number_input(
        "Number of Cases",
        min_value=1,
        value=984,
        step=1,
        help="Total number of cases for Carrier A"
    )

    carrierA_total_pdfs = st.sidebar.number_input(
        "Total PDFs",
        min_value=0,
        value=197,
        step=1,
        help="Total number of PDFs across all cases"
    )

    carrierA_machine_pdfs = st.sidebar.number_input(
        "Machine-Readable PDFs",
        min_value=0,
        value=167,
        step=1,
        help="Number of machine-readable PDFs"
    )

    carrierA_scanned_pdfs = st.sidebar.number_input(
        "Scanned PDFs (OCR required)",
        min_value=0,
        value=30,
        step=1,
        help="Number of scanned PDFs requiring OCR"
    )

    avg_time_machine_seconds = st.sidebar.number_input(
        "Avg Processing Time - Machine PDF (seconds)",
        min_value=0.0,
        value=1.45,
        step=0.1,
        format="%.2f",
        help="Average time to process one machine-readable PDF"
    )

    avg_time_scanned_seconds = st.sidebar.number_input(
        "Avg Processing Time - Scanned PDF (seconds)",
        min_value=0.0,
        value=20.0,
        step=0.1,
        format="%.2f",
        help="Average time to process one scanned PDF"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Human Processing Times")

    human_time_per_case = st.sidebar.number_input(
        "Human Review Time per Case (seconds)",
        min_value=0.0,
        value=5.0,
        step=0.5,
        format="%.1f",
        help="Average time for a human to review one case"
    )

    human_time_per_pdf = st.sidebar.number_input(
        "Human Review Time per PDF (seconds)",
        min_value=0.0,
        value=10.0,
        step=0.5,
        format="%.1f",
        help="Average time for a human to review one PDF (regardless of type)"
    )

    # Excel Formula Reference (collapsible)
    with st.sidebar.expander("📐 Excel Formula Reference"):
        # Calculate current benchmark values for display
        lambda_val = carrierA_total_pdfs / carrierA_cases if carrierA_cases > 0 else 0
        scanned_ratio = carrierA_scanned_pdfs / carrierA_total_pdfs if carrierA_total_pdfs > 0 else 0

        st.markdown("**Current Benchmark Values:**")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| λ (PDFs/Case) | {lambda_val:.4f} |
| Scanned Ratio | {scanned_ratio:.4f} ({scanned_ratio*100:.1f}%) |
| Machine Time | {avg_time_machine_seconds} sec |
| Scanned Time | {avg_time_scanned_seconds} sec |
""")

        st.markdown("**Excel Formulas:**")
        st.code(f"Est. Total PDFs = Cases × {lambda_val:.4f}", language=None)
        st.code(f"Est. Machine PDFs = Total_PDFs × {1-scanned_ratio:.4f}", language=None)
        st.code(f"Est. Scanned PDFs = Total_PDFs × {scanned_ratio:.4f}", language=None)
        st.code(f"Machine Time (sec) = Machine_PDFs × {avg_time_machine_seconds}", language=None)
        st.code(f"Scanned Time (sec) = Scanned_PDFs × {avg_time_scanned_seconds}", language=None)
        st.code("Total Time (sec) = Machine_Time + Scanned_Time", language=None)
        st.code("Total Time (hrs) = Total_Time_sec / 3600", language=None)

        st.markdown("**Example (100 cases):**")
        example_cases = 100
        example_pdfs = example_cases * lambda_val
        example_machine = example_pdfs * (1 - scanned_ratio)
        example_scanned = example_pdfs * scanned_ratio
        example_machine_time = example_machine * avg_time_machine_seconds
        example_scanned_time = example_scanned * avg_time_scanned_seconds
        example_total_time = example_machine_time + example_scanned_time

        st.markdown(f"""
| Step | Calculation | Result |
|------|-------------|--------|
| Total PDFs | 100 × {lambda_val:.4f} | {example_pdfs:.1f} |
| Machine PDFs | {example_pdfs:.1f} × {1-scanned_ratio:.4f} | {example_machine:.1f} |
| Scanned PDFs | {example_pdfs:.1f} × {scanned_ratio:.4f} | {example_scanned:.1f} |
| Machine Time | {example_machine:.1f} × {avg_time_machine_seconds} | {example_machine_time:.1f} sec |
| Scanned Time | {example_scanned:.1f} × {avg_time_scanned_seconds} | {example_scanned_time:.1f} sec |
| **Total Time** | {example_machine_time:.1f} + {example_scanned_time:.1f} | **{example_total_time:.1f} sec** ({example_total_time/60:.1f} min) |
""")

    st.sidebar.markdown("---")

    # Mode-specific inputs
    if analysis_mode == "Single Carrier":
        st.sidebar.subheader("Carrier X Forecast")
        carrierX_cases = st.sidebar.number_input(
            "Number of Cases (Carrier X)",
            min_value=1,
            value=150,
            step=1,
            help="Number of cases for the carrier you want to forecast"
        )
    else:
        st.sidebar.subheader("Multi-Carrier Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Carrier Data (CSV/Excel)",
            type=["csv", "xlsx"],
            help="File must have two columns: 'Carrier ID' and 'Cases'"
        )

    # Validate benchmark inputs
    is_valid, warning_msg, corrected_machine, corrected_scanned = validate_benchmark_inputs(
        carrierA_total_pdfs,
        carrierA_machine_pdfs,
        carrierA_scanned_pdfs,
        carrierA_cases
    )

    if warning_msg:
        st.sidebar.warning(warning_msg)

    if not is_valid:
        st.error(f"❌ Invalid input: {warning_msg}")
        return

    # Compute benchmark metrics
    metrics = compute_benchmark_metrics(
        carrierA_cases,
        carrierA_total_pdfs,
        corrected_machine,
        corrected_scanned
    )

    # Optional override for scanned ratio
    use_override = st.sidebar.checkbox(
        "Override Scanned PDF Ratio",
        value=False,
        help="Override the scanned ratio from Carrier A"
    )

    p_scanned = metrics["p_scanned"]
    if use_override:
        p_scanned = st.sidebar.slider(
            "Scanned PDF Ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(metrics["p_scanned"]),
            step=0.01,
            format="%.2f",
            help="Proportion of PDFs that are scanned (0 = all machine-readable, 1 = all scanned)"
        )

    # Simulations input
    use_simulations = st.sidebar.checkbox(
        "Enable Monte Carlo Simulations",
        value=True,
        help="Uncheck for instant deterministic calculations"
    )

    if use_simulations:
        default_sims = 5000 if analysis_mode == "Single Carrier" else 1000
        simulations = st.sidebar.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=50000,
            value=default_sims,
            step=100,
            help="More simulations = more accurate but slower (use fewer for multi-carrier)"
        )
    else:
        simulations = 0

    # Scenario Analysis settings (only for Multi-Carrier mode)
    enable_scenario_analysis = False
    tier_low_pct = 33
    tier_same_pct = 33
    tier_high_pct = 34

    if analysis_mode == "Multi-Carrier Analysis":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Scenario Analysis")
        enable_scenario_analysis = st.sidebar.checkbox(
            "Enable Scenario Analysis",
            value=False,
            help="Model carrier variability by assigning carriers to Low/Same/High volume tiers"
        )

        if enable_scenario_analysis:
            st.sidebar.markdown("**Tier Distribution (must sum to 100%)**")
            tier_low_pct = st.sidebar.number_input(
                "Low Volume (0.5x benchmark)",
                min_value=0,
                max_value=100,
                value=33,
                step=1,
                help="Percentage of carriers with 50% fewer PDFs per case"
            )
            tier_same_pct = st.sidebar.number_input(
                "Same Volume (1.0x benchmark)",
                min_value=0,
                max_value=100,
                value=33,
                step=1,
                help="Percentage of carriers matching Carrier A benchmark"
            )
            tier_high_pct = st.sidebar.number_input(
                "High Volume (1.5x benchmark)",
                min_value=0,
                max_value=100,
                value=34,
                step=1,
                help="Percentage of carriers with 50% more PDFs per case"
            )

            # Validate percentages sum to 100
            total_pct = tier_low_pct + tier_same_pct + tier_high_pct
            if total_pct != 100:
                st.sidebar.error(f"⚠️ Tier percentages must sum to 100% (currently {total_pct}%)")

    # Run appropriate analysis based on mode
    st.markdown("---")

    if analysis_mode == "Single Carrier":
        # SINGLE CARRIER MODE
        st.header("📈 Single Carrier Forecast Results")

        with st.spinner("Running Monte Carlo simulation..."):
            results = simulate_forecast(
                carrierX_cases=carrierX_cases,
                lambda_pdfs_per_case=metrics["lambda_pdfs_per_case"],
                p_scanned=p_scanned,
                avg_time_machine_seconds=avg_time_machine_seconds,
                avg_time_scanned_seconds=avg_time_scanned_seconds,
                simulations=simulations
            )

        # Compute statistics
        pdf_stats = compute_statistics(results["total_pdfs"])
        time_stats = compute_statistics(results["processing_seconds"])

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "PDFs per Case (Carrier A)",
                f"{metrics['lambda_pdfs_per_case']:.2f}"
            )

        with col2:
            st.metric(
                "Scanned Ratio",
                f"{p_scanned:.1%}",
                delta="Overridden" if use_override else None
            )

        with col3:
            st.metric(
                "Expected Total PDFs",
                f"{pdf_stats['mean']:,.0f}"
            )

        with col4:
            st.metric(
                "Expected Processing Time",
                format_time_hours(time_stats['mean'])
            )

        # Detailed statistics
        st.markdown("---")
        st.subheader("📊 Detailed Statistics")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**PDF Volume Estimates**")
            pdf_df = pd.DataFrame({
                "Metric": ["Mean", "Median (P50)", "P90", "P95"],
                "Total PDFs": [
                    f"{pdf_stats['mean']:,.0f}",
                    f"{pdf_stats['p50']:,.0f}",
                    f"{pdf_stats['p90']:,.0f}",
                    f"{pdf_stats['p95']:,.0f}"
                ]
            })
            st.dataframe(pdf_df, hide_index=True, use_container_width=True)

            st.markdown("**PDF Type Breakdown (Mean)**")
            type_df = pd.DataFrame({
                "Type": ["Machine-Readable", "Scanned (OCR)"],
                "Count": [
                    f"{np.mean(results['machine_pdfs']):,.0f}",
                    f"{np.mean(results['scanned_pdfs']):,.0f}"
                ],
                "Percentage": [
                    f"{(1-p_scanned)*100:.1f}%",
                    f"{p_scanned*100:.1f}%"
                ]
            })
            st.dataframe(type_df, hide_index=True, use_container_width=True)

        with col2:
            st.markdown("**Processing Time Estimates**")
            time_df = pd.DataFrame({
                "Metric": ["Mean", "Median (P50)", "P90", "P95"],
                "Time": [
                    format_time_hours(time_stats['mean']),
                    format_time_hours(time_stats['p50']),
                    format_time_hours(time_stats['p90']),
                    format_time_hours(time_stats['p95'])
                ]
            })
            st.dataframe(time_df, hide_index=True, use_container_width=True)

        # Visualizations
        st.markdown("---")
        st.subheader("📉 Distribution Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Total PDFs Distribution**")
            pdf_chart_data = pd.DataFrame({
                "Total PDFs": results["total_pdfs"]
            })
            st.bar_chart(pdf_chart_data["Total PDFs"].value_counts().sort_index(), height=300)

            # Add percentile markers
            st.caption(
                f"P50: {pdf_stats['p50']:,.0f} | "
                f"P90: {pdf_stats['p90']:,.0f} | "
                f"P95: {pdf_stats['p95']:,.0f}"
            )

        with col2:
            st.markdown("**Processing Time Distribution (hours)**")
            time_hours = results["processing_seconds"] / 3600
            time_chart_data = pd.DataFrame({
                "Processing Time (hours)": time_hours
            })
            st.bar_chart(time_chart_data["Processing Time (hours)"].value_counts().sort_index(), height=300)

            # Add percentile markers
            st.caption(
                f"P50: {format_time_hours(time_stats['p50'])} | "
                f"P90: {format_time_hours(time_stats['p90'])} | "
                f"P95: {format_time_hours(time_stats['p95'])}"
            )

        # Raw simulation data (expandable)
        with st.expander("🔍 View Raw Simulation Data"):
            sim_df = pd.DataFrame({
                "Total PDFs": results["total_pdfs"],
                "Machine PDFs": results["machine_pdfs"],
                "Scanned PDFs": results["scanned_pdfs"],
                "Processing Time (hours)": results["processing_seconds"] / 3600
            })
            st.dataframe(sim_df, height=300, use_container_width=True)

            # Download button
            csv = sim_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="carrier_forecast_simulation.csv",
                mime="text/csv"
            )

    else:
        # MULTI-CARRIER ANALYSIS MODE
        st.header("📊 Multi-Carrier Analysis")

        if uploaded_file is None:
            st.info("👆 Please upload a CSV or Excel file with carrier data to begin analysis.")
            st.markdown("""
            **Expected file format:**
            - Column 1: `Carrier ID` (text)
            - Column 2: `Cases` (numeric)

            Example:
            ```
            Carrier ID,Cases
            Carrier A,100
            Carrier B,150
            Carrier C,80
            ```
            """)
            return

        # Load the uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                carriers_df = pd.read_csv(uploaded_file)
            else:
                carriers_df = pd.read_excel(uploaded_file)

            # Validate columns
            required_cols = ['Carrier ID', 'Cases']
            if not all(col in carriers_df.columns for col in required_cols):
                st.error(f"❌ File must contain columns: {', '.join(required_cols)}")
                st.write("Found columns:", list(carriers_df.columns))
                return

            # Clean data
            carriers_df = carriers_df[required_cols].dropna()
            carriers_df['Cases'] = pd.to_numeric(carriers_df['Cases'], errors='coerce')
            carriers_df = carriers_df[carriers_df['Cases'] > 0]

            if len(carriers_df) == 0:
                st.error("❌ No valid carrier data found in file")
                return

            st.success(f"✅ Loaded {len(carriers_df)} carriers from file")

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return

        # Run batch simulation
        with st.spinner(f"Running Monte Carlo simulations for {len(carriers_df)} carriers..."):
            results_df = batch_simulate_carriers(
                carriers_df=carriers_df,
                lambda_pdfs_per_case=metrics["lambda_pdfs_per_case"],
                p_scanned=p_scanned,
                avg_time_machine_seconds=avg_time_machine_seconds,
                avg_time_scanned_seconds=avg_time_scanned_seconds,
                simulations=simulations
            )

        # Display summary metrics
        st.subheader("📈 Summary Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Total Carriers",
                f"{len(results_df)}"
            )

        with col2:
            st.metric(
                "Total Cases",
                f"{results_df['_cases'].sum():,.0f}"
            )

        with col3:
            st.metric(
                "Est. Total PDFs (Median)",
                f"{results_df['_total_pdfs_p50'].sum():,.0f}"
            )

        with col4:
            total_time = (results_df['_machine_time_p50'].sum() +
                         results_df['_scanned_time_p50'].sum())
            st.metric(
                "Est. Total Processing Time",
                format_time_hours(total_time)
            )

        with col5:
            total_time = (results_df['_machine_time_p50'].sum() +
                         results_df['_scanned_time_p50'].sum())
            total_days = total_time / 86400  # seconds to days
            st.metric(
                "Est. Processing Time (Days)",
                f"{total_days:.2f} days"
            )

        # Display bucketed summary table
        st.markdown("---")
        st.subheader("📊 Bucketed Range Summary")

        # Create case range buckets
        def create_bucket_label(min_cases, max_cases):
            if max_cases == float('inf'):
                return f"{min_cases}+"
            return f"{min_cases}-{max_cases}"

        # Define buckets
        buckets = [
            (1, 50),
            (51, 100),
            (101, 200),
            (201, 500),
            (501, 1000),
            (1001, 2000),
            (2001, 5000),
            (5001, 10000),
            (10001, 25000),
            (25001, 50000),
            (50001, 100000),
            (100001, float('inf'))
        ]

        bucket_data = []
        for min_val, max_val in buckets:
            if max_val == float('inf'):
                bucket_carriers = results_df[results_df['_cases'] >= min_val]
            else:
                bucket_carriers = results_df[(results_df['_cases'] >= min_val) & (results_df['_cases'] <= max_val)]

            if len(bucket_carriers) > 0:
                # Extract numeric values from range strings for min/max calculation
                total_pdfs_p50 = bucket_carriers['_total_pdfs_p50']
                machine_pdfs_p50 = bucket_carriers['_machine_pdfs_p50']
                scanned_pdfs_p50 = bucket_carriers['_scanned_pdfs_p50']
                machine_time_p50 = bucket_carriers['_machine_time_p50']
                scanned_time_p50 = bucket_carriers['_scanned_time_p50']

                bucket_data.append({
                    "Case Range": create_bucket_label(min_val, max_val),
                    "Carriers": len(bucket_carriers),
                    "PDF Range": f"{total_pdfs_p50.min():.0f} - {total_pdfs_p50.max():.0f}",
                    "Machine PDF Range": f"{machine_pdfs_p50.min():.0f} - {machine_pdfs_p50.max():.0f}",
                    "Scanned PDF Range": f"{scanned_pdfs_p50.min():.0f} - {scanned_pdfs_p50.max():.0f}",
                    "Total Time Range": f"{format_time_hours(machine_time_p50.min() + scanned_time_p50.min())} - {format_time_hours(machine_time_p50.max() + scanned_time_p50.max())}"
                })

        bucket_summary_df = pd.DataFrame(bucket_data)

        # Create PDF statistics table (Min/Max/Avg)
        pdf_stats_data = []
        for min_val, max_val in buckets:
            if max_val == float('inf'):
                bucket_carriers = results_df[results_df['_cases'] >= min_val]
            else:
                bucket_carriers = results_df[(results_df['_cases'] >= min_val) & (results_df['_cases'] <= max_val)]

            if len(bucket_carriers) > 0:
                total_pdfs_p50 = bucket_carriers['_total_pdfs_p50']
                cases_per_carrier = bucket_carriers['_cases']
                # Calculate total processing time for each carrier (machine + scanned)
                total_time_per_carrier = bucket_carriers['_machine_time_p50'] + bucket_carriers['_scanned_time_p50']
                # Calculate human processing time for each carrier
                human_time_per_carrier = (cases_per_carrier * human_time_per_case) + (total_pdfs_p50 * human_time_per_pdf)

                pdf_stats_data.append({
                    "Case Range": create_bucket_label(min_val, max_val),
                    "Carriers": len(bucket_carriers),
                    "Min PDFs": f"{total_pdfs_p50.min():.0f}",
                    "Max PDFs": f"{total_pdfs_p50.max():.0f}",
                    "Avg PDFs": f"{total_pdfs_p50.mean():.0f}",
                    "Min Processing Time": format_time_hours(total_time_per_carrier.min()),
                    "Max Processing Time": format_time_hours(total_time_per_carrier.max()),
                    "Avg Processing Time": format_time_hours(total_time_per_carrier.mean()),
                    "Total Processing Time": format_time_hours(total_time_per_carrier.sum()),
                    # Store numeric values for charting
                    "_min_time": total_time_per_carrier.min(),
                    "_max_time": total_time_per_carrier.max(),
                    "_avg_time": total_time_per_carrier.mean(),
                    "_min_human_time": human_time_per_carrier.min(),
                    "_max_human_time": human_time_per_carrier.max(),
                    "_avg_human_time": human_time_per_carrier.mean(),
                    "_total_human_time": human_time_per_carrier.sum()
                })

        if pdf_stats_data:
            st.markdown("**PDF Statistics by Case Bucket**")
            pdf_stats_df = pd.DataFrame(pdf_stats_data)
            # Display table without hidden columns
            display_columns = [col for col in pdf_stats_df.columns if not col.startswith('_')]
            st.dataframe(pdf_stats_df[display_columns], hide_index=True, use_container_width=True)

            # Create grouped bar chart for Min/Max/Avg PDFs
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Min PDFs',
                x=pdf_stats_df['Case Range'],
                y=[int(x) for x in pdf_stats_df['Min PDFs']],
                marker_color='#3498db'
            ))
            fig.add_trace(go.Bar(
                name='Avg PDFs',
                x=pdf_stats_df['Case Range'],
                y=[int(x) for x in pdf_stats_df['Avg PDFs']],
                marker_color='#2ecc71'
            ))
            fig.add_trace(go.Bar(
                name='Max PDFs',
                x=pdf_stats_df['Case Range'],
                y=[int(x) for x in pdf_stats_df['Max PDFs']],
                marker_color='#e74c3c'
            ))
            fig.update_layout(
                barmode='group',
                xaxis_title="Case Range",
                yaxis_title="Number of PDFs",
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            # Create grouped bar chart for Min/Max/Avg Processing Time
            st.markdown("**Processing Time Statistics by Case Bucket**")
            fig_time = go.Figure()
            fig_time.add_trace(go.Bar(
                name='Min Time',
                x=pdf_stats_df['Case Range'],
                y=pdf_stats_df['_min_time'] / 3600,  # Convert to hours for chart
                marker_color='#3498db'
            ))
            fig_time.add_trace(go.Bar(
                name='Avg Time',
                x=pdf_stats_df['Case Range'],
                y=pdf_stats_df['_avg_time'] / 3600,  # Convert to hours for chart
                marker_color='#2ecc71'
            ))
            fig_time.add_trace(go.Bar(
                name='Max Time',
                x=pdf_stats_df['Case Range'],
                y=pdf_stats_df['_max_time'] / 3600,  # Convert to hours for chart
                marker_color='#e74c3c'
            ))
            fig_time.update_layout(
                barmode='group',
                xaxis_title="Case Range",
                yaxis_title="Processing Time (hours)",
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_time, use_container_width=True)

            # Create Human vs Automated Processing Time Comparison Table
            st.markdown("**Human vs Automated Processing Time Comparison**")
            comparison_data = []
            for row in pdf_stats_data:
                auto_total = row['_avg_time']
                human_total = row['_avg_human_time']
                time_saved = human_total - auto_total
                speedup = human_total / auto_total if auto_total > 0 else 0

                comparison_data.append({
                    "Case Range": row['Case Range'],
                    "Carriers": row['Carriers'],
                    "Avg PDFs": row['Avg PDFs'],
                    "Automated Time (Avg)": format_time_hours(auto_total),
                    "Human Time (Avg)": format_time_hours(human_total),
                    "Time Saved (Avg)": format_time_hours(time_saved),
                    "Speedup": f"{speedup:.1f}x",
                    "Total Automated": format_time_hours(row['_avg_time'] * row['Carriers']),
                    "Total Human": format_time_hours(row['_avg_human_time'] * row['Carriers']),
                    # Hidden columns for chart
                    "_auto_hrs": auto_total / 3600,
                    "_human_hrs": human_total / 3600
                })

            comparison_df = pd.DataFrame(comparison_data)
            display_cols = [col for col in comparison_df.columns if not col.startswith('_')]
            st.dataframe(comparison_df[display_cols], hide_index=True, use_container_width=True)

            # Create comparison bar chart
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(
                name='Automated',
                x=comparison_df['Case Range'],
                y=comparison_df['_auto_hrs'],
                marker_color='#2ecc71'
            ))
            fig_compare.add_trace(go.Bar(
                name='Human',
                x=comparison_df['Case Range'],
                y=comparison_df['_human_hrs'],
                marker_color='#e74c3c'
            ))
            fig_compare.update_layout(
                barmode='group',
                xaxis_title="Case Range",
                yaxis_title="Average Processing Time (hours)",
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        # Create carrier distribution bar chart
        if len(bucket_summary_df) > 0:
            st.markdown("**Carrier Distribution by Case Bucket**")
            fig = go.Figure(data=[
                go.Bar(
                    x=bucket_summary_df['Case Range'],
                    y=bucket_summary_df['Carriers'],
                    text=bucket_summary_df['Carriers'],
                    textposition='auto',
                    marker_color='steelblue'
                )
            ])
            fig.update_layout(
                xaxis_title="Case Range",
                yaxis_title="Number of Carriers",
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            bucket_summary_df,
            hide_index=True,
            use_container_width=True
        )

        # Export bucketed summary
        col1, col2 = st.columns(2)
        with col1:
            bucket_csv = bucket_summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Bucketed Summary (CSV)",
                data=bucket_csv,
                file_name="bucketed_summary.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            bucket_buffer = io.BytesIO()
            with pd.ExcelWriter(bucket_buffer, engine='openpyxl') as writer:
                bucket_summary_df.to_excel(writer, index=False, sheet_name='Bucketed Summary')
            bucket_buffer.seek(0)
            st.download_button(
                label="📥 Download Bucketed Summary (Excel)",
                data=bucket_buffer,
                file_name="bucketed_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Display detailed table
        st.markdown("---")
        st.subheader("📋 Detailed Carrier Analysis Table")

        # Create display dataframe (without hidden columns)
        display_cols = [col for col in results_df.columns if not col.startswith('_')]
        display_df = results_df[display_cols].copy()

        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=400
        )

        # Export detailed carrier analysis
        col1, col2 = st.columns(2)

        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Analysis (CSV)",
                data=csv,
                file_name="detailed_carrier_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Create Excel file in memory
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Detailed Analysis')
            buffer.seek(0)

            st.download_button(
                label="📥 Download Detailed Analysis (Excel)",
                data=buffer,
                file_name="detailed_carrier_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

        # Scenario Analysis Section (if enabled)
        if enable_scenario_analysis:
            total_pct = tier_low_pct + tier_same_pct + tier_high_pct
            if total_pct == 100:
                st.markdown("---")
                st.subheader("📊 Scenario Analysis (Variable Distribution)")
                st.markdown(f"""
                *This analysis assumes carriers vary from the benchmark:*
                - **{tier_low_pct}%** of carriers have **Low** volume (0.5x benchmark)
                - **{tier_same_pct}%** of carriers have **Same** volume (1.0x benchmark)
                - **{tier_high_pct}%** of carriers have **High** volume (1.5x benchmark)

                *Carriers are randomly assigned to tiers based on these percentages.*
                """)

                # Assign carriers to tiers
                tier_assignments = assign_carriers_to_tiers(
                    num_carriers=len(carriers_df),
                    tier_low_pct=tier_low_pct,
                    tier_same_pct=tier_same_pct,
                    tier_high_pct=tier_high_pct,
                    random_seed=42
                )

                # Run scenario simulation
                with st.spinner("Running scenario analysis..."):
                    scenario_results_df = batch_simulate_carriers_with_scenarios(
                        carriers_df=carriers_df,
                        lambda_pdfs_per_case=metrics["lambda_pdfs_per_case"],
                        p_scanned=p_scanned,
                        avg_time_machine_seconds=avg_time_machine_seconds,
                        avg_time_scanned_seconds=avg_time_scanned_seconds,
                        tier_assignments=tier_assignments,
                        simulations=simulations
                    )

                # Display tier distribution summary
                st.markdown("**Tier Distribution Summary**")
                tier_counts = scenario_results_df['_tier'].value_counts()
                tier_summary_data = []
                for tier in ['Low', 'Same', 'High']:
                    count = tier_counts.get(tier, 0)
                    tier_carriers = scenario_results_df[scenario_results_df['_tier'] == tier]
                    if len(tier_carriers) > 0:
                        tier_summary_data.append({
                            "Tier": f"{tier} ({'0.5x' if tier == 'Low' else '1.0x' if tier == 'Same' else '1.5x'})",
                            "Carriers": count,
                            "Total Cases": f"{tier_carriers['_cases'].sum():,.0f}",
                            "Est. PDFs (Median)": f"{tier_carriers['_total_pdfs_p50'].sum():,.0f}",
                            "Est. Processing Time": format_time_hours(
                                tier_carriers['_machine_time_p50'].sum() + tier_carriers['_scanned_time_p50'].sum()
                            )
                        })

                tier_summary_df = pd.DataFrame(tier_summary_data)
                st.dataframe(tier_summary_df, hide_index=True, use_container_width=True)

                # Scenario totals comparison
                st.markdown("**Scenario vs Uniform Comparison**")
                uniform_total_cases = results_df['_cases'].sum()
                uniform_total_pdfs = results_df['_total_pdfs_p50'].sum()
                uniform_total_time = results_df['_machine_time_p50'].sum() + results_df['_scanned_time_p50'].sum()
                scenario_total_cases = scenario_results_df['_cases'].sum()
                scenario_total_pdfs = scenario_results_df['_total_pdfs_p50'].sum()
                scenario_total_time = scenario_results_df['_machine_time_p50'].sum() + scenario_results_df['_scanned_time_p50'].sum()

                # Row 1: Uniform Distribution metrics
                st.markdown("*Uniform Distribution (all carriers same as benchmark):*")
                uni_col1, uni_col2, uni_col3, uni_col4, uni_col5 = st.columns(5)
                with uni_col1:
                    st.metric("Total Carriers", f"{len(results_df)}")
                with uni_col2:
                    st.metric("Total Cases", f"{uniform_total_cases:,.0f}")
                with uni_col3:
                    st.metric("Est. Total PDFs", f"{uniform_total_pdfs:,.0f}")
                with uni_col4:
                    st.metric("Est. Processing Time", format_time_hours(uniform_total_time))
                with uni_col5:
                    uniform_total_days = uniform_total_time / 86400
                    st.metric("Est. Time (Days)", f"{uniform_total_days:.2f} days")

                # Row 2: Scenario Distribution metrics
                st.markdown("*Scenario Distribution (carriers vary by tier):*")
                scn_col1, scn_col2, scn_col3, scn_col4, scn_col5 = st.columns(5)
                with scn_col1:
                    st.metric("Total Carriers", f"{len(scenario_results_df)}")
                with scn_col2:
                    st.metric("Total Cases", f"{scenario_total_cases:,.0f}")
                with scn_col3:
                    st.metric("Est. Total PDFs", f"{scenario_total_pdfs:,.0f}",
                              delta=f"{((scenario_total_pdfs/uniform_total_pdfs)-1)*100:+.1f}%" if uniform_total_pdfs > 0 else None)
                with scn_col4:
                    st.metric("Est. Processing Time", format_time_hours(scenario_total_time),
                              delta=f"{((scenario_total_time/uniform_total_time)-1)*100:+.1f}%" if uniform_total_time > 0 else None)
                with scn_col5:
                    scenario_total_days = scenario_total_time / 86400
                    st.metric("Est. Time (Days)", f"{scenario_total_days:.2f} days",
                              delta=f"{((scenario_total_days/(uniform_total_time/86400))-1)*100:+.1f}%" if uniform_total_time > 0 else None)

                # Bucketed scenario summary
                st.markdown("---")
                st.markdown("**Bucketed Scenario Summary**")

                scenario_bucket_data = []
                for min_val, max_val in buckets:
                    if max_val == float('inf'):
                        bucket_carriers = scenario_results_df[scenario_results_df['_cases'] >= min_val]
                    else:
                        bucket_carriers = scenario_results_df[(scenario_results_df['_cases'] >= min_val) & (scenario_results_df['_cases'] <= max_val)]

                    if len(bucket_carriers) > 0:
                        total_pdfs_p50 = bucket_carriers['_total_pdfs_p50']
                        machine_pdfs_p50 = bucket_carriers['_machine_pdfs_p50']
                        scanned_pdfs_p50 = bucket_carriers['_scanned_pdfs_p50']
                        machine_time_p50 = bucket_carriers['_machine_time_p50']
                        scanned_time_p50 = bucket_carriers['_scanned_time_p50']

                        # Count tiers in bucket
                        low_count = len(bucket_carriers[bucket_carriers['_tier'] == 'Low'])
                        same_count = len(bucket_carriers[bucket_carriers['_tier'] == 'Same'])
                        high_count = len(bucket_carriers[bucket_carriers['_tier'] == 'High'])

                        # Calculate total time sum for all carriers in bucket
                        total_time_sum = machine_time_p50.sum() + scanned_time_p50.sum()

                        scenario_bucket_data.append({
                            "Case Range": create_bucket_label(min_val, max_val),
                            "Carriers": len(bucket_carriers),
                            "Tier Mix (L/S/H)": f"{low_count}/{same_count}/{high_count}",
                            "PDF Range": f"{total_pdfs_p50.min():.0f} - {total_pdfs_p50.max():.0f}",
                            "Machine PDF Range": f"{machine_pdfs_p50.min():.0f} - {machine_pdfs_p50.max():.0f}",
                            "Scanned PDF Range": f"{scanned_pdfs_p50.min():.0f} - {scanned_pdfs_p50.max():.0f}",
                            "Total Time Range": f"{format_time_hours(machine_time_p50.min() + scanned_time_p50.min())} - {format_time_hours(machine_time_p50.max() + scanned_time_p50.max())}",
                            "Total Time Sum": format_time_hours(total_time_sum)
                        })

                scenario_bucket_df = pd.DataFrame(scenario_bucket_data)

                # Create PDF statistics table (Min/Max/Avg) for scenario
                scenario_pdf_stats_data = []
                for min_val, max_val in buckets:
                    if max_val == float('inf'):
                        bucket_carriers = scenario_results_df[scenario_results_df['_cases'] >= min_val]
                    else:
                        bucket_carriers = scenario_results_df[(scenario_results_df['_cases'] >= min_val) & (scenario_results_df['_cases'] <= max_val)]

                    if len(bucket_carriers) > 0:
                        total_pdfs_p50 = bucket_carriers['_total_pdfs_p50']
                        cases_per_carrier = bucket_carriers['_cases']
                        # Calculate total processing time for each carrier (machine + scanned)
                        total_time_per_carrier = bucket_carriers['_machine_time_p50'] + bucket_carriers['_scanned_time_p50']
                        # Calculate human processing time for each carrier
                        human_time_per_carrier = (cases_per_carrier * human_time_per_case) + (total_pdfs_p50 * human_time_per_pdf)

                        scenario_pdf_stats_data.append({
                            "Case Range": create_bucket_label(min_val, max_val),
                            "Carriers": len(bucket_carriers),
                            "Min PDFs": f"{total_pdfs_p50.min():.0f}",
                            "Max PDFs": f"{total_pdfs_p50.max():.0f}",
                            "Avg PDFs": f"{total_pdfs_p50.mean():.0f}",
                            "Min Processing Time": format_time_hours(total_time_per_carrier.min()),
                            "Max Processing Time": format_time_hours(total_time_per_carrier.max()),
                            "Avg Processing Time": format_time_hours(total_time_per_carrier.mean()),
                            "Total Processing Time": format_time_hours(total_time_per_carrier.sum()),
                            # Store numeric values for charting
                            "_min_time": total_time_per_carrier.min(),
                            "_max_time": total_time_per_carrier.max(),
                            "_avg_time": total_time_per_carrier.mean(),
                            "_min_human_time": human_time_per_carrier.min(),
                            "_max_human_time": human_time_per_carrier.max(),
                            "_avg_human_time": human_time_per_carrier.mean(),
                            "_total_human_time": human_time_per_carrier.sum()
                        })

                if scenario_pdf_stats_data:
                    st.markdown("**PDF Statistics by Case Bucket (Scenario)**")
                    scenario_pdf_stats_df = pd.DataFrame(scenario_pdf_stats_data)
                    # Display table without hidden columns
                    display_columns = [col for col in scenario_pdf_stats_df.columns if not col.startswith('_')]
                    st.dataframe(scenario_pdf_stats_df[display_columns], hide_index=True, use_container_width=True)

                    # Create grouped bar chart for Min/Max/Avg PDFs (Scenario)
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Min PDFs',
                        x=scenario_pdf_stats_df['Case Range'],
                        y=[int(x) for x in scenario_pdf_stats_df['Min PDFs']],
                        marker_color='#3498db'
                    ))
                    fig.add_trace(go.Bar(
                        name='Avg PDFs',
                        x=scenario_pdf_stats_df['Case Range'],
                        y=[int(x) for x in scenario_pdf_stats_df['Avg PDFs']],
                        marker_color='#2ecc71'
                    ))
                    fig.add_trace(go.Bar(
                        name='Max PDFs',
                        x=scenario_pdf_stats_df['Case Range'],
                        y=[int(x) for x in scenario_pdf_stats_df['Max PDFs']],
                        marker_color='#e74c3c'
                    ))
                    fig.update_layout(
                        barmode='group',
                        xaxis_title="Case Range",
                        yaxis_title="Number of PDFs",
                        height=350,
                        margin=dict(t=20, b=40, l=40, r=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Create grouped bar chart for Min/Max/Avg Processing Time (Scenario)
                    st.markdown("**Processing Time Statistics by Case Bucket (Scenario)**")
                    fig_time = go.Figure()
                    fig_time.add_trace(go.Bar(
                        name='Min Time',
                        x=scenario_pdf_stats_df['Case Range'],
                        y=scenario_pdf_stats_df['_min_time'] / 3600,  # Convert to hours for chart
                        marker_color='#3498db'
                    ))
                    fig_time.add_trace(go.Bar(
                        name='Avg Time',
                        x=scenario_pdf_stats_df['Case Range'],
                        y=scenario_pdf_stats_df['_avg_time'] / 3600,  # Convert to hours for chart
                        marker_color='#2ecc71'
                    ))
                    fig_time.add_trace(go.Bar(
                        name='Max Time',
                        x=scenario_pdf_stats_df['Case Range'],
                        y=scenario_pdf_stats_df['_max_time'] / 3600,  # Convert to hours for chart
                        marker_color='#e74c3c'
                    ))
                    fig_time.update_layout(
                        barmode='group',
                        xaxis_title="Case Range",
                        yaxis_title="Processing Time (hours)",
                        height=350,
                        margin=dict(t=20, b=40, l=40, r=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig_time, use_container_width=True)

                    # Create Human vs Automated Processing Time Comparison Table (Scenario)
                    st.markdown("**Human vs Automated Processing Time Comparison (Scenario)**")
                    scenario_comparison_data = []
                    for row in scenario_pdf_stats_data:
                        auto_total = row['_avg_time']
                        human_total = row['_avg_human_time']
                        time_saved = human_total - auto_total
                        speedup = human_total / auto_total if auto_total > 0 else 0

                        scenario_comparison_data.append({
                            "Case Range": row['Case Range'],
                            "Carriers": row['Carriers'],
                            "Avg PDFs": row['Avg PDFs'],
                            "Automated Time (Avg)": format_time_hours(auto_total),
                            "Human Time (Avg)": format_time_hours(human_total),
                            "Time Saved (Avg)": format_time_hours(time_saved),
                            "Speedup": f"{speedup:.1f}x",
                            "Total Automated": format_time_hours(row['_avg_time'] * row['Carriers']),
                            "Total Human": format_time_hours(row['_avg_human_time'] * row['Carriers']),
                            "_auto_hrs": auto_total / 3600,
                            "_human_hrs": human_total / 3600
                        })

                    scenario_comparison_df = pd.DataFrame(scenario_comparison_data)
                    display_cols = [col for col in scenario_comparison_df.columns if not col.startswith('_')]
                    st.dataframe(scenario_comparison_df[display_cols], hide_index=True, use_container_width=True)

                    # Create comparison bar chart
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        name='Automated',
                        x=scenario_comparison_df['Case Range'],
                        y=scenario_comparison_df['_auto_hrs'],
                        marker_color='#2ecc71'
                    ))
                    fig_compare.add_trace(go.Bar(
                        name='Human',
                        x=scenario_comparison_df['Case Range'],
                        y=scenario_comparison_df['_human_hrs'],
                        marker_color='#e74c3c'
                    ))
                    fig_compare.update_layout(
                        barmode='group',
                        xaxis_title="Case Range",
                        yaxis_title="Average Processing Time (hours)",
                        height=350,
                        margin=dict(t=20, b=40, l=40, r=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig_compare, use_container_width=True)

                # Create stacked bar chart showing carrier distribution by tier
                if len(scenario_bucket_df) > 0:
                    st.markdown("**Carrier Distribution by Case Bucket (with Tier Breakdown)**")

                    # Parse tier mix to get separate counts
                    tier_data = []
                    for _, row in scenario_bucket_df.iterrows():
                        tier_mix = row['Tier Mix (L/S/H)'].split('/')
                        tier_data.append({
                            'Case Range': row['Case Range'],
                            'Low (0.5x)': int(tier_mix[0]),
                            'Same (1.0x)': int(tier_mix[1]),
                            'High (1.5x)': int(tier_mix[2])
                        })
                    tier_chart_df = pd.DataFrame(tier_data)

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name='Low (0.5x)',
                        x=tier_chart_df['Case Range'],
                        y=tier_chart_df['Low (0.5x)'],
                        marker_color='#3498db',
                        text=tier_chart_df['Low (0.5x)'],
                        textposition='auto'
                    ))
                    fig.add_trace(go.Bar(
                        name='Same (1.0x)',
                        x=tier_chart_df['Case Range'],
                        y=tier_chart_df['Same (1.0x)'],
                        marker_color='#2ecc71',
                        text=tier_chart_df['Same (1.0x)'],
                        textposition='auto'
                    ))
                    fig.add_trace(go.Bar(
                        name='High (1.5x)',
                        x=tier_chart_df['Case Range'],
                        y=tier_chart_df['High (1.5x)'],
                        marker_color='#e74c3c',
                        text=tier_chart_df['High (1.5x)'],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        barmode='stack',
                        xaxis_title="Case Range",
                        yaxis_title="Number of Carriers",
                        height=350,
                        margin=dict(t=20, b=40, l=40, r=20),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Display with column tooltips
                st.dataframe(
                    scenario_bucket_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Case Range": st.column_config.TextColumn(
                            "Case Range",
                            help="Range of cases for carriers in this bucket"
                        ),
                        "Carriers": st.column_config.NumberColumn(
                            "Carriers",
                            help="Number of carriers in this case range"
                        ),
                        "Tier Mix (L/S/H)": st.column_config.TextColumn(
                            "Tier Mix (L/S/H)",
                            help="Count of carriers by tier: Low (0.5x) / Same (1.0x) / High (1.5x)"
                        ),
                        "PDF Range": st.column_config.TextColumn(
                            "PDF Range",
                            help="Min to max median PDFs for individual carriers in this bucket"
                        ),
                        "Machine PDF Range": st.column_config.TextColumn(
                            "Machine PDF Range",
                            help="Min to max median machine-readable PDFs for individual carriers in this bucket"
                        ),
                        "Scanned PDF Range": st.column_config.TextColumn(
                            "Scanned PDF Range",
                            help="Min to max median scanned PDFs for individual carriers in this bucket"
                        ),
                        "Total Time Range": st.column_config.TextColumn(
                            "Total Time Range",
                            help="Min to max processing time for individual carriers in this bucket (not summed)"
                        ),
                        "Total Time Sum": st.column_config.TextColumn(
                            "Total Time Sum",
                            help="Combined processing time for ALL carriers in this bucket (aggregate workload)"
                        )
                    }
                )

                # Export scenario analysis
                col1, col2 = st.columns(2)
                with col1:
                    scenario_display_cols = [col for col in scenario_results_df.columns if not col.startswith('_')]
                    scenario_display_df = scenario_results_df[scenario_display_cols].copy()
                    scenario_csv = scenario_display_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Scenario Analysis (CSV)",
                        data=scenario_csv,
                        file_name="scenario_analysis.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    scenario_buffer = io.BytesIO()
                    with pd.ExcelWriter(scenario_buffer, engine='openpyxl') as writer:
                        scenario_display_df.to_excel(writer, index=False, sheet_name='Scenario Analysis')
                        scenario_bucket_df.to_excel(writer, index=False, sheet_name='Scenario Buckets')
                        tier_summary_df.to_excel(writer, index=False, sheet_name='Tier Summary')
                    scenario_buffer.seek(0)
                    st.download_button(
                        label="📥 Download Scenario Analysis (Excel)",
                        data=scenario_buffer,
                        file_name="scenario_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()


"""
How to Run This Application
----------------------------

1. Create and activate a virtual environment:
   python -m venv venv
   venv\\Scripts\\activate  (Windows)
   source venv/bin/activate  (macOS/Linux)

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

4. The app will open in your default browser at http://localhost:8501

Analysis Modes
--------------

Single Carrier Mode:
- Enter Carrier A benchmark data in the left sidebar
- Enter the number of cases for Carrier X
- Optionally override the scanned PDF ratio
- Adjust number of simulations for accuracy vs. speed
- View results, statistics, and distributions in the main panel

Multi-Carrier Analysis Mode:
- Upload a CSV/Excel file with Carrier ID and Cases columns
- View bucketed range summary and detailed carrier analysis
- Export results as CSV or Excel

Scenario Analysis (Multi-Carrier Mode):
- Enable to model carrier variability
- Assign carriers to Low (0.5x), Same (1.0x), or High (1.5x) volume tiers
- Configure tier distribution percentages (must sum to 100%)
- Compare uniform vs scenario distributions

Technical Notes
---------------

Model Assumptions:
- PDFs per case follow a Poisson distribution with λ = (Carrier A total PDFs / Carrier A cases)
- PDF type (scanned vs. machine-readable) follows a Binomial distribution
- Processing times are deterministic averages per PDF type
- All carriers have similar PDF distributions unless overridden or scenario analysis is enabled

Scenario Analysis:
- Carriers are randomly assigned to tiers based on configured percentages
- Low tier: 0.5x benchmark PDFs per case
- Same tier: 1.0x benchmark PDFs per case (unchanged)
- High tier: 1.5x benchmark PDFs per case
- Random seed is fixed for reproducibility

Validation:
- Prevents division by zero when Carrier A cases = 0
- Auto-corrects mismatched PDF counts (machine + scanned ≠ total)
- Validates non-negative inputs
- Warns when expected PDFs are very large (>1M)
- Validates tier percentages sum to 100%

Performance:
- Default 5,000 simulations for single carrier mode
- Default 1,000 simulations for multi-carrier mode
- Increase simulations for more precision (up to 50,000)
- Uses vectorized NumPy operations for speed
"""
