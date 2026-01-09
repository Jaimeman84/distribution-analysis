"""
Carrier Scenario Analysis & Forecasting Tool
A Monte Carlo simulation-based forecasting tool for estimating PDF volumes and processing times
across different carriers using benchmark data.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Dict, Tuple


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


def compute_statistics(data: np.ndarray) -> Dict[str, float]:
    """Compute mean and key percentiles."""
    return {
        "mean": np.mean(data),
        "p50": np.percentile(data, 50),
        "p90": np.percentile(data, 90),
        "p95": np.percentile(data, 95)
    }


def main():
    st.set_page_config(
        page_title="Carrier Forecasting Tool",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Carrier Scenario Analysis & Forecasting")
    st.markdown("""
    **Purpose**: Forecast PDF volumes and processing times for any carrier using Carrier A as a benchmark.

    **Assumptions**:
    - Carrier A is used as the benchmark for all calculations
    - PDFs per case are modeled using a Poisson distribution
    - Scanned vs. machine-readable ratio is modeled using a Binomial distribution
    - Processing times are deterministic per PDF type
    """)

    # Sidebar for inputs
    st.sidebar.header("📥 Input Parameters")

    st.sidebar.subheader("Carrier A Benchmark Data")
    carrierA_cases = st.sidebar.number_input(
        "Number of Cases",
        min_value=1,
        value=100,
        step=1,
        help="Total number of cases for Carrier A"
    )

    carrierA_total_pdfs = st.sidebar.number_input(
        "Total PDFs",
        min_value=0,
        value=500,
        step=1,
        help="Total number of PDFs across all cases"
    )

    carrierA_machine_pdfs = st.sidebar.number_input(
        "Machine-Readable PDFs",
        min_value=0,
        value=350,
        step=1,
        help="Number of machine-readable PDFs"
    )

    carrierA_scanned_pdfs = st.sidebar.number_input(
        "Scanned PDFs (OCR required)",
        min_value=0,
        value=150,
        step=1,
        help="Number of scanned PDFs requiring OCR"
    )

    avg_time_machine_seconds = st.sidebar.number_input(
        "Avg Processing Time - Machine PDF (seconds)",
        min_value=0.0,
        value=2.0,
        step=0.1,
        format="%.2f",
        help="Average time to process one machine-readable PDF"
    )

    avg_time_scanned_seconds = st.sidebar.number_input(
        "Avg Processing Time - Scanned PDF (seconds)",
        min_value=0.0,
        value=8.0,
        step=0.1,
        format="%.2f",
        help="Average time to process one scanned PDF"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Carrier X Forecast")

    carrierX_cases = st.sidebar.number_input(
        "Number of Cases (Carrier X)",
        min_value=1,
        value=150,
        step=1,
        help="Number of cases for the carrier you want to forecast"
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

    simulations = st.sidebar.number_input(
        "Number of Simulations",
        min_value=100,
        max_value=50000,
        value=5000,
        step=100,
        help="More simulations = more accurate but slower"
    )

    # Run simulation
    st.markdown("---")
    st.header("📈 Forecast Results")

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


if __name__ == "__main__":
    main()


"""
How to Run This Application
----------------------------

1. Install dependencies:
   pip install streamlit numpy pandas

2. Run the application:
   streamlit run app.py

3. The app will open in your default browser at http://localhost:8501

4. Usage:
   - Enter Carrier A benchmark data in the left sidebar
   - Enter the number of cases for Carrier X
   - Optionally override the scanned PDF ratio
   - Adjust number of simulations for accuracy vs. speed
   - View results, statistics, and distributions in the main panel

Technical Notes
---------------

Model Assumptions:
- PDFs per case follow a Poisson distribution with λ = (Carrier A total PDFs / Carrier A cases)
- PDF type (scanned vs. machine-readable) follows a Binomial distribution
- Processing times are deterministic averages per PDF type
- All carriers have similar PDF distributions unless overridden

Validation:
- Prevents division by zero when Carrier A cases = 0
- Auto-corrects mismatched PDF counts (machine + scanned ≠ total)
- Validates non-negative inputs
- Warns when expected PDFs are very large (>1M)

Performance:
- Default 5,000 simulations provide good accuracy
- Increase simulations for more precision (up to 50,000)
- Uses vectorized NumPy operations for speed
"""
