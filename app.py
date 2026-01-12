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
        value=900,
        step=1,
        help="Total number of cases for Carrier A"
    )

    carrierA_total_pdfs = st.sidebar.number_input(
        "Total PDFs",
        min_value=0,
        value=403,
        step=1,
        help="Total number of PDFs across all cases"
    )

    carrierA_machine_pdfs = st.sidebar.number_input(
        "Machine-Readable PDFs",
        min_value=0,
        value=309,
        step=1,
        help="Number of machine-readable PDFs"
    )

    carrierA_scanned_pdfs = st.sidebar.number_input(
        "Scanned PDFs (OCR required)",
        min_value=0,
        value=94,
        step=1,
        help="Number of scanned PDFs requiring OCR"
    )

    avg_time_machine_seconds = st.sidebar.number_input(
        "Avg Processing Time - Machine PDF (seconds)",
        min_value=0.0,
        value=0.54,
        step=0.1,
        format="%.2f",
        help="Average time to process one machine-readable PDF"
    )

    avg_time_scanned_seconds = st.sidebar.number_input(
        "Avg Processing Time - Scanned PDF (seconds)",
        min_value=0.0,
        value=8.05,
        step=0.1,
        format="%.2f",
        help="Average time to process one scanned PDF"
    )

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
            (0, 50),
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
        st.dataframe(
            bucket_summary_df,
            hide_index=True,
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

        # Export functionality
        st.markdown("---")
        st.subheader("💾 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="multi_carrier_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            # Create Excel file in memory
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                display_df.to_excel(writer, index=False, sheet_name='Carrier Analysis')
            buffer.seek(0)

            st.download_button(
                label="📥 Download as Excel",
                data=buffer,
                file_name="multi_carrier_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
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
