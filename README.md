# Carrier Distribution Analyzer

A Monte Carlo simulation-based forecasting tool for estimating PDF volumes and processing times across different pharmacy carriers using benchmark data.

## Overview

This application helps pharmacies forecast the workload for processing PDFs from different carriers. Using historical data from one carrier (Carrier A) as a benchmark, the tool predicts PDF volumes and processing times for any other carrier based solely on the number of cases.

### Use Case

- **Context**: A pharmacy processes documents from multiple carriers
- **Problem**: Each carrier has varying numbers of cases containing PDFs (machine-readable or scanned)
- **Solution**: Use benchmark data from one known carrier to forecast workload for unknown carriers
- **Output**: Statistical estimates of PDF volumes and processing times with confidence intervals

## Features

### Analysis Modes

- **Single Carrier Mode**: Detailed forecasting for one specific carrier with full statistical distributions
- **Multi-Carrier Analysis**: Batch process multiple carriers from a spreadsheet upload

### Core Features

- **Benchmark-Based Forecasting**: Use Carrier A data to predict outcomes for any carrier
- **Monte Carlo Simulation**: Run thousands of simulations for robust statistical estimates
- **Dual PDF Types**: Handles both machine-readable PDFs (fast) and scanned PDFs (slower, OCR required)
- **Input Validation**: Auto-corrects invalid inputs and prevents common errors
- **Statistical Analysis**: Provides P10-P90 ranges and median (P50) estimates
- **Visual Distributions**: Interactive charts showing PDF volume and processing time distributions (single carrier mode)
- **Override Capability**: Optionally override the scanned PDF ratio for different scenarios
- **Batch Processing**: Upload CSV/Excel files with multiple carriers for comprehensive analysis
- **Data Export**: Download results as CSV or Excel with formatted tables

## How to Run This Application

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository:
```bash
git clone <repository-url>
cd distribution-analyzer
```

2. Create a Python virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Make sure your virtual environment is activated, then start the Streamlit app:
```bash
streamlit run app.py
```

The application will automatically open in your default browser at `http://localhost:8501`

To stop the application, press `Ctrl+C` in the terminal.

To deactivate the virtual environment when done:
```bash
deactivate
```

### Input Parameters

#### Carrier A Benchmark Data (Sidebar)

Required for both modes:
1. **Number of Cases**: Total cases processed for Carrier A
2. **Total PDFs**: Total number of PDFs across all cases
3. **Machine-Readable PDFs**: Number of PDFs that don't require OCR
4. **Scanned PDFs**: Number of PDFs requiring OCR processing
5. **Avg Processing Time - Machine PDF**: Average seconds to process one machine-readable PDF
6. **Avg Processing Time - Scanned PDF**: Average seconds to process one scanned PDF (typically higher)

#### Single Carrier Mode

7. **Number of Cases (Carrier X)**: Number of cases for the carrier you want to forecast
8. **Override Scanned PDF Ratio** *(optional)*: Check to manually set a different scanned ratio
9. **Number of Simulations**: More simulations = more accurate (default: 5,000)

#### Multi-Carrier Analysis Mode

7. **Upload Carrier Data**: CSV or Excel file with two columns:
   - `Carrier ID`: Text identifier for each carrier
   - `Cases`: Number of cases per carrier
8. **Override Scanned PDF Ratio** *(optional)*: Check to manually set a different scanned ratio
9. **Number of Simulations**: Simulations per carrier (default: 1,000 for performance)

### Example Scenarios

#### Single Carrier Example

**Benchmark Data (Carrier A)**:
- Cases: 100
- Total PDFs: 500
- Machine-Readable: 350
- Scanned: 150
- Machine processing time: 2 seconds
- Scanned processing time: 8 seconds

**Forecast (Carrier X)**:
- Cases: 150

**Expected Results**:
- Estimated PDFs: ~750 (150 cases × 5 PDFs/case)
- Estimated scanned: ~225 (30% scanned ratio)
- Estimated machine: ~525
- Processing time: ~3.0 hours

#### Multi-Carrier Example

**Benchmark Data (Carrier A)**: Same as above

**Upload File** (`carriers.csv`):
```csv
Carrier ID,Cases
Carrier B,150
Carrier C,80
Carrier D,200
Carrier E,120
```

**Expected Output Table**:
| Carrier ID | Cases | PDF Range (P10-P90) | Machine PDF Range | Scanned PDF Range | Machine Time Range | Scanned Time Range |
|------------|-------|---------------------|-------------------|-------------------|-------------------|-------------------|
| Carrier B  | 150   | 720-780            | 504-546          | 216-234          | 17.0-18.2 min     | 28.8-31.2 min     |
| Carrier C  | 80    | 384-416            | 269-291          | 115-125          | 9.0-9.7 min       | 15.3-16.7 min     |
| Carrier D  | 200   | 960-1040           | 672-728          | 288-312          | 22.4-24.3 min     | 38.4-41.6 min     |
| Carrier E  | 120   | 576-624            | 403-437          | 173-187          | 13.4-14.6 min     | 23.1-25.0 min     |

## Technical Notes

### Statistical Model

The application uses a two-stage Monte Carlo simulation:

1. **PDF Volume Modeling**:
   - Calculates λ (lambda) = Total PDFs / Cases from Carrier A
   - Samples total PDFs for Carrier X using Poisson distribution with λ

2. **PDF Type Distribution**:
   - Calculates p_scanned = Scanned PDFs / Total PDFs from Carrier A
   - Samples scanned PDFs using Binomial distribution (n=total_pdfs, p=p_scanned)
   - Machine PDFs = Total PDFs - Scanned PDFs

3. **Processing Time**:
   - Total time = (Machine PDFs × Machine time) + (Scanned PDFs × Scanned time)

### Assumptions

- **PDFs per case** follow a Poisson distribution (appropriate for count data)
- **Scanned vs. machine-readable ratio** follows a Binomial distribution
- **Processing times** are deterministic averages per PDF type
- **Carrier similarity**: Other carriers have similar PDF distributions to Carrier A (unless overridden)

### Validation & Error Handling

- Prevents division by zero when cases = 0
- Auto-corrects mismatched PDF counts (machine + scanned ≠ total)
- Validates all inputs are non-negative
- Warns when expected PDFs exceed 1 million
- Caps simulations at 50,000 to prevent performance issues

## Output Interpretation

### Key Metrics

- **Mean**: Average across all simulations (expected value)
- **P50 (Median)**: 50% of simulations fall below this value
- **P90**: 90% of simulations fall below this value (planning buffer)
- **P95**: 95% of simulations fall below this value (conservative estimate)

### Recommended Usage

- Use **Mean** for best-guess estimates
- Use **P90** for capacity planning with reasonable buffer
- Use **P95** for worst-case scenario planning

## Project Structure

```
distribution-analyzer/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── sample_carriers.csv     # Sample multi-carrier data file
└── README.md              # This file
```

## Functions Reference

### Core Functions

- `validate_benchmark_inputs()`: Validates and auto-corrects benchmark inputs
- `compute_benchmark_metrics()`: Calculates λ and probability distributions from Carrier A
- `simulate_forecast()`: Runs Monte Carlo simulation for a single carrier
- `batch_simulate_carriers()`: Runs batch simulations for multiple carriers
- `compute_statistics()`: Calculates mean and percentiles
- `format_time_hours()`: Converts seconds to human-readable format

## Troubleshooting

### Common Issues

**Issue**: "Carrier A cases cannot be zero"
- **Solution**: Enter a valid number of cases (minimum 1)

**Issue**: Warning about PDF count mismatch
- **Solution**: The app auto-corrects this, but verify your input data

**Issue**: Very large expected PDFs warning
- **Solution**: Double-check your inputs; the forecast may be correct for large-scale operations

**Issue**: Slow performance with multi-carrier analysis
- **Solution**: Reduce the number of simulations (500-1,000 per carrier is sufficient for batch processing)

**Issue**: "File must contain columns: Carrier ID, Cases"
- **Solution**: Ensure your CSV/Excel file has exactly these two column headers (case-sensitive)

**Issue**: Excel export not working
- **Solution**: Make sure openpyxl is installed: `pip install openpyxl>=3.1.0`

## Contributing

Suggestions for improvements:
- Add support for multiple benchmark carriers
- Include confidence intervals on charts
- Add time-series forecasting for seasonal variations
- Export results as PDF report
- Add support for parallel processing cores

## License

This project is provided as-is for internal pharmacy operations and analysis.

## Support

For questions or issues, please contact your development team or open an issue in the repository.

---

**Version**: 1.0.0
**Last Updated**: January 2026
**Built with**: Python 3.8+, Streamlit, NumPy, Pandas
