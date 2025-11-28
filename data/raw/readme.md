# Aircraft Engine Prognostics Data Set

## Overview

This dataset contains **multivariate time series data from aircraft engine simulations**. The goal is to predict **Remaining Useful Life (RUL)** - how many operational cycles an engine has left before failure. This is a critical problem in predictive maintenance for aircraft engines.

### What is RUL?
**Remaining Useful Life (RUL)** is the number of operational cycles remaining before an engine fails. For example, if an engine fails at cycle 200, and we're currently at cycle 150, the RUL is 50 cycles.

---

## Dataset Variants

The data comes in four variants (FD001-FD004) with increasing complexity:

### Data Set: FD001
- **Train trajectories**: 100 engines
- **Test trajectories**: 100 engines
- **Conditions**: ONE (Sea Level only)
- **Fault Modes**: ONE (HPC Degradation only)
- **Difficulty**: Simplest - single operating condition, single fault type

### Data Set: FD002
- **Train trajectories**: 260 engines
- **Test trajectories**: 259 engines
- **Conditions**: SIX (multiple operating conditions)
- **Fault Modes**: ONE (HPC Degradation only)
- **Difficulty**: Medium - multiple conditions, single fault type

### Data Set: FD003
- **Train trajectories**: 100 engines
- **Test trajectories**: 100 engines
- **Conditions**: ONE (Sea Level only)
- **Fault Modes**: TWO (HPC Degradation, Fan Degradation)
- **Difficulty**: Medium - single condition, multiple fault types

### Data Set: FD004
- **Train trajectories**: 248 engines
- **Test trajectories**: 249 engines
- **Conditions**: SIX (multiple operating conditions)
- **Fault Modes**: TWO (HPC Degradation, Fan Degradation)
- **Difficulty**: Hardest - multiple conditions, multiple fault types

**Note**: 
- **HPC** = High Pressure Compressor
- **Fan** = Turbofan engine fan
- **Conditions** refer to different operational environments (altitude, temperature, etc.)

---

## Data Structure

### File Organization

Each dataset has three files:
- `train_FD00X.txt` - Training data (engines run to failure)
- `test_FD00X.txt` - Test data (engines stopped before failure)
- `RUL_FD00X.txt` - True RUL values for test engines (ground truth)

### Data Format

- **Format**: Space-separated text files
- **Rows**: Each row = one operational cycle (one time step)
- **Columns**: 26 columns total (see below)
- **Time Series**: Each engine has multiple rows (one per cycle until failure)

### Column Descriptions

The data has **26 columns** organized as follows:

#### 1. Unit Number (Column 1)
- **What it is**: Unique identifier for each engine
- **Example**: 1, 2, 3, ... (each number represents a different engine)
- **Usage**: Group rows by `unit_number` to get time series for each engine

#### 2. Time Cycles (Column 2)
- **What it is**: Operational cycle number for that engine
- **Example**: 1, 2, 3, ... (starts at 1 for each engine)
- **Usage**: Time index within each engine's life cycle
- **Note**: Each engine starts at cycle 1 and runs until failure (training) or is stopped early (test)

#### 3-5. Operational Settings (Columns 3-5)
- **What they are**: Three operational parameters that affect engine performance
- **Examples**: 
  - `setting_1`: Typically around -0.0007 to 0.003 (normalized)
  - `setting_2`: Typically around -0.0004 to 0.0005 (normalized)
  - `setting_3`: Typically around 100.0 or 518.67 (may be in original units)
- **Important**: These settings can change during operation and affect sensor readings
- **Usage**: Account for these when analyzing sensor trends (they're confounding variables)

#### 6-26. Sensor Measurements (Columns 6-26)
- **What they are**: 21 different sensor readings from the engine (in original data)
- **Note**: When loaded via `load_dataset.py`, only 20 sensors are loaded (s_1 through s_20)
- **Examples of sensor types** (typical ranges observed):
  - Temperature sensors: ~500-600°F range
  - Pressure sensors: ~8000-9000 psi range
  - Vibration sensors: ~20-50 range
  - Speed sensors: ~2000-2500 RPM range
  - Other physical measurements
- **Note**: Sensor values are **normalized** (scaled) in this dataset
- **Usage**: These sensors show degradation patterns as the engine approaches failure

---

## Understanding the Data

### Key Concepts

1. **Each Engine is a Time Series**
   - Each engine (unit_number) has multiple rows
   - Rows are ordered by time_cycles (1, 2, 3, ...)
   - Early cycles = healthy engine
   - Later cycles = degraded engine approaching failure

2. **Training vs Test Data**
   - **Training**: Engines run until complete failure (you know when they failed)
   - **Test**: Engines stopped before failure (you need to predict when they would fail)
   - **RUL file**: Contains true remaining cycles for test engines (for evaluation)

3. **Degradation Pattern**
   - Engines start healthy (normal sensor values)
   - Over time, sensors show degradation (values drift from normal)
   - Degradation accelerates as failure approaches
   - Different sensors degrade at different rates

4. **Normal Variation vs Fault**
   - Each engine starts with different initial wear (normal variation)
   - Manufacturing differences cause initial sensor value differences
   - This is NOT a fault - it's expected variation
   - The fault develops gradually during operation

5. **Sensor Noise**
   - Data contains measurement noise
   - Sensor readings fluctuate even when engine state is constant
   - This makes the problem harder (need to filter noise from signal)

---

## Working with the Data

### Loading the Data

The data files are space-separated. When loaded, you'll have:
- **unit_number**: Engine ID
- **time_cycles**: Cycle number (1, 2, 3, ...)
- **setting_1, setting_2, setting_3**: Operational settings
- **s_1 through s_21**: Sensor measurements (21 sensors)

### Data Exploration Tips

1. **Group by Engine**
   ```python
   # Get all cycles for engine 1
   engine_1 = df[df['unit_number'] == 1]
   ```

2. **Track Degradation Over Time**
   - Plot sensor values vs time_cycles for each engine
   - Look for trends: increasing, decreasing, or stable
   - Degrading sensors will show clear trends over time

3. **Compare Engines**
   - Different engines have different lifespans
   - Some engines fail at cycle 100, others at cycle 300
   - Initial sensor values vary between engines (normal)

---

## Identifying Outliers and Data Quality Issues

### What to Look For

1. **Sensor Outliers**
   - **Sudden spikes**: Values that jump dramatically in one cycle
   - **Impossible values**: Negative temperatures, pressures, or speeds (if not normalized)
   - **Missing values**: NaN or empty entries
   - **Constant values**: Sensors that never change (may be broken)

2. **Temporal Outliers**
   - **Time gaps**: Missing cycles (e.g., cycle 5, then cycle 7)
   - **Negative cycles**: time_cycles < 1
   - **Non-sequential cycles**: Cycles that don't increment by 1

3. **Engine-Level Outliers**
   - **Very short lifespans**: Engines that fail in < 10 cycles (may be data error)
   - **Very long lifespans**: Engines with > 500 cycles (check if realistic)
   - **Duplicate engines**: Same unit_number appearing multiple times

### How to Identify Outliers

#### Method 1: Statistical Methods
```python
# Z-score method (values > 3 standard deviations)
from scipy import stats
z_scores = np.abs(stats.zscore(df['s_1']))
outliers = df[z_scores > 3]

# IQR method (Interquartile Range)
Q1 = df['s_1'].quantile(0.25)
Q3 = df['s_1'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['s_1'] < Q1 - 1.5*IQR) | (df['s_1'] > Q3 + 1.5*IQR)]
```

#### Method 2: Domain Knowledge
- **Sensor ranges**: Check if values are within expected physical ranges
- **Trend analysis**: Sudden changes in sensor trends may indicate outliers
- **Cross-sensor validation**: If one sensor shows anomaly, check related sensors

#### Method 3: Time Series Methods
```python
# Check for sudden changes in sensor values
df['s_1_diff'] = df.groupby('unit_number')['s_1'].diff()
large_changes = df[abs(df['s_1_diff']) > threshold]
```

### Common Data Quality Issues

1. **Normalized Data**
   - Sensor values are normalized (scaled to similar ranges)
   - Don't expect physical units (e.g., don't expect temperature in °F)
   - Focus on **relative changes** and **trends**, not absolute values

2. **Operational Setting Changes**
   - Settings can change during operation
   - This causes sensor values to shift (not a fault!)
   - Account for setting changes when analyzing sensor trends

3. **Sensor Noise**
   - Small fluctuations are normal (measurement noise)
   - Use smoothing or filtering to reduce noise
   - Don't treat all small variations as outliers

4. **Missing Sensors**
   - Some sensors may be constant (not useful for prediction)
   - Some sensors may have no variation (broken or not relevant)
   - Identify and potentially remove constant sensors

---

## Practical Analysis Workflow

### Step 1: Load and Inspect
- Load the data files
- Check basic statistics (mean, std, min, max for each column)
- Check for missing values
- Verify data types

### Step 2: Understand Engine Lifecycles
- Group by unit_number
- Calculate lifespan for each engine (max time_cycles)
- Plot distribution of engine lifespans
- Identify any unusually short/long engines

### Step 3: Analyze Sensor Behavior
- For each sensor, plot values over time for a few engines
- Identify which sensors show degradation trends
- Identify which sensors are constant or noisy
- Note which sensors are most predictive of failure

### Step 4: Handle Outliers
- Use statistical methods to identify outliers
- Decide on handling strategy (remove, cap, or impute)
- Document any outliers found

### Step 5: Feature Engineering
- Create features from raw sensors (e.g., rolling averages, rates of change)
- Account for operational settings
- Normalize/scale features if needed

### Step 6: Model Development
- Split training data appropriately
- Train models to predict RUL
- Evaluate on test data using RUL file

---

## Important Notes

1. **Data is Normalized**: Sensor values are scaled, so don't expect physical units
2. **Time Series Nature**: Each engine is a separate time series - don't mix cycles from different engines
3. **Operational Settings Matter**: Settings affect sensor readings - account for them in analysis
4. **Noise is Present**: Sensor noise is expected - use appropriate filtering/smoothing
5. **Initial Variation is Normal**: Different initial sensor values between engines is expected, not a fault
6. **Degradation is Gradual**: Faults develop slowly over many cycles, not instantly

---

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

---

## Quick Reference: Column Mapping

| Column | Name | Type | Description |
|--------|------|------|-------------|
| 1 | unit_number | ID | Engine identifier |
| 2 | time_cycles | Time | Operational cycle number |
| 3 | setting_1 | Setting | Operational setting 1 (normalized) |
| 4 | setting_2 | Setting | Operational setting 2 (normalized) |
| 5 | setting_3 | Setting | Operational setting 3 |
| 6-26 | s_1 to s_21 | Sensor | 21 sensor measurements in original data (normalized) |

**Note**: When loaded via `load_dataset.py`, columns are named: `unit_number`, `time_cycles`, `setting_1`, `setting_2`, `setting_3`, `s_1`, `s_2`, ..., `s_20` (only 20 sensors are loaded, the last sensor from the original 21 is not included in the loaded version).
