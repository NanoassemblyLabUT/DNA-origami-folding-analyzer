import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
import tkinter as tk
from tkinter import filedialog, simpledialog

# GUI setup
root = tk.Tk()
root.withdraw()

# Prompt for raw data file
raw_data_file = filedialog.askopenfilename(title="Select the raw data", filetypes=[("Excel files", "*.xlsx")])
if not raw_data_file:
    raise Exception("No raw data file selected.")

# Prompt for sheet name
xls = pd.ExcelFile(raw_data_file, engine='openpyxl')
sheet_names = xls.sheet_names
sheet_to_analyze = simpledialog.askstring("Input", f"Enter the sheet name to analyze:\nAvailable sheets: {', '.join(sheet_names)}")
if sheet_to_analyze not in sheet_names:
    raise Exception(f"Sheet '{sheet_to_analyze}' not found in the selected file.")

# Prompt for structure label to use in output filename
structure_label = simpledialog.askstring("Input", "Enter a label for the structure (this will be used to name your output file):")
if not structure_label:
    structure_label = "structure"

# Prompt for dummy Tfold file
Akseloutput_file = filedialog.askopenfilename(title="Select the output file from Aksel et al.'s code", filetypes=[("Excel files", "*.xlsx")])
if not Akseloutput_file:
    raise Exception("No file selected.")

# Load and process data
data = pd.read_excel(raw_data_file, sheet_name=sheet_to_analyze, engine='openpyxl')
temperature = data.iloc[2:, 0].to_numpy(dtype=float)
fluor_structure = data.iloc[2:, 1].to_numpy(dtype=float)
fluor_staples = data.iloc[2:, 2].to_numpy(dtype=float)
fluorescence = fluor_structure - fluor_staples

# Compute rate of change and smooth
fluor_ROC = np.gradient(fluorescence, temperature) * -1
smoothed = savgol_filter(fluor_ROC, window_length=13, polyorder=3)

# Find peaks
peaks, _ = find_peaks(smoothed, prominence=25)
valid_peaks = [i for i in peaks if temperature[i] > 40]
tfold_temps = []
tfold_fluors = []

for i, peak_idx in enumerate(valid_peaks):
    if i == 0:
        window_size = 40
    else:
        prev_peak_temp = temperature[valid_peaks[i - 1]]
        curr_peak_temp = temperature[peak_idx]
        delta_temp = curr_peak_temp - prev_peak_temp
        window_size = int(np.clip((delta_temp / (temperature[1] - temperature[0])) * 0.8, 10, 30))
    window_start = max(0, peak_idx - window_size - 5)
    window_end = peak_idx
    window = smoothed[window_start:window_end]
    if len(window) == 0:
        continue
    min_idx = np.argmin(window)
    tfold_index = window_start + min_idx
    tfold_temps.append(temperature[tfold_index])
    tfold_fluors.append(fluor_ROC[tfold_index])

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(temperature, fluor_ROC, label='Raw Fluorescence Rate of Change', alpha=0.5)
plt.plot(temperature, smoothed, label='Smoothed Data', linewidth=2)
plt.scatter(temperature[peaks], smoothed[peaks], color='orange', label='Detected Peaks')
for i, (t, f) in enumerate(zip(tfold_temps, tfold_fluors)):
    plt.scatter(t, f, color='green', zorder=5, label='Tfold' if i == 0 else "")
    offset_y = 20 if f < 0 else -30
    plt.annotate(f'T{i+1}\n{t:.1f}°C', (t, f), textcoords="offset points", xytext=(0, offset_y),
                 ha='center', fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5))
plt.xlabel('Temperature (°C)')
plt.ylabel('Fluorescence ROC')
plt.title(f'{sheet_to_analyze} - Tfold Detection')
plt.legend()
plt.grid(False)
plt.tight_layout()
plot_filename = f'tfold_plot_{sheet_to_analyze}.png'
plt.savefig(plot_filename)

# Print results
print(f"\nSheet: {sheet_to_analyze}")
for i, (t, f) in enumerate(zip(tfold_temps, tfold_fluors)):
    print(f" Tfold {i+1}: Temperature = {t:.2f} °C, Fluorescence ROC = {f:.2f}")

# Prompt for custom Tfold for cooperativity
custom_tfold_input = simpledialog.askstring("Input", "Override Tfold value for cooperativity calculation (leave blank to use calculated minimum):")
if custom_tfold_input:
    try:
        T1 = float(custom_tfold_input)
    except ValueError:
        raise ValueError("Invalid input for custom Tfold value.")
else:
    T1 = min(tfold_temps)

# Save to Excel with refined formatting
output_excel = f'Tfold_results_{structure_label}.xlsx'
staple_tfolds = pd.read_excel(Akseloutput_file, sheet_name="Final", engine="openpyxl").iloc[:, 13].dropna().values

# Calculate cooperativity
below_T1 = sum(1 for t in staple_tfolds if t < T1)
total = len(staple_tfolds)
cooperativity = (below_T1 / total) * 100 if total > 0 else None

# Build DataFrame
rows = len(tfold_temps)
output_df = pd.DataFrame({
    'Calculated Tfold': tfold_temps,
    'Overridden Tfold': [T1 if custom_tfold_input else ""] + [""] * (rows - 1),
    'Cooperativity (%)': [round(cooperativity, 2)] + [""] * (rows - 1)


})

# Save to Excel
output_df.to_excel(output_excel, sheet_name=sheet_to_analyze, index=False)

print(f"\nCooperativity calculated using Tfold = {T1:.2f} °C: {cooperativity:.2f}%")
print(f"Results saved to {output_excel} and plot saved to {plot_filename}")
