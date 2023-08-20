# Given data: Vmp and Imp
vmp_values = [26.90, 29.80, 31.24, 32.10, 30.11, 31.96, 32.15, 32.49, 32.08, 32.73, 32.98, 33.00, 32.89, 32.71, 33.18, 32.44, 32.42, 31.75, 30.72, 28.83, 24.28]
imp_values = [0.17, 0.55, 0.98, 1.40, 0.62, 1.32, 1.44, 1.66, 1.39, 1.86, 2.08, 2.27, 2.00, 1.84, 2.29, 1.63, 1.61, 1.21, 0.79, 0.38, 0.06]

# Calculate total wattage
total_wattage = sum(vmp * imp for vmp, imp in zip(vmp_values, imp_values))

# Calculate average wattage
average_wattage = total_wattage / len(vmp_values)

print("Average Wattage:", average_wattage, "W")
