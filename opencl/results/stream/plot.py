import pandas as pd
import matplotlib.pyplot as plt
import sys

# Parse args
if len(sys.argv) != 3:
    print("Usage: python plot.py <benchmark.csv> <output_file>")
    sys.exit(1)

file = sys.argv[1]
output_file = sys.argv[2]


# Load CSV
df = pd.read_csv(file)

# Quick sanity check
print(df.head())
print(df.info())


def plot_bandwidth(df):
    plt.figure(figsize=(10, 6))
    for function in df['function'].unique():
        subset = df[df['function'] == function]
        plt.plot(subset['n_elements'], subset['max_GB_per_sec'], marker='o', label=function)

    plt.xscale('log')
    plt.xlabel('Number of Elements')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('STREAM benchmark, Mali-G610')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_file}.png')
    plt.show()


plot_bandwidth(df)
