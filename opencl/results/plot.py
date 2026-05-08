import pandas as pd
import matplotlib.pyplot as plt
import sys

# Parse args
if len(sys.argv) != 4:
    print("Usage: python plot.py <benchmark.csv> <device> <kernel>")
    sys.exit(1)

file = sys.argv[1]
device = sys.argv[2]
kernel = sys.argv[3]


# Load CSV
df = pd.read_csv(file)

# Quick sanity check
print(df.head())
print(df.info())

def sort_by_batch_size(df):
    df['Batch Size'] = df['Batch Size'].astype(int)
    return df.sort_values(by='Batch Size')

def filter_by_device_kernel(df, kernel):
    return df[(df['Kernel'] == kernel)]

def plot_bandwidth(df):
    df = sort_by_batch_size(df)
    df = filter_by_device_kernel(df, kernel)
    plt.figure(figsize=(10, 6))
    for device in df['Device'].unique():
        subset = df[df['Device'] == device]
        plt.plot(subset['Batch Size'], subset['Bandwidth (GB/s)'], marker='o', label=device)

    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Bandwidth vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'bandwidth_plot_{kernel}.png')
    plt.show()


plot_bandwidth(df)
