import pandas as pd
import matplotlib.pyplot as plt
import sys
import itertools
import numpy as np

# Parse args
if len(sys.argv) != 4:
    print("Usage: python plot.py <benchmark.csv> <device> <kernel>")
    sys.exit(1)

file = sys.argv[1]
filterparam = sys.argv[2]
outputfile = sys.argv[3]

markers = itertools.cycle(['o', 's', '^', 'D', 'v', 'x', '*'])

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

def filter_by_device(df, device):
    return df[(df['Device'] == device)]

def plot_bandwidth(df):
    df = sort_by_batch_size(df)
    df = filter_by_device_kernel(df, filterparam)
    plt.figure(figsize=(10, 6))
    for device in df['Device'].unique():
        subset = df[df['Device'] == device]
        plt.plot(subset['Batch Size'], subset['Bandwidth (GB/s)'], marker=next(markers), label=device)

    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Bandwidth vs Batch Size')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{outputfile}.png')
    plt.show()

def plot_bandwidth_all(df):
    df = sort_by_batch_size(df)
    df = filter_by_device(df, filterparam)
    plt.figure(figsize=(10, 6))
    for kernel in df['Kernel'].unique():
        subset = df[df['Kernel'] == kernel]
        plt.plot(subset['Batch Size'], subset['Bandwidth (GB/s)'], marker=next(markers), label=kernel)

    plt.xscale('log')
    plt.xlabel('Batch Size')
    plt.ylabel('Bandwidth (GB/s)')
    plt.title('Kernel Throughput vs Batch Size (Intel Iris Xe) GPU')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{outputfile}.png')
    plt.show()


def plot_throughput_all(df):
    df = df[df['Batch Size'] == 512]

    kernels = df['Kernel'].unique()
    devices = {'CPU', 'GPU'}

    x = np.arange(len(devices))
    width = 0.8 / len(kernels)

    plt.figure(figsize=(10, 6))

    for i, kernel in enumerate(kernels):
        subset = df[df['Kernel'] == kernel]

        # align values per device explicitly
        values = [
            subset[(subset['Device'] == d)]['Throughput (elements/s)'].values[0]
            if not subset[(subset['Device'] == d)].empty else 0
            for d in devices
        ]

        plt.bar(
            x + i * width,
            values,
            width=width,
            label=kernel
        )

    plt.xticks(x + width * (len(kernels) - 1) / 2, devices)

    plt.xlabel("Device")
    plt.ylabel("Throughput (elements/s)")
    plt.title("Kernel Throughput at Batch Size 512")

    plt.grid(axis='y', alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{outputfile}.png')
    plt.show()

plot_bandwidth_all(df)
