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
    for device in {'CPU', 'GPU'}:
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
    plt.title('Kernel Throughput vs Batch Size (Mali-G610) GPU')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{outputfile}.png')
    plt.show()


def plot_throughput_ntt_multiple_files(files, labels):
    # plt.figure(figsize=(10, 6))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for file, label in zip(files, labels):
        df = pd.read_csv(file)

        df = sort_by_batch_size(df)
        df = filter_by_device(df, filterparam)

        # Only NTT kernel
        subset = df[df['Kernel'] == 'NTT']
        subset['Throughput (Mpoly/s)'] = (
            subset['Throughput (elements/s)'] / ( 1e6)
        )

        ax1.plot(
            subset['Batch Size'],
            subset['Throughput (Mpoly/s)'],
            marker=next(markers),
            label=label
        )
        ax2.plot(
            subset['Batch Size'],
            subset['Bandwidth (GB/s)'],
            marker=next(markers),
            label=label
        )
    
    #plot CPU
    df = pd.read_csv(files[0])  # just read one file to get the CPU data")
    subset = df[(df['Device'] == 'CPU') & (df['Kernel'] == 'NTT')]
    subset['Throughput (Mpoly/s)'] = (
            subset['Throughput (elements/s)'] / ( 1e6)
        )
    
    ax1.plot(
        subset['Batch Size'],
        subset['Throughput (Mpoly/s)'],
        marker=next(markers),
        label='CPU (NTT)',
        linestyle='--'
    )
    ax2.plot(
        subset['Batch Size'],
        subset['Bandwidth (GB/s)'],
        marker=next(markers),
        label='CPU (NTT)',
        linestyle='--'
    )

    plt.xscale('log')

    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax2.set_ylabel('Throughput (Mpoly/s)')
    plt.title('NTT Bandwidth vs Batch Size (Mali-G610)')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{outputfile}.png')
    plt.show()

def plot_throughput_multiple_files_endtoend(files, labels):
    plt.figure(figsize=(10, 6))

    for file, label in zip(files, labels):
        df = pd.read_csv(file)

        df = sort_by_batch_size(df)
        df = filter_by_device(df, filterparam)

        # Only NTT kernel
        subset = df[df['Kernel'] == 'NTT']

        plt.plot(
            subset['Batch Size'],
            subset['Throughput (elements/s)'],
            marker=next(markers),
            label=label
        )
    
    #plot CPU
    df = pd.read_csv(files[0])  # just read one file to get the CPU data")
    subset = df[(df['Device'] == 'CPU') & (df['Kernel'] == 'NTT')]
    plt.plot(
        subset['Batch Size'],
        subset['Throughput (elements/s)'],
        marker=next(markers),
        label='CPU',
        linestyle='--'
    )

    plt.xscale('log')

    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (polynomials/s)')
    plt.title('End-To-End Throughput (Mali-G610)')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{outputfile}.png')
    plt.show()


def plot_throughput_all(df):
    df = df[df['Batch Size'] == 512]

    kernels = df['Kernel'].unique()
    devices = {'CPU', 'GPU'}

    x = np.arange(len(devices))
    width = 0.8 / len(kernels)

    plt.figure(figsize=(10, 6))
    for file, label in zip(files, labels):
        df = pd.read_csv(file)

        df = sort_by_batch_size(df)
        df = filter_by_device(df, filterparam)

        # Only NTT kernel
        subset = df[df['Kernel'] == 'NTT']
        

        plt.plot(
            subset['Batch Size'],
            subset['Throughput (elements/s)'],
            marker=next(markers),
            label=label
        )
    
    #plot CPU
    df = pd.read_csv(files[0])  # just read one file to get the CPU data"
    subset = df[(df['Device'] == 'CPU') & (df['Kernel'] == 'NTT')]
    plt.plot(
        subset['Batch Size'],
        subset['Throughput (elements/s)'],
        marker=next(markers),
        label='CPU (NTT)',
        linestyle='--'
    )
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

def plot_throughput_all(df):
    df = df[df['Batch Size'] == 512]

    kernels = df['Kernel'].unique()
    devices = {'CPU', 'GPU'}

    x = np.arange(len(devices))
    width = 0.8 / len(kernels)

    plt.figure(figsize=(10, 6))
    for file, label in zip(files, labels):
        df = pd.read_csv(file)

        df = sort_by_batch_size(df)
        df = filter_by_device(df, filterparam)

        # Only NTT kernel
        subset = df[df['Kernel'] == 'NTT']
        

        plt.plot(
            subset['Batch Size'],
            subset['Throughput (elements/s)'],
            marker=next(markers),
            label=label
        )
    
    #plot CPU
    df = pd.read_csv(files[0])  # just read one file to get the CPU data"
    subset = df[(df['Device'] == 'CPU') & (df['Kernel'] == 'NTT')]
    plt.plot(
        subset['Batch Size'],
        subset['Throughput (elements/s)'],
        marker=next(markers),
        label='CPU (NTT)',
        linestyle='--'
    )
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

def get_throughput(file, batch_size, kernel, device):
    df = pd.read_csv(file)

    subset = df[
        (df['Batch Size'] == batch_size) &
        (df['Kernel'] == kernel) &
        (df['Device'] == device)
    ]

    if subset.empty:
        return 0
    subset['Throughput (Mpoly/s)'] = (
        subset['Throughput (elements/s)'] / (1e6)
    )

    return subset['Throughput (Mpoly/s)'].values[0]


def plot_throughput_bar(
    intel_files,
    intel_labels,
    amd_files,
    amd_labels,
    batch_size,
    kernel,
    outputfile
):
    labels = []
    values = []

    # ----- Intel section -----
    for file, label in zip(intel_files, intel_labels):
        labels.append(f"{label}")
        values.append(
            get_throughput(file, batch_size, kernel, 'GPU')
        )

    # Intel CPU (take from first Intel file)
    labels.append("CPU")
    values.append(
        get_throughput(
            intel_files[0],
            batch_size,
            kernel,
            'CPU'
        )
    )

    # spacer
    labels.append("")
    values.append(0)

    # ----- AMD section -----
    for file, label in zip(amd_files, amd_labels):
        labels.append(f" {label}")
        values.append(
            get_throughput(file, batch_size, kernel, 'GPU')
        )

    # AMD CPU (take from first AMD file)
    labels.append("CPU")
    values.append(
        get_throughput(
            amd_files[0],
            batch_size,
            kernel,
            'CPU'
        )
    )

    # Plot
    x = np.arange(len(values))

    plt.figure(figsize=(14, 6))

    plt.bar(x, values)

    plt.xticks(x, labels)

    plt.ylabel("Throughput (Million polynomials/s)")
    plt.xlabel("Implementation")
    plt.title(
        f"Throughput for kernel '{kernel}' at batch size {batch_size}"
    )

    intel_center = (0 + (len(intel_files))) / 2
    arm_start = len(intel_files) + 1  # +1 for CPU + spacer
    arm_center = (arm_start + len(values) - 1) / 2

    plt.text(
        intel_center, 
        -max(values)*0.08, 
        "Intel Iris Xe",
        ha='center',
        va='top',
        fontsize=11,
        fontweight='bold'
    )

    plt.text(
        arm_center,
        -max(values)*0.08,
        "ARM Mali-G610 (Radxa Rock 5B)",
        ha='center',
        va='top',
        fontsize=11,
        fontweight='bold'
    )

    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{outputfile}.png")
    plt.show()


def plot_throughput_endtoend(file, outputfile):
    df = pd.read_csv(file)
    plt.figure(figsize=(10, 6))

    baseline = df[
    (df['Kernel'] == 'Chain') &
    (df['Device'] == 'GPU')
    ].copy()

    combined = df[
        (df['Kernel'] == 'Fused') &
        (df['Device'] == 'GPU')
    ].copy()

    cpu = df[
        (df['Kernel'] == 'Chain') &
        (df['Device'] == 'CPU')
    ].copy()

    scale = 1e6

    baseline['Throughput (elements/s)'] /= scale
    combined['Throughput (elements/s)'] /= scale
    cpu['Throughput (elements/s)'] /= scale

    plt.plot(
        baseline['Batch Size'],
        baseline['Throughput (elements/s)'],
        marker=next(markers),
        label='GPU Baseline',
    )
    plt.plot(
        combined['Batch Size'],
        combined['Throughput (elements/s)'],
        marker=next(markers),
        label='GPU Combined',
    )
    plt.plot(
        cpu['Batch Size'],
        cpu['Throughput (elements/s)'],
        marker=next(markers),
        label='CPU',
        
    )

    plt.xscale('log')

    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (Million polynomials/s)')
    plt.title('End-To-End Throughput (Mali-G610)')

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{outputfile}.png')
    plt.show()

    

# plot_throughput_ntt_multiple_files(["final_csv/ntt512_radxa_origin", "ntt512_mali_combined", "radxa_short4"], ["FP-NTT", "SF-NTT", "V-NTT"])
# plot_throughput_multiple_files_endtoend(["final_csv/ntt512_radxa_origin"], ["FP-NTT End-to-End"])
# plot_throughput_bar(["ntt512_main", "ntt512_combining", "ntt512_short4", "ntt512_radxa_origin", "ntt512_mali_combined", "radxa_short4"], ["FP-NTT Intel", "SF-NTT Intel", "V-NTT Intel","FP-NTT AMD", "SF-NTT AMD", "V-NTT AMD"], 512, "NTT", outputfile)
# plot_bandwidth_all(df)
# plot_throughput_endtoend("final_csv/ntt512_radxa_origin", "ntt512_endtoend_mali")

intel_files = [
    "final_csv/intel_main_lower_mem", 
    "final_csv/intel_combined", 
    "final_csv/intel_short4",
]
intel_labels = [
    "FP-NTT",
    "SF-NTT",
    "V-NTT"
]

amd_files = [
    "final_csv/ntt512_radxa_origin", 
    "final_csv/mali_combined", 
    "radxa_short4"
] 
amd_labels = [
    "FP-NTT",
    "SF-NTT",
    "V-NTT"
]

plot_throughput_ntt_multiple_files(amd_files, amd_labels)

# plot_throughput_bar(
#     intel_files,
#     intel_labels,
#     amd_files,
#     amd_labels,
#     batch_size=1024,
#     kernel="NTT",
#     outputfile=outputfile
# )

