#!/usr/bin/env python3

import argparse
import subprocess
import os
import sys
# Set the matplotlib backend to 'Agg' to prevent X forwarding attempts
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Run SYEVX benchmarks and generate plots')
    parser.add_argument('--sizes', '-s', type=str, default='128,256,512',
                        help='Comma-separated matrix sizes (default: 128,256,512)')
    parser.add_argument('--batches', '-b', type=str, default='1,2,4,8,16',
                        help='Comma-separated batch sizes (default: 1,2,4,8,16)')
    parser.add_argument('--sparsity', '-p', type=str, default='0.01,0.05,0.1',
                        help='Comma-separated sparsity levels (default: 0.01,0.05,0.1)')
    parser.add_argument('--neigs', '-n', type=str, default='5,10,15',
                        help='Comma-separated numbers of eigenvalues (default: 5,10,15)')
    parser.add_argument('--output', '-o', type=str, default='',
                        help='CSV output file path (default: benchmark_results_YYYY-MM-DD_HH-MM-SS.csv)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--benchmark-path', type=str, default='',
                        help='Path to the syevx_benchmark executable (default: auto-detect)')
    parser.add_argument('--import-csv', type=str, default='',
                        help='Import existing CSV results and generate plots without running benchmark')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip benchmarking and only generate plots from existing CSV')
    parser.add_argument('--output-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    parser.add_argument('--compare-csv', type=str, default='',
                        help='Comma-separated list of CSV files to compare (used with --import-csv)')
    
    return parser.parse_args()

def find_benchmark_executable():
    """Find the benchmark executable in common locations"""
    search_paths = [
        # Build directories
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "tests", "syevx_benchmark"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "bin", "syevx_benchmark"),
        # Installation directories
        "/usr/local/bin/syevx_benchmark",
        "/usr/bin/syevx_benchmark",
    ]
    
    for path in search_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    
    return None

def run_benchmark(args):
    # Create output filename if not specified
    if not args.output:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.output = f"benchmark_results_{timestamp}.csv"
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find benchmark executable
    benchmark_path = args.benchmark_path
    if not benchmark_path:
        benchmark_path = find_benchmark_executable()
        if not benchmark_path:
            raise FileNotFoundError("Could not find syevx_benchmark executable. Please specify with --benchmark-path.")
    
    print(f"Using benchmark executable: {benchmark_path}")
    
    # Build the command
    cmd = [
        benchmark_path,
        "-s", args.sizes,
        "-b", args.batches,
        "-p", args.sparsity,
        "-n", args.neigs,
        "-o", args.output
    ]
    
    # Run the benchmark
    print(f"Running benchmark with command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Benchmark completed. Results saved to {args.output}")
        return args.output
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed with error code {e.returncode}")
        return None

def create_plots(csv_path, args):
    """Generate plots from benchmark results"""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    print(f"Generating plots from {csv_path}...")
    
    # Try to import pandas, exit gracefully if not available
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not found, cannot generate plots. Install with 'pip install pandas'")
        return None
    
    # Load the results
    df = pd.read_csv(csv_path)
    
    # Add throughput metric (time per matrix) if it doesn't exist
    if 'throughput' not in df.columns:
        df['throughput'] = df['time_seconds'] / df['batch_size']
    
    # Check if we're comparing multiple CSVs
    comparison_dfs = []
    comparison_labels = []
    
    if args.compare_csv:
        csv_files = args.compare_csv.split(',')
        for i, csv_file in enumerate(csv_files):
            if os.path.exists(csv_file):
                try:
                    comp_df = pd.read_csv(csv_file)
                    if 'throughput' not in comp_df.columns:
                        comp_df['throughput'] = comp_df['time_seconds'] / comp_df['batch_size']
                    
                    # Add a label - use filename without extension
                    label = os.path.splitext(os.path.basename(csv_file))[0]
                    comparison_dfs.append(comp_df)
                    comparison_labels.append(label)
                    print(f"Loaded comparison CSV: {csv_file}")
                except Exception as e:
                    print(f"Error loading comparison CSV {csv_file}: {e}")
    
    # Create output directory for plots
    # If output_dir is specified, use that, otherwise use a plots subdirectory in the CSV directory
    if args.output_dir:
        plots_dir = args.output_dir
    else:
        plots_dir = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique values for parameters
    sizes = sorted(df['matrix_size'].unique())
    batches = sorted(df['batch_size'].unique())
    neigs_values = sorted(df['neigs'].unique())
    sparsities = sorted(df['sparsity'].unique())
    
    # Select default values for fixing parameters
    default_size = sizes[0] if sizes else None
    default_batch = batches[0] if batches else None
    default_neig = neigs_values[0] if neigs_values else None
    default_sparsity = sparsities[0] if sparsities else None
    
    # Define markers and colors
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    
    # Helper function to create parameter combinations plots
    def create_parameter_plot(x_param, y_param, fixed_params, title_prefix, filename_prefix):
        plt.figure(figsize=(15, 10))
        
        # Get unique values for the parameters
        x_values = sorted(df[x_param].unique())
        param_values = sorted(df[y_param].unique())
        
        # Limit to first 5 values for clarity
        param_values = param_values[:min(5, len(param_values))]
        
        # Plot data for each parameter value
        for i, param_val in enumerate(param_values):
            # Filter data for this parameter value and fixed params
            filter_dict = {y_param: param_val}
            filter_dict.update(fixed_params)
            
            subset = df
            for key, val in filter_dict.items():
                subset = subset[subset[key] == val]
            
            if not subset.empty:
                plt.plot(subset[x_param], subset['throughput'], 
                        marker=markers[i % len(markers)],
                        linestyle=line_styles[i % len(line_styles)],
                        linewidth=2,
                        color=plt.cm.viridis(0.1 + 0.8 * i / len(param_values)),
                        label=f'{y_param}={param_val}')
        
        # Add comparison plots if available
        for i, (comp_df, label) in enumerate(zip(comparison_dfs, comparison_labels)):
            for j, param_val in enumerate(param_values):
                # Filter data for this parameter value and fixed params
                filter_dict = {y_param: param_val}
                filter_dict.update(fixed_params)
                
                subset = comp_df
                for key, val in filter_dict.items():
                    if key in comp_df.columns:
                        subset = subset[subset[key] == val]
                    else:
                        subset = pd.DataFrame()  # Empty if column doesn't exist
                
                if not subset.empty:
                    plt.plot(subset[x_param], subset['throughput'], 
                            marker=markers[j % len(markers)],
                            linestyle=line_styles[(j+i+1) % len(line_styles)],
                            linewidth=2,
                            alpha=0.7,
                            color=plt.cm.plasma(0.1 + 0.8 * j / len(param_values)),
                            label=f'{label}: {y_param}={param_val}')
        
        # Configure the plot
        plt.xlabel(f'{x_param.replace("_", " ").title()}', fontsize=14)
        plt.ylabel('Time per Matrix (seconds)', fontsize=14)
        
        # Create title with fixed parameters
        fixed_params_str = ', '.join([f"{k.replace('_', ' ').title()}={v}" 
                                     for k, v in fixed_params.items()])
        plt.title(f'{title_prefix} ({fixed_params_str})', fontsize=16)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        
        # Use log scale for appropriate parameters
        if x_param in ['batch_size', 'matrix_size']:
            plt.xscale('log', base=2)
        plt.yscale('log')
        
        plt.tight_layout()
        plt_path = os.path.join(plots_dir, f'{filename_prefix}.png')
        plt.savefig(plt_path)
        print(f"Saved plot: {plt_path}")
        plt.close()
    
    # Create the standard parameter plots
    # 1. Batch Size as independent variable
    create_parameter_plot('batch_size', 'matrix_size', 
                         {'neigs': default_neig, 'sparsity': default_sparsity}, 
                         'Throughput vs Batch Size for Different Matrix Sizes', 
                         'throughput_batchsize_vs_matrix_size')
    
    create_parameter_plot('batch_size', 'neigs', 
                         {'matrix_size': default_size, 'sparsity': default_sparsity},
                         'Throughput vs Batch Size for Different Eigenvalue Counts', 
                         'throughput_batchsize_vs_neigs')
    
    create_parameter_plot('batch_size', 'sparsity', 
                         {'matrix_size': default_size, 'neigs': default_neig},
                         'Throughput vs Batch Size for Different Sparsity Levels', 
                         'throughput_batchsize_vs_sparsity')
    
    # 2. Matrix Size as independent variable
    create_parameter_plot('matrix_size', 'batch_size', 
                         {'neigs': default_neig, 'sparsity': default_sparsity}, 
                         'Throughput vs Matrix Size for Different Batch Sizes', 
                         'throughput_matrixsize_vs_batch')
    
    create_parameter_plot('matrix_size', 'neigs', 
                         {'batch_size': default_batch, 'sparsity': default_sparsity},
                         'Throughput vs Matrix Size for Different Eigenvalue Counts', 
                         'throughput_matrixsize_vs_neigs')
    
    create_parameter_plot('matrix_size', 'sparsity', 
                         {'batch_size': default_batch, 'neigs': default_neig},
                         'Throughput vs Matrix Size for Different Sparsity Levels', 
                         'throughput_matrixsize_vs_sparsity')
    
    # 3. Number of Eigenvalues as independent variable
    create_parameter_plot('neigs', 'matrix_size', 
                         {'batch_size': default_batch, 'sparsity': default_sparsity}, 
                         'Throughput vs Number of Eigenvalues for Different Matrix Sizes', 
                         'throughput_neigs_vs_matrix_size')
    
    create_parameter_plot('neigs', 'batch_size', 
                         {'matrix_size': default_size, 'sparsity': default_sparsity},
                         'Throughput vs Number of Eigenvalues for Different Batch Sizes', 
                         'throughput_neigs_vs_batch')
    
    create_parameter_plot('neigs', 'sparsity', 
                         {'matrix_size': default_size, 'batch_size': default_batch},
                         'Throughput vs Number of Eigenvalues for Different Sparsity Levels', 
                         'throughput_neigs_vs_sparsity')
    
    # 4. Sparsity as independent variable
    create_parameter_plot('sparsity', 'matrix_size', 
                         {'batch_size': default_batch, 'neigs': default_neig}, 
                         'Throughput vs Sparsity for Different Matrix Sizes', 
                         'throughput_sparsity_vs_matrix_size')
    
    create_parameter_plot('sparsity', 'batch_size', 
                         {'matrix_size': default_size, 'neigs': default_neig},
                         'Throughput vs Sparsity for Different Batch Sizes', 
                         'throughput_sparsity_vs_batch')
    
    create_parameter_plot('sparsity', 'neigs', 
                         {'matrix_size': default_size, 'batch_size': default_batch},
                         'Throughput vs Sparsity for Different Eigenvalue Counts', 
                         'throughput_sparsity_vs_neigs')
    
    # Create a multi-line plot showing batch processing efficiency across different parameters
    try:
        plt.figure(figsize=(15, 10))
        
        # For each matrix size and neig value, calculate throughput relative to batch_size=1
        color_idx = 0
        param_combinations = []
        
        # Create combinations of parameters to plot
        for size in sizes[:3]:  # Limit to 3 sizes
            for neig in neigs_values[:2]:  # Limit to 2 neig values
                param_combinations.append((size, neig, default_sparsity))
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(param_combinations)))
        
        for idx, (size, neig, sparsity) in enumerate(param_combinations):
            # Plot main dataset
            subset = df[(df['matrix_size'] == size) & 
                      (df['neigs'] == neig) & 
                      (df['sparsity'] == sparsity)]
            
            if not subset.empty and 1 in subset['batch_size'].values:
                # Get the base throughput for batch_size=1
                base_throughput = subset[subset['batch_size'] == 1]['throughput'].values[0]
                
                # Calculate relative efficiency
                subset = subset.copy()
                subset['relative_throughput'] = subset['throughput'] / base_throughput
                
                plt.plot(subset['batch_size'], subset['relative_throughput'], 
                        marker=markers[idx % len(markers)],
                        linestyle=line_styles[idx % len(line_styles)],
                        color=colors[idx],
                        label=f'n={size}, neigs={neig}')

            # Plot comparison datasets
            for i, (comp_df, label) in enumerate(zip(comparison_dfs, comparison_labels)):
                comp_subset = comp_df[(comp_df['matrix_size'] == size) & 
                                    (comp_df['neigs'] == neig) & 
                                    (comp_df['sparsity'] == sparsity)]
                
                if not comp_subset.empty and 1 in comp_subset['batch_size'].values:
                    # Get the base throughput for batch_size=1
                    comp_base = comp_subset[comp_subset['batch_size'] == 1]['throughput'].values[0]
                    
                    # Calculate relative efficiency
                    comp_subset = comp_subset.copy()
                    comp_subset['relative_throughput'] = comp_subset['throughput'] / comp_base
                    
                    plt.plot(comp_subset['batch_size'], comp_subset['relative_throughput'], 
                            marker=markers[(idx+i+1) % len(markers)],
                            linestyle=line_styles[(idx+i+1) % len(line_styles)],
                            color=plt.cm.plasma(0.1 + 0.8 * idx / len(param_combinations)),
                            alpha=0.8,
                            label=f'{label}: n={size}, neigs={neig}')
        
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Ideal scaling')
        plt.xlabel('Batch Size', fontsize=14)
        plt.ylabel('Relative Time per Matrix\n(normalized to batch_size=1)', fontsize=14)
        plt.title(f'Batch Processing Efficiency (Sparsity={default_sparsity})', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12, loc='best')
        plt.xscale('log', base=2)
        plt.tight_layout()
        plt_path = os.path.join(plots_dir, 'batch_scaling_efficiency_multi.png')
        plt.savefig(plt_path)
        print(f"Saved plot: {plt_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating batch scaling efficiency plot: {e}")
    
    # Return the directory containing the plots
    print(f"All plots saved to: {plots_dir}")
    return plots_dir

def main():
    # Set the environment variable to disable X11 display
    os.environ['MPLBACKEND'] = 'Agg'
    
    args = parse_args()
    
    csv_path = None
    
    # If import-csv is specified, use that file instead of running benchmark
    if args.import_csv:
        if os.path.exists(args.import_csv):
            csv_path = args.import_csv
            print(f"Using existing CSV file: {csv_path}")
        else:
            print(f"Error: Specified CSV file not found: {args.import_csv}")
            return
    elif not args.plot_only:
        # Run the benchmark only if neither import_csv nor plot_only are specified
        csv_path = run_benchmark(args)
    else:
        # If plot_only is specified but no CSV file provided, show error
        print("Error: When using --plot-only, you must specify a CSV file with --import-csv")
        return
    
    # Generate plots if we have a valid CSV path and plots are requested
    if csv_path and not args.no_plot:
        try:
            plots_dir = create_plots(csv_path, args)
            if plots_dir:
                print(f"Plots generated in {plots_dir}")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
if __name__ == '__main__':
    main()
