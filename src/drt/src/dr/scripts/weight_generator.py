#!/usr/bin/env python3
import os
"""
Weight Generator for Circuit Routing

This script generates perturbed weights for circuit routing based on initial weights.
It creates variations around the default weights while respecting the relationships
between different parameters and the constraints of the routing process.

Usage: python weight_generator.py output_file.csv
"""

# ======= CONFIGURATION OPTIONS =======
# Set this to True to use manual weights instead of generating them
USE_MANUAL_WEIGHTS = False

# Path to manual weights file (only used if USE_MANUAL_WEIGHTS is True)
MANUAL_WEIGHTS_FILE = "manual_weights.csv"

# ======= PERTURBATION CONTROL =======
# Controls the amount of perturbation (will increase with each run)
PERTURBATION_FACTOR = 1.0
# Set to True to reset perturbation factor back to 1.0
RESET_PERTURBATION = False
# Path to store state between runs (use absolute path to ensure consistency)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Default state file (will be overridden if design name is provided)
STATE_FILE = os.path.join(SCRIPT_DIR, "weight_generator_state.txt")
# Current mode (auto, manual, or default)
CURRENT_MODE = "auto"
# Enable quasi-random sequence for better exploration (Sobol sequences)
USE_QUASI_RANDOM = True
# Current design name (for design-specific state tracking)
DESIGN_NAME = None
# ====================================

import pandas as pd
import numpy as np
import random
import sys
import argparse
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import scipy for quasi-random sequences
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Falling back to pure random sampling.")
    USE_QUASI_RANDOM = False

def load_initial_weights():
    """
    Load the initial weights from CSV file in the same directory as the script.
    If file doesn't exist, generate default weights.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "initial_weights.csv")
    
    if os.path.exists(input_path):
        print(f"Loading initial weights from {input_path}")
        return pd.read_csv(input_path)
    else:
        print(f"Initial weights file not found at {input_path}, generating default weights")
        # Create default weights if file doesn't exist
        default_weights = []
        # Define default weights with all three RipUpMode types
        modes = ["ALL", "DRC", "NEARDRC"]
        
        # Add some variation similar to what we observed in the real data
        for mode in modes:
            if mode == "ALL":
                drc_weights = [1]
                marker_weights = [0, 1]
                fixed_weights = [1, 4, 10, 50, 100]
                decay_weights = [0.99, 0.999, 1.0]
            elif mode == "NEARDRC":
                drc_weights = [1]
                marker_weights = [1]
                fixed_weights = [10, 50, 100]
                decay_weights = [0.99, 0.999, 1.0]
            else:  # DRC
                drc_weights = [1, 2, 4, 8, 16, 32, 64]
                marker_weights = [1, 2, 4, 8, 16]
                fixed_weights = [2, 3, 4, 10, 50, 100]
                decay_weights = [0.99, 0.999, 1.0]
            
            for drc_w in drc_weights:
                for marker_w in marker_weights:
                    for fixed_w in fixed_weights:
                        for decay_w in decay_weights:
                            # Add some randomness to avoid creating too many entries
                            if random.random() < 0.2:
                                default_weights.append({
                                    'drc_weight': drc_w,
                                    'marker_weight': marker_w,
                                    'fixed_weight': fixed_w,
                                    'decay_weight': decay_w,
                                    'RipUpMode': mode
                                })
        
        # Create a DataFrame and sample if we have too many rows
        df = pd.DataFrame(default_weights)
        if len(df) > 66:  # Original file had 66 rows
            df = df.sample(66)
        return df

def load_manual_weights(file_path):
    """
    Load manual weights from a CSV file.
    Expected format: drc_cost,marker_cost,fixed_cost,decay_cost
    """
    try:
        # Try to read the file both with and without headers
        try:
            df = pd.read_csv(file_path, names=['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost'])
        except:
            df = pd.read_csv(file_path)
            # If headers were included, rename them to expected format
            if list(df.columns) != ['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost']:
                df.columns = ['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost']
        
        # Rename columns to match internal naming
        df = df.rename(columns={
            'drc_cost': 'drc_weight', 
            'marker_cost': 'marker_weight', 
            'fixed_cost': 'fixed_weight', 
            'decay_cost': 'decay_weight'
        })
        
        # Assign a default RipUpMode (doesn't matter for output)
        df['RipUpMode'] = 'DRC'
        
        print(f"Loaded {len(df)} manual weight combinations from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading manual weights: {str(e)}")
        sys.exit(1)

def get_weight_stats(df):
    """Calculate statistics for each weight column by RipUpMode."""
    stats = {}
    
    for mode in df['RipUpMode'].unique():
        mode_df = df[df['RipUpMode'] == mode]
        stats[mode] = {
            'drc_weight': {
                'values': sorted(mode_df['drc_weight'].unique()),
                'min': mode_df['drc_weight'].min(),
                'max': mode_df['drc_weight'].max(),
                'mean': mode_df['drc_weight'].mean()
            },
            'marker_weight': {
                'values': sorted(mode_df['marker_weight'].unique()),
                'min': mode_df['marker_weight'].min(),
                'max': mode_df['marker_weight'].max(),
                'mean': mode_df['marker_weight'].mean()
            },
            'fixed_weight': {
                'values': sorted(mode_df['fixed_weight'].unique()),
                'min': mode_df['fixed_weight'].min(),
                'max': mode_df['fixed_weight'].max(),
                'mean': mode_df['fixed_weight'].mean()
            },
            'decay_weight': {
                'values': sorted(mode_df['decay_weight'].unique()),
                'min': mode_df['decay_weight'].min(),
                'max': mode_df['decay_weight'].max(),
                'mean': mode_df['decay_weight'].mean()
            }
        }
    
    return stats

def perturb_drc_weight(original, mode):
    """
    Perturb drc_weight while generally maintaining the power-of-2 pattern,
    but with added variability for more exploration.
    """
    powers = [1, 2, 4, 8, 16, 32, 64]
    
    if mode in ["ALL", "NEARDRC"]:
        # For ALL and NEARDRC, drc_weight tends to be lower (mostly 1)
        # But add more exploration with a chance for higher values
        if random.random() < 0.7:  # 70% chance to stay in lower range
            choices = [1, 2, 4]
            return random.choice(choices)
        else:  # 30% chance to explore higher values
            choices = [8, 16]
            return random.choice(choices)
    else:  # DRC
        # For DRC, we see values ranging from 1 to 64 (powers of 2)
        
        # 20% chance for complete random exploration across all powers
        if random.random() < 0.2:
            return random.choice(powers)
        
        # Otherwise, perturb around the original value with a wider range
        original_idx = powers.index(original) if original in powers else 0
        
        # Define a range of indices to choose from (centered around original but wider)
        # Apply perturbation factor to increase the range
        range_shift = int(3 * PERTURBATION_FACTOR)
        min_idx = max(0, original_idx - range_shift)
        max_idx = min(len(powers) - 1, original_idx + range_shift)
        
        return powers[random.randint(min_idx, max_idx)]

def perturb_marker_weight(original, mode, drc_weight):
    """
    Perturb marker_weight while respecting the relationship with RipUpMode.
    ALL mode should have the smallest marker_weight, then NEARDRC, then DRC.
    marker_weight usually follows powers of 2 pattern, but we'll add variability for exploration.
    """
    if mode == "ALL":
        # For ALL, keep marker_weight consistently low (0-1) as requested
        return random.choice([0, 1])
    elif mode == "NEARDRC":
        # For NEARDRC, add more exploration (original data had only 1)
        # Apply perturbation factor to potentially include higher values
        max_val = max(4, int(4 * PERTURBATION_FACTOR))
        powers = [p for p in [1, 2, 4, 8, 16] if p <= max_val]
        return random.choice(powers)
    else:  # DRC
        # For DRC, marker_weight ranges from 1 to 16
        # Add more exploration while loosely maintaining relationship with drc_weight
        powers = [1, 2, 4, 8, 16]
        
        # Less strict relationship for more exploration
        # Apply perturbation factor to increase the maximum
        base_max = drc_weight // 2 if drc_weight >= 2 else 1
        max_power = min(int(base_max * PERTURBATION_FACTOR), 16)
        
        # Allow some probability of choosing any value for exploration
        # Increase this probability with higher perturbation factor
        explore_prob = 0.2 * PERTURBATION_FACTOR
        if random.random() < explore_prob:  # Chance for complete exploration
            return random.choice(powers)
        
        # Otherwise, maintain loose relationship
        valid_powers = [p for p in powers if p <= max_power]
        if not valid_powers:  # Safety check
            valid_powers = [1]
            
        return random.choice(valid_powers)

def perturb_fixed_weight(original):
    """
    Perturb fixed_weight based on the observed patterns but with significant exploration.
    Values are typically 1, 2, 3, 4, 10, 50, 100, but we'll explore more values.
    """
    common_values = [1, 2, 3, 4, 10, 50, 100]
    
    # Increase exploration probability with perturbation factor
    explore_prob = 0.3 * PERTURBATION_FACTOR
    if random.random() < explore_prob:
        # Add some in-between values that weren't in the original data
        additional_values = [5, 15, 20, 25, 30, 40, 60, 75, 90]
        expanded_values = common_values + additional_values
        return random.choice(expanded_values)
    
    # Chance of a completely random integer value between 1-100
    # Increase max range with perturbation factor
    if random.random() < 0.2 * PERTURBATION_FACTOR:
        max_val = int(100 * PERTURBATION_FACTOR)
        return random.randint(1, max_val)
    
    # Otherwise perturb around the original value
    closest_idx = min(range(len(common_values)), 
                     key=lambda i: abs(common_values[i] - original))
    
    # Define a wider range of indices to choose from
    # Apply perturbation factor to increase range
    range_shift = int(2 * PERTURBATION_FACTOR)
    min_idx = max(0, closest_idx - range_shift)
    max_idx = min(len(common_values) - 1, closest_idx + range_shift)
    
    return common_values[random.randint(min_idx, max_idx)]

def perturb_decay_weight(original):
    """
    Perturb decay_weight with significant exploration in the 0-1 range.
    Values are typically 0.99, 0.999, or 1.0, but we'll explore more values.
    """
    common_values = [0.99, 0.999, 1.0]
    
    # With higher perturbation, explore wider range of values
    if random.random() < 0.4:
        return random.choice(common_values)
    
    if random.random() < 0.3:
        # Expand range based on perturbation factor
        min_val = max(0.8, 0.9 - (0.1 * PERTURBATION_FACTOR))
        other_values = [0.9, 0.95, 0.98, 0.995, 0.998]
        # Add more values with higher perturbation
        if PERTURBATION_FACTOR > 2:
            other_values.extend([0.85, 0.88, 0.92, 0.94, 0.96, 0.97])
        if PERTURBATION_FACTOR > 3:
            other_values.extend([min_val, min_val + 0.02, min_val + 0.04])
        return random.choice(other_values)
    
    # Random float in a range that expands with perturbation factor
    min_val = max(0.8, 0.9 - (0.1 * PERTURBATION_FACTOR))
    return round(min_val + random.random() * (1.0 - min_val), 3)

def generate_perturbed_weights(initial_df, num_samples=100):
    """
    Generate perturbed weights based on the initial weights.
    
    Args:
        initial_df: DataFrame with the initial weights
        num_samples: Number of perturbed samples to generate
        
    Returns:
        DataFrame with perturbed weights
    """
    # Get statistics for each weight by RipUpMode
    weight_stats = get_weight_stats(initial_df)
    
    # Get unique combinations from initial data to use as seeds
    initial_combinations = initial_df.drop_duplicates().to_dict('records')
    
    # Generate perturbed samples
    perturbed_samples = []
    
    # Use each unique initial combination as a seed at least once
    for combo in initial_combinations:
        # Create a perturbed version of this combination
        drc_weight = perturb_drc_weight(combo['drc_weight'], combo['RipUpMode'])
        marker_weight = perturb_marker_weight(combo['marker_weight'], combo['RipUpMode'], drc_weight)
        fixed_weight = perturb_fixed_weight(combo['fixed_weight'])
        decay_weight = perturb_decay_weight(combo['decay_weight'])
        
        perturbed_samples.append({
            'drc_weight': drc_weight,
            'marker_weight': marker_weight,
            'fixed_weight': fixed_weight,
            'decay_weight': decay_weight,
            'RipUpMode': combo['RipUpMode']
        })
    
    # Determine if we should use quasi-random sequences for better exploration
    remaining_samples = num_samples - len(perturbed_samples)
    if USE_QUASI_RANDOM and HAS_SCIPY and remaining_samples > 0:
        # Use Sobol sequence for better exploration of the weight space
        return generate_sobol_weights(initial_df, weight_stats, num_samples, perturbed_samples)
    else:
        # Fall back to original method
        return generate_random_weights(initial_df, initial_combinations, num_samples, perturbed_samples)


def generate_random_weights(initial_df, initial_combinations, num_samples, perturbed_samples):
    """
    Generate weight samples using traditional random sampling (original method).
    """
    # Generate the rest of the samples
    remaining_samples = num_samples - len(perturbed_samples)
    
    for _ in range(remaining_samples):
        # Randomly select a RipUpMode with probability proportional to frequency in initial data
        mode_counts = initial_df['RipUpMode'].value_counts(normalize=True)
        mode = np.random.choice(mode_counts.index, p=mode_counts.values)
        
        # Randomly select a seed from the initial combinations with matching mode
        mode_combos = [c for c in initial_combinations if c['RipUpMode'] == mode]
        if not mode_combos:
            continue
            
        seed = random.choice(mode_combos)
        
        # Create a perturbed version of this seed
        drc_weight = perturb_drc_weight(seed['drc_weight'], mode)
        marker_weight = perturb_marker_weight(seed['marker_weight'], mode, drc_weight)
        fixed_weight = perturb_fixed_weight(seed['fixed_weight'])
        decay_weight = perturb_decay_weight(seed['decay_weight'])
        
        perturbed_samples.append({
            'drc_weight': drc_weight,
            'marker_weight': marker_weight,
            'fixed_weight': fixed_weight,
            'decay_weight': decay_weight,
            'RipUpMode': mode
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(perturbed_samples)
    
    # Remove duplicates if any
    result_df = result_df.drop_duplicates()
    
    return result_df


def generate_sobol_weights(initial_df, weight_stats, num_samples, existing_samples):
    """
    Generate weight samples using Sobol sequences for better space exploration.
    
    This method ensures maximum exploration of the weight space by using
    low-discrepancy sequences (Sobol) which provide more uniform coverage
    compared to pure random sampling.
    
    Args:
        initial_df: DataFrame with the initial weights
        weight_stats: Statistics for each weight by RipUpMode
        num_samples: Total number of samples to generate
        existing_samples: List of already generated samples
        
    Returns:
        DataFrame with perturbed weights using Sobol sequences
    """
    # Determine how many more samples we need
    remaining_samples = num_samples - len(existing_samples)
    if remaining_samples <= 0:
        return pd.DataFrame(existing_samples)
        
    print(f"Using Sobol sequences for generating {remaining_samples} additional weight combinations")
    
    # Define weight ranges for each RipUpMode
    mode_ranges = {}
    for mode in initial_df['RipUpMode'].unique():
        # Get the stats for this mode
        stats = weight_stats[mode]
        
        # Use full ranges for maximum exploration as requested
        mode_ranges[mode] = {
            'drc_weight': (0, 100),     # Full range 0-100 for maximum exploration
            'marker_weight': (0, 100),  # Full range 0-100 for maximum exploration
            'fixed_weight': (0, 100),   # Full range 0-100 for maximum exploration
            'decay_weight': (0, 1.0)    # Full range 0-1 for maximum exploration
        }
    
    # Create a Sobol sequence generator 
    # Use dimension 4 for the 4 weights we need to generate
    sampler = qmc.Sobol(d=4, scramble=True, seed=int(100 * PERTURBATION_FACTOR))
    
    # Generate base samples in [0, 1) range
    # We generate extra samples as some may be filtered out due to constraints
    # Ensure n is a power of 2 for optimal balance (Sobol requirement)
    n_samples = remaining_samples * 3
    # Find the next power of 2
    power_of_2 = 2 ** int(np.ceil(np.log2(n_samples)))
    sample_points = sampler.random(n=power_of_2)
    
    # Container for new samples
    sobol_samples = []
    
    # Generate samples for each RipUpMode proportionally
    mode_proportions = initial_df['RipUpMode'].value_counts(normalize=True)
    samples_per_mode = {mode: int(remaining_samples * prop) for mode, prop in mode_proportions.items()}
    
    # Ensure we have at least a few samples per mode
    for mode in samples_per_mode:
        if samples_per_mode[mode] < 3:
            samples_per_mode[mode] = 3
    
    # Adjust total to match requested number
    total_allocated = sum(samples_per_mode.values())
    if total_allocated < remaining_samples:
        # Add the remainder to the most common mode
        most_common_mode = mode_proportions.idxmax()
        samples_per_mode[most_common_mode] += (remaining_samples - total_allocated)
    
    # Track used indices in the sample_points array
    used_indices = 0
    
    # Generate samples for each mode
    for mode, num_mode_samples in samples_per_mode.items():
        mode_ranges_dict = mode_ranges[mode]
        
        for i in range(num_mode_samples):
            if used_indices >= len(sample_points):
                break
                
            # Get the Sobol point
            point = sample_points[used_indices]
            used_indices += 1
            
            # Scale the point to the actual ranges for this mode
            # Apply non-linear scaling to favor certain values within the ranges
            if mode == "DRC":
                # For DRC mode, prefer powers of 2 for drc_weight
                powers = [1, 2, 4, 8, 16, 32, 64]
                drc_idx = int(point[0] * len(powers))
                drc_idx = min(drc_idx, len(powers) - 1)
                drc_weight = powers[drc_idx]
            else:
                # Full range exploration (0-100) as requested
                # Properly format as 2D array for qmc.scale
                drc_weight = int(qmc.scale(np.array([[point[0]]]), 0, 100)[0][0])
            
            # Marker weight with appropriate ranges per mode
            marker_min, marker_max = mode_ranges_dict['marker_weight']
            # Full range exploration (0-100) as requested
            # Properly format as 2D array for qmc.scale
            marker_weight = int(qmc.scale(np.array([[point[1]]]), 0, 100)[0][0])
            
            # Fixed weight with common values plus exploration
            fixed_values = [1, 2, 3, 4, 10, 50, 100]
            if random.random() < 0.7:  # 70% chance to use common values
                fixed_idx = int(point[2] * len(fixed_values))
                fixed_idx = min(fixed_idx, len(fixed_values) - 1)
                fixed_weight = fixed_values[fixed_idx]
            else:  # 30% chance to explore between ranges
                # Full range exploration (0-100) as requested
                # Properly format as 2D array for qmc.scale
                fixed_weight = int(qmc.scale(np.array([[point[2]]]), 0, 100)[0][0])
            
            # Decay weight with typical values
            decay_values = [0.99, 0.995, 0.997, 0.999, 1.0]
            if random.random() < 0.8:  # 80% chance to use common values
                decay_idx = int(point[3] * len(decay_values))
                decay_idx = min(decay_idx, len(decay_values) - 1)
                decay_weight = decay_values[decay_idx]
            else:  # 20% chance to explore between range
                # Full range exploration (0-1) as requested
                # Properly format as 2D array for qmc.scale
                decay_weight = float(qmc.scale(np.array([[point[3]]]), 0, 1.0)[0][0])
                decay_weight = round(decay_weight, 3)  # Round to 3 decimal places
            
            # Create the sample
            sobol_samples.append({
                'drc_weight': drc_weight,
                'marker_weight': marker_weight,
                'fixed_weight': fixed_weight,
                'decay_weight': decay_weight,
                'RipUpMode': mode
            })
    
    # Combine existing and new samples
    all_samples = existing_samples + sobol_samples
    
    # Convert to DataFrame
    result_df = pd.DataFrame(all_samples)
    
    # Ensure we have no duplicates
    result_df = result_df.drop_duplicates()
    
    # If we still don't have enough samples, add some using original method
    if len(result_df) < num_samples:
        print(f"Need {num_samples - len(result_df)} more samples, generating using traditional method")
        # Add more samples using traditional method
        additional_df = generate_random_weights(
            initial_df, 
            initial_df.drop_duplicates().to_dict('records'),
            num_samples - len(result_df), 
            []
        )
        result_df = pd.concat([result_df, additional_df]).drop_duplicates()
    
    # If we have too many samples, take a random subset
    if len(result_df) > num_samples:
        result_df = result_df.sample(num_samples)
    
    return result_df

def visualize_weights(initial_df, perturbed_df, output_dir=None):
    """
    Create visualizations to compare initial and perturbed weights.
    
    Args:
        initial_df: DataFrame with initial weights
        perturbed_df: DataFrame with perturbed weights
        output_dir: Directory to save the plots (defaults to script directory/weight_plots)
    """
    # Set up the visualization
    sns.set(style="whitegrid")
    
    # Create output directory for plots
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "weight_plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add a source column to identify initial vs perturbed
    initial_df_copy = initial_df.copy()
    initial_df_copy['source'] = 'Initial'
    perturbed_df_copy = perturbed_df.copy()
    perturbed_df_copy['source'] = 'Perturbed'
    
    # Combine for comparison
    combined_df = pd.concat([initial_df_copy, perturbed_df_copy])
    
    # Plot distributions for each weight by RipUpMode
    for col in ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']:
        plt.figure(figsize=(10, 6))
        
        for mode in combined_df['RipUpMode'].unique():
            # Create subplot for each RipUpMode
            plt.subplot(len(combined_df['RipUpMode'].unique()), 1, 
                       list(combined_df['RipUpMode'].unique()).index(mode) + 1)
            
            mode_initial = initial_df_copy[initial_df_copy['RipUpMode'] == mode]
            mode_perturbed = perturbed_df_copy[perturbed_df_copy['RipUpMode'] == mode]
            
            # Plot histograms
            if len(mode_initial) > 0:
                sns.histplot(mode_initial[col], color='blue', label='Initial', 
                           alpha=0.5, discrete=(col != 'decay_weight'))
            
            if len(mode_perturbed) > 0:
                sns.histplot(mode_perturbed[col], color='red', label='Perturbed', 
                           alpha=0.5, discrete=(col != 'decay_weight'))
            
            plt.title(f'{mode} - {col}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{col}_distribution.png"))
        plt.close()
    
    # Plot 2D relationships between key weights
    key_pairs = [
        ('drc_weight', 'marker_weight'),
        ('drc_weight', 'fixed_weight'),
        ('marker_weight', 'fixed_weight'),
        ('fixed_weight', 'decay_weight')
    ]
    
    for x_col, y_col in key_pairs:
        plt.figure(figsize=(12, 10))
        
        for mode in combined_df['RipUpMode'].unique():
            plt.subplot(len(combined_df['RipUpMode'].unique()), 1, 
                       list(combined_df['RipUpMode'].unique()).index(mode) + 1)
            
            # Plot initial points
            initial_mode = initial_df_copy[initial_df_copy['RipUpMode'] == mode]
            plt.scatter(initial_mode[x_col], initial_mode[y_col], 
                      color='blue', label='Initial', alpha=0.7)
            
            # Plot perturbed points
            perturbed_mode = perturbed_df_copy[perturbed_df_copy['RipUpMode'] == mode]
            plt.scatter(perturbed_mode[x_col], perturbed_mode[y_col], 
                      color='red', label='Perturbed', alpha=0.7)
            
            plt.title(f'{mode} - {x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{x_col}_vs_{y_col}.png"))
        plt.close()
    
    print(f"Generated visualizations in {output_dir}")

def load_state():
    """Load the current mode and perturbation factor from the state file."""
    global CURRENT_MODE, PERTURBATION_FACTOR, RESET_PERTURBATION
    
    # If we're resetting perturbation, do that regardless of the saved state
    if RESET_PERTURBATION:
        print("Resetting perturbation factor to 1.0")
        PERTURBATION_FACTOR = 1.0
        # We'll save this reset state when save_state is called
        return
    
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                lines = f.readlines()
                
                # Parse mode (first line)
                if len(lines) > 0:
                    saved_mode = lines[0].strip()
                    if saved_mode in ["auto", "manual", "default"]:
                        CURRENT_MODE = saved_mode
                        print(f"Loaded previous mode: {CURRENT_MODE}")
                    else:
                        print(f"Invalid mode in state file: {saved_mode}, using 'auto'")
                
                # Parse perturbation factor (second line)
                if len(lines) > 1:
                    saved_factor = float(lines[1].strip())
                    # Increase the factor for this run if in auto mode
                    if CURRENT_MODE == "auto":
                        # Cap the perturbation factor to prevent extreme values
                        # but still allow it to grow for long runs (150-200)
                        PERTURBATION_FACTOR = min(saved_factor + 1.0, 200.0)
                        print(f"Loaded perturbation factor {saved_factor}, increasing to {PERTURBATION_FACTOR}")
                    else:
                        PERTURBATION_FACTOR = saved_factor
                        print(f"Loaded perturbation factor {PERTURBATION_FACTOR}")
                else:
                    print(f"No perturbation factor found, using default: {PERTURBATION_FACTOR}")
        else:
            print(f"No state file found, using defaults: mode={CURRENT_MODE}, factor={PERTURBATION_FACTOR}")
    except Exception as e:
        print(f"Error loading state: {str(e)}")
        print(f"Using defaults: mode={CURRENT_MODE}, factor={PERTURBATION_FACTOR}")

def save_state():
    """Save the current mode and perturbation factor to the state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            # Write mode on first line
            f.write(f"{CURRENT_MODE}\n")
            # Write perturbation factor on second line
            f.write(f"{PERTURBATION_FACTOR}\n")
        print(f"Saved state: mode={CURRENT_MODE}, factor={PERTURBATION_FACTOR}")
    except Exception as e:
        print(f"Error saving state: {str(e)}")

def main():
    """Main function to generate perturbed weights and save to CSV."""
    parser = argparse.ArgumentParser(description='Generate weights for detailed routing')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of weight combinations to generate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--manual', action='store_true', help='Use manual weights instead of generating them')
    parser.add_argument('--default', action='store_true', help='Use default weights from flow/default_weights.csv')
    parser.add_argument('--auto', action='store_true', help='Force automatic weight generation mode')
    parser.add_argument('--design', type=str, help='Design name for design-specific state tracking')
    parser.add_argument('--reset', action='store_true', help='Reset perturbation factor back to 1.0')
    args = parser.parse_args()
    
    try:
        print(f"Weight generator starting - will save results to {args.output_file}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Set mode based on flags
        global CURRENT_MODE, RESET_PERTURBATION, DESIGN_NAME, STATE_FILE
        
        # Auto-detect design from environment variables or output file path
        global DESIGN_NAME, STATE_FILE
        
        # First try to get design from command line argument
        if args.design:
            DESIGN_NAME = args.design
        else:
            # Try to detect from environment variables
            try:
                # Try to get design from DESIGN_NAME environment variable
                env_design = os.environ.get("DESIGN_NAME", "")
                if env_design:
                    DESIGN_NAME = env_design
                    print(f"Detected design '{DESIGN_NAME}' from DESIGN_NAME environment variable")
                else:
                    # Try to extract from DESIGN_CONFIG environment variable
                    design_config = os.environ.get("DESIGN_CONFIG", "")
                    if design_config:
                        # Format is typically designs/platform/design/config.mk
                        parts = design_config.split('/')
                        if len(parts) >= 3:
                            DESIGN_NAME = parts[-2]  # Extract design name
                            print(f"Detected design '{DESIGN_NAME}' from DESIGN_CONFIG environment variable")
                    else:
                        # Try to extract from the output file path
                        # Typical path structure: /path/to/flow/designs/platform/design/...
                        path_parts = args.output_file.split('/')
                        if 'designs' in path_parts:
                            designs_idx = path_parts.index('designs')
                            if len(path_parts) > designs_idx + 2:
                                DESIGN_NAME = path_parts[designs_idx + 2]
                                print(f"Detected design '{DESIGN_NAME}' from output file path")
            except Exception as e:
                print(f"Warning: Failed to auto-detect design name: {str(e)}")
        
        # If design name was detected, create design-specific state file
        if DESIGN_NAME:
            STATE_FILE = os.path.join(SCRIPT_DIR, f"weight_generator_state_{DESIGN_NAME}.txt")
            print(f"Using design-specific state file for design: {DESIGN_NAME}")
        else:
            print("No design name detected, using default state file")
        
        # Check if reset flag is set
        if args.reset:
            RESET_PERTURBATION = True
            print("Resetting perturbation factor to 1.0")
            
        if args.default:
            CURRENT_MODE = "default"
            RESET_PERTURBATION = True
            print("Setting mode to 'default' and resetting perturbation")
        elif args.manual:
            CURRENT_MODE = "manual"
            print("Setting mode to 'manual'")
        elif args.auto:
            CURRENT_MODE = "auto"
            print("Setting mode to 'auto'")
        else:
            # No flag specified, load previous state
            load_state()
            print(f"No mode flag specified, using previous mode: {CURRENT_MODE}")
            
        # Process based on current mode
        if CURRENT_MODE == "default":
            # For default weights, use the exact path to default_weights.csv
            default_weights_path = "/home/atk331/OpenROAD-flow-scripts/flow/default_weights.csv"
            print(f"Using default weights from {default_weights_path}")
            
            # Load default weights
            weights_df = load_manual_weights(default_weights_path)
            
            # Create output DataFrame with renamed columns
            output_df = weights_df[['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']].copy()
            output_df.columns = ['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost']
            
            # Save to CSV
            output_df.to_csv(args.output_file, index=False, header=False)
            print(f"Saved {len(output_df)} weight combinations to {args.output_file}")
            
        elif CURRENT_MODE == "manual" or USE_MANUAL_WEIGHTS:
            # For manual weights
            print(f"Using manual weights from {MANUAL_WEIGHTS_FILE}")
            
            # Load manual weights
            weights_df = load_manual_weights(MANUAL_WEIGHTS_FILE)
            
            # Create output DataFrame with renamed columns
            output_df = weights_df[['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']].copy()
            output_df.columns = ['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost']
            
            # Save to CSV
            output_df.to_csv(args.output_file, index=False, header=False)
            print(f"Saved {len(output_df)} manual weight combinations to {args.output_file}")
            
        else:  # Auto mode
            # Set random seed if provided
            if args.seed is not None:
                random.seed(args.seed)
                np.random.seed(args.seed)
                print(f"Using random seed: {args.seed}")
            
            print(f"Current perturbation factor: {PERTURBATION_FACTOR}")
            
            # If not using manual weights, proceed with normal generation
            # Load initial weights
            initial_df = load_initial_weights()
            print(f"Loaded {len(initial_df)} initial weight combinations")
            
            # Get unique combinations
            unique_initial = initial_df.drop_duplicates()
            print(f"Found {len(unique_initial)} unique weight combinations")
            
            # Generate perturbed weights
            perturbed_df = generate_perturbed_weights(initial_df, num_samples=args.num_samples)
            print(f"Generated {len(perturbed_df)} perturbed weight combinations")
        
            # Ensure we have exactly the requested number of samples
            if len(perturbed_df) < args.num_samples:
                print(f"Warning: Only generated {len(perturbed_df)} unique samples, adding duplicates")
                # Add duplicates of random samples until we have the requested number
                while len(perturbed_df) < args.num_samples:
                    perturbed_df = pd.concat([perturbed_df, perturbed_df.sample(1)])
            elif len(perturbed_df) > args.num_samples:
                print(f"Warning: Generated {len(perturbed_df)} samples, sampling down to {args.num_samples}")
                perturbed_df = perturbed_df.sample(args.num_samples)
            
            # Ensure integer columns are integers (not floats)
            perturbed_df['drc_weight'] = perturbed_df['drc_weight'].astype(int)
            perturbed_df['marker_weight'] = perturbed_df['marker_weight'].astype(int)
            perturbed_df['fixed_weight'] = perturbed_df['fixed_weight'].astype(int)
            
            # Create output DataFrame with renamed columns and without RipUpMode
            output_df = perturbed_df[['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']].copy()
            output_df.columns = ['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost']
            
            # Save to CSV
            output_df.to_csv(args.output_file, index=False, header=False)
            print(f"Saved {len(output_df)} weight combinations to {args.output_file}")
        
            # Summary of generated weights
            print("\nSummary of generated weights:")
            for mode in perturbed_df['RipUpMode'].unique():
                mode_df = perturbed_df[perturbed_df['RipUpMode'] == mode]
                print(f"\n{mode} mode ({len(mode_df)} samples):")
                for col in ['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']:
                    print(f"  {col}: min={mode_df[col].min()}, max={mode_df[col].max()}, mean={mode_df[col].mean():.2f}")
            
            # Generate visualizations if requested
            if args.visualize:
                output_dir = os.path.dirname(args.output_file)
                visualize_dir = os.path.join(output_dir, "weight_plots") if output_dir else "weight_plots"
                visualize_weights(initial_df, perturbed_df, output_dir=visualize_dir)
        
        # Save the state for next run
        save_state()
        
        # Print information about the state file used
        if DESIGN_NAME:
            print(f"State saved to design-specific file: {STATE_FILE}")
        else:
            print(f"State saved to default file: {STATE_FILE}")
            print("Warning: Using default state file. Future runs will all share the same perturbation factor.")
            print("To use design-specific tracking, set DESIGN_NAME environment variable before running make.")
        
    except Exception as e:
        print(f"Error generating weights: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()