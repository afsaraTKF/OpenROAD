#!/usr/bin/env python3
"""
Weight Generator for Circuit Routing

This script generates perturbed weights for circuit routing based on initial weights.
It creates variations around the default weights while respecting the relationships
between different parameters and the constraints of the routing process.

Usage: python weight_generator.py output_file.csv
"""

# ======= CONFIGURATION OPTIONS =======
# Set this to True to use manual weights instead of generating them
USE_MANUAL_WEIGHTS = True

# Path to manual weights file (only used if USE_MANUAL_WEIGHTS is True)
MANUAL_WEIGHTS_FILE = "manual_weights.csv"

# ======= PERTURBATION CONTROL =======
# Controls the amount of perturbation (will increase with each run)
PERTURBATION_FACTOR = 1.0
# Set to True to reset perturbation factor back to 1.0
RESET_PERTURBATION = False
# Path to store perturbation state between runs
PERTURBATION_STATE_FILE = "perturbation_state.txt"
# ====================================

import pandas as pd
import numpy as np
import os
import random
import sys
import argparse
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_perturbation_state():
    """Load the perturbation factor from the state file or use default if not found."""
    global PERTURBATION_FACTOR, RESET_PERTURBATION
    
    if RESET_PERTURBATION:
        print("Resetting perturbation factor to 1.0")
        PERTURBATION_FACTOR = 1.0
        # Save the reset state
        with open(PERTURBATION_STATE_FILE, 'w') as f:
            f.write(str(PERTURBATION_FACTOR))
        return
    
    try:
        if os.path.exists(PERTURBATION_STATE_FILE):
            with open(PERTURBATION_STATE_FILE, 'r') as f:
                saved_factor = float(f.read().strip())
                # Increase the factor for this run
                PERTURBATION_FACTOR = saved_factor + 1.0
                print(f"Loaded perturbation factor {saved_factor}, increasing to {PERTURBATION_FACTOR}")
        else:
            print(f"No perturbation state found, using default factor {PERTURBATION_FACTOR}")
    except Exception as e:
        print(f"Error loading perturbation state: {str(e)}")
        print(f"Using default perturbation factor {PERTURBATION_FACTOR}")

def save_perturbation_state():
    """Save the current perturbation factor to the state file."""
    try:
        with open(PERTURBATION_STATE_FILE, 'w') as f:
            f.write(str(PERTURBATION_FACTOR))
        print(f"Saved perturbation factor {PERTURBATION_FACTOR} for next run")
    except Exception as e:
        print(f"Error saving perturbation state: {str(e)}")

def main():
    """Main function to generate perturbed weights and save to CSV."""
    parser = argparse.ArgumentParser(description='Generate weights for detailed routing')
    parser.add_argument('output_file', help='Output CSV file path')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of weight combinations to generate')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    try:
        # Load and update perturbation state
        load_perturbation_state()
        
        # Set random seed if provided
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            
        print(f"Weight generator starting - will save results to {args.output_file}")
        print(f"Current perturbation factor: {PERTURBATION_FACTOR}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if we're using manual weights (based on the flag at the top of the file)
        if USE_MANUAL_WEIGHTS:
            print(f"Using manual weights from {MANUAL_WEIGHTS_FILE}")
            # Load manual weights
            weights_df = load_manual_weights(MANUAL_WEIGHTS_FILE)
            
            # Create output DataFrame with renamed columns
            output_df = weights_df[['drc_weight', 'marker_weight', 'fixed_weight', 'decay_weight']].copy()
            output_df.columns = ['drc_cost', 'marker_cost', 'fixed_cost', 'decay_cost']
            
            # Save to CSV
            output_df.to_csv(args.output_file, index=False, header=False)
            print(f"Saved {len(output_df)} manual weight combinations to {args.output_file}")
            
            # Save the perturbation state for next run
            save_perturbation_state()
            
            # Exit since we've done what was requested
            return
        
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
        
        # Save the perturbation state for next run
        save_perturbation_state()
        
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
        
    except Exception as e:
        print(f"Error generating weights: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()