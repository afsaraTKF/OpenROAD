import csv
import random
import math
import datetime
import sys
import os
import argparse
import traceback

def generate_initial_weights():
    """Generate random initial weights instead of using fixed values"""
    initial_weights = []
    
    # Generate 100 different weight combinations
    for _ in range(100):
        drc_weight = random.randint(0, 100)  
        marker_weight = random.randint(0, 100) 
        fixed_weight = random.randint(0, 100)  
        decay_weight = random.uniform(0, 1.0)  # Decay weight between 0-1.0
        
        initial_weights.append((drc_weight, marker_weight, fixed_weight, decay_weight))
    
    return initial_weights

def generate_weights(initial_weights):
    num_rows = len(initial_weights)
    num_cols = len(initial_weights[0])
    weights = [initial_weights[0]]

    # Increase base jump and randomness for more diversity
    base_jump = 0.5
    jump_randomness = 0.4
    acceleration = 0.05
    large_jump_prob = 0.3
    large_jump_factor = 5.0

    # Wider range of integer jumps
    small_int_jump = [-10, -5, 0, 5, 10, 15]  # Reduced negative jumps
    medium_int_jump = [-15, -10, -5, 0, 5, 10, 15, 20, 25]  # More positive jumps
    large_int_jump = [-25, -15, -10, 0, 10, 20, 30, 40]  # Biased towards positive

    # Higher change probability for more frequent changes
    change_prob = 0.9

    for i in range(1, num_rows):
        new_weights = list(weights[-1])
        max_jump = base_jump + random.uniform(0, jump_randomness)
        max_jump += acceleration * i
        
        if random.random() < large_jump_prob:
            max_jump *= large_jump_factor
            if random.random() < 0.3:
                max_jump *= 1.5

        for j in range(num_cols):
            if random.random() < change_prob:
                if j < 3:  # Integer weights (0-100)
                    if i < num_rows // 3:
                        possible_jumps = small_int_jump
                    elif i < 2 * num_rows // 3:
                        possible_jumps = medium_int_jump
                    else:
                        possible_jumps = large_int_jump
                        
                    # Special handling for marker cost (j == 1)
                    if j == 1:  # Marker weight
                        current_weight = new_weights[j]
                        
                        # If marker weight is low, bias towards increase
                        if current_weight < 15:
                            if random.random() < 0.8:  # 80% chance to increase
                                jump = random.randint(5, 20)  # Force positive jump
                            else:
                                jump = random.randint(-5, 10)
                        else:
                            # Normal weight generation logic
                            if random.random() < large_jump_prob:
                                jump = random.randint(-30, 30)
                            else:
                                jump = random.choice(possible_jumps)
                                # Add some extra randomness
                                jump += random.randint(-5, 5)
                    else:
                        # Normal weight generation logic
                        if random.random() < large_jump_prob:
                            jump = random.randint(-50, 50)
                        else:
                            jump = random.choice(possible_jumps)
                            jump += random.randint(-5, 5)

                    new_weights[j] += jump

                else:  # Float weights (0-1)
                    if random.random() < large_jump_prob:
                        jump = random.uniform(-0.5, 0.5)
                    else:
                        jump = random.uniform(-max_jump, max_jump)
                    
                    new_weights[j] += jump

                # Clamp values to valid ranges
                if j < 3:  # Integer weights
                    new_weights[j] = max(0, min(100, int(round(new_weights[j]))))
                else:  # Float weights
                    new_weights[j] = max(0.0, min(1.0, new_weights[j]))

        # Ensure marker cost doesn't stay too low for consecutive iterations
        if i > 1 and weights[-1][1] <= 10 and weights[-2][1] <= 10:  # If marker cost has been low for 2 iterations
            new_weights[1] = random.randint(20, 40)  # Force it higher

        weights.append(tuple(new_weights))

    return weights


initial_weights = [
    (1, 0, 1, 1.0),
    (1, 1, 1, 1.0),
    (1, 1, 1, 1.0), 
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 4, 1.0),
    (2, 1, 4, 1.0),
    (1, 1, 4, 1.0),
    (4, 1, 4, 1.0),
    (4, 1, 4, 1.0),
    (4, 1, 10, 1.0),
    (4, 1, 10, 1.0),
    (4, 1, 10, 1.0),
    (1, 1, 10, 1.0),
    (4, 1, 10, 1.0),
    (1, 1, 10, 1.0),
    (8, 2, 10, 1.0),
    (8, 2, 10, 1.0),
    (8, 2, 10, 1.0),
    (8, 2, 10, 1.0),
    (1, 1, 50, 1.0),
    (8, 2, 50, 1.0),
    (8, 2, 50, 1.0),
    (1, 1, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 50, 1.0),
    (1, 1, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 100, 1.0),
    (1, 1, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (1, 1, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (1, 1, 100, 1.0),
    (32, 8, 100, 1.0),
    (1, 1, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (1, 1, 100, 1.0),
    (1, 1, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0) 
]

# Direct weights for our experiment
our_weights = [
    (1, 0, 1, 1.0),      # iteration 0
    (1, 1, 1, 1.0),      # iteration 1 - default weights
    (2, 2, 2, 1.8084736),  # iteration 2 - RL weights
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (1, 1, 2, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 3, 1.0),
    (2, 1, 4, 1.0),
    (2, 1, 4, 1.0),
    (1, 1, 4, 1.0),
    (4, 1, 4, 1.0),
    (4, 1, 4, 1.0),
    (4, 1, 10, 1.0),
    (4, 1, 10, 1.0),
    (4, 1, 10, 1.0),
    (1, 1, 10, 1.0),
    (4, 1, 10, 1.0),
    (1, 1, 10, 1.0),
    (8, 2, 10, 1.0),
    (8, 2, 10, 1.0),
    (8, 2, 10, 1.0),
    (8, 2, 10, 1.0),
    (1, 1, 50, 1.0),
    (8, 2, 50, 1.0),
    (8, 2, 50, 1.0),
    (1, 1, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 50, 1.0),
    (1, 1, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 50, 1.0),
    (16, 4, 100, 1.0),
    (1, 1, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (1, 1, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (16, 4, 100, 1.0),
    (1, 1, 100, 1.0),
    (32, 8, 100, 1.0),
    (1, 1, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (32, 8, 100, 1.0),
    (1, 1, 100, 1.0),
    (1, 1, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0),
    (64, 16, 100, 1.0) 
]

# Validate and convert weights to ensure they match WeightMultipliers struct
validated_weights = []
for i, (drc, marker, fixed, decay) in enumerate(our_weights):
    # Ensure first three fields are non-negative integers
    drc = max(0, int(drc))
    marker = max(0, int(marker))
    fixed = max(0, int(fixed))
    # Keep decay as float
    validated_weights.append((drc, marker, fixed, float(decay)))

our_weights = validated_weights

def main():
    # Get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', help='Output file path')
    args = parser.parse_args()

    print("\n=== Debug Information ===")
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Output file (as passed): {args.output_file}")
    print(f"Output file (absolute): {os.path.abspath(args.output_file)}")
    print(f"Output directory exists: {os.path.exists(os.path.dirname(args.output_file))}")
    print(f"Output directory is writable: {os.access(os.path.dirname(args.output_file), os.W_OK)}")
    print("=======================\n")
    
    # Generate weights based on initial weights list
    weights = generate_weights(our_weights)  # Use our_weights instead of generate_initial_weights

    print(f"\n=== Writing Weights ===")
    print(f"About to write {len(weights)} weights")
    print(f"First few weights: {weights[:3]}")
    print(f"Last few weights: {weights[-3:]}")

    # Write to the specified output file only once
    try:
        with open(args.output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(weights)
            print(f"Successfully wrote weights to: {args.output_file}")
    except Exception as e:
        print(f"ERROR writing to {args.output_file}: {str(e)}")
        print(f"Stack trace: {traceback.format_exc()}")
        sys.exit(1)

    # Verify the file was written
    if os.path.exists(args.output_file):
        print(f"Verified: File exists at {args.output_file}")
        print(f"File size: {os.path.getsize(args.output_file)} bytes")
    else:
        print(f"WARNING: File does not exist at {args.output_file}")

if __name__ == "__main__":
    main()
