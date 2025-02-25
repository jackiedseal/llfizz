import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_side_by_side_bar(output_file, *csv_files):
    if len(csv_files) < 2:
        print("Error: At least two CSV files are required.")
        sys.exit(1)
    
    # Read CSV files and extract Experiment names
    dataframes = []
    experiment_names = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dataframes.append(df)
        # Extract the 'Experiment' name (assuming it's in the first column)
        experiment_names.append(df['experiment'].iloc[0])

    # Ensure column names match
    prob_columns = dataframes[0].columns[4:]
    for df in dataframes[1:]:
        if not np.array_equal(prob_columns, df.columns[4:]):
            print("Error: Column names in all CSV files must match.")
            sys.exit(1)
    
    # Compute average probabilities and standard deviations
    avg_probs = [df[prob_columns].mean() for df in dataframes]
    std_devs = [df[prob_columns].std() for df in dataframes]
    
    # Define bar positions
    x = np.arange(len(prob_columns))
    width = 0.8 / len(csv_files)  # Adjust width based on number of files
    
    # Plot side-by-side bar chart with error bars
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    for i, (avg, std) in enumerate(zip(avg_probs, std_devs)):
        plt.bar(x + (i - len(csv_files) / 2) * width, avg, width, yerr=std, capsize=5, 
                label=f'{experiment_names[i]}', color=colors[i % len(colors)])
    
    plt.xlabel("Localization")
    plt.ylabel("Average Probability")
    plt.title("DeepLoc predicted localization by feature mimic design approach")
    plt.xticks(x, prob_columns, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot to a file
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <output_file> <csv_file1> <csv_file2> [<csv_file3> ...]")
        sys.exit(1)
    
    output_file = sys.argv[1]
    csv_files = sys.argv[2:]
    plot_side_by_side_bar(output_file, *csv_files)
