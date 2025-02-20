import pandas as pd

# Load the CSV files into DataFrames
df1 = pd.read_csv('cox15_output.csv')
df2 = pd.read_csv('wildtype_fts.csv')

# Specify the column(s) to sort by; replace 'id' with your unique identifier column
sort_column = 'id'

# Sort both DataFrames by the specified column(s)
df1_sorted = df1.sort_values(by=sort_column).reset_index(drop=True)
df2_sorted = df2.sort_values(by=sort_column).reset_index(drop=True)

# Get the set of columns in each DataFrame
columns_df1 = set(df1.columns)
columns_df2 = set(df2.columns)

# Find columns present in one DataFrame but not the other
unique_to_df1 = columns_df1 - columns_df2
unique_to_df2 = columns_df2 - columns_df1

if unique_to_df1:
    print(f"Columns unique to file1.csv: {unique_to_df1}")
if unique_to_df2:
    print(f"Columns unique to file2.csv: {unique_to_df2}")

# Check if the DataFrames have the same shape
if df1_sorted.shape != df2_sorted.shape:
    print(f"The CSV files have different shapes. {df1_sorted.shape} vs {df2_sorted.shape}.")
else:
    # Compare the DataFrames
    comparison_result = df1_sorted.compare(df2_sorted)

    # Check if there are any differences
    if comparison_result.empty:
        print("The CSV files are identical.")
    else:
        print("Differences found:")
        print(comparison_result)

        # Identify columns with differences
        differing_columns = comparison_result.columns.get_level_values(0).unique()
        print("\nColumns with differences:")
        print(differing_columns.tolist())
