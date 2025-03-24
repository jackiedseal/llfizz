import argparse
import pandas as pd
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform UMAP on protein feature embeddings.")
    parser.add_argument("input_file", type=str, help="Path to the CSV file containing protein feature embeddings.")
    parser.add_argument("embedding", type=str, help="Type of embeddings to use (ie. one of 'original', 'LLPhyScore', 'hybrid').")
    parser.add_argument("output_png", type=str, help="Path and name of the output PNG file for the plot.")
    return parser.parse_args()

embedding_to_column = {
    'original': (1, 127),
    'LLPhyScore': (127, 143),
    'hybrid': (1, 143)
}

def main():
    args = parse_arguments()

    df = pd.read_csv(args.input_file)

    column_range = embedding_to_column[args.embedding]
    features = df.iloc[:, column_range[0]:column_range[1]]
    print(features.columns)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply UMAP to reduce the features to 2 dimensions
    umap_model = umap.UMAP(n_components=2, random_state=42) 
    umap_embeddings = umap_model.fit_transform(features_scaled)

    df['UMAP1'] = umap_embeddings[:, 0]
    df['UMAP2'] = umap_embeddings[:, 1]

    sns.set(style="whitegrid")

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='localization', data=df, palette="Set1", s=20, marker="o", alpha=0.5)

    plt.title(f"UMAP projection of {args.embedding} features embedding of DeepLoc training sequences", fontsize=14)
    plt.xlabel("UMAP1", fontsize=14)
    plt.ylabel("UMAP2", fontsize=14)
    plt.legend(title='Localization', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(args.output_png, bbox_inches='tight')

    fig = px.scatter(df, x='UMAP1', y='UMAP2', color='localization', 
                     title=f"UMAP projection of {args.embedding} features embedding of DeepLoc training sequences",
                     labels={'UMAP1': 'UMAP1', 'UMAP2': 'UMAP2'},
                     color_continuous_scale='Set1')  
    fig.show()


if __name__ == "__main__":
    main()
