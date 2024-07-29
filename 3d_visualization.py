import pandas as pd
import plotly.express as px

def load_structures(file_path):
    return pd.read_csv(file_path)

def visualize_molecule(structures, molecule_name):
    # Filter the data for the given molecule
    molecule_data = structures[structures["molecule_name"] == molecule_name]
    
    # Create a 3D Scatter Plot
    fig = px.scatter_3d(
        molecule_data,
        x="x",
        y="y",
        z="z",
        color="atom",
        symbol="atom",
        title=f'3D Visualization of Molecule: {molecule_name}',
        labels={'x': 'X Coordinate', 'y': 'Y Coordinate', 'z': 'Z Coordinate'}
    ) 
    
    fig.update_traces(marker=dict(size=8))
    fig.show()
    
# Example usage: visualize a specific molecule
if __name__ == "__main__":
    # Load the structures data
    file_path = "dataset/champs-scalar-coupling/structures.csv"
    structures = load_structures(file_path)
    visualize_molecule(structures, 'dsgdb9nsd_000017')