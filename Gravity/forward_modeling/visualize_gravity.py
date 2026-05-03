import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize Gravity Forward Modeling Results")
    parser.add_argument("--file_path", type=str, help="Path to the .npz output file")
    parser.add_argument("--height_idx", type=int, default=0, help="Index of height level to plot (default: 0)")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the figure (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File {args.file_path} not found.")
        return

    print(f"Loading {args.file_path}...")
    try:
        data = np.load(args.file_path)
    except Exception as e:
        print(f"Failed to load numpy file: {e}")
        return

    print("Keys in npz:", data.files)
    
    # 'data' shape depends on layout. 
    # Grid: (1, n_heights, n_x, n_y) or similar depending on how it was saved
    gz_data = data['data']
    print(f"Data shape: {gz_data.shape}")
    
    layout = str(data['layout']) if 'layout' in data else 'grid'
    unit = str(data['output_unit']) if 'output_unit' in data else 'mgal'
    heights = data['heights_m'] if 'heights_m' in data else [0]
    
    # Extract meta info for axes if available
    d_x = data['d_x'] if 'd_x' in data else 1
    d_y = data['d_y'] if 'd_y' in data else 1
    # dx/dy might be array, take first elements if so
    if np.ndim(d_x) > 0: d_x = d_x.flatten()[0]
    if np.ndim(d_y) > 0: d_y = d_y.flatten()[0]

    obs_x = data['obs_x_idx'] if 'obs_x_idx' in data else None
    obs_y = data['obs_y_idx'] if 'obs_y_idx' in data else None

    if layout == 'grid':
        # Expecting (1, n_heights, nx, ny)
        if gz_data.ndim == 4:
                gz_data = gz_data[0] # remove batch dim
        
        n_heights = gz_data.shape[0]
        if args.height_idx >= n_heights:
            print(f"Error: height_idx {args.height_idx} out of range (max {n_heights-1})")
            return
        
        grid_slice = gz_data[args.height_idx] # (nx, ny)
        h_val = heights[args.height_idx] if len(heights) > args.height_idx else '?'
        
        plt.figure(figsize=(10, 8))
        
        # Plot with extent if possible
        extent = None
        # if obs_x is not None and obs_y is not None:
        #      # Assume uniform grid for extent calculation
        #      x_min, x_max = obs_x.min(), obs_x.max()
        #      y_min, y_max = obs_y.min(), obs_y.max()
        #      extent = [x_min, x_max, y_min, y_max] # [left, right, bottom, top]
             
        # grid_slice is (nx, ny). imshow expects (rows, cols) -> (ny, nx) for "Y vertical, X horizontal"
        # So we transpose.
        im = plt.imshow(grid_slice.T, cmap='jet', extent=extent)
        cbar = plt.colorbar(im)
        cbar.set_label(f"Gravity Anomaly ({unit})")
        
        plt.title(f"Gravity Anomaly at Height {h_val}m")
        plt.xlabel("X Index" if extent is None else "X Coordinate")
        plt.ylabel("Y Index" if extent is None else "Y Coordinate")
        
        output_file = args.save_path
        if not output_file:
            base_name = os.path.splitext(args.file_path)[0]
            output_file = f"{base_name}_h{args.height_idx}.png"
            
        plt.savefig(output_file)
        print(f"Figure saved to {output_file}")
        
    elif layout == 'points':
        # (1, n_heights, n_points)
        if gz_data.ndim == 3:
             gz_data = gz_data[0]
        
        n_heights = gz_data.shape[0]
        if args.height_idx >= n_heights:
            print(f"Error: height_idx {args.height_idx} out of range (max {n_heights-1})")
            return

        vals = gz_data[args.height_idx] # (n_points,)
        
        if obs_x is not None and obs_y is not None:
            plt.figure(figsize=(10,8))
            sc = plt.scatter(obs_x, obs_y, c=vals, cmap='jet')
            plt.colorbar(sc, label=f"Gravity Anomaly ({unit})")
            plt.title(f"Gravity Anomaly (Points) at Height {heights[args.height_idx]}m")
            
            output_file = args.save_path or f"{os.path.splitext(args.file_path)[0]}_points.png"
            plt.savefig(output_file)
            print(f"Figure saved to {output_file}")
        else:
            print("Cannot plot points layout without obs_x_idx and obs_y_idx metadata.")

    else:
        print(f"Unknown layout: {layout}")

if __name__ == "__main__":
    main()
