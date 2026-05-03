import os
import numpy as np

import cigvis

from core.builder import DatasetBuilder
from core.presets import build_default_viz_presets, read_segy_volume
from core.viz_utils import SUBTYPE_COLORS, extract_subtype_labels


def generate_and_plot(anomaly, name_en, name_zh, vp_bg, label_vol, dx, dy, dz, run_app=False):
    builder = DatasetBuilder(dx, dy, dz)
    print(f"\n-> Generating {name_zh}...")
    
    # Inject individually
    vp_final, mask_final, X, Y, Z = builder.inject_anomalies(vp_bg, [anomaly])
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "outputs", "ModelPreview")
    os.makedirs(save_dir, exist_ok=True)
    
    nodes1 = cigvis.create_slices(vp_bg, cmap='jet')
    nodes2 = cigvis.create_slices(vp_final, cmap='jet')
    
    sub_labels = extract_subtype_labels(anomaly, X, Y, Z, mask_final, vp_bg=vp_bg)
    
    # To plot anomaly masks we can add it to the nodes
    if sub_labels is not None and len(np.unique(sub_labels)) > 1:
        for code in np.unique(sub_labels):
            if code == 0: continue
            
            c = SUBTYPE_COLORS.get(code, 'magenta') # Default magenta
            sub_mask = (sub_labels == code).astype(np.float32)
            
            mask_nodes = cigvis.create_bodys(sub_mask, level=0.5, color=c, **{"alpha": 0.3})
            if isinstance(mask_nodes, list):
                nodes2.extend(mask_nodes)
            else:
                nodes2.append(mask_nodes)
            
    # Draw two side-by-side canvases and save
    out_file = f"Anomaly_{name_en}-{name_zh}.png"
    out_path = os.path.join(save_dir, out_file)
    cigvis.plot3D([nodes1, nodes2], grid=(1, 2), savename=out_file, savedir=save_dir, run_app=run_app, title=["原始速度模型", name_zh])
    
    print(f"Saved: {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_app", action="store_true", help="Launch cigvis GUI for checking visuals interactively.")
    parser.add_argument("--vp_segy", type=str, required=True, help="Path to background Vp SEGY.")
    parser.add_argument("--label_segy", type=str, default=None, help="Optional path to background label SEGY.")
    args = parser.parse_args()
    
    vp_bg, (dx, dy, dz) = read_segy_volume(args.vp_segy)
    label_vol, _ = read_segy_volume(args.label_segy)
    nx, ny, nz = vp_bg.shape
    print(f'velocity shape = {nx,ny,nz}, dx={dx}, dy={dy}, dz={dz}')

    for preset in build_default_viz_presets(vp_bg, label_vol, (dx, dy, dz), include_stock=True):
        generate_and_plot(
            preset.anomaly,
            preset.name_en,
            preset.name_zh,
            vp_bg,
            label_vol,
            dx,
            dy,
            dz,
            run_app=args.run_app,
        )


if __name__ == '__main__':
    main()
