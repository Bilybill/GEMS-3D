import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
import cigvis

from core.builder import DatasetBuilder
from core.presets import build_default_viz_presets, read_segy_volume

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vp_segy", type=str, required=True, help="Path to background Vp SEGY.")
    parser.add_argument("--label_segy", type=str, default=None, help="Optional path to background label SEGY.")
    args = parser.parse_args()

    print("=== Small-Scale Visualization Verification with all anomalies ===")
    
    vp_bg, (dx, dy, dz) = read_segy_volume(args.vp_segy)
    label_vol, _ = read_segy_volume(args.label_segy)
    print(f"Volume loaded with shape: {vp_bg.shape}")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "outputs", "ModelPreview")
    os.makedirs(save_dir, exist_ok=True)

    builder = DatasetBuilder(dx, dy, dz)
    presets = build_default_viz_presets(vp_bg, label_vol, (dx, dy, dz), include_stock=True)
    anomalies = []
    for idx, preset in enumerate(presets, start=1):
        print(f"--- {idx}. Adding {preset.name_en} ({preset.name_zh}) ---")
        anomalies.append(preset.anomaly)

    print("-> Injecting anomalies into background Vp...")
    vp_final, mask_final, _, _, _ = builder.inject_anomalies(vp_bg, anomalies)

    vp_save_path = os.path.join(save_dir, "vp_all_anomalies.npy")
    np.save(vp_save_path, vp_final)
    print(f"Saved {vp_save_path} successfully.")

    print("-> Start visualization using cigvis (Saving to png)...")
    try:
        # Transpose to Match CigVis X,Y,Z expected formats if needed, or directly pass.
        # usually (X, Y, Z) is what segyio.tools.cube returns and cigvis expects, 
        # but cigvis docs use segy shape. Let's see visulize.py logic:
        # (NZ, NY, NX) -> transposes (2,1,0). But if SEGY cube is already INLINE(x), CROSSLINE(y), TIME(z),
        # shape is (NX, NY, NZ), we can pass directly.
        # We will use center coordinates for slicing
        pos_vol = [vp_final.shape[0]//2, vp_final.shape[1]//2, vp_final.shape[2]//2]
        
        slices_bg = cigvis.create_slices(vp_bg, cmap='jet', pos=pos_vol, vmin=vp_bg.min(), vmax=vp_bg.max())
        slices_final = cigvis.create_slices(vp_final, cmap='jet', pos=pos_vol, vmin=vp_final.min(), vmax=vp_final.max())
        
        cbar_bg = cigvis.create_colorbar(cmap='jet', clim=[vp_bg.min(), vp_bg.max()], label_str='Background Vp (m/s)')
        cbar_final = cigvis.create_colorbar(cmap='jet', clim=[vp_final.min(), vp_final.max()], label_str='Final Anomaly Vp (m/s)')
        
        # 提取各个异常体的掩码 (用于 cigvis 高亮显示)
        # mask_final records the index of applied anomalies (1 for first, 2 for second, etc.)
        anomaly_bodies = []
        colors = ['red', 'yellow', 'cyan', 'magenta', 'green', 'orange', 'purple', 'blue']
        for i, anomaly in enumerate(anomalies):
            idx = i + 1
            # 取出当前异常体的布尔掩码
            sub_mask = (mask_final == idx).astype(np.float32)
            if sub_mask.sum() > 0:
                c = colors[i % len(colors)]
                # 使用 cigvis 创建该异常体的 3D 轮廓体
                anomaly_bodies += cigvis.create_bodys(sub_mask, level=0.5, color=c, alpha=0.3)
                print(f"Adding anomaly outline for: {anomaly.type} (cigvis body, color={c})")

        # Plot and save
        try:
            # We already have run_app=False, but cigvis/vispy might crash if no OpenGL Context exists in pure headless Linux.
            nodes1 = slices_bg + [cbar_bg]
            nodes2 = slices_final + anomaly_bodies + [cbar_final]
            cigvis.plot3D(
                [nodes1, nodes2],
                grid=(1, 2),
                share=True,
                size=(1200, 600),
                savename="3D_Anomalies_Preview.png",
                savedir=save_dir,
                run_app=False,
            )
        except Exception as vis_e:
            print(f"cigvis plot3D directly failed ({vis_e}), falling back to matplotlib 3D slice plotting...")
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(16, 8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')
            
            x_mid, y_mid, z_mid = pos_vol
            
            def plot_3d_slices(ax, vol, title, with_anomalies=False):
                nx, ny, nz = vol.shape
                x, y, z = np.arange(nx), np.arange(ny), np.arange(nz)
                cmap_m = plt.get_cmap('jet')
                norm = plt.Normalize(vmin=vol.min(), vmax=vol.max())
                
                stride = max(1, nx // 64)  # downsample for plotting speed so it does not hang

                # Z slice (horizontal) - Move to the bottom of the cube
                z_bottom = nz - 1
                yy, xx = np.meshgrid(y, x)
                zz = np.full_like(xx, z_bottom)
                ax.plot_surface(xx, yy, zz, facecolors=cmap_m(norm(vol[:, :, z_bottom])), rstride=stride, cstride=stride, shade=False, zorder=1)

                # X slice (vertical) - X slice lies on YZ plane
                zz_x, yy_x = np.meshgrid(z, y)
                xx_x = np.full_like(yy_x, x_mid)
                ax.plot_surface(xx_x, yy_x, zz_x, facecolors=cmap_m(norm(vol[x_mid, :, :])), rstride=stride, cstride=stride, shade=False, zorder=2)

                # Y slice (vertical) - Y slice lies on XZ plane
                zz_y, xx_y = np.meshgrid(z, x)
                yy_y = np.full_like(xx_y, y_mid)
                ax.plot_surface(xx_y, yy_y, zz_y, facecolors=cmap_m(norm(vol[:, y_mid, :])), rstride=stride, cstride=stride, shade=False, zorder=3)

                # 如果带有异常，我们在表面上方叠加半透明散点来勾勒出异常体形态
                if with_anomalies:
                    colors = ['red', 'yellow', 'cyan', 'magenta', 'green', 'orange', 'purple', 'blue']
                    for i in range(len(anomalies)):
                        idx = i + 1
                        pts = np.where(mask_final == idx)
                        if len(pts[0]) > 0:
                            # 降采样散点以防止卡死
                            skip = stride * 3
                            pts_ds = (pts[0][::skip], pts[1][::skip], pts[2][::skip])
                            c = colors[i % len(colors)]
                            ax.scatter(pts_ds[0], pts_ds[1], pts_ds[2], c=c, s=1, alpha=0.1, label=anomalies[i].type, zorder=4)
                    
                    # 取消注释以下代码将显示图例，但可能遮挡图形
                    # ax.legend(loc='upper right', fontsize='small')

                ax.set_title(title)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Depth (Z)')
                ax.invert_zaxis() # Invert Z axis so depth goes down
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap_m, norm=norm)
                sm.set_array([])
                plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label='Vp (m/s)')

            print("Rendering 3D slices with Matplotlib (this may take a few seconds)...")
            plot_3d_slices(ax1, vp_bg, "Background Vp Slices", with_anomalies=False)
            plot_3d_slices(ax2, vp_final, "Final Anomaly Vp Slices", with_anomalies=True)
            
            plt.tight_layout()
            fallback_path = os.path.join(save_dir, "3D_Slices_Matplotlib.png")
            plt.savefig(fallback_path, dpi=150)
            print(f"Done! Fallback 3D visualization saved to '{fallback_path}'")

        print(f"Done! Visualization saved to '{os.path.join(save_dir, '3D_Anomalies_Preview.png')}'")
        
    except Exception as e:
        print(f"Failed to visualize: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
