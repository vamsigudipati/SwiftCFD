def plot_scatter(model, dataX, dataY, device="cpu",
                 title_prefix="SwiftCFD", save_path=None,
                 save_individual=False, out_dir="."):
    """
    Scatter plot of predicted vs ground truth values across the
    entire test set.
    - save_path      → saves the combined 3-panel figure
    - save_individual → additionally saves scatter_ux.png,
                        scatter_uy.png, scatter_p.png separately
    """
    field_names  = ["Ux (x-velocity)", "Uy (y-velocity)", "p (pressure)"]
    field_units  = ["m/s", "m/s", "Pa"]
    field_tags   = ["ux", "uy", "p"]
    field_colors = ["#1565C0", "#1565C0", "#E65100"]

    test_x, test_y = get_test_split(dataX, dataY)
    loader = DataLoader(TensorDataset(test_x, test_y),
                        batch_size=32, shuffle=False)

    all_gt   = [[], [], []]
    all_pred = [[], [], []]

    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            pb = model(xb.to(device)).cpu()
            for i in range(3):
                all_gt[i].append(yb[:, i].numpy().flatten())
                all_pred[i].append(pb[:, i].numpy().flatten())

    all_gt   = [np.concatenate(v) for v in all_gt]
    all_pred = [np.concatenate(v) for v in all_pred]

    # ── Precompute stats for all fields ────────────────────────
    stats = []
    for i in range(3):
        gt_flat   = all_gt[i]
        pred_flat = all_pred[i]
        if len(gt_flat) > 200_000:
            idx = np.random.choice(len(gt_flat), 200_000, replace=False)
            gt_flat   = gt_flat[idx]
            pred_flat = pred_flat[idx]
        r2    = r2_score(gt_flat, pred_flat)
        coeff = np.polyfit(gt_flat, pred_flat, 1)
        stats.append((gt_flat, pred_flat, r2, coeff))

    # ── Combined 3-panel figure ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{title_prefix} — Predicted vs CFD Ground Truth (Test Set)",
                 fontsize=14, fontweight="bold")

    for i, (ax, name, unit, color) in enumerate(
            zip(axes, field_names, field_units, field_colors)):
        gt_flat, pred_flat, r2, coeff = stats[i]
        xline = np.linspace(gt_flat.min(), gt_flat.max(), 200)
        ax.scatter(gt_flat, pred_flat, s=2, alpha=0.15, color=color,
                   label="Test predictions")
        ax.plot(xline, xline, "k--", lw=1.5, label="Perfect (y = x)")
        ax.plot(xline, np.polyval(coeff, xline), "r-", lw=2,
                label=f"Fit: y = {coeff[0]:.3f}x + {coeff[1]:.4f}")
        ax.set_title(f"{name}\n$R^2$ = **{r2:.4f}** | slope = {coeff[0]:.4f}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(f"CFD Ground Truth ({unit})")
        ax.set_ylabel(f"{title_prefix} Prediction ({unit})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved combined → {save_path}")
    plt.show()
    plt.close()

    # ── Individual per-field figures ────────────────────────────
    if save_individual:
        for i, (name, unit, tag, color) in enumerate(
                zip(field_names, field_units, field_tags, field_colors)):
            gt_flat, pred_flat, r2, coeff = stats[i]
            xline = np.linspace(gt_flat.min(), gt_flat.max(), 200)

            fig, ax = plt.subplots(figsize=(7, 6))
            fig.suptitle(
                f"{title_prefix} — Predicted vs CFD Ground Truth\n{name}",
                fontsize=13, fontweight="bold")

            ax.scatter(gt_flat, pred_flat, s=2, alpha=0.15, color=color,
                       label="Test predictions")
            ax.plot(xline, xline, "k--", lw=1.5, label="Perfect (y = x)")
            ax.plot(xline, np.polyval(coeff, xline), "r-", lw=2,
                    label=f"Fit: y = {coeff[0]:.3f}x + {coeff[1]:.4f}")
            ax.set_title(f"$R^2$ = {r2:.4f}  |  slope = {coeff[0]:.4f}",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel(f"CFD Ground Truth ({unit})", fontsize=11)
            ax.set_ylabel(f"{title_prefix} Prediction ({unit})", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            ind_path = f"{out_dir}/scatter_{tag}.png"
            plt.savefig(ind_path, dpi=150, bbox_inches="tight")
            print(f"✅ Saved individual → {ind_path}")
            plt.show()
            plt.close()