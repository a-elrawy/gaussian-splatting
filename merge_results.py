import os
import json

# Define the datasets and their respective scenes
datasets = {
    "mipnerf360": ["bicycle", "flowers", "garden", "stump", "treehill", "room", "counter", "kitchen", "bonsai"],
    # "mipnerf360_indoor": ["room", "counter", "kitchen", "bonsai"],
    "tanks_and_temples": ["truck", "train"],
    "deep_blending": ["drjohnson", "playroom"]
}

# Define the experiment configurations
experiment_configs = [
    ("random", 10, 0, "full"),
    ("random", 20, 0, "full"),
    ("random", 30, 0, "full"),
    ("random", 40, 0, "full"),
    ("random", 50, 0, "full"),
    ("random", 60, 0, "full"),
    ("random", 70, 0, "full"),
    ("random", 80, 0, "full"),
    ("random", 90, 0, "full"),
    ("random", 100, 0, "full"),
    ("random", 120, 0, "full"),
    ("random", 140, 0, "full"),
    # ("random", 160, 0, "full")
    # ("structured", 20, 360, "full"),
    # # ("structured", 50, 360, "full"),
    # ("structured", 100, 360, "full")
]

# Prepare to collect results
results = []

for config in experiment_configs:
    sampling_type, num_views, angular_coverage, range_type = config
    exp_dir = f"{sampling_type}_{num_views}_{angular_coverage}_{range_type}"
    exp_metrics = {}

    for dataset_name, scenes in datasets.items():
        ssim_values = []
        psnr_values = []
        lpips_values = []

        for scene in scenes:
            scene_path = os.path.join("experiments", exp_dir, scene, "results.json")
            if not os.path.exists(scene_path):
                print(f"Warning: File {scene_path} does not exist. Skipping.")
                continue

            with open(scene_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {scene_path}. Skipping.")
                    continue

            # Extract metrics for 'ours_30000'
            if "ours_30000" not in data:
                print(f"Warning: 'ours_30000' not found in {scene_path}. Skipping.")
                continue

            metrics = data["ours_30000"]
            ssim = metrics.get("SSIM", 0)
            psnr = metrics.get("PSNR", 0)
            lpips = metrics.get("LPIPS", 0)

            ssim_values.append(ssim)
            psnr_values.append(psnr)
            lpips_values.append(lpips)

        # Calculate averages for the current dataset
        avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
        avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
        avg_lpips = sum(lpips_values) / len(lpips_values) if lpips_values else 0

        exp_metrics[dataset_name] = {
            "SSIM": avg_ssim,
            "PSNR": avg_psnr,
            "LPIPS": avg_lpips
        }

    results.append({
        "experiment": exp_dir,
        "metrics": exp_metrics
    })

# Output the results in CSV format
print("Experiment,Dataset,SSIM,PSNR,LPIPS")
for entry in results:
    exp_name = entry["experiment"]
    metrics = entry["metrics"]
    for dataset_name in datasets:
        dataset_metrics = metrics.get(dataset_name, {})
        line = f"{exp_name},{dataset_name},{dataset_metrics.get('SSIM', 0):.2f},{dataset_metrics.get('PSNR', 0):.2f},{dataset_metrics.get('LPIPS', 0):.2f}"
        print(line)