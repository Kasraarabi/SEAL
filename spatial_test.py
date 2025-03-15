import numpy as np
from scipy.stats import mannwhitneyu
from skimage import measure, filters
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def compute_advanced_statistics(data):
    # Normalize data to 0-1 range
    data_norm = (data - data.min()) / (data.max() - data.min())
    
    # 1. Multi-scale cluster analysis
    thresholds = [60, 70, 80, 90]  # Multiple percentile thresholds
    cluster_features = []
    
    for thresh in thresholds:
        binary = data_norm > np.percentile(data_norm, thresh)
        

        labels = measure.label(binary, connectivity=2)
        
        if labels.max() > 0:
            regions = measure.regionprops(labels)
            # Sort regions by area in descending order
            regions.sort(key=lambda x: x.area, reverse=True)
            
            # Features from largest cluster
            largest = regions[0]
            cluster_features.extend([
                largest.area,
                largest.perimeter,
                largest.eccentricity,
                largest.solidity
            ])
            
            # Features from all clusters
            total_area = sum(r.area for r in regions)
            n_clusters = len(regions)
            
            cluster_features.extend([
                total_area,
                n_clusters,
                total_area/n_clusters if n_clusters > 0 else 0
            ])
        else:
            cluster_features.extend([0] * 7)  # Padding for no clusters
    
    # 2. Gradient-based features
    gx, gy = np.gradient(data_norm)
    gradient_mag = np.sqrt(gx**2 + gy**2)
    
    gradient_features = [
        np.mean(gradient_mag),
        np.std(gradient_mag),
        np.percentile(gradient_mag, 90),
        np.sum(gradient_mag > np.mean(gradient_mag))
    ]
    
    # 3. Local variance features
    from scipy.ndimage import generic_filter
    local_var = generic_filter(data_norm, np.var, size=3)
    
    variance_features = [
        np.mean(local_var),
        np.std(local_var),
        np.percentile(local_var, 90),
        np.sum(local_var > np.mean(local_var))
    ]
    
    # 4. Spatial continuity features
    continuity_score = np.sum(data_norm[1:, :] * data_norm[:-1, :]) + \
                      np.sum(data_norm[:, 1:] * data_norm[:, :-1])
    
    # Combine all features
    all_features = {
        f'cluster_{i}': v for i, v in enumerate(cluster_features)
    }
    all_features.update({
        f'gradient_{i}': v for i, v in enumerate(gradient_features)
    })
    all_features.update({
        f'variance_{i}': v for i, v in enumerate(variance_features)
    })
    all_features['continuity'] = continuity_score
    
    return all_features

def plot_roc_curves(metrics_data, save_path='roc_curves.png'):
    plt.figure(figsize=(10, 8))
    
    # Sort metrics by AUC for better visualization
    sorted_metrics = {}
    for metric_name, (y_true, y_scores) in metrics_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        sorted_metrics[metric_name] = (y_true, y_scores, roc_auc)
    
    # Plot in order of decreasing AUC
    for metric_name, (y_true, y_scores, roc_auc) in sorted(
        sorted_metrics.items(), key=lambda x: x[1][2], reverse=True)[:5]:  # Show top 5
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plt.plot(fpr, tpr, label=f'{metric_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Top 5 Metrics')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Load datasets
    with_cat = None
    without_cat = None

    if with_cat is None or without_cat is None:
        raise ValueError("Please collect the data from the cat_attack_patch_search.py first and fill with_cat and without_cat with the data.")
    
    with_cat_stats = []
    without_cat_stats = []
    
    # Process images
    print("Processing with_cat images...")
    for key in with_cat.files:
        stats = compute_advanced_statistics(with_cat[key].reshape(32, 32))
        with_cat_stats.append(stats)
    
    print("Processing without_cat images...")
    for key in without_cat.files:
        stats = compute_advanced_statistics(without_cat[key].reshape(32, 32))
        without_cat_stats.append(stats)
    
    # Prepare data for ROC curves
    metrics_data = {}
    
    # Get all metric names from the first stats dictionary
    metrics = list(with_cat_stats[0].keys())
    
    print("\nROC-AUC Results:")
    print("-" * 50)
    
    # Calculate ROC-AUC for each metric
    for metric in metrics:
        with_cat_values = [s[metric] for s in with_cat_stats]
        without_cat_values = [s[metric] for s in without_cat_stats]
        
        y_true = np.array([1] * len(with_cat_values) + [0] * len(without_cat_values))
        y_scores = np.array(with_cat_values + without_cat_values)
        

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        metrics_data[metric] = (y_true, y_scores)
        
        # Only print top performing metrics
        if roc_auc > 0.7:
            print(f"\nMetric: {metric}")
            print(f"ROC-AUC Score: {roc_auc:.3f}")
    
    # Plot ROC curves
    plot_roc_curves(metrics_data)
    print("\nROC curves have been saved as 'roc_curves.png'")

if __name__ == "__main__":
    main()