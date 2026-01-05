import pandas as pd
from sklearn.preprocessing import StandardScaler

# 导入 UMAP
import umap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import os
import traceback


def run_umap_kmeans_analysis(k_final,umap_n_components_cluster,n_neighbors,min_dist):
    """
    基于 UMAP (cosine) + K-Means 的聚类分析:
    - 目标: 基于 'events_get_all_features.csv' (每场比赛每队) 的数据，对 '球队' 进行聚类。
    - 步骤 1: 将比赛数据按 team_id 聚合为球队平均特征。
    - 步骤 2 (修改): 使用 UMAP (metric='cosine') 降维用于聚类 (n_components=10) 和可视化 (n_components=2)。
    - 步骤 3: 绘制轮廓系数随 k 变化的图像 (内部指标)。
    - 步骤 4 (修改): 使用 k=8 (或指定值) 进行最终聚类。
    - 步骤 5: 绘制 2D UMAP 可视化图，并添加球队标注。
    - 步骤 6: 保存聚类结果到 CSV。
    - 步骤 7: 所有输出均保存到指定文件夹。
    """

    # 1. 配置路径 (与原脚本一致)
    BASE_OUTPUT_DIR = r'E:\code\2025\FootballAnalysis\TeamClustering\clustering for teams v1.4'
    INPUT_CSV_NAME = r'style_features_v1.4.csv'
    INPUT_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, INPUT_CSV_NAME)

    # 输出文件 (添加 'umap' 标识)
    SILHOUETTE_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, f'kmeans_silhouette_plot_k{k_final}_umap.png')
    UMAP_2D_PLOT_PATH = os.path.join(BASE_OUTPUT_DIR, f'kmeans_2d_umap_cluster_plot_k{k_final}.png')
    RESULTS_CSV_PATH = os.path.join(BASE_OUTPUT_DIR, f'team_cluster_assignments_k{k_final}_umap.csv')

    print(
        f"--- UMAP (cosine) + K-Means Clustering Analysis (k={k_final}, UMAP components={umap_n_components_cluster}) ---")
    print(f"Output directory: {BASE_OUTPUT_DIR}")

    # 确保输出目录存在
    try:
        os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
        print(f"Output directory verified.")
    except Exception as e:
        print(f"Error creating directory {BASE_OUTPUT_DIR}: {e}")
        return

    try:
        # 2. 加载数据 (与原脚本一致)
        print(f"\nLoading data from '{INPUT_CSV_PATH}'...")
        data = pd.read_csv(INPUT_CSV_PATH)
        print(f"Successfully loaded. Shape: {data.shape}")

        # 3. 聚合数据 (与原脚本一致)
        identifier_cols = ['match_id', 'team_id', 'team_name', 'opponent_id', 'opponent_team_name']
        feature_cols = [col for col in data.columns if col not in identifier_cols]

        if not feature_cols:
            print("Error: No feature columns found. Check CSV content and identifier_cols list.")
            return

        print(f"Found {len(feature_cols)} feature columns to aggregate.")

        print("Aggregating match data to team-level averages...")
        team_avg_features = data.groupby(['team_id', 'team_name'])[feature_cols].mean(numeric_only=True).reset_index()
        print(f"Aggregation complete. New shape (teams, features): {team_avg_features.shape}")

        # 4. 数据预处理 (与原脚本一致)
        team_ids = team_avg_features['team_id']
        team_names = team_avg_features['team_name']
        features = team_avg_features.drop(['team_id', 'team_name'], axis=1)

        if features.isnull().values.any():
            print("Missing values found after aggregation. Filling with column mean.")
            features = features.fillna(features.mean())
        else:
            print("No missing values found in aggregated features.")

        # 标准化 (UMAP 同样受益于标准化数据)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print("Aggregated features standardized.")

        # 5. UMAP 降维 (替换 PCA)

        # 5a. UMAP (用于聚类 - 目标 n_components_cluster)
        print(f"\nApplying UMAP (for clustering) to {umap_n_components_cluster} components...")
        print(f"(Metric: cosine, n_neighbors: {n_neighbors}, min_dist: {min_dist})")

        umap_cluster_model = umap.UMAP(
            n_components=umap_n_components_cluster,
            metric='cosine',  # 关键: 按要求使用余弦距离
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        features_umap_cluster = umap_cluster_model.fit_transform(features_scaled)
        print(f"UMAP (Clustering): Shape {features_umap_cluster.shape}")

        # 5b. UMAP (用于 2D 可视化)
        print("Applying UMAP (for visualization) to get 2 components...")
        print(f"(Metric: cosine, n_neighbors: {n_neighbors}, min_dist: {min_dist})")

        umap_2d_model = umap.UMAP(
            n_components=2,
            metric='cosine',  # 关键: 按要求使用余弦距离
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        features_umap_2d = umap_2d_model.fit_transform(features_scaled)
        print(f"UMAP (2D Viz): Shape {features_umap_2d.shape}")

        # 6. 计算不同 k 值的轮廓系数 (内部指标)
        # (使用高维 UMAP 数据进行计算)
        k_range = range(2, 15)
        silhouette_scores = []
        print(
            f"\nCalculating Silhouette Scores for k in {list(k_range)} (using {umap_n_components_cluster}D UMAP data)...")

        for k_val in k_range:
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_umap_cluster)

            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(features_umap_cluster, cluster_labels)
                silhouette_scores.append(score)
                print(f"  k={k_val}, Silhouette Score: {score:.4f}")
            else:
                silhouette_scores.append(np.nan)
                print(f"  k={k_val}, Silhouette Score: N/A (Only 1 cluster found)")

        # 7. 绘制轮廓系数随 k 变化的图像
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Score vs. k (on {umap_n_components_cluster}D UMAP data, cosine metric)')
        plt.xticks(k_range)
        plt.grid(True)

        plt.savefig(SILHOUETTE_PLOT_PATH)
        print(f"\nSilhouette plot saved to '{SILHOUETTE_PLOT_PATH}'")

        # 8. 使用 k_final 运行最终聚类
        print(f"\nRunning final clustering with k={k_final} on {umap_n_components_cluster}D UMAP data...")
        kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(features_umap_cluster)

        # 9. 准备用于可视化和保存的数据 (使用 umap_1, umap_2)
        results_df = pd.DataFrame({
            'team_id': team_ids,
            'team_name': team_names,
            'cluster': final_labels,
            'umap_1': features_umap_2d[:, 0],  # 替换 pca_1
            'umap_2': features_umap_2d[:, 1]  # 替换 pca_2
        })

        # 10. 绘制 2D 可视化聚类图
        print(f"Generating 2D cluster visualization plot (k={k_final})...")
        plt.figure(figsize=(14, 10))

        cmap = plt.get_cmap('tab10')
        norm = mcolors.BoundaryNorm(np.arange(-0.5, k_final, 1), cmap.N)

        scatter = plt.scatter(
            results_df['umap_1'],  # 替换
            results_df['umap_2'],  # 替换
            c=results_df['cluster'],
            cmap=cmap,
            norm=norm,
            alpha=0.7
        )

        # 添加球队名称标注
        print("Adding team name annotations to the 2D plot...")
        for i, row in results_df.iterrows():
            plt.text(
                row['umap_1'],  # 替换
                row['umap_2'],  # 替换
                row['team_name'],
                fontsize=6,
                alpha=0.6,
                ha='center',
                va='bottom'
            )

        plt.xlabel('UMAP Component 1 (cosine metric)')
        plt.ylabel('UMAP Component 2 (cosine metric)')
        plt.title(f'Team Clusters (k={k_final}) visualized on 2D UMAP (cosine metric)')

        # 生成图例
        handles, _ = scatter.legend_elements()
        legend_labels = [f'Cluster {i}' for i in range(k_final)]
        plt.legend(handles, legend_labels, title="Clusters")

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(UMAP_2D_PLOT_PATH, dpi=300)  # 使用新路径
        print(f"2D cluster visualization saved to '{UMAP_2D_PLOT_PATH}'")

        # 11. 给出聚类示例 (与原脚本一致)
        print("\n--- Cluster Examples (Sample of 5 teams per cluster) ---")
        for i in range(k_final):
            print(f"\nCluster {i}:")
            cluster_teams = results_df[results_df['cluster'] == i]['team_name']

            if cluster_teams.empty:
                print("  (No teams in this cluster)")
            else:
                sample_size = min(5, len(cluster_teams))
                print(cluster_teams.sample(sample_size, random_state=42).to_string(index=False))

        # 12. 保存聚类结果到 CSV
        print(f"\nSaving clustering results to '{RESULTS_CSV_PATH}'...")

        # 保存时不包含 umap_1 和 umap_2
        results_to_save = results_df.drop(['umap_1', 'umap_2'], axis=1)
        results_to_save.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8-sig')

        print(f"Successfully saved results to '{RESULTS_CSV_PATH}'.")
        print("\n--- Analysis Complete ---")

    except FileNotFoundError:
        print(f"Error: The file '{INPUT_CSV_PATH}' was not found.")
        print(
            "Please ensure 'json2csv_get_all_features.py' has been run and the output CSV is in the correct location.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 运行分析，使用 k=7 (与原脚本的 __main__ 一致)
    # 并指定用于聚类的 UMAP 组件数为 10 (这是一个可调参数)
    run_umap_kmeans_analysis(
        k_final=9,
        umap_n_components_cluster=12,
        n_neighbors=12,  #
        min_dist=0.1  # UMAP 默认值，可调
    )