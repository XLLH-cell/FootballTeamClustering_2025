import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# 设置中文字体 (Windows 11 环境)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def analyze_clustering_features(k_val):
    """
    分析哪些特征对聚类结果影响最大
    :param k_val: 对应之前运行聚类时的 k 值 (用于读取正确的文件)
    """

    # 1. 配置路径
    BASE_DIR = r'E:\code\2025\FootballAnalysis\TeamClustering\clustering for teams v1.3'
    RAW_DATA_PATH = os.path.join(BASE_DIR, 'style_features_v1.4.csv')
    # 读取之前脚本生成的聚类结果文件 (注意文件名中的 _umap 后缀)
    CLUSTER_RESULT_PATH = os.path.join(BASE_DIR, f'team_cluster_assignments_k{k_val}_umap.csv')

    OUTPUT_DIR = os.path.join(BASE_DIR, 'analysis_results')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- Starting Feature Importance Analysis for k={k_val} ---")

    # 2. 加载并准备数据
    try:
        # 加载原始数据
        print("Loading and aggregating raw data...")
        raw_df = pd.read_csv(RAW_DATA_PATH)

        # --- 修改开始: 更严谨的特征列筛选 ---

        # 1. 定义非特征列（ID类、名称类）
        # 注意：这里把所有可能的非数值列都列出来，防止漏网之鱼
        exclude_cols = ['match_id', 'team_id', 'team_name', 'opponent_id', 'opponent_team_name', 'opponent_name',
                        'date', 'competition']

        # 2. 筛选出所有数值类型的列作为候选特征
        # 只保留 float 和 int 类型的列，自动过滤掉所有字符串列
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()

        # 3. 从数值列中排除掉 ID 类列 (如 match_id, team_id, opponent_id)
        # 这样 feature_cols 就只包含真正的特征数据，且一定存在于聚合后的结果中
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        print(f"Identified {len(feature_cols)} numeric feature columns.")

        # --- 修改结束 ---

        # 按球队聚合取平均 (只对 feature_cols 进行聚合)
        # 此时 feature_cols 全是数值，不会被 mean() 丢弃
        team_features = raw_df.groupby(['team_id', 'team_name'])[feature_cols].mean().reset_index()

        # 加载聚类结果
        print(f"Loading cluster assignments from {CLUSTER_RESULT_PATH}...")
        cluster_df = pd.read_csv(CLUSTER_RESULT_PATH)

        # 合并数据
        merged_df = pd.merge(team_features, cluster_df[['team_id', 'cluster']], on='team_id', how='inner')
        print(f"Merged Data Shape: {merged_df.shape}")

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
        return

    # 准备 X (特征) 和 y (聚类标签)
    X = merged_df[feature_cols].fillna(0)  # 简单填充缺失值
    y = merged_df['cluster']

    # ---------------------------------------------------------
    # 分析方法 1: 随机森林特征重要性 (Random Forest Feature Importance)
    # 逻辑: 训练一个分类器去预测"这个球队属于哪个聚类"。
    # 如果某个特征能很好地帮助分类，说明它对聚类结果有很大影响。
    # ---------------------------------------------------------
    print("\nCalculating Feature Importance using Random Forest...")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]  # 降序排列索引

    # 提取前 20 个最重要的特征
    top_n = 20
    top_features = [feature_cols[i] for i in indices[:top_n]]
    top_importances = importances[indices[:top_n]]

    # 绘图 1: 特征重要性条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.title(f'Top {top_n} Features Influencing Clustering (Random Forest Importance)')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    save_path_rf = os.path.join(OUTPUT_DIR, 'feature_importance_rf.png')
    plt.savefig(save_path_rf)
    print(f"Saved Feature Importance Plot to: {save_path_rf}")

    # ---------------------------------------------------------
    # 分析方法 2: 聚类特征热力图 (Cluster Profile Heatmap)
    # 逻辑: 计算每个聚类在所有特征上的平均值，并进行标准化(Z-Score)，
    # 这样可以看出"Cluster 0 在 控球率 上偏高，在 抢断 上偏低"。
    # ---------------------------------------------------------
    print("\nGenerating Cluster Profile Heatmap...")

    # 计算每个聚类的特征均值
    cluster_means = merged_df.groupby('cluster')[feature_cols].mean()

    # 标准化 (按列/特征标准化)，为了让不同量纲的特征可以在一张图上比较
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )

    # 为了图表可读性，只展示最重要的 20 个特征 (基于上面的 RF 结果)
    heatmap_data = cluster_means_scaled[top_features].transpose()  # 转置: 行是特征，列是聚类

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, annot=True, fmt=".2f", linewidths=.5)
    plt.title('Cluster Profiles (Standardized Means of Top Features)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Feature')
    plt.tight_layout()
    save_path_heatmap = os.path.join(OUTPUT_DIR, 'cluster_profile_heatmap.png')
    plt.savefig(save_path_heatmap)
    print(f"Saved Heatmap to: {save_path_heatmap}")

    # ---------------------------------------------------------
    # 分析方法 3: 关键特征的箱线图 (Boxplots)
    # 逻辑: 展示最重要的 6 个特征在不同聚类下的具体分布情况。
    # ---------------------------------------------------------
    print("\nGenerating Boxplots for Top 6 Features...")

    num_boxplots = 6
    top_6_features = top_features[:num_boxplots]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(top_6_features):
        sns.boxplot(x='cluster', y=feature, data=merged_df, ax=axes[i], palette="tab10")
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].set_xlabel('Cluster')
        axes[i].set_ylabel('Value')

    plt.suptitle(f'Distribution of Top {num_boxplots} Features across Clusters', fontsize=16)
    plt.tight_layout()
    save_path_box = os.path.join(OUTPUT_DIR, 'top_features_boxplots.png')
    plt.savefig(save_path_box)
    print(f"Saved Boxplots to: {save_path_box}")

    print("\nAnalysis Complete.")


if __name__ == "__main__":
    # 请确保这里的 k 值与您生成的 CSV 文件名中的 k 值一致
    # 例如：如果文件是 team_cluster_assignments_k9_umap.csv，则 k_val=9
    analyze_clustering_features(k_val=9)