修改：

1. 移除xg相关特征，避免强弱影响球队。
2. 移除传球数等，改为基于回合数量的比例
3. 移除Dominate特征，
4. 移除交互特征
5. 建议构建以下几组核心比率型特征来替代当前的绝对值特征：

    进攻构建 (Build-up Style):

        Directness (直接度): 向前传球距离 / 总传球距离。

        Long Ball Preference: 长传次数 / 总传球次数。

        Width Usage: 边路(Wing)区域的触球次数 / 总触球次数。

    防守强度 (Defensive Intensity):

        High Press Intensity: 进攻三区施压次数 / 总施压次数。

        Aggression: 犯规次数 / 抢断次数。

    网络特征优化:

        保留 clustering_coefficient，这通常代表球队配合的复杂程度，与对手关系较小。
6. 