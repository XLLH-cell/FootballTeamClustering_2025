import json
import math
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any


# import networkx as nx # 暂时注释掉，聚焦于核心风格特征

class FootballFeatureExtractor:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = {
            'long_pass_threshold': 30,
            'far_shot_threshold': 20,
            'data_dir': './data',
            'output_file': './football_styles.csv',
            'epsilon': 1e-6,
            # 场地定义 (StatsBomb 坐标系: 120x80)
            'pitch_length': 120,
            'pitch_width': 80,
            'def_third_x': 40,
            'att_third_x': 80,
            'left_channel_y': 25,  # < 25 为左路
            'right_channel_y': 55,  # > 55 为右路
        }
        if config:
            self.config.update(config)

    def load_match_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def identify_teams(self, events: List[Dict]):
        """识别主队和客队"""
        home_team = None
        home_team_name = "Unknown"
        away_team = None
        away_team_name = "Unknown"

        for event in events:
            if event.get('type', {}).get('id') == 35:  # Starting XI
                if home_team is None:
                    home_team = event['team']['id']
                    home_team_name = event['team']['name']
                elif away_team is None and event['team']['id'] != home_team:
                    away_team = event['team']['id']
                    away_team_name = event['team']['name']
                    break

        if home_team is None or away_team is None:
            # Fallback logic if Starting XI not found quickly
            pass

        return home_team, home_team_name, away_team, away_team_name

    def calculate_possession_time(self, events: List[Dict], home_team: int, away_team: int) -> Tuple[float, float]:
        """
        (修改) 计算双方控球的绝对时间（秒），用于计算节奏 (Passes per minute)。
        """
        home_seconds = 0
        away_seconds = 0
        current_possession_team = None
        possession_start = 0

        # 简单的时间累积逻辑 (基于 possession_team 字段)
        # 注意：为了性能和简化，这里假设事件已排序且利用 duration
        # StatsBomb 数据通常包含 duration，或者通过下一事件时间差计算
        # 这里沿用你之前的逻辑框架，但简化为累加 duration

        for event in events:
            duration = event.get('duration', 0.0)
            p_team = event.get('possession_team', {}).get('id')

            if p_team == home_team:
                home_seconds += duration
            elif p_team == away_team:
                away_seconds += duration

        # 如果 duration 缺失严重，可能需要回退到 timestamp 差值法
        # 这里假设 duration 可用。如果总时间太短（数据问题），给予保底值
        total_time = home_seconds + away_seconds
        if total_time < 3000:  # 假如一场比赛记录时间少于50分钟，可能有误，做归一化处理
            # 简易处理：不处理，依赖外部数据质量
            pass

        return home_seconds, away_seconds

    def _extract_raw_metrics(self, events: List[Dict], team_id: int) -> Dict[str, Any]:
        """
        (核心修改) 步骤1：提取用于计算风格的原始数据 (Counts & Lists)。
        这些数据不会直接输出到 CSV，而是传给 calculate_style_metrics。
        """
        raw = {
            # 1. 传球基础
            'total_passes': 0,
            'long_passes': 0,
            'forward_passes': 0,
            'backward_passes': 0,
            'crosses': 0,
            'att_third_passes': 0,  # 进攻三区传球总数 (分母)

            # 2. 门将
            'gk_total_passes': 0,
            'gk_long_passes': 0,

            # 3. 宽度/区域
            'passes_left': 0,
            'passes_middle': 0,
            'passes_right': 0,

            # 4. 射门
            'total_shots': 0,
            'box_shots': 0,  # 禁区内射门
            'total_xg': 0.0,

            # 5. 防守
            'def_action_x_coords': [],  # 用于计算防线高度
            'def_action_count': 0,  # 用于计算 PPDA
            'foul_count': 0,
            'duel_count': 0,
            'interception_count': 0
        }

        team_events = [e for e in events if e.get('team', {}).get('id') == team_id]

        # 预定义区域边界
        ATT_X = self.config['att_third_x']
        LEFT_Y = self.config['left_channel_y']
        RIGHT_Y = self.config['right_channel_y']
        LONG_THR = self.config['long_pass_threshold']

        for e in team_events:
            type_id = e.get('type', {}).get('id')
            location = e.get('location', [])

            # --- Pass Logic (ID 30) ---
            if type_id == 30:
                raw['total_passes'] += 1
                pass_info = e.get('pass', {})
                end_loc = pass_info.get('end_location', [])

                # 长传
                if pass_info.get('length', 0) > LONG_THR:
                    raw['long_passes'] += 1

                # 方向 (向前/向后)
                if location and end_loc:
                    if end_loc[0] > location[0]:
                        raw['forward_passes'] += 1
                    elif end_loc[0] < location[0]:  # 严格回传
                        raw['backward_passes'] += 1

                # 传中
                if pass_info.get('cross'):
                    raw['crosses'] += 1

                # 进攻三区传球 (作为传中率的分母)
                if location and location[0] >= ATT_X:
                    raw['att_third_passes'] += 1

                # 宽度利用 (基于起点 y)
                if location:
                    y = location[1]
                    if y <= LEFT_Y:
                        raw['passes_left'] += 1
                    elif y >= RIGHT_Y:
                        raw['passes_right'] += 1
                    else:
                        raw['passes_middle'] += 1

                # 门将分布
                if e.get('position', {}).get('id') == 1:  # GK
                    raw['gk_total_passes'] += 1
                    if pass_info.get('length', 0) > LONG_THR:
                        raw['gk_long_passes'] += 1

            # --- Shot Logic (ID 16) ---
            elif type_id == 16:
                raw['total_shots'] += 1
                shot_info = e.get('shot', {})
                raw['total_xg'] += shot_info.get('statsbomb_xg', 0.0)

                # 禁区内射门 (简单判定：x > 102, abs(y-40) < 18) -> 或者直接判定距离
                # 这里使用简单的 x 坐标判定，120码场地的禁区线大约在 102
                if location and location[0] >= 102:
                    raw['box_shots'] += 1

            # --- Defensive Logic ---
            # 包含: Duel(4), Block(6), Interception(10), Pressure(17), Foul Committed(22)
            # 这些是 "Defensive Actions" 用于计算防线高度和 PPDA 分母
            if type_id in [4, 6, 10, 17, 22]:
                raw['def_action_count'] += 1
                if location:
                    raw['def_action_x_coords'].append(location[0])

                if type_id == 22:
                    raw['foul_count'] += 1
                if type_id == 4:
                    raw['duel_count'] += 1
                if type_id == 10:
                    raw['interception_count'] += 1

        return raw

    def _calculate_style_metrics(self, my_raw: Dict, opp_raw: Dict, possession_seconds: float) -> Dict[str, float]:
        """
        (核心修改) 步骤2：计算纯粹的风格特征 (0.0 - 1.0 或 标准化数值)。
        """
        styles = {}
        eps = self.config['epsilon']

        # -----------------------------
        # 1. 进攻构建风格 (Build-up)
        # -----------------------------
        total_passes = max(my_raw['total_passes'], 1)

        # 长传比: 喜欢长传冲吊还是短传？
        styles['style_long_pass_ratio'] = my_raw['long_passes'] / total_passes

        # 向前/向后倾向: 激进还是稳健？
        styles['style_forward_pass_ratio'] = my_raw['forward_passes'] / total_passes
        styles['style_backward_pass_ratio'] = my_raw['backward_passes'] / total_passes

        # 门将出球: 大脚还是短传？
        gk_passes = max(my_raw['gk_total_passes'], 1)
        styles['style_gk_long_ratio'] = my_raw['gk_long_passes'] / gk_passes

        # -----------------------------
        # 2. 进攻区域风格 (Zones)
        # -----------------------------
        # 宽度利用率 (归一化)
        styles['style_width_left'] = my_raw['passes_left'] / total_passes
        styles['style_width_center'] = my_raw['passes_middle'] / total_passes
        styles['style_width_right'] = my_raw['passes_right'] / total_passes

        # 传中倾向 (在进攻三区的传球中，有多少是传中？)
        att_passes = max(my_raw['att_third_passes'], 1)
        styles['style_cross_ratio'] = my_raw['crosses'] / att_passes

        # -----------------------------
        # 3. 射门风格 (Shooting)
        # -----------------------------
        total_shots = max(my_raw['total_shots'], 1)

        # 禁区内射门比 (耐心度/渗透能力)
        styles['style_box_shot_ratio'] = my_raw['box_shots'] / total_shots

        # 平均射门质量 (每次射门的 xG) - 衡量是滥射还是寻找绝对机会
        styles['style_xg_per_shot'] = my_raw['total_xg'] / total_shots

        # -----------------------------
        # 4. 防守风格 (Defense)
        # -----------------------------
        # 防线高度 (平均防守动作 X 坐标)
        if my_raw['def_action_x_coords']:
            styles['style_def_line_height'] = np.mean(my_raw['def_action_x_coords'])
        else:
            styles['style_def_line_height'] = 0.0  # 异常处理

        # PPDA (Passes Per Defensive Action) - 核心压迫指标
        # 数值越小，压迫越强。公式：对手传球数 / 我的防守动作数
        my_def_actions = max(my_raw['def_action_count'], 1)
        styles['style_ppda'] = opp_raw['total_passes'] / my_def_actions

        # 侵略性 (犯规占比) - 是干净的拦截还是凶狠的犯规？
        # 使用 犯规 / (对抗+拦截+犯规)
        physical_actions = max(my_raw['duel_count'] + my_raw['interception_count'] + my_raw['foul_count'], 1)
        styles['style_aggression_ratio'] = my_raw['foul_count'] / physical_actions

        # -----------------------------
        # 5. 比赛节奏 (Tempo)
        # -----------------------------
        # 每分钟控球传球数
        possession_minutes = max(possession_seconds / 60.0, 1.0)  # 避免除零
        styles['style_tempo'] = my_raw['total_passes'] / possession_minutes

        return styles

    def process_single_match(self, file_path: str) -> List[Dict]:
        try:
            events = self.load_match_data(file_path)

            # 1. 基础信息
            home_team, home_name, away_team, away_name = self.identify_teams(events)
            if not home_team or not away_team:
                return []

            # 过滤常规时间
            events = [e for e in events if e.get('period') in [1, 2]]

            # 2. 计算控球时间 (秒)
            home_sec, away_sec = self.calculate_possession_time(events, home_team, away_team)

            # 3. 提取 Raw Data
            raw_home = self._extract_raw_metrics(events, home_team)
            raw_away = self._extract_raw_metrics(events, away_team)

            # 4. 计算 Style Metrics (传入对手数据用于 PPDA)
            home_style = self._calculate_style_metrics(raw_home, raw_away, home_sec)
            away_style = self._calculate_style_metrics(raw_away, raw_home, away_sec)

            match_id = os.path.basename(file_path).replace('.json', '')

            # 5. 组装最终记录
            # 只包含 ID, Name 和 style_ 开头的特征
            records = []

            home_record = {
                'match_id': match_id,
                'team_id': home_team,
                'team_name': home_name,
                'opponent_name': away_name,  # 保留名字方便核对，不进模型
                **home_style
            }
            records.append(home_record)

            away_record = {
                'match_id': match_id,
                'team_id': away_team,
                'team_name': away_name,
                'opponent_name': home_name,
                **away_style
            }
            records.append(away_record)

            return records

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def process_all_matches(self) -> pd.DataFrame:
        """处理文件夹下所有文件"""
        all_records = []
        data_dir = self.config['data_dir']

        if not os.path.exists(data_dir):
            print(f"Error: Directory {data_dir} not found.")
            return pd.DataFrame()

        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files for Style Extraction...")

        for i, filename in enumerate(json_files):
            if (i + 1) % 50 == 0:
                print(f"Processing {i + 1}/{len(json_files)}...")

            file_path = os.path.join(data_dir, filename)
            records = self.process_single_match(file_path)
            all_records.extend(records)

        df = pd.DataFrame(all_records)

        if not df.empty:
            output_path = self.config['output_file']
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"Style features saved to {output_path}")
            print(f"Total records: {len(df)}")
            print("Features list:", list(df.columns))

        return df


# 使用示例
if __name__ == "__main__":
    config = {
        'data_dir': 'E:\\code\\2025\\FootballAnalysis\\TeamClustering\\events',
        'output_file': 'E:\\code\\2025\\FootballAnalysis\\TeamClustering\\style_features_v1.4.csv'
    }

    extractor = FootballFeatureExtractor(config)
    df = extractor.process_all_matches()