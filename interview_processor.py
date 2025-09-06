import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime
import json
import traceback
import random
from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class InterviewDataProcessor:
    """面试数据处理器 - 专门处理interview_data.csv"""

    def __init__(self, csv_path: str = "interview_data.csv"):
        """
        初始化面试数据处理器

        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = Path(csv_path)
        self.df = None

        # Token统计相关
        self.token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "api_calls": 0,
        }

        # Checkpoint相关
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.load_data()

    def load_data(self) -> pd.DataFrame:
        """加载CSV数据，自动处理编码问题"""
        try:
            # 尝试不同的编码方式
            encodings = ["utf-8", "gbk", "gb2312", "utf-8-sig"]

            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    print(f"✅ 成功使用 {encoding} 编码加载数据")
                    break
                except UnicodeDecodeError:
                    continue

            if self.df is None:
                raise ValueError("无法使用任何编码方式读取文件")

            # 数据清理
            self._clean_data()

            print(f"📊 数据加载完成: {len(self.df)} 行 × {len(self.df.columns)} 列")
            return self.df

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            raise

    def _clean_data(self):
        """清理数据"""
        try:
            # 移除完全空的行和列
            self.df = self.df.dropna(how="all").dropna(axis=1, how="all")

            # 标准化列名 - 使用实际的列名
            print("原始列名:", list(self.df.columns))

            # 保持原始列名，只做基本清理
            self.df.columns = [col.strip() for col in self.df.columns]
        except Exception as e:
            print(f"❌ 数据清理失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            raise

    def _extract_response_text(self, response) -> str:
        """从ChatResponse对象中提取纯文本内容"""
        try:
            # 尝试多种方式提取text内容
            if hasattr(response, "message") and hasattr(response.message, "content"):
                # 处理 message.content 结构
                content = response.message.content
                if hasattr(content, "text"):
                    return content.text
                elif isinstance(content, str):
                    return content
                elif hasattr(content, "__str__"):
                    return str(content)

            # 尝试直接获取content属性
            if hasattr(response, "content"):
                content = response.content
                if hasattr(content, "text"):
                    return content.text
                elif isinstance(content, str):
                    return content
                elif hasattr(content, "__str__"):
                    return str(content)

            # 尝试获取text属性
            if hasattr(response, "text"):
                return response.text

            # 如果都没有，尝试从字符串表示中提取
            response_str = str(response)

            # 尝试从字符串中提取text内容
            import re

            text_match = re.search(r"text='([^']*)'", response_str)
            if text_match:
                return text_match.group(1)

            text_match = re.search(r'text="([^"]*)"', response_str)
            if text_match:
                return text_match.group(1)

            # 尝试提取content中的text
            content_match = re.search(r"Content\(text='([^']*)'", response_str)
            if content_match:
                return content_match.group(1)

            content_match = re.search(r'Content\(text="([^"]*)"', response_str)
            if content_match:
                return content_match.group(1)

            # 最后的fallback，返回处理过的字符串
            print("⚠️ 无法提取text内容，使用字符串表示")
            return response_str

        except Exception as e:
            print(f"❌ 提取响应文本失败: {e}")
            return str(response)

    def _parse_indices_input(self, input_str: str, max_index: int) -> List[int]:
        """
        解析用户输入的序号字符串，支持单个、多个、范围

        Args:
            input_str: 用户输入，如 "2", "2,4,6", "2-5", "1,3-5,8"
            max_index: 最大有效索引

        Returns:
            解析后的索引列表
        """
        if not input_str.strip():
            return []

        indices = []
        try:
            # 分割逗号分隔的部分
            parts = input_str.strip().split(",")

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # 检查是否是范围（包含连字符）
                if "-" in part:
                    range_parts = part.split("-")
                    if len(range_parts) == 2:
                        start = int(range_parts[0].strip())
                        end = int(range_parts[1].strip())
                        # 确保范围有效
                        if (
                            start <= end
                            and 1 <= start <= max_index
                            and 1 <= end <= max_index
                        ):
                            indices.extend(range(start, end + 1))  # 闭区间
                        else:
                            print(f"⚠️ 无效范围: {part} (有效范围: 1-{max_index})")
                    else:
                        print(f"⚠️ 无效范围格式: {part}")
                else:
                    # 单个数字
                    index = int(part)
                    if 1 <= index <= max_index:
                        indices.append(index)
                    else:
                        print(f"⚠️ 无效索引: {index} (有效范围: 1-{max_index})")

            # 去重并排序
            return sorted(list(set(indices)))

        except ValueError as e:
            print(f"❌ 解析索引时出错: {e}")
            return []

    def display_data_preview(self, max_records: int = 10) -> int:
        """
        显示数据预览，供用户选择

        Args:
            max_records: 最多显示的记录数

        Returns:
            有效数据的总数量
        """
        if self.df is None or len(self.df) == 0:
            print("❌ 数据为空")
            return 0

        # 筛选有效数据
        required_columns = [
            "候选人脱敏简历",
            "岗位脱敏jd",
            "一面面试对话",
            "一面面试评价",
        ]

        missing_columns = [
            col for col in required_columns if col not in self.df.columns
        ]
        if missing_columns:
            print(f"❌ 缺少必要列: {missing_columns}")
            return 0

        valid_data = self.df.dropna(subset=required_columns)
        total_valid = len(valid_data)

        if total_valid == 0:
            print("❌ 没有包含完整信息的有效数据")
            return 0

        print(f"\n📊 数据预览 (共{total_valid}条有效记录):")
        print("=" * 80)

        # 显示预览
        display_count = min(max_records, total_valid)
        for i in range(display_count):
            row = valid_data.iloc[i]
            position = row.get("岗位名称", "未知岗位")

            # 获取简历摘要（前50字符）
            resume = str(row.get("候选人脱敏简历", ""))
            resume_preview = resume[:50] + "..." if len(resume) > 50 else resume

            # 获取评价
            evaluation = str(row.get("一面面试评价", "无评价"))
            evaluation_preview = (
                evaluation[:80] + "..." if len(evaluation) > 80 else evaluation
            )

            print(f"{i+1:2d}. 岗位: {position}")
            print(f"    简历: {resume_preview}")
            print(f"    评价: {evaluation_preview}")
            print()

        if total_valid > display_count:
            print(f"... 还有 {total_valid - display_count} 条记录未显示")

        print("=" * 80)
        print("📝 选择说明:")
        print("  单个记录: 2")
        print("  多个记录: 2,4,6,9")
        print("  连续范围: 2-5 (包含2,3,4,5)")
        print("  混合使用: 1,3-5,8 (包含1,3,4,5,8)")
        print()

        return total_valid

    # def _estimate_tokens(self, text: str) -> int:
    #     """估算文本的token数量（粗略估算）"""
    #     # 简单估算：英文约4字符=1token，中文约1.5字符=1token
    #     chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    #     english_chars = len(text) - chinese_chars
    #     estimated_tokens = int(chinese_chars / 1.5 + english_chars / 4)
    #     return estimated_tokens

    def _calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    ) -> float:
        """计算API调用成本"""
        # 价格表（每1K tokens的价格，单位：美元）
        pricing = {
            "us.anthropic.claude-sonnet-4-20250514-v1:0": {
                "input": 0.003,
                "output": 0.015,
            },
            "claude-sonnet-4-20250514": {
                "input": 0.003,
                "output": 0.015,
            },
        }

        if model_name not in pricing:
            model_name = (
                "us.anthropic.claude-sonnet-4-20250514-v1:0"  # 默认使用claude价格
            )

        input_cost = (input_tokens / 1000) * pricing[model_name]["input"]
        output_cost = (output_tokens / 1000) * pricing[model_name]["output"]

        return input_cost + output_cost

    def _update_token_stats_from_response(
        self, response, model_name: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    ):
        """从chat response中更新token统计"""
        try:
            # 尝试从response中获取usage信息
            usage = None
            if hasattr(response, "usage"):
                usage = response.usage
            elif (
                hasattr(response, "response_metadata")
                and "usage" in response.response_metadata
            ):
                usage = response.response_metadata["usage"]
            elif hasattr(response, "_raw_response") and hasattr(
                response._raw_response, "usage"
            ):
                usage = response._raw_response.usage

            if usage:
                # 从usage中获取token数量
                input_tokens = getattr(usage, "input_tokens", 0) or getattr(
                    usage, "prompt_tokens", 0
                )
                output_tokens = getattr(usage, "output_tokens", 0) or getattr(
                    usage, "completion_tokens", 0
                )

                if input_tokens > 0 or output_tokens > 0:
                    cost = self._calculate_cost(input_tokens, output_tokens, model_name)

                    self.token_stats["total_input_tokens"] += input_tokens
                    self.token_stats["total_output_tokens"] += output_tokens
                    self.token_stats["total_cost"] += cost
                    self.token_stats["api_calls"] += 1

                    print(
                        f"💰 本次调用: 输入{input_tokens}tokens, 输出{output_tokens}tokens, 成本${cost:.4f}"
                    )
                    return True

            # 如果无法获取usage信息，使用估算方法作为fallback
            print("⚠️ 无法从response中获取usage信息，使用估算方法")
            return False

        except Exception as e:
            print(f"❌ 更新token统计失败: {e}")
            return False

    # def _update_token_stats(
    #     self,
    #     input_text: str,
    #     output_text: str,
    #     model_name: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    # ):
    #     """更新token统计（备用方法）"""
    #     # input_tokens = self._estimate_tokens(input_text)
    #     # output_tokens = self._estimate_tokens(output_text)
    #     cost = self._calculate_cost(input_tokens, output_tokens, model_name)

    #     self.token_stats["total_input_tokens"] += input_tokens
    #     self.token_stats["total_output_tokens"] += output_tokens
    #     self.token_stats["total_cost"] += cost
    #     self.token_stats["api_calls"] += 1

    #     print(
    #         f"💰 本次调用(估算): 输入{input_tokens}tokens, 输出{output_tokens}tokens, 成本${cost:.4f}"
    #     )

    def _save_single_experience(
        self, experience: Dict, record_id: int, sample_name: str
    ):
        """保存单个经验到独立的JSON文件"""
        try:
            # 新命名规范：个人面试经验总结
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
            filename = f"individual_interview_experience_{record_id}_{timestamp}.json"
            experience_file = self.checkpoint_dir / filename

            experience_data = {
                "record_id": record_id,
                "sample_name": sample_name,
                "timestamp": datetime.now().isoformat(),
                "experience": experience,
                "token_stats_snapshot": self.token_stats.copy(),
            }

            with open(experience_file, "w", encoding="utf-8") as f:
                json.dump(experience_data, f, ensure_ascii=False, indent=2)

            print(f"💾 经验已保存: {experience_file.name}")
            return experience_file

        except Exception as e:
            print(f"❌ 保存经验失败: {e}")
            return None

    def _save_checkpoint(self, experiences: List[Dict], checkpoint_name: str):
        """保存checkpoint（保持兼容性）"""
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"

            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "experiences": experiences,
                "token_stats": self.token_stats.copy(),
                "total_processed": len(experiences),
            }

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            print(f"💾 Checkpoint已保存: {checkpoint_file}")

        except Exception as e:
            print(f"❌ 保存checkpoint失败: {e}")

    def list_experience_files(self) -> List[Dict]:
        """列出所有单个经验文件"""
        try:
            experience_files = []
            pattern = "*_record_*.json"

            for file_path in self.checkpoint_dir.glob(pattern):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    experience_files.append(
                        {
                            "filename": file_path.name,
                            "record_id": data.get("record_id"),
                            "sample_name": data.get("sample_name"),
                            "timestamp": data.get("timestamp"),
                            "file_path": str(file_path),
                        }
                    )
                except Exception as e:
                    print(f"⚠️ 读取经验文件失败 {file_path}: {e}")
                    continue

            # 按时间戳排序
            experience_files.sort(key=lambda x: x["timestamp"])
            return experience_files

        except Exception as e:
            print(f"❌ 列出经验文件失败: {e}")
            return []

    def load_all_experiences(self) -> List[Dict]:
        """加载所有单个经验文件"""
        try:
            all_experiences = []
            experience_files = self.list_experience_files()

            for file_info in experience_files:
                try:
                    with open(file_info["file_path"], "r", encoding="utf-8") as f:
                        data = json.load(f)
                    all_experiences.append(data)
                except Exception as e:
                    print(f"⚠️ 加载经验文件失败 {file_info['filename']}: {e}")
                    continue

            print(f"📁 已加载 {len(all_experiences)} 个经验文件")
            return all_experiences

        except Exception as e:
            print(f"❌ 加载所有经验失败: {e}")
            return []

    def get_basic_info(self) -> Dict:
        """获取数据基本信息"""
        try:
            if self.df is None:
                return {}

            # 转换numpy类型为Python原生类型
            def convert_numpy(obj):
                # 首先检查是否是numpy数组
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # 检查是否是pandas类型的缺失值
                elif isinstance(obj, (float, np.floating)):
                    try:
                        if pd.isna(obj) or np.isnan(obj):
                            return None
                    except (ValueError, TypeError):
                        pass
                    return float(obj)
                # 检查其他类型的缺失值
                elif obj is None:
                    return None
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    # 最后尝试检查pandas的缺失值
                    try:
                        if pd.isna(obj):
                            return None
                    except (ValueError, TypeError):
                        pass
                return obj

            info = {
                "total_records": len(self.df),
                "total_columns": len(self.df.columns),
                "columns": list(self.df.columns),
                "data_types": dict(self.df.dtypes.astype(str)),
                "missing_values": convert_numpy(dict(self.df.isnull().sum())),
                "sample_data": (
                    convert_numpy(self.df.head(2).to_dict("records"))
                    if len(self.df) > 0
                    else []
                ),
            }

            return info
        except Exception as e:
            print(f"❌ 获取基本信息失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            return {}

    def analyze_positions(self) -> pd.DataFrame:
        """分析岗位分布"""
        try:
            # 寻找包含岗位信息的列
            position_cols = [
                col
                for col in self.df.columns
                if "岗位" in col or "position" in col.lower()
            ]

            if not position_cols:
                print("❌ 未找到岗位信息列")
                return pd.DataFrame()

            position_col = position_cols[0]
            print(f"📍 使用列 '{position_col}' 进行岗位分析")

            # 基本统计
            position_stats = self.df[position_col].value_counts().to_frame("候选人数量")

            # 如果有聪明度要求列
            intelligence_cols = [col for col in self.df.columns if "聪明度" in col]
            if intelligence_cols:
                intelligence_col = intelligence_cols[0]
                avg_intelligence = (
                    self.df.groupby(position_col)[intelligence_col]
                    .agg(["mean", "std"])
                    .round(2)
                )
                avg_intelligence.columns = ["平均聪明度要求", "聪明度标准差"]
                position_stats = position_stats.join(avg_intelligence)

            # 如果有面试结果列
            result_cols = [
                col for col in self.df.columns if "通过" in col or "结果" in col
            ]
            for result_col in result_cols[:2]:  # 最多处理2个结果列
                pass_count = self.df.groupby(position_col)[result_col].apply(
                    lambda x: x.astype(str).str.contains("通过", na=False).sum()
                )
                position_stats[f"{result_col}_通过人数"] = pass_count
                position_stats[f"{result_col}_通过率"] = (
                    pass_count / position_stats["候选人数量"] * 100
                ).round(1)

            return position_stats
        except Exception as e:
            print(f"❌ 岗位分析失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            return pd.DataFrame()

    def analyze_interview_results(self) -> Dict:
        """分析面试结果"""
        try:
            results = {}

            # 寻找面试结果相关列
            result_cols = [
                col for col in self.df.columns if "通过" in col or "结果" in col
            ]

            for i, col in enumerate(result_cols[:2]):  # 最多分析2轮面试
                round_name = f"第{i+1}轮面试" if i < 2 else col

                if col in self.df.columns:
                    col_results = self.df[col].astype(str).value_counts()
                    total = len(self.df[col].dropna())
                    pass_count = (
                        self.df[col].astype(str).str.contains("通过", na=False).sum()
                    )

                    results[round_name] = {
                        "统计": {str(k): int(v) for k, v in dict(col_results).items()},
                        "通过率": (
                            f"{(pass_count / total * 100):.1f}%" if total > 0 else "0%"
                        ),
                        "总人数": int(total),
                    }

            return results
        except Exception as e:
            print(f"❌ 面试结果分析失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            return {}

    def search_candidates(self, keyword: str, column: str = None) -> pd.DataFrame:
        """搜索候选人"""
        if self.df is None:
            return pd.DataFrame()

        if column and column in self.df.columns:
            # 在指定列中搜索
            mask = (
                self.df[column].astype(str).str.contains(keyword, case=False, na=False)
            )
        else:
            # 在所有文本列中搜索
            mask = pd.Series([False] * len(self.df))
            for col in self.df.select_dtypes(include=["object"]).columns:
                mask |= (
                    self.df[col].astype(str).str.contains(keyword, case=False, na=False)
                )

        return self.df[mask]

    def extract_candidate_info(self, row_index: int) -> Dict:
        """提取单个候选人的详细信息"""
        if row_index >= len(self.df):
            return {}

        candidate = self.df.iloc[row_index]

        # 寻找简历相关列
        resume_cols = [col for col in self.df.columns if "简历" in col]
        resume_info = {}

        if resume_cols and pd.notna(candidate[resume_cols[0]]):
            resume_text = str(candidate[resume_cols[0]])

            # 提取基本信息
            patterns = {
                "求职意向": r"求职意向：(.+?)(?:\n|$)",
                "籍贯": r"籍贯：(.+?)(?:\n|$)",
                "出生年月": r"出生年月：(.+?)(?:\n|$)",
                "学历": r"学历：(.+?)(?:\n|$)",
                "政治面貌": r"政治面貌：(.+?)(?:\n|$)",
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, resume_text)
                if match:
                    resume_info[key] = match.group(1).strip()

        return {
            "index": row_index,
            "resume_info": resume_info,
            "raw_data": candidate.to_dict(),
        }

    def plot_position_distribution(self, save_path: str = None):
        """绘制岗位分布图"""
        position_cols = [col for col in self.df.columns if "岗位" in col]

        if not position_cols:
            print("❌ 未找到岗位信息")
            return

        position_col = position_cols[0]

        plt.figure(figsize=(15, 5))

        # 岗位分布饼图
        plt.subplot(1, 3, 1)
        position_counts = self.df[position_col].value_counts()
        plt.pie(position_counts.values, labels=position_counts.index, autopct="%1.1f%%")
        plt.title("岗位分布")

        # 岗位数量柱状图
        plt.subplot(1, 3, 2)
        position_counts.plot(kind="bar")
        plt.title("各岗位候选人数量")
        plt.ylabel("候选人数")
        plt.xticks(rotation=45)

        # 通过率分析（如果有结果数据）
        result_cols = [col for col in self.df.columns if "通过" in col]
        if result_cols:
            plt.subplot(1, 3, 3)
            position_stats = self.analyze_positions()

            pass_rate_cols = [col for col in position_stats.columns if "通过率" in col]
            if pass_rate_cols:
                position_stats[pass_rate_cols[0]].plot(kind="bar")
                plt.title(f"{pass_rate_cols[0]}")
                plt.ylabel("通过率 (%)")
                plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def extract_interview_experience(
        self,
        selected_indices: List[int] = None,
        resume_from_checkpoint: bool = True,
        summary_mode: str = "incremental",
    ) -> Dict:
        """
        从面试数据中提取经验总结

        Args:
            selected_indices: 选择的记录索引列表（从1开始），如果为None则随机选择5条
            resume_from_checkpoint: 是否从checkpoint恢复，默认True
            summary_mode: 总结模式
                - "incremental": 增量更新（默认，新经验与已有经验合并）
                - "review": 温故知新（包含历史经验的重新总结）
                - "full_refresh": 全量重新总结（完全重新分析所有经验）

        Returns:
            包含经验总结的字典
        """
        try:
            # 如果没有指定索引，使用默认随机选择5条
            if selected_indices is None:
                selected_indices = []
                sample_size = 5
                print(
                    f"🔍 开始提取面试经验，默认随机选择 {sample_size} 条，总结模式: {summary_mode}"
                )
            else:
                sample_size = len(selected_indices)
                print(
                    f"🔍 开始提取面试经验，选择记录: {selected_indices}，总结模式: {summary_mode}"
                )

            # 重置token统计
            self.token_stats = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0,
                "api_calls": 0,
            }

            if self.df is None or len(self.df) == 0:
                return {"error": "数据为空"}

            # 确保需要的列存在
            required_columns = [
                "候选人脱敏简历",
                "岗位脱敏jd",
                "一面面试对话",
                "一面面试评价",
            ]

            missing_columns = [
                col for col in required_columns if col not in self.df.columns
            ]
            if missing_columns:
                print(f"❌ 缺少必要列: {missing_columns}")
                return {"error": f"缺少必要列: {missing_columns}"}

            # 筛选有效数据（四个字段都不为空）
            valid_data = self.df.dropna(subset=required_columns)
            print(f"📊 有效数据行数: {len(valid_data)}/{len(self.df)}")

            if len(valid_data) == 0:
                return {"error": "没有包含完整信息的有效数据"}

            # 根据选择的索引获取数据
            if selected_indices:
                # 转换为DataFrame索引（从1开始转为从0开始）
                df_indices = [
                    i - 1 for i in selected_indices if 1 <= i <= len(valid_data)
                ]
                if not df_indices:
                    return {
                        "error": f"所选择的索引都无效，有效范围: 1-{len(valid_data)}"
                    }

                # 根据索引选择数据
                sampled_data = valid_data.iloc[df_indices]
                print(f"📝 已选择 {len(sampled_data)} 条记录: {selected_indices}")
            else:
                # 默认随机采样
                if len(valid_data) > sample_size:
                    sampled_data = valid_data.sample(n=sample_size, random_state=42)
                    print(f"📝 已随机采样到 {sample_size} 条记录")
                else:
                    sampled_data = valid_data
                    print(f"📝 使用全部 {len(sampled_data)} 条有效记录")

            sample_name = f"experience_sample_{len(sampled_data)}"

            # 如果是全量重新总结模式，检查是否有足够的历史经验
            if summary_mode == "full_refresh":
                existing_experiences = self.load_all_experiences()
                if len(existing_experiences) < 2:
                    print("⚠️ 历史经验不足，改为增量模式")
                    summary_mode = "incremental"

            # 处理新记录（每条经验保存为单独文件）
            new_experiences = []
            model = Model()

            for i, (idx, row) in enumerate(sampled_data.iterrows()):
                print(f"🤖 正在分析第 {i + 1}/{len(sampled_data)} 条记录...")

                try:
                    # 构造提示词
                    prompt = self._build_experience_extraction_prompt(
                        resume=row["候选人脱敏简历"],
                        jd=row["岗位脱敏jd"],
                        conversation=row["一面面试对话"],
                        evaluation=row["一面面试评价"],
                    )

                    # 调用AI模型
                    response = model.chat([user(content=prompt)])

                    # 提取响应文本内容（只保存text，不保存整个对象）
                    response_text = self._extract_response_text(response)

                    # 更新token统计 - 优先使用response中的usage信息
                    if not self._update_token_stats_from_response(response):
                        # 如果无法从response获取usage，使用估算方法
                        # self._update_token_stats(prompt, response_text)
                        print("⚠️ 无法获取token使用情况，跳过统计")

                    experience_data = {
                        "record_id": idx,
                        "resume_summary": str(row["候选人脱敏简历"])[:200] + "...",
                        "jd_summary": str(row["岗位脱敏jd"])[:200] + "...",
                        "evaluation": str(row["一面面试评价"]),
                        "extracted_experience": response_text,
                        "analysis_time": datetime.now().isoformat(),
                    }

                    # 保存单个经验文件
                    experience_file = self._save_single_experience(
                        experience_data, idx, sample_name
                    )
                    if experience_file:
                        new_experiences.append(experience_data)

                except Exception as e:
                    print(f"❌ 分析记录 {idx} 时出错: {e}")
                    continue

            # 显示总token统计
            print(f"\n💰 总计Token使用统计:")
            print(f"输入tokens: {self.token_stats['total_input_tokens']:,}")
            print(f"输出tokens: {self.token_stats['total_output_tokens']:,}")
            print(f"API调用次数: {self.token_stats['api_calls']}")
            print(f"预估总成本: ${self.token_stats['total_cost']:.4f}")

            # 根据模式选择要整合的经验
            if summary_mode == "incremental":
                # 增量模式：只整合新经验
                experiences_to_integrate = new_experiences
                print(f"📊 增量模式：整合 {len(new_experiences)} 条新经验")

            elif summary_mode == "review":
                # 温故知新模式：新经验 + 部分历史经验
                all_experiences = self.load_all_experiences()
                # 取最新的历史经验和所有新经验
                historical_experiences = [
                    exp["experience"] for exp in all_experiences[-5:]
                ]
                experiences_to_integrate = historical_experiences + new_experiences
                print(
                    f"📊 温故知新模式：整合 {len(historical_experiences)} 条历史经验 + {len(new_experiences)} 条新经验"
                )

            elif summary_mode == "full_refresh":
                # 全量重新总结模式：所有经验
                all_experiences = self.load_all_experiences()
                experiences_to_integrate = [
                    exp["experience"] for exp in all_experiences
                ] + new_experiences
                print(
                    f"📊 全量重新总结模式：整合所有 {len(experiences_to_integrate)} 条经验"
                )

            else:
                print(f"⚠️ 未知总结模式 {summary_mode}，使用增量模式")
                experiences_to_integrate = new_experiences

            # 整合经验
            if experiences_to_integrate:
                print(f"🔄 正在整合 {len(experiences_to_integrate)} 条经验...")
                integrated_experience = self._integrate_experiences(
                    experiences_to_integrate, summary_mode
                )
            else:
                integrated_experience = "暂无经验可整合"

            result = {
                "total_records": len(self.df),
                "valid_records": len(valid_data),
                "sampled_records": len(sampled_data),
                "successful_extractions": len(new_experiences),
                "new_experiences": new_experiences,
                "integrated_experience": integrated_experience,
                "summary_mode": summary_mode,
                "token_stats": self.token_stats.copy(),
                "extraction_time": datetime.now().isoformat(),
            }

            print(f"✅ 经验提取完成! 成功提取 {len(new_experiences)} 条新经验")
            return result

        except Exception as e:
            print(f"❌ 提取面试经验失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            return {"error": str(e), "traceback": traceback.format_exc()}

    def _build_experience_extraction_prompt(
        self, resume: str, jd: str, conversation: str, evaluation: str
    ) -> str:
        """构建经验提取的提示词"""
        return f"""
请基于以下面试数据，提取HR在面试中识别候选人"聪明度"、"皮实"和"勤奋"这三个指标的经验技巧。

## 背景信息

### 候选人简历
{resume}

### 岗位JD
{jd}

### 面试对话
{conversation}

### HR评价
{evaluation}

## 任务要求

请分析这次面试中HR是如何通过提问和互动来评估候选人的：
1. **聪明度** - 逻辑思维、学习能力、问题分析能力
2. **皮实** - 抗压能力、韧性、面对困难的态度
3. **勤奋** - 工作热情、主动性、持续学习意愿

请从以下角度提取经验：

### 1. 有效提问技巧
- 针对聪明度的提问方式和角度
- 针对皮实的提问方式和角度  
- 针对勤奋的提问方式和角度

### 2. 关键追问策略
- 当候选人回答不够深入时的追问技巧
- 如何通过追问挖掘真实能力
- 什么样的回答需要进一步验证

### 3. 评估要点
- 每个指标的关键观察点
- 优秀回答的特征
- 需要警惕的回答模式

### 4. 适用场景
- 这些技巧适合什么类型的岗位
- 什么背景的候选人
- 什么阶段使用

请用结构化的方式输出，包含具体的问题示例和判断标准。
"""

    def _integrate_experiences(
        self, experiences: List[Dict], mode: str = "incremental"
    ) -> str:
        """
        整合多个经验成为综合指导

        Args:
            experiences: 经验列表
            mode: 整合模式
                - "incremental": 增量更新（默认，新经验与已有经验合并）
                - "review": 温故知新（包含历史经验的重新总结）
                - "full_refresh": 全量重新总结（完全重新分析所有经验）
        """
        try:
            if not experiences:
                return "暂无经验可整合"

            print(f"🔄 正在整合 {len(experiences)} 条经验，模式: {mode}...")

            # 根据模式选择不同的整合策略
            if mode == "incremental":
                prompt = self._build_incremental_prompt(experiences)
            elif mode == "review":
                prompt = self._build_review_prompt(experiences)
            elif mode == "full_refresh":
                prompt = self._build_full_refresh_prompt(experiences)
            else:
                print(f"⚠️ 未知模式 {mode}，使用默认增量模式")
                prompt = self._build_incremental_prompt(experiences)

            print(f"📝 整合prompt长度: {len(prompt)} 字符")

            model = Model()
            response = model.chat([user(content=prompt)])

            # 从response中提取纯文本内容
            integrated_experience = self._extract_response_text(response)

            # 更新token统计
            if not self._update_token_stats_from_response(response):
                # 如果无法获取真实usage，使用估算
                # self._update_token_stats(prompt, integrated_experience)
                print("⚠️ 无法获取token使用情况，跳过统计")

            return integrated_experience

        except Exception as e:
            print(f"❌ 整合经验失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            return "整合经验时发生错误"

    def _build_incremental_prompt(self, experiences: List[Dict]) -> str:
        """构建增量更新的prompt"""
        # 限制经验数量和内容长度以避免超时
        max_experiences = 5
        max_chars_per_experience = 3000

        if len(experiences) > max_experiences:
            experiences = experiences[-max_experiences:]  # 使用最新的经验

        experiences_text = ""
        for i, exp in enumerate(experiences, 1):
            exp_text = str(exp.get("extracted_experience", exp.get("experience", "")))
            if len(exp_text) > max_chars_per_experience:
                exp_text = exp_text[:max_chars_per_experience] + "..."
            experiences_text += f"\n## 经验{i}:\n{exp_text}\n"

        return f"""请将以下面试经验进行整合，形成一份简洁的HR面试指导手册。

要求：
1. 聚焦于"聪明度"、"皮实"、"勤奋"三个核心维度的评估
2. 提取具体可操作的面试技巧和问题
3. 避免重复，突出核心要点
4. 输出长度控制在2000字以内

经验内容：
{experiences_text}

请整合成一份实用的面试指导手册。"""

    def _build_review_prompt(self, experiences: List[Dict]) -> str:
        """构建温故知新的prompt"""
        # 选择部分历史经验和新经验
        max_experiences = 8
        max_chars_per_experience = 2000

        if len(experiences) > max_experiences:
            # 取前面的历史经验和最新的经验
            historical = experiences[: max_experiences // 2]
            recent = experiences[-max_experiences // 2 :]
            experiences = historical + recent

        experiences_text = ""
        for i, exp in enumerate(experiences, 1):
            exp_text = str(exp.get("extracted_experience", exp.get("experience", "")))
            if len(exp_text) > max_chars_per_experience:
                exp_text = exp_text[:max_chars_per_experience] + "..."
            timestamp = exp.get("timestamp", exp.get("analysis_time", "未知时间"))
            experiences_text += f"\n## 经验{i} ({timestamp[:10] if timestamp else '未知时间'}):\n{exp_text}\n"

        return f"""请对以下面试经验进行温故知新式的整合，既要保留经典的面试技巧，也要融入新的洞察。

要求：
1. 重点关注"聪明度"、"皮实"、"勤奋"的评估方法
2. 总结历史经验中的成功模式
3. 识别新经验中的创新点
4. 形成既有传承又有发展的指导手册
5. 输出长度控制在2500字以内

经验内容（按时间排序）：
{experiences_text}

请整合成一份兼具经典与创新的面试指导手册。"""

    def _build_full_refresh_prompt(self, experiences: List[Dict]) -> str:
        """构建全量重新总结的prompt"""
        # 更严格的限制，因为要处理所有经验
        max_experiences = 10
        max_chars_per_experience = 1500

        if len(experiences) > max_experiences:
            # 均匀采样
            step = len(experiences) // max_experiences
            experiences = experiences[::step][:max_experiences]

        experiences_text = ""
        for i, exp in enumerate(experiences, 1):
            exp_text = str(exp.get("extracted_experience", exp.get("experience", "")))
            if len(exp_text) > max_chars_per_experience:
                exp_text = exp_text[:max_chars_per_experience] + "..."
            experiences_text += f"\n## 经验{i}:\n{exp_text}\n"

        return f"""请对以下所有面试经验进行全面重新分析和总结，形成一份全新的综合性HR面试指导手册。

要求：
1. 从"聪明度"、"皮实"、"勤奋"三个维度进行系统性分析
2. 提取共性规律和差异化洞察
3. 构建完整的评估框架和操作指南
4. 去除冗余，突出精华
5. 输出长度控制在3000字以内

所有经验内容：
{experiences_text}

请重新整合成一份系统性的面试指导手册。"""

    def list_checkpoints(self) -> List[str]:
        """列出所有可用的checkpoint"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("*.json"))
            checkpoint_names = [f.stem for f in checkpoint_files]

            if checkpoint_names:
                print("📁 可用的checkpoint:")
                for name in checkpoint_names:
                    checkpoint_data = self._load_checkpoint(name)
                    if checkpoint_data:
                        total_processed = checkpoint_data.get("total_processed", 0)
                        timestamp = checkpoint_data.get("timestamp", "未知")
                        token_stats = checkpoint_data.get("token_stats", {})
                        cost = token_stats.get("total_cost", 0)
                        print(
                            f"  - {name}: {total_processed}条记录, 成本${cost:.4f}, 时间:{timestamp}"
                        )
            else:
                print("📁 没有找到checkpoint文件")

            return checkpoint_names

        except Exception as e:
            print(f"❌ 列出checkpoint失败: {e}")
            return []

    def clear_checkpoints(self):
        """清空所有checkpoint"""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("*.json"))
            for file in checkpoint_files:
                file.unlink()
            print(f"🗑️ 已清除 {len(checkpoint_files)} 个checkpoint文件")
        except Exception as e:
            print(f"❌ 清除checkpoint失败: {e}")

    def save_experience_report(
        self, experience_data: Dict, output_path: str = None
    ) -> str:
        """保存经验报告到文件"""
        try:
            if output_path is None:
                # 新命名规范：通用面试提问经验
                version = "1.0"  # 可以根据需要调整版本号
                output_path = f"general_interview_guidelines_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # 转换numpy类型
            def convert_numpy(obj):
                # 首先检查是否是numpy数组
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                # 检查是否是pandas类型的缺失值
                elif isinstance(obj, (float, np.floating)):
                    try:
                        if pd.isna(obj) or np.isnan(obj):
                            return None
                    except (ValueError, TypeError):
                        pass
                    return float(obj)
                # 检查其他类型的缺失值
                elif obj is None:
                    return None
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                else:
                    # 最后尝试检查pandas的缺失值
                    try:
                        if pd.isna(obj):
                            return None
                    except (ValueError, TypeError):
                        pass
                return obj

            clean_data = convert_numpy(experience_data)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(clean_data, f, ensure_ascii=False, indent=2)

            print(f"📋 经验报告已保存到: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ 保存经验报告失败: {e}")
            print("详细错误信息:")
            print(traceback.format_exc())
            return ""

    def export_analysis_report(self, output_path: str = None):
        """导出分析报告"""
        if output_path is None:
            output_path = f"interview_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            # 首先检查是否是numpy数组
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # 检查是否是pandas类型的缺失值
            elif isinstance(obj, (float, np.floating)):
                try:
                    if pd.isna(obj) or np.isnan(obj):
                        return None
                except (ValueError, TypeError):
                    pass
                return float(obj)
            # 检查其他类型的缺失值
            elif obj is None:
                return None
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                # 最后尝试检查pandas的缺失值
                try:
                    if pd.isna(obj):
                        return None
                except (ValueError, TypeError):
                    pass
            return obj

        basic_info = self.get_basic_info()
        position_analysis = (
            self.analyze_positions().to_dict("index")
            if not self.analyze_positions().empty
            else {}
        )
        interview_results = self.analyze_interview_results()

        report = {
            "analysis_time": datetime.now().isoformat(),
            "basic_info": convert_numpy(basic_info),
            "position_analysis": convert_numpy(position_analysis),
            "interview_results": convert_numpy(interview_results),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"📋 分析报告已保存到: {output_path}")
        return output_path

    def get_summary(self):
        """获取数据摘要"""
        if self.df is None:
            return "数据未加载"

        position_cols = [col for col in self.df.columns if "岗位" in col]
        position_info = ""
        if position_cols:
            position_counts = self.df[position_cols[0]].value_counts()
            position_info = f"\n主要岗位:\n{position_counts.head()}"

        summary = f"""
📊 面试数据摘要
===============
总记录数: {len(self.df)}
数据列数: {len(self.df.columns)}
列名: {list(self.df.columns)}
{position_info}

面试结果概览:
{self.analyze_interview_results()}
        """
        return summary


# 使用示例
def main():
    """使用示例"""
    try:
        # 初始化处理器
        print("🚀 开始处理面试数据...")
        processor = InterviewDataProcessor("interview_data.csv")

        # 打印数据摘要
        print(processor.get_summary())

        # 分析岗位
        print("\n🎯 岗位分析:")
        position_analysis = processor.analyze_positions()
        if not position_analysis.empty:
            print(position_analysis)
        else:
            print("无岗位分析数据")

        # 搜索候选人示例
        print("\n🔍 搜索包含'市场'的记录:")
        market_candidates = processor.search_candidates("市场")
        print(f"找到 {len(market_candidates)} 个相关记录")

        # 查看第一个记录详情
        if len(processor.df) > 0:
            print("\n👤 第一条记录详情:")
            candidate_info = processor.extract_candidate_info(0)
            if candidate_info["resume_info"]:
                print(
                    "简历信息:",
                    json.dumps(
                        candidate_info["resume_info"], ensure_ascii=False, indent=2
                    ),
                )
            else:
                print("前几列数据:", dict(list(candidate_info["raw_data"].items())[:5]))

        # 绘制图表
        print("\n📊 生成数据可视化...")
        processor.plot_position_distribution()

        # 导出报告
        print("\n📋 导出分析报告...")
        report_path = processor.export_analysis_report()

        print("✅ 数据处理完成！")
        return processor

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        print("详细错误信息:")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    processor = main()
