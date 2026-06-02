"""
数据处理器模块

保留原有的数据分析处理逻辑，包括统计分析、搜索和可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional
import re

from manager.models import StatisticsResult, SearchQuery, QueryResult, Record

logger = logging.getLogger(__name__)

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class DataProcessor:
    """
    数据处理器

    功能:
    - 数据统计分析
    - 岗位分析
    - 面试结果分析
    - 候选人搜索
    - 数据可视化
    """

    def __init__(self):
        """初始化数据处理器"""
        pass

    def analyze_positions(self, df: pd.DataFrame) -> StatisticsResult:
        """
        分析岗位分布

        Args:
            df: 数据DataFrame

        Returns:
            StatisticsResult: 分析结果
        """
        try:
            # 寻找包含岗位信息的列
            job_cols = [col for col in df.columns if "Job Title" in col]

            if not job_cols:
                logger.warning("未找到岗位信息列")
                return StatisticsResult(metrics={}, summary="未找到岗位信息")

            job_col = job_cols[0]
            logger.info(f"使用列 '{job_col}' 进行岗位分析")

            # 基本统计
            job_counts = df[job_col].value_counts()

            metrics = {
                "job_distribution": job_counts.to_dict(),
                "total_job": len(job_counts),
                "total_candidates": int(job_counts.sum()),
            }

            # 如果有聪明度要求列
            intelligence_cols = [col for col in df.columns if "Intelligence" in col]
            if intelligence_cols:
                intelligence_col = intelligence_cols[0]
                avg_intelligence = (
                    df.groupby(job_col)[intelligence_col].agg(["mean", "std"]).round(2)
                )
                metrics["average_intelligence"] = avg_intelligence.to_dict()

            # 如果有面试结果列
            result_cols = [col for col in df.columns if "Result" in col]
            if result_cols:
                pass_rates = {}
                for result_col in result_cols[:3]:  # 最多处理3个结果列
                    pass_count = df.groupby(job_col)[result_col].apply(
                        lambda x: x.astype(str).str.contains("通过", na=False).sum()
                    )
                    total_count = df.groupby(job_col).size()
                    pass_rate = (pass_count / total_count * 100).round(1)
                    pass_rates[result_col] = pass_rate.to_dict()

                metrics["pass_rates"] = pass_rates

            # 生成摘要
            summary = self._generate_position_summary(metrics)

            return StatisticsResult(metrics=metrics, summary=summary)

        except Exception as e:
            logger.error(f"岗位分析失败: {e}")
            return StatisticsResult(metrics={}, summary=f"分析失败: {str(e)}")

    def analyze_interview_results(self, df: pd.DataFrame) -> StatisticsResult:
        """
        分析面试结果

        Args:
            df: 数据DataFrame

        Returns:
            StatisticsResult: 分析结果
        """
        try:
            metrics = {}

            # 寻找面试结果相关列
            result_cols = [col for col in df.columns if "Result" in col]

            for i, col in enumerate(result_cols[:3]):  # 最多分析3轮面试
                round_name = f"第{i + 1}轮面试" if i < 3 else col

                if col in df.columns:
                    col_results = df[col].astype(str).value_counts()
                    total = len(df[col].dropna())
                    pass_count = (
                        df[col].astype(str).str.contains("通过", na=False).sum()
                    )

                    metrics[round_name] = {
                        "统计": {str(k): int(v) for k, v in dict(col_results).items()},
                        "通过率": f"{(pass_count / total * 100):.1f}%"
                        if total > 0
                        else "0%",
                        "总人数": int(total),
                    }

            summary = self._generate_interview_summary(metrics)

            return StatisticsResult(metrics=metrics, summary=summary)

        except Exception as e:
            logger.error(f"面试结果分析失败: {e}")
            return StatisticsResult(metrics={}, summary=f"分析失败: {str(e)}")

    def search_candidates(self, df: pd.DataFrame, query: SearchQuery) -> QueryResult:
        """
        搜索候选人

        Args:
            df: 数据DataFrame
            query: 搜索查询参数

        Returns:
            QueryResult: 查询结果
        """
        try:
            # 应用过滤条件
            filtered_df = df.copy()

            # 关键词搜索
            if query.keyword:
                if query.fields:
                    # 在指定字段中搜索
                    mask = pd.Series([False] * len(filtered_df))
                    for field in query.fields:
                        if field in filtered_df.columns:
                            mask |= (
                                filtered_df[field]
                                .astype(str)
                                .str.contains(query.keyword, case=False, na=False)
                            )
                    filtered_df = filtered_df[mask]
                else:
                    # 在所有文本列中搜索
                    mask = pd.Series([False] * len(filtered_df))
                    for col in filtered_df.select_dtypes(include=["object"]).columns:
                        mask |= (
                            filtered_df[col]
                            .astype(str)
                            .str.contains(query.keyword, case=False, na=False)
                        )
                    filtered_df = filtered_df[mask]

            # 应用其他过滤条件
            for field, value in query.filters.items():
                if field in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[field] == value]

            total = len(filtered_df)

            # 分页
            page_size = query.limit
            page = query.offset // page_size + 1 if page_size > 0 else 1
            start_idx = query.offset
            end_idx = start_idx + query.limit

            paginated_df = filtered_df.iloc[start_idx:end_idx]

            # 转换为Record对象
            records = []
            for idx, row in paginated_df.iterrows():
                record = Record.from_dataframe_row(idx, row.to_dict())
                records.append(record)

            return QueryResult(
                records=records,
                total=total,
                page=page,
                page_size=page_size,
                filters_applied=query.filters,
            )

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return QueryResult(
                records=[],
                total=0,
                page=1,
                page_size=query.limit,
                filters_applied=query.filters,
            )

    def extract_candidate_info(self, row_data: Dict) -> Dict:
        """
        提取单个候选人的详细信息

        Args:
            row_data: 候选人数据字典

        Returns:
            包含提取信息的字典
        """
        resume_info = {}

        # 寻找简历相关列
        resume_text = None
        for key in row_data.keys():
            if "Resume" in key and pd.notna(row_data[key]):
                resume_text = str(row_data[key])
                break

        if resume_text:
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

        return {"resume_info": resume_info, "raw_data": row_data}

    def plot_position_distribution(
        self, df: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        绘制岗位分布图

        Args:
            df: 数据DataFrame
            save_path: 保存路径，如果为None则显示图表
        """
        try:
            job_cols = [col for col in df.columns if "Job Title" in col]

            if not job_cols:
                logger.warning("未找到岗位信息")
                return

            job_col = job_cols[0]
            plt.figure(figsize=(15, 5))

            # 岗位分布饼图
            plt.subplot(1, 3, 1)
            job_counts = df[job_col].value_counts()
            plt.pie(job_counts.values, labels=job_counts.index, autopct="%1.1f%%")
            plt.title("岗位分布")

            # 岗位数量柱状图
            plt.subplot(1, 3, 2)
            job_counts.plot(kind="bar")
            plt.title("各岗位候选人数量")
            plt.ylabel("候选人数")
            plt.xticks(rotation=45)

            # 通过率分析（如果有结果数据）
            result_cols = [col for col in df.columns if "通过" in col]
            if result_cols:
                plt.subplot(1, 3, 3)
                result_col = result_cols[0]

                # 计算每个岗位的通过率
                pass_rate = (
                    df.groupby(job_col)[result_col]
                    .apply(
                        lambda x: x.astype(str).str.contains("通过", na=False).sum()
                        / len(x)
                        * 100
                    )
                    .round(1)
                )

                pass_rate.plot(kind="bar")
                plt.title(f"{result_col} - 通过率")
                plt.ylabel("通过率 (%)")
                plt.xticks(rotation=45)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"图表已保存: {save_path}")
            else:
                plt.show()

        except Exception as e:
            logger.error(f"绘制图表失败: {e}")

    def _generate_position_summary(self, metrics: Dict) -> str:
        """生成岗位分析摘要"""
        summary_parts = []

        if "position_distribution" in metrics:
            dist = metrics["position_distribution"]
            summary_parts.append(f"共有 {metrics['total_positions']} 个岗位，")
            summary_parts.append(f"总候选人数 {metrics['total_candidates']} 人。")

            # 列举主要岗位
            top_positions = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:3]
            summary_parts.append("\n主要岗位: ")
            for pos, count in top_positions:
                summary_parts.append(f"{pos}({count}人) ")

        return "".join(summary_parts)

    def _generate_interview_summary(self, metrics: Dict) -> str:
        """生成面试结果摘要"""
        summary_parts = []

        for round_name, data in metrics.items():
            summary_parts.append(f"{round_name}: ")
            summary_parts.append(f"总人数 {data['总人数']}人, ")
            summary_parts.append(f"通过率 {data['通过率']}\n")

        return "".join(summary_parts)

    def get_overview(self, df: pd.DataFrame) -> Dict:
        """
        获取数据总览

        Args:
            df: 数据DataFrame

        Returns:
            包含总览信息的字典
        """
        overview = {
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
        }

        # 岗位信息
        job_cols = [col for col in df.columns if "Job Title" in col]
        if job_cols:
            job_col = job_cols[0]
            job_counts = df[job_col].value_counts()
            overview["positions"] = job_counts.to_dict()
        # 面试结果概览
        result_cols = [col for col in df.columns if "Result" in col]
        if result_cols:
            results_overview = {}
            for col in result_cols[:3]:
                value_counts = df[col].value_counts()
                results_overview[col] = value_counts.to_dict()
            overview["interview_results"] = results_overview

        return overview
