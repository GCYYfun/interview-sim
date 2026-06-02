"""
数据分析模块

功能：
- 面试数据统计分析
- 数据可视化
- 候选人搜索
- 使用 manager.InterviewDataManager
"""

from typing import Dict, List, Optional
from pathlib import Path

from manager.interview_data_manager import InterviewDataManager
from manager.models import ManagerConfig
from menglong.utils.log import print_message


class DataAnalyzer:
    """数据分析器 - 面试数据统计和分析"""

    def __init__(self, csv_path: str = "interview_data.csv"):
        """
        初始化数据分析器

        Args:
            csv_path: CSV数据文件路径
        """
        self.csv_path = csv_path
        config = ManagerConfig(data_source=csv_path)
        self.manager = InterviewDataManager(config)

        # 加载数据
        print_message(f"📂 加载数据: {csv_path}")
        load_result = self.manager.load_data()
        if load_result.success:
            print_message(f"✓ 成功加载 {load_result.total_records} 条记录")
        else:
            print_message(f"❌ 加载失败: {load_result.error_message}")

    def show_statistics(self) -> Dict:
        """
        显示数据统计信息

        Returns:
            统计结果字典
        """
        print_message("\n" + "=" * 60)
        print_message("📊 数据统计")
        print_message("=" * 60)

        stats = self.manager.get_statistics()

        # 基本统计
        print_message(f"\n总记录数: {stats.total_records}")
        print_message(f"数据时间范围: {stats.date_range}")

        # 岗位分布
        print_message("\n岗位分布:")
        for position, count in stats.position_distribution.items():
            percentage = (
                (count / stats.total_records * 100) if stats.total_records > 0 else 0
            )
            print_message(f"  • {position}: {count} ({percentage:.1f}%)")

        # 面试结果分布
        if stats.result_distribution:
            print_message("\n面试结果分布:")
            for result, count in stats.result_distribution.items():
                percentage = (
                    (count / stats.total_records * 100)
                    if stats.total_records > 0
                    else 0
                )
                print_message(f"  • {result}: {count} ({percentage:.1f}%)")

        # 聪明度要求分布
        if stats.intelligence_distribution:
            print_message("\n聪明度要求分布:")
            for level, count in stats.intelligence_distribution.items():
                print_message(f"  • {level}分: {count}")

        print_message("=" * 60)

        return {
            "total_records": stats.total_records,
            "position_distribution": stats.position_distribution,
            "result_distribution": stats.result_distribution,
            "intelligence_distribution": stats.intelligence_distribution,
        }

    def search_candidates(
        self, keyword: str, field: Optional[str] = None
    ) -> List[Dict]:
        """
        搜索候选人

        Args:
            keyword: 搜索关键词
            field: 搜索字段（可选），如 'position', 'resume'

        Returns:
            匹配的候选人记录列表
        """
        print_message(f"\n🔍 搜索关键词: '{keyword}'")

        if field:
            print_message(f"搜索字段: {field}")

        # 使用manager的搜索功能
        from manager.models import SearchQuery

        query = SearchQuery(keyword=keyword, field=field)
        results = self.manager.search(query)

        if results.total_results == 0:
            print_message("❌ 未找到匹配结果")
            return []

        print_message(f"\n✓ 找到 {results.total_results} 条匹配记录:\n")

        matched_records = []
        for i, record in enumerate(results.records[:10], 1):  # 最多显示10条
            print_message(
                f"{i}. ID:{record.id} | "
                f"岗位:{record.position} | "
                f"聪明度要求:{record.intelligence_requirement}/100"
            )

            matched_records.append(
                {
                    "id": record.id,
                    "position": record.position,
                    "resume": record.resume[:100] + "...",
                    "jd": record.jd[:100] + "...",
                    "intelligence_requirement": record.intelligence_requirement,
                }
            )

        if results.total_results > 10:
            print_message(f"\n... 还有 {results.total_results - 10} 条结果未显示")

        return matched_records

    def generate_visualizations(self, output_dir: str = "reports") -> Dict:
        """
        生成数据可视化图表

        Args:
            output_dir: 输出目录

        Returns:
            生成的图表文件路径
        """
        print_message("\n📈 生成可视化图表...")

        try:
            # 使用manager生成图表
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # 获取统计数据
            stats = self.manager.get_statistics()

            # 生成岗位分布图
            position_chart = self._generate_position_chart(
                stats.position_distribution, output_dir
            )

            # 生成结果分布图
            result_chart = self._generate_result_chart(
                stats.result_distribution, output_dir
            )

            print_message(f"✓ 图表已生成到 {output_dir}/ 目录")

            return {
                "position_chart": position_chart,
                "result_chart": result_chart,
            }

        except Exception as e:
            print_message(f"❌ 生成图表失败: {str(e)}")
            return {"error": str(e)}

    def _generate_position_chart(
        self, distribution: Dict[str, int], output_dir: str
    ) -> str:
        """生成岗位分布图（简化版，实际可以使用matplotlib）"""
        # 这里简化为文本输出，实际项目中可以使用matplotlib/plotly
        chart_file = Path(output_dir) / "position_distribution.txt"

        with open(chart_file, "w", encoding="utf-8") as f:
            f.write("岗位分布统计\n")
            f.write("=" * 50 + "\n\n")

            max_count = max(distribution.values()) if distribution else 1

            for position, count in sorted(
                distribution.items(), key=lambda x: x[1], reverse=True
            ):
                bar_length = int((count / max_count) * 40)
                bar = "█" * bar_length
                f.write(f"{position:20s} {bar} {count}\n")

        print_message(f"  • 岗位分布图: {chart_file}")
        return str(chart_file)

    def _generate_result_chart(
        self, distribution: Dict[str, int], output_dir: str
    ) -> str:
        """生成结果分布图（简化版）"""
        chart_file = Path(output_dir) / "result_distribution.txt"

        with open(chart_file, "w", encoding="utf-8") as f:
            f.write("面试结果分布\n")
            f.write("=" * 50 + "\n\n")

            total = sum(distribution.values()) if distribution else 1

            for result, count in sorted(
                distribution.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total) * 100
                bar_length = int(percentage / 2.5)
                bar = "█" * bar_length
                f.write(f"{result:15s} {bar} {count} ({percentage:.1f}%)\n")

        print_message(f"  • 结果分布图: {chart_file}")
        return str(chart_file)

    def export_data(
        self, format: str = "json", output_path: Optional[str] = None
    ) -> str:
        """
        导出数据

        Args:
            format: 导出格式 (json/csv/excel)
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        print_message(f"\n💾 导出数据为 {format.upper()} 格式...")

        try:
            from manager.models import ExportFormat, ExportType

            # 映射格式
            format_map = {
                "json": ExportFormat.JSON,
                "csv": ExportFormat.CSV,
                "excel": ExportFormat.EXCEL,
            }

            export_format = format_map.get(format.lower(), ExportFormat.JSON)

            # 导出
            result_path = self.manager.export(
                export_type=ExportType.ALL,
                export_format=export_format,
                output_path=output_path,
            )

            print_message(f"✓ 数据已导出: {result_path}")
            return result_path

        except Exception as e:
            print_message(f"❌ 导出失败: {str(e)}")
            return ""

    def get_record_by_id(self, record_id: int) -> Optional[Dict]:
        """
        根据ID获取记录详情

        Args:
            record_id: 记录ID

        Returns:
            记录详情字典
        """
        try:
            records = self.manager.get_all_records()
            if 0 <= record_id < len(records):
                record = records[record_id]
                return {
                    "id": record.id,
                    "position": record.position,
                    "resume": record.resume,
                    "jd": record.jd,
                    "intelligence_requirement": record.intelligence_requirement,
                    "conversation": record.conversation,
                    "evaluation": record.evaluation,
                    "result": record.result,
                }
            else:
                print_message(f"❌ 记录ID {record_id} 不存在")
                return None
        except Exception as e:
            print_message(f"❌ 获取记录失败: {str(e)}")
            return None

    def __repr__(self):
        return f"DataAnalyzer(csv_path='{self.csv_path}')"
