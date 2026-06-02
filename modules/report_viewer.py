"""
报告查看模块

功能：
- 列出所有报告文件
- 查看报告详情
- 搜索报告
- 导出报告
"""

from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import json

from menglong.utils.log import print_message


class ReportViewer:
    """报告查看器 - 查看和管理面试报告"""

    def __init__(self, report_dirs: Optional[List[str]] = None):
        """
        初始化报告查看器

        Args:
            report_dirs: 报告目录列表，默认为 ["./", "temp/", "reports/", "experiences/"]
        """
        if report_dirs is None:
            report_dirs = ["./", "temp/", "reports/", "experiences/"]

        self.report_dirs = [Path(d) for d in report_dirs]

    def list_all_reports(self, report_type: Optional[str] = None) -> List[Dict]:
        """
        列出所有报告

        Args:
            report_type: 报告类型过滤
                - "interview": 面试结果
                - "experience": 经验提取
                - "plan": 面试方案
                - None: 所有类型

        Returns:
            报告列表
        """
        print_message("\n" + "=" * 60)
        print_message("📁 可用报告")
        print_message("=" * 60)

        # 定义报告模式
        patterns = {
            "interview": ["interview_result_*.json", "real_data_interview_*.json"],
            "experience": [
                "exp_*.json",
                "interview_experience_*.json",
                "general_interview_*.json",
            ],
            "plan": ["interview_plan_*.json", "complete_plan_*.json"],
        }

        # 收集报告
        all_reports = []

        if report_type:
            search_patterns = patterns.get(report_type, [])
        else:
            search_patterns = []
            for pats in patterns.values():
                search_patterns.extend(pats)

        for directory in self.report_dirs:
            if not directory.exists():
                continue

            for pattern in search_patterns:
                for filepath in directory.glob(pattern):
                    # 获取文件信息
                    stat = filepath.stat()
                    report_info = {
                        "filename": filepath.name,
                        "path": str(filepath),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                        "type": self._detect_report_type(filepath.name),
                    }
                    all_reports.append(report_info)

        # 按修改时间排序
        all_reports.sort(key=lambda x: x["modified"], reverse=True)

        # 显示
        if not all_reports:
            print_message("\n暂无报告文件")
            return []

        print_message(f"\n共找到 {len(all_reports)} 个报告:\n")

        for i, report in enumerate(all_reports[:20], 1):  # 最多显示20个
            size_kb = report["size"] / 1024
            print_message(
                f"{i:2d}. [{report['type']:12s}] {report['filename']:50s} "
                f"({size_kb:.1f}KB, {report['modified'].strftime('%Y-%m-%d %H:%M')})"
            )

        if len(all_reports) > 20:
            print_message(f"\n... 还有 {len(all_reports) - 20} 个报告未显示")

        print_message("\n" + "=" * 60)

        return all_reports

    def view_report(
        self, filepath: str, show_full: bool = False, max_length: int = 2000
    ) -> Dict:
        """
        查看报告内容

        Args:
            filepath: 报告文件路径
            show_full: 是否显示完整内容
            max_length: 非完整模式下的最大显示长度

        Returns:
            报告内容
        """
        print_message("\n" + "=" * 60)
        print_message(f"📄 报告: {Path(filepath).name}")
        print_message("=" * 60)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 显示基本信息
            report_type = self._detect_report_type(Path(filepath).name)
            print_message(f"\n类型: {report_type}")

            # 根据类型显示关键信息
            if report_type == "面试结果":
                self._display_interview_result(data, show_full, max_length)
            elif report_type == "经验提取":
                self._display_experience_report(data, show_full, max_length)
            elif report_type == "面试方案":
                self._display_interview_plan(data, show_full, max_length)
            else:
                self._display_generic_report(data, show_full, max_length)

            return data

        except Exception as e:
            print_message(f"\n❌ 读取报告失败: {str(e)}")
            return {"error": str(e)}

    def search_reports(self, keyword: str) -> List[Dict]:
        """
        搜索报告

        Args:
            keyword: 搜索关键词

        Returns:
            匹配的报告列表
        """
        print_message(f"\n🔍 搜索关键词: '{keyword}'")

        all_reports = self.list_all_reports()
        matched = []

        for report in all_reports:
            try:
                with open(report["path"], "r", encoding="utf-8") as f:
                    content = f.read()

                if keyword.lower() in content.lower():
                    matched.append(report)
            except Exception:
                continue

        print_message(f"\n找到 {len(matched)} 个匹配的报告")

        return matched

    def export_report(
        self,
        filepath: str,
        output_format: str = "txt",
        output_path: Optional[str] = None,
    ) -> str:
        """
        导出报告为其他格式

        Args:
            filepath: 报告文件路径
            output_format: 输出格式 (txt/md/json)
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        print_message(f"\n💾 导出报告为 {output_format.upper()} 格式...")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 生成输出路径
            if output_path is None:
                stem = Path(filepath).stem
                output_path = f"reports/{stem}.{output_format}"

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # 导出
            if output_format == "txt":
                self._export_to_text(data, output_path)
            elif output_format == "md":
                self._export_to_markdown(data, output_path)
            elif output_format == "json":
                self._export_to_json(data, output_path)
            else:
                print_message(f"❌ 不支持的格式: {output_format}")
                return ""

            print_message(f"✓ 已导出到: {output_path}")
            return output_path

        except Exception as e:
            print_message(f"❌ 导出失败: {str(e)}")
            return ""

    def _detect_report_type(self, filename: str) -> str:
        """检测报告类型"""
        if "interview_result" in filename or "real_data_interview" in filename:
            return "面试结果"
        elif "exp_" in filename or "experience" in filename:
            return "经验提取"
        elif "plan" in filename:
            return "面试方案"
        else:
            return "其他"

    def _display_interview_result(self, data: Dict, show_full: bool, max_length: int):
        """显示面试结果"""
        print_message(
            f"\n岗位: {data.get('candidate_info', {}).get('position', 'N/A')}"
        )
        print_message(f"模式: {data.get('interview_mode', 'N/A')}")
        print_message(f"轮次: {data.get('total_rounds', 0)}")

        final_eval = data.get("final_evaluation", {})
        avg_scores = final_eval.get("average_scores", {})

        print_message("\n平均评分:")
        for dimension, score in avg_scores.items():
            print_message(f"  • {dimension}: {score}/100")

        if show_full:
            print_message("\n完整评估:")
            print_message(final_eval.get("final_evaluation", "N/A"))
        else:
            eval_text = final_eval.get("final_evaluation", "")
            if len(eval_text) > max_length:
                print_message(f"\n评估摘要:\n{eval_text[:max_length]}...")
            else:
                print_message(f"\n评估:\n{eval_text}")

    def _display_experience_report(self, data: Dict, show_full: bool, max_length: int):
        """显示经验提取报告"""
        print_message(f"\n经验ID: {data.get('id', 'N/A')}")
        print_message(f"岗位: {data.get('position', 'N/A')}")
        print_message(f"提取时间: {data.get('extracted_at', 'N/A')}")

        content = data.get("content", "")
        if show_full:
            print_message(f"\n经验内容:\n{content}")
        else:
            if len(content) > max_length:
                print_message(f"\n经验摘要:\n{content[:max_length]}...")
            else:
                print_message(f"\n经验内容:\n{content}")

    def _display_interview_plan(self, data: Dict, show_full: bool, max_length: int):
        """显示面试方案"""
        print_message(f"\n岗位: {data.get('position', 'N/A')}")
        print_message(f"生成时间: {data.get('prepared_at', 'N/A')}")

        questions = data.get("questions", {}).get("generated_questions", "")
        if show_full:
            print_message(f"\n面试问题:\n{questions}")
        else:
            if len(questions) > max_length:
                print_message(f"\n问题摘要:\n{questions[:max_length]}...")
            else:
                print_message(f"\n面试问题:\n{questions}")

    def _display_generic_report(self, data: Dict, show_full: bool, max_length: int):
        """显示通用报告"""
        import json

        json_str = json.dumps(data, ensure_ascii=False, indent=2)

        if show_full:
            print_message(f"\n{json_str}")
        else:
            if len(json_str) > max_length:
                print_message(f"\n{json_str[:max_length]}...")
            else:
                print_message(f"\n{json_str}")

    def _export_to_text(self, data: Dict, output_path: str):
        """导出为文本"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))

    def _export_to_markdown(self, data: Dict, output_path: str):
        """导出为Markdown"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 面试报告\n\n")
            f.write(f"```json\n{json.dumps(data, ensure_ascii=False, indent=2)}\n```\n")

    def _export_to_json(self, data: Dict, output_path: str):
        """导出为JSON"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def __repr__(self):
        return f"ReportViewer(dirs={len(self.report_dirs)})"
