"""
CLI 主应用

使用 Rich 创建的控制台交互应用，统一调用所有功能模块
"""

import sys

from rich.console import Console
from rich.traceback import install

from cli.menus import MenuSystem
from modules import (
    InterviewSimulator,
    DataAnalyzer,
    ExperienceExtractor,
    InterviewAssistant,
    ReportViewer,
    ConversationEvaluator,
)

# 安装 Rich 异常处理
install(show_locals=True)

console = Console()


class InterviewSimApp:
    """面试模拟系统主应用"""

    def __init__(self):
        """初始化应用"""
        self.menu = MenuSystem()

        # 默认路径
        self.default_csv = "interview_data.csv"
        self.default_exp_dir = "experiences/"

        # 延迟初始化模块（按需创建）
        self._simulator = None
        self._analyzer = None
        self._extractor = None
        self._assistant = None
        self._viewer = None
        self._evaluator = None

    @property
    def simulator(self):
        """延迟初始化面试模拟器"""
        if self._simulator is None:
            self._simulator = InterviewSimulator()
        return self._simulator

    @property
    def analyzer(self):
        """延迟初始化数据分析器"""
        if self._analyzer is None:
            self._analyzer = DataAnalyzer(csv_path=self.default_csv)
        return self._analyzer

    @property
    def extractor(self):
        """延迟初始化经验提取器"""
        if self._extractor is None:
            self._extractor = ExperienceExtractor(
                csv_path=self.default_csv, exp_dir=self.default_exp_dir
            )
        return self._extractor

    @property
    def assistant(self):
        """延迟初始化面试助手"""
        if self._assistant is None:
            self._assistant = InterviewAssistant()
        return self._assistant

    @property
    def viewer(self):
        """延迟初始化报告查看器"""
        if self._viewer is None:
            self._viewer = ReportViewer()
        return self._viewer

    @property
    def evaluator(self):
        """延迟初始化对话评估器"""
        if self._evaluator is None:
            self._evaluator = ConversationEvaluator(csv_path=self.default_csv)
        return self._evaluator

    def run(self):
        """运行主应用"""
        while True:
            choice = self.menu.show_main_menu()

            if choice == "0":
                self.menu.show_info("感谢使用！再见！")
                break
            elif choice == "1":
                self.handle_interview_simulation()
            elif choice == "2":
                self.handle_data_analysis()
            elif choice == "3":
                self.handle_experience_extraction()
            elif choice == "4":
                self.handle_interview_assistant()
            elif choice == "5":
                self.handle_report_viewer()
            elif choice == "6":
                self.handle_eval_conversation()

    def handle_interview_simulation(self):
        """处理面试模拟"""
        try:
            # 获取参数
            params = self.menu.show_interview_menu()

            if not params:
                return

            # 确认执行
            if not self.menu.confirm_action("开始面试模拟？", default=True):
                return

            self.menu.show_progress_message("准备面试环境")

            # 运行面试
            result = self.simulator.run_interview(
                resume=params["resume"],
                jd=params.get("jd"),
                mode=params["mode"],
                max_rounds=params["max_rounds"],
            )

            if result:
                self.menu.show_success("面试模拟完成！")
                console.print(f"\n结果已保存: {result.get('output_file', 'N/A')}")
            else:
                self.menu.show_error("面试模拟失败")

        except Exception as e:
            self.menu.show_error(f"面试模拟出错: {str(e)}")
        finally:
            self.menu.pause()

    def handle_data_analysis(self):
        """处理数据分析"""
        while True:
            choice = self.menu.show_data_analysis_menu()

            if choice == "0":
                break

            try:
                if choice == "1":
                    # 显示统计
                    self.analyzer.show_statistics()

                elif choice == "2":
                    # 搜索候选人
                    criteria = self.menu.get_search_criteria()
                    results = self.analyzer.search_candidates(**criteria)

                    if results:
                        self.menu.display_table(
                            [r.__dict__ for r in results], title="搜索结果"
                        )

                elif choice == "3":
                    # 生成可视化
                    output_dir = "charts/"
                    self.menu.show_progress_message("生成可视化图表")
                    self.analyzer.generate_visualizations(output_dir=output_dir)
                    self.menu.show_success(f"图表已生成到: {output_dir}")

                elif choice == "4":
                    # 导出摘要
                    output_path = "reports/summary.json"
                    self.analyzer.export_summary(output_path=output_path)
                    self.menu.show_success(f"摘要已导出到: {output_path}")

            except Exception as e:
                self.menu.show_error(f"操作失败: {str(e)}")
            finally:
                self.menu.pause()

    def handle_experience_extraction(self):
        """处理经验提取"""
        while True:
            choice = self.menu.show_experience_menu()

            if choice == "0":
                break

            try:
                if choice == "1":
                    # 从记录提取
                    record_ids_str = console.input("\n请输入记录ID（逗号分隔）: ")
                    record_ids = [int(x.strip()) for x in record_ids_str.split(",")]
                    position = console.input("岗位名称（可选）: ").strip()

                    self.menu.show_progress_message("提取经验")
                    experience = self.extractor.extract_from_records(
                        record_ids=record_ids, position=position if position else None
                    )

                    if experience:
                        self.menu.show_success("经验提取完成！")

                elif choice == "2":
                    # 整合所有经验
                    if self.menu.confirm_action("整合所有经验？这可能需要一些时间"):
                        self.menu.show_progress_message("整合经验")
                        result = self.extractor.consolidate_all_experiences()
                        if result:
                            self.menu.show_success("经验整合完成！")

                elif choice == "3":
                    # 查看经验
                    self.extractor.view_experiences()

                elif choice == "4":
                    # 导出经验
                    output_path = "reports/experiences.json"
                    self.extractor.export_experiences(output_path=output_path)
                    self.menu.show_success(f"经验已导出到: {output_path}")

            except Exception as e:
                self.menu.show_error(f"操作失败: {str(e)}")
            finally:
                self.menu.pause()

    def handle_interview_assistant(self):
        """处理面试助手"""
        while True:
            choice = self.menu.show_assistant_menu()

            if choice == "0":
                break

            try:
                if choice == "1":
                    # 生成问题
                    resume = console.input("\n简历文件路径: ").strip()
                    jd = console.input("JD文件路径: ").strip()

                    self.menu.show_progress_message("生成面试问题")
                    questions = self.assistant.generate_questions(resume, jd)

                    if questions:
                        self.menu.show_success("问题生成完成！")

                elif choice == "2":
                    # 分析匹配度
                    resume = console.input("\n简历文件路径: ").strip()
                    jd = console.input("JD文件路径: ").strip()

                    self.menu.show_progress_message("分析匹配度")
                    result = self.assistant.analyze_match(resume, jd)

                    if result:
                        self.menu.show_success("匹配度分析完成！")

                elif choice == "3":
                    # 完整准备
                    resume = console.input("\n简历文件路径: ").strip()
                    jd = console.input("JD文件路径: ").strip()

                    self.menu.show_progress_message("准备面试方案")
                    plan = self.assistant.prepare_interview(resume, jd)

                    if plan:
                        self.menu.show_success("面试准备完成！")
                        console.print(f"\n方案已保存: {plan.get('output_file', 'N/A')}")

                elif choice == "4":
                    # 从历史加载
                    record_id = int(console.input("\n请输入记录ID: ").strip())

                    self.menu.show_progress_message("加载历史数据")
                    result = self.assistant.load_from_history(record_id)

                    if result:
                        self.menu.show_success("历史数据加载完成！")

            except Exception as e:
                self.menu.show_error(f"操作失败: {str(e)}")
            finally:
                self.menu.pause()

    def handle_report_viewer(self):
        """处理报告查看"""
        while True:
            choice = self.menu.show_report_menu()

            if choice == "0":
                break

            try:
                if choice == "1":
                    # 列出报告
                    type_filter = console.input(
                        "\n报告类型（interview/experience/plan，留空显示全部）: "
                    ).strip()
                    self.viewer.list_all_reports(
                        report_type=type_filter if type_filter else None
                    )

                elif choice == "2":
                    # 查看报告
                    filepath = console.input("\n报告文件路径: ").strip()
                    show_full = self.menu.confirm_action(
                        "显示完整内容？", default=False
                    )

                    self.viewer.view_report(filepath, show_full=show_full)

                elif choice == "3":
                    # 搜索报告
                    keyword = console.input("\n搜索关键词: ").strip()
                    self.viewer.search_reports(keyword)

                elif choice == "4":
                    # 导出报告
                    filepath = console.input("\n报告文件路径: ").strip()
                    output_format = (
                        console.input("导出格式（txt/md/json）[json]: ").strip()
                        or "json"
                    )
                    output_path = console.input("输出路径（可选）: ").strip()

                    self.viewer.export_report(
                        filepath=filepath,
                        output_format=output_format,
                        output_path=output_path if output_path else None,
                    )

            except Exception as e:
                self.menu.show_error(f"操作失败: {str(e)}")
            finally:
                self.menu.pause()

    def handle_eval_conversation(self):
        """处理对话评估"""
        while True:
            choice = self.menu.show_eval_conversation_menu()

            if choice == "0":
                break

            try:
                if choice == "1":
                    # 评估单条记录
                    record_id = int(console.input("\n请输入记录ID: ").strip())
                    round_name = (
                        console.input(
                            "选择轮次 (First Round/Second Round/Final Round) [First Round]: "
                        ).strip()
                        or "First Round"
                    )

                    self.menu.show_progress_message("评估对话中")
                    result = self.evaluator.evaluate_record_by_id(
                        record_id=record_id, round_name=round_name
                    )

                    if "error" in result:
                        self.menu.show_error(f"评估失败: {result['error']}")
                    else:
                        self.menu.show_success("对话评估完成！")
                        console.print(f"\n{result.get('summary', 'N/A')}")

                        # 询问是否保存
                        if self.menu.confirm_action("是否保存评估报告？", default=True):
                            output_path = self.evaluator.export_evaluation_report(
                                result
                            )
                            console.print(f"\n已保存到: {output_path}")

                elif choice == "2":
                    # 评估所有轮次
                    record_id = int(console.input("\n请输入记录ID: ").strip())

                    self.menu.show_progress_message("评估所有轮次中")
                    result = self.evaluator.evaluate_all_rounds(record_id=record_id)

                    if "error" in result:
                        self.menu.show_error(f"评估失败: {result['error']}")
                    else:
                        self.menu.show_success("所有轮次评估完成！")

                        # 显示总体摘要
                        summary = result.get("summary", {})
                        console.print("\n" + "=" * 60)
                        console.print(f"岗位: {result.get('job_title', 'N/A')}")
                        console.print(
                            f"有效轮次: {summary.get('valid_rounds', 0)}/{summary.get('total_rounds', 3)}"
                        )

                        if summary.get("average_scores"):
                            console.print("\n各维度平均分:")
                            for dimension, score in summary["average_scores"].items():
                                console.print(f"  • {dimension}: {score:.1f}/100")
                            console.print(
                                f"\n总体平均分: {summary.get('overall_average', 0):.1f}/100"
                            )

                        if summary.get("performance_trend"):
                            console.print(f"\n{summary['performance_trend']}")

                        console.print("=" * 60)

                        # 显示各轮次详情
                        console.print("\n各轮次详情:")
                        for round_name, round_result in result.get(
                            "rounds", {}
                        ).items():
                            status = round_result.get("status", "unknown")
                            if status == "success":
                                eval_data = round_result.get("evaluation", {})
                                scores = eval_data.get("scores", {})
                                avg = (
                                    sum(scores.values()) / len(scores) if scores else 0
                                )
                                console.print(
                                    f"\n  [{round_name}] ✓ 平均分: {avg:.1f}/100"
                                )
                            elif status == "no_data":
                                console.print(f"\n  [{round_name}] ⚠️ 无数据")
                            else:
                                console.print(f"\n  [{round_name}] ✗ 失败")

                        # 询问是否保存
                        if self.menu.confirm_action("是否保存评估报告？", default=True):
                            output_path = self.evaluator.export_evaluation_report(
                                result
                            )
                            console.print(f"\n已保存到: {output_path}")

                elif choice == "3":
                    # 批量评估
                    ids_input = console.input(
                        "\n请输入记录ID列表（逗号分隔，留空评估前10条）: "
                    ).strip()

                    if ids_input:
                        record_ids = [int(x.strip()) for x in ids_input.split(",")]
                    else:
                        record_ids = None

                    round_name = (
                        console.input(
                            "选择轮次 (First Round/Second Round/Final Round) [First Round]: "
                        ).strip()
                        or "First Round"
                    )

                    max_records = int(
                        console.input("最大评估数量 [10]: ").strip() or "10"
                    )

                    self.menu.show_progress_message("批量评估中")
                    results = self.evaluator.batch_evaluate(
                        record_ids=record_ids,
                        round_name=round_name,
                        max_records=max_records,
                    )

                    success_count = sum(1 for r in results if "error" not in r)
                    self.menu.show_success(
                        f"批量评估完成！成功: {success_count}/{len(results)}"
                    )

                    # 询问是否保存
                    if self.menu.confirm_action("是否保存批量评估报告？", default=True):
                        import json
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = f"reports/batch_eval_{timestamp}.json"

                        import os

                        os.makedirs("reports", exist_ok=True)

                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)

                        console.print(f"\n已保存到: {output_path}")

                elif choice == "4":
                    # 查看评估报告
                    import glob
                    import json

                    reports = glob.glob("reports/eval_conversation_*.json") + glob.glob(
                        "reports/batch_eval_*.json"
                    )

                    if not reports:
                        self.menu.show_info("暂无评估报告")
                    else:
                        console.print(f"\n找到 {len(reports)} 个评估报告：\n")
                        for i, report_path in enumerate(reports[:20], 1):
                            console.print(f"{i}. {report_path}")

                        choice_idx = console.input("\n选择报告编号（0取消）: ").strip()
                        if choice_idx and choice_idx != "0":
                            idx = int(choice_idx) - 1
                            if 0 <= idx < len(reports):
                                with open(reports[idx], "r", encoding="utf-8") as f:
                                    data = json.load(f)
                                console.print(
                                    f"\n{json.dumps(data, ensure_ascii=False, indent=2)}"
                                )

                elif choice == "5":
                    # 导出评估结果
                    self.menu.show_info("评估结果已自动保存在 reports/ 目录")

            except Exception as e:
                self.menu.show_error(f"操作失败: {str(e)}")
            finally:
                self.menu.pause()


def main():
    """主入口函数"""
    try:
        app = InterviewSimApp()
        app.run()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]程序已中断[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]程序异常: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
