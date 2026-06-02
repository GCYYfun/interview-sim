"""
Rich 交互菜单

使用 Rich 库创建美观的控制台菜单界面
"""

from typing import Optional, Callable
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box


console = Console()


class MenuSystem:
    """菜单系统 - 使用 Rich 创建交互式菜单"""

    @staticmethod
    def show_main_menu() -> str:
        """显示主菜单"""
        console.clear()

        # 标题
        title = Panel.fit(
            "[bold cyan]面试模拟系统[/bold cyan]\n"
            "[dim]Interview Simulation System[/dim]",
            border_style="cyan",
            box=box.DOUBLE,
        )
        console.print(title)
        console.print()

        # 菜单选项
        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("选项", style="cyan", width=8)
        table.add_column("功能", style="white")

        table.add_row("1", "🎭 面试模拟 - 运行完整面试流程")
        table.add_row("2", "📊 数据分析 - 查看统计和分析数据")
        table.add_row("3", "💡 经验提取 - 从面试中提取经验")
        table.add_row("4", "🤝 面试助手 - 准备面试问题和分析")
        table.add_row("5", "📄 报告查看 - 浏览和导出报告")
        table.add_row("6", "🔍 对话评估 - 评估已有面试对话")
        table.add_row("0", "❌ 退出系统")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "[bold yellow]请选择功能[/bold yellow]",
            choices=["0", "1", "2", "3", "4", "5", "6"],
            default="0",
        )

        return choice

    @staticmethod
    def show_interview_menu() -> dict:
        """面试模拟菜单"""
        console.print("\n[bold cyan]═══ 面试模拟 ═══[/bold cyan]\n")

        # 选择模式
        mode = Prompt.ask("选择面试模式", choices=["auto", "manual"], default="auto")

        # 获取文件路径
        resume_path = Prompt.ask("简历文件路径", default="temp/demo_data.csv")
        jd_path = Prompt.ask("JD文件路径（可选，按Enter跳过）", default="")

        # 面试轮次
        max_rounds = int(Prompt.ask("最大面试轮次", default="5"))

        return {
            "mode": mode,
            "resume": resume_path,
            "jd": jd_path if jd_path else None,
            "max_rounds": max_rounds,
        }

    @staticmethod
    def show_data_analysis_menu() -> str:
        """数据分析菜单"""
        console.print("\n[bold cyan]═══ 数据分析 ═══[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("选项", style="cyan", width=8)
        table.add_column("功能", style="white")

        table.add_row("1", "📈 显示统计信息")
        table.add_row("2", "🔍 搜索候选人")
        table.add_row("3", "📊 生成可视化图表")
        table.add_row("4", "💾 导出摘要报告")
        table.add_row("0", "🔙 返回主菜单")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "请选择操作", choices=["0", "1", "2", "3", "4"], default="0"
        )

        return choice

    @staticmethod
    def show_experience_menu() -> str:
        """经验提取菜单"""
        console.print("\n[bold cyan]═══ 经验提取 ═══[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("选项", style="cyan", width=8)
        table.add_column("功能", style="white")

        table.add_row("1", "📝 从面试记录提取经验")
        table.add_row("2", "🔗 整合所有经验")
        table.add_row("3", "👁️  查看经验列表")
        table.add_row("4", "💾 导出经验")
        table.add_row("0", "🔙 返回主菜单")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "请选择操作", choices=["0", "1", "2", "3", "4"], default="0"
        )

        return choice

    @staticmethod
    def show_assistant_menu() -> str:
        """面试助手菜单"""
        console.print("\n[bold cyan]═══ 面试助手 ═══[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("选项", style="cyan", width=8)
        table.add_column("功能", style="white")

        table.add_row("1", "❓ 生成面试问题")
        table.add_row("2", "🎯 分析简历和JD匹配度")
        table.add_row("3", "📋 完整面试准备")
        table.add_row("4", "📜 从历史记录加载")
        table.add_row("0", "🔙 返回主菜单")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "请选择操作", choices=["0", "1", "2", "3", "4"], default="0"
        )

        return choice

    @staticmethod
    def show_report_menu() -> str:
        """报告查看菜单"""
        console.print("\n[bold cyan]═══ 报告查看 ═══[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("选项", style="cyan", width=8)
        table.add_column("功能", style="white")

        table.add_row("1", "📑 列出所有报告")
        table.add_row("2", "👁️  查看报告详情")
        table.add_row("3", "🔍 搜索报告")
        table.add_row("4", "💾 导出报告")
        table.add_row("0", "🔙 返回主菜单")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "请选择操作", choices=["0", "1", "2", "3", "4"], default="0"
        )

        return choice

    @staticmethod
    def show_eval_conversation_menu() -> str:
        """对话评估菜单"""
        console.print("\n[bold cyan]═══ 对话评估 ═══[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        table.add_column("选项", style="cyan", width=8)
        table.add_column("功能", style="white")

        table.add_row("1", "📝 评估单条记录（单轮次）")
        table.add_row("2", "🎯 评估单条记录（所有轮次）")
        table.add_row("3", "🔄 批量评估记录")
        table.add_row("4", "👁️  查看评估报告")
        table.add_row("5", "💾 导出评估结果")
        table.add_row("0", "🔙 返回主菜单")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "请选择操作", choices=["0", "1", "2", "3", "4", "5"], default="0"
        )

        return choice

    @staticmethod
    def get_search_criteria() -> dict:
        """获取搜索条件"""
        console.print("\n[bold]搜索条件[/bold]")

        position = Prompt.ask("岗位名称（可选）", default="")
        name = Prompt.ask("候选人姓名（可选）", default="")

        return {
            "position": position if position else None,
            "name": name if name else None,
        }

    @staticmethod
    def select_from_list(items: list, title: str = "选择项目") -> Optional[int]:
        """从列表中选择项目"""
        if not items:
            console.print("\n[yellow]没有可用项目[/yellow]")
            return None

        console.print(f"\n[bold]{title}[/bold]\n")

        for i, item in enumerate(items[:20], 1):  # 最多显示20个
            console.print(f"{i:2d}. {item}")

        if len(items) > 20:
            console.print(f"\n... 还有 {len(items) - 20} 个项目")

        console.print()

        choice = Prompt.ask("请选择（输入0取消）", default="0")

        try:
            idx = int(choice)
            if idx == 0:
                return None
            if 1 <= idx <= min(20, len(items)):
                return idx - 1
            else:
                console.print("[red]无效选择[/red]")
                return None
        except ValueError:
            console.print("[red]请输入数字[/red]")
            return None

    @staticmethod
    def confirm_action(message: str, default: bool = False) -> bool:
        """确认操作"""
        return Confirm.ask(f"\n{message}", default=default)

    @staticmethod
    def show_success(message: str):
        """显示成功消息"""
        console.print(f"\n[green]✓[/green] {message}\n")

    @staticmethod
    def show_error(message: str):
        """显示错误消息"""
        console.print(f"\n[red]✗[/red] {message}\n")

    @staticmethod
    def show_info(message: str):
        """显示信息"""
        console.print(f"\n[blue]ℹ[/blue] {message}\n")

    @staticmethod
    def pause():
        """暂停等待用户"""
        console.print()
        Prompt.ask("[dim]按 Enter 继续[/dim]", default="")

    @staticmethod
    def show_progress_message(message: str):
        """显示进度消息"""
        console.print(f"\n[yellow]⏳[/yellow] {message}...")

    @staticmethod
    def display_table(data: list[dict], title: str = "数据表格"):
        """显示数据表格"""
        if not data:
            console.print("\n[yellow]暂无数据[/yellow]")
            return

        # 创建表格
        table = Table(title=title, box=box.ROUNDED, show_lines=True)

        # 添加列
        if data:
            for key in data[0].keys():
                table.add_column(key, style="cyan", no_wrap=False)

        # 添加行
        for row in data[:50]:  # 最多显示50行
            table.add_row(*[str(v) for v in row.values()])

        console.print()
        console.print(table)

        if len(data) > 50:
            console.print(f"\n[dim]... 还有 {len(data) - 50} 行数据未显示[/dim]")


def create_menu_wrapper(menu_func: Callable) -> Callable:
    """创建菜单包装器，统一处理异常"""

    def wrapper(*args, **kwargs):
        try:
            return menu_func(*args, **kwargs)
        except KeyboardInterrupt:
            MenuSystem.show_info("操作已取消")
            return None
        except Exception as e:
            MenuSystem.show_error(f"发生错误: {str(e)}")
            return None

    return wrapper
