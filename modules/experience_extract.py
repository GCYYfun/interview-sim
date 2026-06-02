"""
经验提取模块

功能：
- 从面试记录中提取经验
- 整合多个经验形成通用指南
- 使用 ExperienceAgent 和 manager.InterviewDataManager
"""

from typing import List, Dict, Optional
from pathlib import Path

from agents import ExperienceAgent
from manager.interview_data_manager import InterviewDataManager
from manager.models import ManagerConfig, ExtractionMode, ExtractionParams
from menglong.utils.log import print_message


class ExperienceExtractor:
    """经验提取器 - 从面试数据中提取通用经验"""

    def __init__(self, csv_path: str = "interview_data.csv"):
        """
        初始化经验提取器

        Args:
            csv_path: CSV数据文件路径
        """
        self.csv_path = csv_path
        config = ManagerConfig(data_source=csv_path)
        self.manager = InterviewDataManager(config)
        self.exp_agent = ExperienceAgent()

        # 加载数据
        print_message(f"📂 加载数据: {csv_path}")
        load_result = self.manager.load_data()
        if load_result.success:
            print_message(f"✓ 成功加载 {load_result.total_records} 条记录")
        else:
            print_message(f"❌ 加载失败: {load_result.error_message}")

    def extract_from_records(
        self,
        record_indices: List[int],
        mode: str = "incremental",
        save_results: bool = True,
    ) -> Dict:
        """
        从指定记录中提取经验

        Args:
            record_indices: 记录索引列表（从0开始）
            mode: 提取模式
                - "incremental": 增量提取（基于已有经验）
                - "review": 复盘模式（重新审视）
                - "full_refresh": 全量刷新
            save_results: 是否保存结果

        Returns:
            提取结果字典
        """
        print_message("\n" + "=" * 60)
        print_message(f"🧠 经验提取 - {mode.upper()}模式")
        print_message("=" * 60)

        print_message(f"\n已选择 {len(record_indices)} 条记录进行提取")

        # 映射提取模式
        mode_map = {
            "incremental": ExtractionMode.INCREMENTAL,
            "review": ExtractionMode.REVIEW,
            "full_refresh": ExtractionMode.FULL_REFRESH,
        }

        extraction_mode = mode_map.get(mode, ExtractionMode.INCREMENTAL)

        # 使用manager提取经验
        try:
            params = ExtractionParams(
                record_indices=record_indices, mode=extraction_mode
            )

            print_message("\n🤖 开始提取经验...")
            result = self.manager.extract_experiences(params)

            if result.success:
                print_message("\n✓ 经验提取完成！")
                print_message(f"  • 处理记录数: {result.processed_count}")
                print_message(f"  • 提取经验数: {result.experience_count}")
                print_message(f"  • Token使用: {result.total_tokens}")
                print_message(f"  • 预估成本: ${result.total_cost:.4f}")

                if save_results and result.output_path:
                    print_message(f"  • 保存路径: {result.output_path}")

                return {
                    "success": True,
                    "processed_count": result.processed_count,
                    "experience_count": result.experience_count,
                    "total_tokens": result.total_tokens,
                    "total_cost": result.total_cost,
                    "output_path": result.output_path,
                }
            else:
                print_message(f"\n❌ 提取失败: {result.error_message}")
                return {
                    "success": False,
                    "error": result.error_message,
                }

        except Exception as e:
            print_message(f"\n❌ 提取出错: {str(e)}")
            return {"success": False, "error": str(e)}

    def extract_from_selection(
        self, selection_str: str, mode: str = "incremental"
    ) -> Dict:
        """
        从选择字符串提取经验

        Args:
            selection_str: 选择字符串，如 "1,3,5-10"
            mode: 提取模式

        Returns:
            提取结果
        """
        # 解析选择字符串
        indices = self._parse_selection(selection_str)

        if not indices:
            print_message("❌ 选择格式错误或为空")
            return {"success": False, "error": "Invalid selection"}

        return self.extract_from_records(indices, mode)

    def consolidate_all_experiences(self, output_path: Optional[str] = None) -> Dict:
        """
        整合所有经验形成通用指南

        Args:
            output_path: 输出路径

        Returns:
            整合结果
        """
        print_message("\n" + "=" * 60)
        print_message("📚 整合所有经验")
        print_message("=" * 60)

        # 获取所有经验
        all_experiences = self.exp_agent.get_all_experiences()

        if not all_experiences:
            print_message("\n⚠️ 暂无可整合的经验，请先提取经验")
            return {"success": False, "error": "No experiences to consolidate"}

        print_message(f"\n发现 {len(all_experiences)} 个经验记录")

        # 整合
        print_message("\n🤖 开始整合...")
        guide = self.exp_agent.consolidate_experiences()

        if "error" in guide:
            print_message(f"\n❌ 整合失败: {guide['error']}")
            return {"success": False, "error": guide["error"]}

        print_message("\n✓ 整合完成！")
        print_message(f"  • 源经验数: {guide.get('source_count', 0)}")
        print_message(f"  • 覆盖岗位: {', '.join(guide.get('source_positions', []))}")

        # 保存
        if output_path is None:
            output_path = "experiences/consolidated_guide.json"

        save_result = self.exp_agent.save_experience(guide, output_path)

        if save_result.get("success"):
            print_message(f"  • 保存路径: {save_result['filepath']}")

        return {
            "success": True,
            "guide": guide,
            "output_path": save_result.get("filepath"),
        }

    def list_available_records(self) -> List[Dict]:
        """
        列出可用的面试记录

        Returns:
            记录列表
        """
        print_message("\n📋 可用的面试记录:")
        print_message("=" * 60)

        records = self.manager.get_all_records()

        if not records:
            print_message("暂无记录")
            return []

        record_list = []
        for i, record in enumerate(records):
            print_message(
                f"{i + 1}. ID:{record.id} | "
                f"岗位:{record.position} | "
                f"聪明度:{record.intelligence_requirement}/100"
            )

            record_list.append(
                {
                    "index": i,
                    "id": record.id,
                    "position": record.position,
                    "intelligence_requirement": record.intelligence_requirement,
                }
            )

        print_message(f"\n共 {len(records)} 条记录")
        print_message("=" * 60)

        return record_list

    def _parse_selection(self, selection_str: str) -> List[int]:
        """
        解析选择字符串

        Args:
            selection_str: 如 "1,3,5-10"

        Returns:
            索引列表（从0开始）
        """
        indices = []
        parts = selection_str.split(",")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            if "-" in part:
                try:
                    start, end = part.split("-")
                    start_idx = int(start.strip()) - 1  # 转为0-based
                    end_idx = int(end.strip()) - 1
                    indices.extend(range(start_idx, end_idx + 1))
                except ValueError:
                    print_message(f"⚠️ 忽略无效范围: {part}")
            else:
                try:
                    idx = int(part) - 1  # 转为0-based
                    indices.append(idx)
                except ValueError:
                    print_message(f"⚠️ 忽略无效索引: {part}")

        # 去重并排序
        return sorted(set(indices))

    def get_extraction_history(self) -> List[Dict]:
        """获取提取历史"""
        # 查找所有经验文件
        experiences_dir = Path("experiences")
        if not experiences_dir.exists():
            return []

        history = []
        for exp_file in experiences_dir.glob("exp_*.json"):
            import json

            try:
                with open(exp_file, "r", encoding="utf-8") as f:
                    exp_data = json.load(f)
                    history.append(
                        {
                            "id": exp_data.get("id"),
                            "position": exp_data.get("position"),
                            "extracted_at": exp_data.get("extracted_at"),
                            "file": str(exp_file),
                        }
                    )
            except Exception:
                continue

        return sorted(history, key=lambda x: x.get("extracted_at", ""), reverse=True)

    def __repr__(self):
        return f"ExperienceExtractor(csv_path='{self.csv_path}')"
