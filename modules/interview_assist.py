"""
面试助手模块

功能：
- 生成面试问题
- 分析候选人匹配度
- 保存面试方案
- 使用 InterviewAgent
"""

from typing import Dict, Optional
from pathlib import Path

from agents import InterviewAgent
from manager.interview_data_manager import InterviewDataManager
from manager.models import ManagerConfig
from menglong.utils.log import print_message


class InterviewAssistant:
    """面试助手 - 面试准备和问题生成"""

    def __init__(self, csv_path: Optional[str] = None):
        """
        初始化面试助手

        Args:
            csv_path: CSV数据文件路径（可选，用于从历史数据加载候选人）
        """
        self.interview_agent = InterviewAgent()
        self.csv_path = csv_path
        self.manager = None

        if csv_path:
            config = ManagerConfig(data_source=csv_path)
            self.manager = InterviewDataManager(config)
            self.manager.load_data()

    def generate_questions(
        self,
        resume: str,
        jd: str,
        position: str = "未指定",
        focus_areas: Optional[list] = None,
        save_plan: bool = True,
    ) -> Dict:
        """
        生成面试问题

        Args:
            resume: 候选人简历
            jd: 岗位描述
            position: 岗位名称
            focus_areas: 重点评估维度
            save_plan: 是否保存面试方案

        Returns:
            问题生成结果
        """
        print_message("\n" + "=" * 60)
        print_message("💡 面试问题生成")
        print_message("=" * 60)

        if focus_areas is None:
            focus_areas = ["聪明度", "皮实", "勤奋"]

        print_message(f"\n岗位: {position}")
        print_message(f"评估维度: {', '.join(focus_areas)}")

        # 生成问题
        print_message("\n🤖 生成针对性面试问题...")
        result = self.interview_agent.generate_interview_questions(
            resume=resume, jd=jd, focus_areas=focus_areas, position=position
        )

        if "error" in result:
            print_message(f"\n❌ 生成失败: {result['error']}")
            return result

        print_message("\n✓ 问题生成完成！")

        # 显示摘要
        print_message("\n📋 问题方案摘要:")
        questions_text = result.get("generated_questions", "")
        print_message(questions_text[:500] + "...")

        # Token使用
        token_usage = result.get("token_usage", {})
        print_message(f"\nToken使用: {token_usage.get('total_tokens', 0)}")

        # 保存
        if save_plan:
            filepath = self.interview_agent.save_interview_plan(result)
            print_message(f"\n💾 面试方案已保存: {filepath}")
            result["saved_path"] = filepath

        return result

    def analyze_match(self, resume: str, jd: str, position: str = "未指定") -> Dict:
        """
        分析候选人匹配度

        Args:
            resume: 候选人简历
            jd: 岗位描述
            position: 岗位名称

        Returns:
            匹配度分析结果
        """
        print_message("\n" + "=" * 60)
        print_message("🔍 候选人匹配度分析")
        print_message("=" * 60)

        print_message(f"\n岗位: {position}")

        # 分析
        print_message("\n🤖 分析中...")
        result = self.interview_agent.analyze_candidate_match(
            resume=resume, jd=jd, position=position
        )

        if "error" in result:
            print_message(f"\n❌ 分析失败: {result['error']}")
            return result

        print_message("\n✓ 分析完成！")

        # 显示摘要
        print_message("\n📊 匹配度分析摘要:")
        analysis_text = result.get("match_analysis", "")
        print_message(analysis_text[:500] + "...")

        # Token使用
        token_usage = result.get("token_usage", {})
        print_message(f"\nToken使用: {token_usage.get('total_tokens', 0)}")

        return result

    def prepare_interview(
        self,
        resume: str,
        jd: str,
        position: str = "未指定",
        candidate_id: Optional[str] = None,
    ) -> Dict:
        """
        完整的面试准备（问题生成 + 匹配度分析）

        Args:
            resume: 候选人简历
            jd: 岗位描述
            position: 岗位名称
            candidate_id: 候选人ID

        Returns:
            完整的准备结果
        """
        print_message("\n" + "=" * 60)
        print_message("🎯 面试准备（完整方案）")
        print_message("=" * 60)

        # 1. 生成问题
        questions_result = self.generate_questions(
            resume, jd, position, save_plan=False
        )

        # 2. 分析匹配度
        match_result = self.analyze_match(resume, jd, position)

        # 3. 整合结果
        complete_plan = {
            "position": position,
            "candidate_id": candidate_id,
            "questions": questions_result,
            "match_analysis": match_result,
            "prepared_at": questions_result.get("generation_time"),
        }

        # 4. 保存完整方案
        filepath = self._save_complete_plan(complete_plan, candidate_id)
        print_message(f"\n💾 完整面试方案已保存: {filepath}")
        complete_plan["saved_path"] = filepath

        return complete_plan

    def load_candidate_from_data(self, record_id: int) -> Optional[Dict]:
        """
        从历史数据加载候选人

        Args:
            record_id: 记录ID（从0开始）

        Returns:
            候选人数据
        """
        if not self.manager:
            print_message("❌ 未加载数据文件")
            return None

        try:
            records = self.manager.get_all_records()
            if 0 <= record_id < len(records):
                record = records[record_id]
                return {
                    "resume": record.resume,
                    "jd": record.jd,
                    "position": record.position,
                    "intelligence_requirement": record.intelligence_requirement,
                }
            else:
                print_message(f"❌ 记录ID {record_id} 不存在")
                return None
        except Exception as e:
            print_message(f"❌ 加载失败: {str(e)}")
            return None

    def quick_generate_for_record(self, record_id: int) -> Dict:
        """
        快速为历史记录生成面试方案

        Args:
            record_id: 记录ID

        Returns:
            面试方案
        """
        candidate_data = self.load_candidate_from_data(record_id)
        if not candidate_data:
            return {"error": "无法加载候选人数据"}

        return self.prepare_interview(
            resume=candidate_data["resume"],
            jd=candidate_data["jd"],
            position=candidate_data["position"],
            candidate_id=f"record_{record_id}",
        )

    def _save_complete_plan(
        self, plan: Dict, candidate_id: Optional[str] = None
    ) -> str:
        """保存完整面试方案"""
        from datetime import datetime
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if candidate_id:
            filename = f"complete_plan_{candidate_id}_{timestamp}.json"
        else:
            filename = f"complete_plan_{timestamp}.json"

        filepath = Path("temp") / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        return str(filepath)

    def __repr__(self):
        data_status = "已加载数据" if self.manager else "未加载数据"
        return (
            f"InterviewAssistant(interview_agent={self.interview_agent}, {data_status})"
        )
