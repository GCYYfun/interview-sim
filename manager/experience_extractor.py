"""
经验提取器模块

使用AI模型提取和整合面试经验
"""

import logging
import re
from typing import List, Optional
from datetime import datetime

from menglong import Model
from menglong.ml_model.schema.ml_request import UserMessage as user

from manager.models import (
    Experience,
    ExtractionParams,
    ExtractionResult,
    TokenStats,
    ManagerConfig,
    ExtractionMode,
    Record,
)

logger = logging.getLogger(__name__)


class ExperienceExtractor:
    """
    经验提取器

    功能:
    - 从单个面试记录提取经验
    - 批量提取经验
    - 整合多个经验
    - Token使用统计
    """

    # 定价表（每1K tokens的价格，单位：美元）
    PRICING = {
        "us.anthropic.claude-sonnet-4-20250514-v1:0": {
            "input": 0.003,
            "output": 0.015,
        },
        "claude-sonnet-4-20250514": {
            "input": 0.003,
            "output": 0.015,
        },
    }

    def __init__(self, config: ManagerConfig):
        """
        初始化经验提取器

        Args:
            config: 管理器配置
        """
        self.config = config
        self.model = Model()
        self.token_stats = TokenStats()

    def extract_single(self, record: Record) -> Optional[Experience]:
        """
        从单个记录提取经验

        Args:
            record: 面试记录

        Returns:
            Experience对象，如果提取失败返回None
        """
        try:
            logger.info(f"正在提取记录 {record.id} 的经验...")

            # 构建提示词
            prompt = self._build_extraction_prompt(
                resume=record.resume,
                jd=record.jd,
                conversation=record.conversation,
                evaluation=record.evaluation,
            )

            # 调用AI模型
            response = self.model.chat([user(content=prompt)])

            # 提取响应文本
            response_text = self._extract_response_text(response)

            # 更新Token统计
            self._update_token_stats(response)

            # 创建Experience对象
            experience_id = (
                f"exp_{record.id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
            )

            experience = Experience(
                id=experience_id,
                record_id=record.id,
                content=response_text,
                timestamp=datetime.now(),
                token_stats=self.token_stats.to_dict(),
                metadata={
                    "resume_summary": record.resume[:200] + "..."
                    if len(record.resume) > 200
                    else record.resume,
                    "position": record.position,
                },
            )

            logger.info(f"✅ 成功提取记录 {record.id} 的经验")
            return experience

        except Exception as e:
            logger.error(f"提取记录 {record.id} 的经验失败: {e}")
            return None

    def extract_batch(
        self, records: List[Record], params: ExtractionParams
    ) -> ExtractionResult:
        """
        批量提取经验

        Args:
            records: 记录列表
            params: 提取参数

        Returns:
            ExtractionResult: 提取结果
        """
        logger.info(f"开始批量提取 {len(records)} 条记录的经验...")

        # 重置token统计
        self.token_stats = TokenStats()

        experiences = []
        errors = []

        for i, record in enumerate(records, 1):
            logger.info(f"处理第 {i}/{len(records)} 条记录 (ID: {record.id})...")

            experience = self.extract_single(record)

            if experience:
                experiences.append(experience)
            else:
                error_msg = f"记录 {record.id} 提取失败"
                errors.append(error_msg)

        # 如果需要自动整合
        integrated_experience = None
        if params.auto_integrate and experiences:
            logger.info(f"正在整合 {len(experiences)} 条经验...")
            integrated_experience = self.integrate_experiences(experiences, params.mode)

        result = ExtractionResult(
            total_records=len(records),
            successful_extractions=len(experiences),
            failed_extractions=len(records) - len(experiences),
            experiences=experiences,
            integrated_experience=integrated_experience,
            token_stats=self.token_stats,
            errors=errors,
        )

        # 打印统计信息
        self._print_stats()

        return result

    def integrate_experiences(
        self,
        experiences: List[Experience],
        mode: ExtractionMode = ExtractionMode.INCREMENTAL,
    ) -> str:
        """
        整合多个经验成为综合指导

        Args:
            experiences: 经验列表
            mode: 整合模式

        Returns:
            整合后的经验文本
        """
        try:
            if not experiences:
                return "暂无经验可整合"

            logger.info(f"正在整合 {len(experiences)} 条经验，模式: {mode.value}...")

            # 根据模式选择不同的整合策略
            if mode == ExtractionMode.INCREMENTAL:
                prompt = self._build_incremental_prompt(experiences)
            elif mode == ExtractionMode.REVIEW:
                prompt = self._build_review_prompt(experiences)
            elif mode == ExtractionMode.FULL_REFRESH:
                prompt = self._build_full_refresh_prompt(experiences)
            else:
                logger.warning(f"未知模式 {mode}，使用默认增量模式")
                prompt = self._build_incremental_prompt(experiences)

            # 调用AI进行整合
            response = self.model.chat([user(content=prompt)])

            # 提取响应文本
            integrated_text = self._extract_response_text(response)

            # 更新Token统计
            self._update_token_stats(response)

            logger.info("✅ 经验整合完成")
            return integrated_text

        except Exception as e:
            logger.error(f"整合经验失败: {e}")
            return "整合经验时发生错误"

    def _build_extraction_prompt(
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

    def _build_incremental_prompt(self, experiences: List[Experience]) -> str:
        """构建增量更新的prompt"""
        max_experiences = 5
        max_chars_per_experience = 3000

        if len(experiences) > max_experiences:
            experiences = experiences[-max_experiences:]

        experiences_text = ""
        for i, exp in enumerate(experiences, 1):
            exp_text = exp.content
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

    def _build_review_prompt(self, experiences: List[Experience]) -> str:
        """构建温故知新的prompt"""
        max_experiences = 8
        max_chars_per_experience = 2000

        if len(experiences) > max_experiences:
            historical = experiences[: max_experiences // 2]
            recent = experiences[-max_experiences // 2 :]
            experiences = historical + recent

        experiences_text = ""
        for i, exp in enumerate(experiences, 1):
            exp_text = exp.content
            if len(exp_text) > max_chars_per_experience:
                exp_text = exp_text[:max_chars_per_experience] + "..."
            timestamp = (
                exp.timestamp.strftime("%Y-%m-%d") if exp.timestamp else "未知时间"
            )
            experiences_text += f"\n## 经验{i} ({timestamp}):\n{exp_text}\n"

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

    def _build_full_refresh_prompt(self, experiences: List[Experience]) -> str:
        """构建全量重新总结的prompt"""
        max_experiences = 10
        max_chars_per_experience = 1500

        if len(experiences) > max_experiences:
            step = len(experiences) // max_experiences
            experiences = experiences[::step][:max_experiences]

        experiences_text = ""
        for i, exp in enumerate(experiences, 1):
            exp_text = exp.content
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

    def _extract_response_text(self, response) -> str:
        """从ChatResponse对象中提取纯文本内容"""
        try:
            # 尝试多种方式提取text内容
            if hasattr(response, "message") and hasattr(response.message, "content"):
                content = response.message.content
                if hasattr(content, "text"):
                    return content.text
                elif isinstance(content, str):
                    return content

            if hasattr(response, "content"):
                content = response.content
                if hasattr(content, "text"):
                    return content.text
                elif isinstance(content, str):
                    return content

            if hasattr(response, "text"):
                return response.text

            # 尝试从字符串中提取
            response_str = str(response)
            text_match = re.search(r"text='([^']*)'", response_str)
            if text_match:
                return text_match.group(1)

            text_match = re.search(r'text="([^"]*)"', response_str)
            if text_match:
                return text_match.group(1)

            content_match = re.search(r"Content\(text='([^']*)'", response_str)
            if content_match:
                return content_match.group(1)

            logger.warning("无法提取text内容，使用字符串表示")
            return response_str

        except Exception as e:
            logger.error(f"提取响应文本失败: {e}")
            return str(response)

    def _update_token_stats(self, response):
        """从chat response中更新token统计"""
        try:
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
                input_tokens = getattr(usage, "input_tokens", 0) or getattr(
                    usage, "prompt_tokens", 0
                )
                output_tokens = getattr(usage, "output_tokens", 0) or getattr(
                    usage, "completion_tokens", 0
                )

                if input_tokens > 0 or output_tokens > 0:
                    cost = self._calculate_cost(input_tokens, output_tokens)
                    self.token_stats.add_usage(input_tokens, output_tokens, cost)

                    logger.info(
                        f"💰 本次调用: 输入{input_tokens}tokens, 输出{output_tokens}tokens, 成本${cost:.4f}"
                    )
                    return True

            logger.warning("⚠️ 无法从response中获取usage信息")
            return False

        except Exception as e:
            logger.error(f"更新token统计失败: {e}")
            return False

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算API调用成本"""
        model_name = self.config.ai_model

        if model_name not in self.PRICING:
            model_name = "us.anthropic.claude-sonnet-4-20250514-v1:0"

        input_cost = (input_tokens / 1000) * self.PRICING[model_name]["input"]
        output_cost = (output_tokens / 1000) * self.PRICING[model_name]["output"]

        return input_cost + output_cost

    def _print_stats(self):
        """打印Token使用统计"""
        logger.info("\n💰 总计Token使用统计:")
        logger.info(f"输入tokens: {self.token_stats.total_input_tokens:,}")
        logger.info(f"输出tokens: {self.token_stats.total_output_tokens:,}")
        logger.info(f"API调用次数: {self.token_stats.api_calls}")
        logger.info(f"预估总成本: ${self.token_stats.total_cost:.4f}")

    def get_token_stats(self) -> TokenStats:
        """获取Token统计信息"""
        return self.token_stats

    def reset_token_stats(self):
        """重置Token统计"""
        self.token_stats = TokenStats()
