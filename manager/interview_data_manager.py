"""
面试数据管理器 - 主模块

统一管理面试数据的加载、处理、分析和存储
"""

import logging
from typing import List, Dict, Optional

from manager.models import (
    ManagerConfig,
    DataLoadResult,
    QueryResult,
    SearchQuery,
    StatisticsResult,
    ExtractionParams,
    ExtractionResult,
    ExportFormat,
    ExportType,
    CheckpointInfo,
    Record,
)

from manager.data_loader import DataLoader
from manager.data_processor import DataProcessor
from manager.experience_extractor import ExperienceExtractor
from manager.storage_manager import StorageManager

logger = logging.getLogger(__name__)


class InterviewDataManager:
    """
    面试数据管理器

    职责:
    - 协调各个子模块
    - 提供统一的API接口
    - 管理生命周期

    使用示例:
        >>> config = ManagerConfig(data_source="interview_data.csv")
        >>> manager = InterviewDataManager(config)
        >>> result = manager.load_data()
        >>> stats = manager.get_statistics()
        >>> extraction_result = manager.extract_experiences(...)
    """

    def __init__(self, config: Optional[ManagerConfig] = None):
        """
        初始化管理器

        Args:
            config: 管理器配置，如果为None则使用默认配置
        """
        self.config = config or ManagerConfig()

        # 配置日志
        self._setup_logging()

        # 初始化各个子模块
        self.loader = DataLoader(self.config)
        self.processor = DataProcessor()
        self.extractor = ExperienceExtractor(self.config)
        self.storage = StorageManager(self.config)

        logger.info("=" * 60)
        logger.info("InterviewDataManager 初始化完成")
        logger.info("=" * 60)

    # ==================== 数据管理接口 ====================

    def load_data(self, source: Optional[str] = None) -> DataLoadResult:
        """
        加载数据

        Args:
            source: 数据源路径，如果为None则使用配置中的路径

        Returns:
            DataLoadResult: 加载结果
        """
        logger.info("🚀 开始加载数据...")
        result = self.loader.load_from_csv(source)

        if result.success:
            logger.info(f"✅ 数据加载成功: {result.total_records} 条记录")
        else:
            logger.error(f"❌ 数据加载失败: {result.errors}")

        return result

    def get_overview(self) -> Dict:
        """
        获取数据概览

        Returns:
            包含数据概览信息的字典
        """
        df = self.loader.get_dataframe()
        if df is None:
            return {"error": "数据未加载"}

        return self.processor.get_overview(df)

    def get_summary(self) -> Dict:
        """
        获取数据摘要（包括加载器的统计信息）

        Returns:
            数据摘要字典
        """
        return self.loader.get_summary()

    # ==================== 查询接口 ====================

    def query(
        self,
        filters: Optional[Dict] = None,
        fields: Optional[List[str]] = None,
        limit: int = 10,
        offset: int = 0,
    ) -> QueryResult:
        """
        查询数据

        Args:
            filters: 过滤条件，如 {"position": "产品经理"}
            fields: 要返回的字段
            limit: 返回结果数量限制
            offset: 结果偏移量

        Returns:
            QueryResult: 查询结果
        """
        df = self.loader.get_valid_data()

        if df.empty:
            return QueryResult(records=[], total=0, page=1, page_size=limit)

        # 应用过滤条件
        if filters:
            for field, value in filters.items():
                if field in df.columns:
                    df = df[df[field] == value]

        total = len(df)
        page = offset // limit + 1 if limit > 0 else 1

        # 分页
        paginated_df = df.iloc[offset : offset + limit]

        # 转换为Record对象
        records = []
        for idx, row in paginated_df.iterrows():
            record = Record.from_dataframe_row(idx, row.to_dict())
            records.append(record)

        return QueryResult(
            records=records,
            total=total,
            page=page,
            page_size=limit,
            filters_applied=filters or {},
        )

    def search(
        self,
        keyword: str,
        fields: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        limit: int = 10,
    ) -> QueryResult:
        """
        搜索数据

        Args:
            keyword: 搜索关键词
            fields: 搜索字段，如 ["resume", "jd"]
            filters: 额外的过滤条件
            limit: 返回结果数量限制

        Returns:
            QueryResult: 搜索结果
        """
        df = self.loader.get_valid_data()

        if df.empty:
            return QueryResult(records=[], total=0, page=1, page_size=limit)

        # 创建搜索查询
        query = SearchQuery(
            keyword=keyword, fields=fields, filters=filters or {}, limit=limit
        )

        return self.processor.search_candidates(df, query)

    # ==================== 分析接口 ====================

    def get_statistics(
        self, analysis_type: str = "position_distribution", **kwargs
    ) -> StatisticsResult:
        """
        获取统计分析结果

        Args:
            analysis_type: 分析类型
                - "position_distribution": 岗位分布分析
                - "interview_results": 面试结果分析
            **kwargs: 其他参数

        Returns:
            StatisticsResult: 统计结果
        """
        df = self.loader.get_valid_data()

        if df.empty:
            return StatisticsResult(metrics={}, summary="数据为空")

        if analysis_type == "position_distribution":
            return self.processor.analyze_positions(df)
        elif analysis_type == "interview_results":
            return self.processor.analyze_interview_results(df)
        else:
            return StatisticsResult(
                metrics={}, summary=f"未知的分析类型: {analysis_type}"
            )

    def visualize(
        self, chart_type: str = "position_distribution", save_path: Optional[str] = None
    ):
        """
        生成可视化图表

        Args:
            chart_type: 图表类型
            save_path: 保存路径，如果为None则显示图表
        """
        df = self.loader.get_valid_data()

        if df.empty:
            logger.warning("数据为空，无法生成图表")
            return

        if chart_type == "position_distribution":
            self.processor.plot_position_distribution(df, save_path)
        else:
            logger.warning(f"未知的图表类型: {chart_type}")

    # ==================== 经验提取接口 ====================

    def extract_experiences(
        self,
        record_ids: Optional[List[int]] = None,
        mode: str = "incremental",
        save_individual: bool = True,
        auto_integrate: bool = True,
        batch_size: int = 5,
    ) -> ExtractionResult:
        """
        提取面试经验

        Args:
            record_ids: 要提取的记录ID列表，如果为None则随机选择
            mode: 提取模式 ("incremental" | "review" | "full_refresh")
            save_individual: 是否保存单个经验文件
            auto_integrate: 是否自动整合
            batch_size: 批处理大小

        Returns:
            ExtractionResult: 提取结果
        """
        logger.info("🤖 开始提取面试经验...")

        # 如果没有指定记录ID，随机选择
        if record_ids is None:
            valid_count = len(self.loader.get_valid_data())
            if valid_count == 0:
                return ExtractionResult(
                    total_records=0,
                    successful_extractions=0,
                    failed_extractions=0,
                    experiences=[],
                    errors=["没有有效数据"],
                )

            # 随机选择最多batch_size条记录
            import random

            record_ids = random.sample(range(valid_count), min(batch_size, valid_count))
            logger.info(f"随机选择了 {len(record_ids)} 条记录: {record_ids}")

        # 获取记录
        records = self.loader.get_records(record_ids)

        if not records:
            return ExtractionResult(
                total_records=0,
                successful_extractions=0,
                failed_extractions=0,
                experiences=[],
                errors=["未找到有效记录"],
            )

        # 创建提取参数
        params = ExtractionParams(
            record_ids=record_ids,
            mode=mode,
            save_individual=save_individual,
            auto_integrate=auto_integrate,
            batch_size=batch_size,
        )

        # 提取经验
        result = self.extractor.extract_batch(records, params)

        # 如果需要保存单个经验文件
        if save_individual:
            for experience in result.experiences:
                self.storage.save_experience(experience)

        logger.info(
            f"✅ 经验提取完成: 成功 {result.successful_extractions}/{result.total_records}"
        )

        return result

    def integrate_experiences(
        self,
        experience_ids: Optional[List[str]] = None,
        mode: str = "incremental",
        output_format: str = "markdown",
    ) -> str:
        """
        整合经验

        Args:
            experience_ids: 要整合的经验ID列表，如果为None则整合所有经验
            mode: 整合模式
            output_format: 输出格式

        Returns:
            整合后的经验文本
        """
        # 加载经验
        if experience_ids:
            filters = {"experience_ids": experience_ids}
        else:
            filters = None

        experiences = self.storage.load_experiences(filters)

        if not experiences:
            logger.warning("没有找到可整合的经验")
            return "暂无经验可整合"

        # 整合经验
        integrated = self.extractor.integrate_experiences(experiences, mode)

        return integrated

    # ==================== Checkpoint管理接口 ====================

    def create_checkpoint(
        self, name: Optional[str] = None, include_state: bool = True
    ) -> str:
        """
        创建checkpoint

        Args:
            name: checkpoint名称
            include_state: 是否包含状态信息

        Returns:
            checkpoint文件路径
        """
        data = {
            "config": self.config.__dict__,
            "data_summary": self.get_summary(),
        }

        if include_state:
            data["token_stats"] = self.extractor.get_token_stats().to_dict()

        return self.storage.save_checkpoint(data, name)

    def restore_checkpoint(self, name: str) -> bool:
        """
        从checkpoint恢复

        Args:
            name: checkpoint名称

        Returns:
            是否成功恢复
        """
        data = self.storage.load_checkpoint(name)

        if data is None:
            return False

        logger.info(f"✅ 已从checkpoint恢复: {name}")
        return True

    def list_checkpoints(self, filters: Optional[Dict] = None) -> List[CheckpointInfo]:
        """
        列出所有checkpoint

        Args:
            filters: 过滤条件

        Returns:
            CheckpointInfo列表
        """
        return self.storage.list_checkpoints(filters)

    def delete_checkpoint(self, name: str) -> bool:
        """
        删除checkpoint

        Args:
            name: checkpoint名称

        Returns:
            是否成功删除
        """
        return self.storage.delete_checkpoint(name)

    # ==================== 导出接口 ====================

    def export(
        self,
        export_type: str = "report",
        export_format: str = "json",
        filters: Optional[Dict] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        导出数据

        Args:
            export_type: 导出类型 ("report" | "data" | "experiences")
            export_format: 导出格式 ("json" | "csv" | "excel" | "markdown")
            filters: 过滤条件
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        # 转换为枚举
        export_type_enum = ExportType(export_type)
        export_format_enum = ExportFormat(export_format)

        # 根据类型准备数据
        if export_type == "report":
            data = {
                "overview": self.get_overview(),
                "statistics": self.get_statistics().to_dict(),
                "summary": self.get_summary(),
            }
        elif export_type == "data":
            df = self.loader.get_valid_data()
            data = df
        elif export_type == "experiences":
            experiences = self.storage.load_experiences(filters)
            data = [exp.to_dict() for exp in experiences]
        else:
            raise ValueError(f"未知的导出类型: {export_type}")

        return self.storage.export_report(
            data=data,
            export_type=export_type_enum,
            export_format=export_format_enum,
            output_path=output_path,
        )

    # ==================== 辅助方法 ====================

    def _setup_logging(self):
        """配置日志"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)

        # 配置根logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # 如果配置了日志文件
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            logging.getLogger().addHandler(file_handler)

    def __repr__(self) -> str:
        """字符串表示"""
        return f"InterviewDataManager(config={self.config})"
