"""
数据模型定义模块

定义InterviewDataManager系统中使用的所有数据模型
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


"""
"Job Title", - 岗位名称
"Required Intelligence", - 岗位要求的聪明度
"Candidate Resume", - 候选人简历
"Job Description", - 岗位JD
"First Round Interview Dialogue", - 一面面试对话
"First Round Interview Evaluation", - 一面面试评价
"First Round Result", - 一面结果
"Second Round Interview Dialogue", - 二面面试对话
"Second Round Interview Evaluation", - 二面面试评价
"Second Round Result", - 二面结果
"Final Round Interview Dialogue", - 三面面试对话
"Final Round Interview Evaluation", - 三面面试评价
"Final Round Result", - 三面结果
"""


class ExtractionMode(Enum):
    """经验提取模式"""

    INCREMENTAL = "incremental"  # 增量更新
    REVIEW = "review"  # 温故知新
    FULL_REFRESH = "full_refresh"  # 全量重新总结


class ExportFormat(Enum):
    """导出格式"""

    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    MARKDOWN = "markdown"


class ExportType(Enum):
    """导出类型"""

    REPORT = "report"
    DATA = "data"
    EXPERIENCES = "experiences"


@dataclass
class Record:
    """面试记录数据模型"""

    id: int
    resume: str  # 候选人简历
    jd: str  # 岗位JD
    conversation: str  # 面试对话
    evaluation: str  # 面试评价
    position: Optional[str] = None  # 岗位名称
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据

    @classmethod
    def from_dataframe_row(cls, row_id: int, row_data: Dict) -> "Record":
        """从DataFrame行数据创建Record对象"""
        return cls(
            id=row_id,
            resume=str(row_data.get("Candidate Resume", "")),
            jd=str(row_data.get("Job Description", "")),
            conversation=str(row_data.get("First Round Interview Dialogue", "")),
            evaluation=str(row_data.get("First Round Interview Evaluation", "")),
            position=row_data.get("Job Title"),
            metadata={
                k: v
                for k, v in row_data.items()
                if k
                not in [
                    "Candidate Resume",
                    "Job Description",
                    "First Round Interview Dialogue",
                    "First Round Interview Evaluation",
                    "Job Title",
                ]
            },
        )

    def is_valid(self) -> bool:
        """检查记录是否包含完整信息"""
        return bool(self.resume and self.jd and self.conversation and self.evaluation)


@dataclass
class Experience:
    """提取的面试经验数据模型"""

    id: str  # 经验唯一标识
    record_id: int  # 关联的记录ID
    content: str  # 经验内容
    timestamp: datetime  # 创建时间
    token_stats: Dict[str, int] = field(default_factory=dict)  # Token统计
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "record_id": self.record_id,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "token_stats": self.token_stats,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Experience":
        """从字典创建Experience对象"""
        return cls(
            id=data["id"],
            record_id=data["record_id"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            token_stats=data.get("token_stats", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionParams:
    """经验提取参数"""

    record_ids: List[int]  # 要提取的记录ID列表
    mode: ExtractionMode = ExtractionMode.INCREMENTAL  # 提取模式
    save_individual: bool = True  # 是否保存单个经验文件
    auto_integrate: bool = True  # 是否自动整合
    prompt_template: Optional[str] = None  # 自定义提示词模板
    batch_size: int = 5  # 批处理大小

    def __post_init__(self):
        """确保mode是ExtractionMode枚举"""
        if isinstance(self.mode, str):
            self.mode = ExtractionMode(self.mode)


@dataclass
class TokenStats:
    """Token使用统计"""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    api_calls: int = 0

    def add_usage(self, input_tokens: int, output_tokens: int, cost: float):
        """添加一次API调用的使用统计"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.api_calls += 1

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "api_calls": self.api_calls,
        }


@dataclass
class SearchQuery:
    """搜索查询参数"""

    keyword: str  # 搜索关键词
    fields: Optional[List[str]] = None  # 搜索字段
    filters: Dict[str, Any] = field(default_factory=dict)  # 过滤条件
    limit: int = 10  # 返回结果数量限制
    offset: int = 0  # 结果偏移量


@dataclass
class QueryResult:
    """查询结果"""

    records: List[Record]  # 记录列表
    total: int  # 总记录数
    page: int  # 当前页码
    page_size: int  # 每页大小
    filters_applied: Dict[str, Any] = field(default_factory=dict)  # 应用的过滤条件

    @property
    def total_pages(self) -> int:
        """总页数"""
        return (
            (self.total + self.page_size - 1) // self.page_size
            if self.page_size > 0
            else 0
        )


@dataclass
class StatisticsResult:
    """统计分析结果"""

    metrics: Dict[str, Any]  # 统计指标
    charts: List[Dict] = field(default_factory=list)  # 图表数据
    summary: str = ""  # 文字总结
    timestamp: datetime = field(default_factory=datetime.now)  # 生成时间

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "metrics": self.metrics,
            "charts": self.charts,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExtractionResult:
    """经验提取结果"""

    total_records: int  # 总记录数
    successful_extractions: int  # 成功提取数
    failed_extractions: int  # 失败提取数
    experiences: List[Experience]  # 提取的经验列表
    integrated_experience: Optional[str] = None  # 整合后的经验
    token_stats: TokenStats = field(default_factory=TokenStats)  # Token统计
    errors: List[str] = field(default_factory=list)  # 错误信息列表
    timestamp: datetime = field(default_factory=datetime.now)  # 提取时间

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_records": self.total_records,
            "successful_extractions": self.successful_extractions,
            "failed_extractions": self.failed_extractions,
            "experiences": [exp.to_dict() for exp in self.experiences],
            "integrated_experience": self.integrated_experience,
            "token_stats": self.token_stats.to_dict(),
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValidationResult:
    """数据验证结果"""

    is_valid: bool  # 是否有效
    total_records: int  # 总记录数
    valid_records: int  # 有效记录数
    invalid_records: int  # 无效记录数
    missing_columns: List[str] = field(default_factory=list)  # 缺失的列
    errors: List[str] = field(default_factory=list)  # 错误信息
    warnings: List[str] = field(default_factory=list)  # 警告信息


@dataclass
class CheckpointInfo:
    """Checkpoint信息"""

    name: str  # Checkpoint名称
    file_path: str  # 文件路径
    timestamp: datetime  # 创建时间
    total_processed: int  # 已处理记录数
    token_stats: Dict[str, Any] = field(default_factory=dict)  # Token统计
    metadata: Dict[str, Any] = field(default_factory=dict)  # 其他元数据

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "name": self.name,
            "file_path": self.file_path,
            "timestamp": self.timestamp.isoformat(),
            "total_processed": self.total_processed,
            "token_stats": self.token_stats,
            "metadata": self.metadata,
        }


@dataclass
class ManagerConfig:
    """管理器配置"""

    # 数据源配置
    data_source: str = "interview_data.csv"
    encoding: str = "auto"  # auto | utf-8 | gbk | gb2312 | utf-8-sig

    # AI配置
    ai_model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7

    # 存储配置
    checkpoint_dir: str = "checkpoints"
    experience_dir: str = "experiences"
    report_dir: str = "reports"

    # 性能配置
    batch_size: int = 10
    max_workers: int = 4
    timeout: int = 300  # API调用超时时间（秒）

    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存过期时间（秒）

    # 日志配置
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def validate(self) -> ValidationResult:
        """验证配置"""
        errors = []
        warnings = []

        if self.batch_size <= 0:
            errors.append("batch_size必须大于0")

        if self.max_workers <= 0:
            errors.append("max_workers必须大于0")

        if self.timeout <= 0:
            warnings.append("timeout应该大于0")

        if self.cache_ttl < 0:
            errors.append("cache_ttl不能为负数")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            total_records=0,
            valid_records=0,
            invalid_records=0,
            errors=errors,
            warnings=warnings,
        )


@dataclass
class DataLoadResult:
    """数据加载结果"""

    success: bool  # 是否成功
    total_records: int  # 总记录数
    valid_records: int  # 有效记录数
    encoding_used: str  # 使用的编码
    columns: List[str] = field(default_factory=list)  # 数据列
    errors: List[str] = field(default_factory=list)  # 错误信息
    warnings: List[str] = field(default_factory=list)  # 警告信息
    load_time: float = 0.0  # 加载耗时（秒）

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "success": self.success,
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "encoding_used": self.encoding_used,
            "columns": self.columns,
            "errors": self.errors,
            "warnings": self.warnings,
            "load_time": self.load_time,
        }
