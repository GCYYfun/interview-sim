"""
数据加载器模块

负责从各种数据源加载面试数据，支持多种编码格式
"""

import pandas as pd
from pathlib import Path
import time
import logging
from typing import Optional, List
from manager.models import DataLoadResult, ValidationResult, ManagerConfig, Record

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    功能:
    - 从CSV文件加载数据
    - 自动检测和处理编码问题
    - 数据验证
    - 数据清洗
    """

    # 必需的列名
    REQUIRED_COLUMNS = [
        "Candidate Resume",
        "Job Description",
        "First Round Interview Dialogue",
        "First Round Interview Evaluation",
    ]

    # 支持的编码列表
    SUPPORTED_ENCODINGS = ["utf-8", "gbk", "gb2312", "utf-8-sig", "latin1"]

    def __init__(self, config: ManagerConfig):
        """
        初始化数据加载器

        Args:
            config: 管理器配置
        """
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.encoding_used: Optional[str] = None

    def load_from_csv(self, file_path: Optional[str] = None) -> DataLoadResult:
        """
        从CSV文件加载数据

        Args:
            file_path: CSV文件路径，如果为None则使用配置中的路径

        Returns:
            DataLoadResult: 加载结果
        """
        start_time = time.time()

        if file_path is None:
            file_path = self.config.data_source

        csv_path = Path(file_path)

        if not csv_path.exists():
            return DataLoadResult(
                success=False,
                total_records=0,
                valid_records=0,
                encoding_used="",
                errors=[f"文件不存在: {file_path}"],
            )

        logger.info(f"开始加载数据: {file_path}")

        # 尝试使用不同编码加载
        df = None
        encoding_used = None
        errors = []

        # 确定要尝试的编码列表
        if self.config.encoding == "auto":
            encodings_to_try = self.SUPPORTED_ENCODINGS
        else:
            encodings_to_try = [self.config.encoding] + [
                enc for enc in self.SUPPORTED_ENCODINGS if enc != self.config.encoding
            ]

        # 尝试不同编码
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                encoding_used = encoding
                logger.info(f"✅ 成功使用 {encoding} 编码加载数据")
                break
            except UnicodeDecodeError:
                logger.debug(f"尝试 {encoding} 编码失败")
                continue
            except Exception as e:
                error_msg = f"使用 {encoding} 编码加载失败: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue

        if df is None:
            return DataLoadResult(
                success=False,
                total_records=0,
                valid_records=0,
                encoding_used="",
                errors=errors if errors else ["无法使用任何编码方式读取文件"],
                load_time=time.time() - start_time,
            )

        # 清洗数据
        df, warnings = self._clean_data(df)

        # 验证数据
        validation = self._validate_data(df)

        # 保存数据
        self.df = df
        self.encoding_used = encoding_used

        load_time = time.time() - start_time

        logger.info(
            f"数据加载完成: {len(df)} 行 × {len(df.columns)} 列, 耗时 {load_time:.2f}秒"
        )

        return DataLoadResult(
            success=validation.is_valid,
            total_records=len(df),
            valid_records=validation.valid_records,
            encoding_used=encoding_used,
            columns=list(df.columns),
            errors=validation.errors,
            warnings=warnings + validation.warnings,
            load_time=load_time,
        )

    def _clean_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
        """
        清洗数据

        Args:
            df: 原始DataFrame

        Returns:
            (清洗后的DataFrame, 警告信息列表)
        """
        warnings = []
        original_rows = len(df)
        original_cols = len(df.columns)

        # 移除完全空的行
        df = df.dropna(how="all")
        removed_rows = original_rows - len(df)
        if removed_rows > 0:
            warnings.append(f"移除了 {removed_rows} 个完全空的行")

        # 移除完全空的列
        df = df.dropna(axis=1, how="all")
        removed_cols = original_cols - len(df.columns)
        if removed_cols > 0:
            warnings.append(f"移除了 {removed_cols} 个完全空的列")

        # 标准化列名（去除首尾空格）
        df.columns = [col.strip() for col in df.columns]

        # 重置索引
        df = df.reset_index(drop=True)

        logger.info(f"数据清洗完成: {len(df)} 行 × {len(df.columns)} 列")

        return df, warnings

    def _validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        验证数据完整性

        Args:
            df: 要验证的DataFrame

        Returns:
            ValidationResult: 验证结果
        """
        errors = []
        warnings = []

        # 检查必需列是否存在
        missing_columns = [
            col for col in self.REQUIRED_COLUMNS if col not in df.columns
        ]

        if missing_columns:
            errors.append(f"缺少必需列: {', '.join(missing_columns)}")
            return ValidationResult(
                is_valid=False,
                total_records=len(df),
                valid_records=0,
                invalid_records=len(df),
                missing_columns=missing_columns,
                errors=errors,
                warnings=warnings,
            )

        # 检查有效记录数（必需字段都不为空）
        valid_mask = df[self.REQUIRED_COLUMNS].notna().all(axis=1)
        valid_records = valid_mask.sum()
        invalid_records = len(df) - valid_records

        if invalid_records > 0:
            warnings.append(f"发现 {invalid_records} 条不完整的记录（缺少必需字段）")

        # 检查数据类型
        for col in self.REQUIRED_COLUMNS:
            if df[col].dtype != "object":
                warnings.append(f"列 '{col}' 的数据类型不是字符串类型")

        is_valid = len(errors) == 0 and valid_records > 0

        return ValidationResult(
            is_valid=is_valid,
            total_records=len(df),
            valid_records=valid_records,
            invalid_records=invalid_records,
            missing_columns=missing_columns,
            errors=errors,
            warnings=warnings,
        )

    def get_valid_data(self) -> pd.DataFrame:
        """
        获取有效数据（必需字段都不为空的记录）

        Returns:
            包含有效记录的DataFrame
        """
        if self.df is None:
            return pd.DataFrame()

        # 筛选有效数据
        valid_mask = self.df[self.REQUIRED_COLUMNS].notna().all(axis=1)
        return self.df[valid_mask].copy()

    def get_records(self, record_ids: Optional[List[int]] = None) -> List[Record]:
        """
        获取Record对象列表

        Args:
            record_ids: 要获取的记录ID列表，如果为None则返回所有有效记录

        Returns:
            Record对象列表
        """
        valid_df = self.get_valid_data()

        if valid_df.empty:
            return []

        records = []

        if record_ids is None:
            # 返回所有有效记录
            for idx, row in valid_df.iterrows():
                record = Record.from_dataframe_row(idx, row.to_dict())
                records.append(record)
        else:
            # 返回指定ID的记录
            for record_id in record_ids:
                if record_id < len(valid_df):
                    row = valid_df.iloc[record_id]
                    record = Record.from_dataframe_row(record_id, row.to_dict())
                    records.append(record)
                else:
                    logger.warning(f"记录ID {record_id} 超出范围")

        return records

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """获取原始DataFrame"""
        return self.df

    def get_summary(self) -> dict:
        """
        获取数据摘要

        Returns:
            包含摘要信息的字典
        """
        if self.df is None:
            return {"error": "数据未加载"}

        valid_df = self.get_valid_data()

        summary = {
            "total_records": len(self.df),
            "valid_records": len(valid_df),
            "invalid_records": len(self.df) - len(valid_df),
            "total_columns": len(self.df.columns),
            "columns": list(self.df.columns),
            "encoding_used": self.encoding_used,
            "data_types": dict(self.df.dtypes.astype(str)),
        }

        # 添加缺失值统计
        missing_values = {}
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            if missing_count > 0:
                missing_values[col] = int(missing_count)

        if missing_values:
            summary["missing_values"] = missing_values

        return summary
