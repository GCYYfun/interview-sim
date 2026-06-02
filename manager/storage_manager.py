"""
存储管理器模块

管理checkpoint、经验文件和报告的持久化存储
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np

from manager.models import (
    Experience,
    CheckpointInfo,
    ManagerConfig,
    ExportFormat,
    ExportType,
    ExtractionResult,
)

logger = logging.getLogger(__name__)


class StorageManager:
    """
    存储管理器

    功能:
    - Checkpoint管理
    - 经验文件管理
    - 报告导出
    - 版本控制
    """

    def __init__(self, config: ManagerConfig):
        """
        初始化存储管理器

        Args:
            config: 管理器配置
        """
        self.config = config

        # 创建存储目录
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.experience_dir = Path(config.experience_dir)
        self.report_dir = Path(config.report_dir)

        self.checkpoint_dir.mkdir(exist_ok=True)
        self.experience_dir.mkdir(exist_ok=True)
        self.report_dir.mkdir(exist_ok=True)

        logger.info("存储目录初始化完成:")
        logger.info(f"  Checkpoint: {self.checkpoint_dir}")
        logger.info(f"  Experience: {self.experience_dir}")
        logger.info(f"  Report: {self.report_dir}")

    # ==================== Checkpoint管理 ====================

    def save_checkpoint(self, data: Dict[str, Any], name: Optional[str] = None) -> str:
        """
        保存checkpoint

        Args:
            data: 要保存的数据
            name: checkpoint名称，如果为None则自动生成

        Returns:
            checkpoint文件路径
        """
        try:
            if name is None:
                name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            checkpoint_file = self.checkpoint_dir / f"{name}.json"

            # 添加元数据
            checkpoint_data = {
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "data": data,
            }

            # 转换numpy类型
            checkpoint_data = self._convert_numpy(checkpoint_data)

            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 Checkpoint已保存: {checkpoint_file.name}")
            return str(checkpoint_file)

        except Exception as e:
            logger.error(f"保存checkpoint失败: {e}")
            raise

    def load_checkpoint(self, name: str) -> Optional[Dict]:
        """
        加载checkpoint

        Args:
            name: checkpoint名称

        Returns:
            checkpoint数据，如果不存在返回None
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{name}.json"

            if not checkpoint_file.exists():
                logger.warning(f"Checkpoint不存在: {name}")
                return None

            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"📁 已加载checkpoint: {name}")
            return data.get("data")

        except Exception as e:
            logger.error(f"加载checkpoint失败: {e}")
            return None

    def list_checkpoints(self, filters: Optional[Dict] = None) -> List[CheckpointInfo]:
        """
        列出所有checkpoint

        Args:
            filters: 过滤条件

        Returns:
            CheckpointInfo列表
        """
        try:
            checkpoints = []

            for checkpoint_file in self.checkpoint_dir.glob("*.json"):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    checkpoint_info = CheckpointInfo(
                        name=data.get("name", checkpoint_file.stem),
                        file_path=str(checkpoint_file),
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        total_processed=data.get("data", {}).get("total_processed", 0),
                        token_stats=data.get("data", {}).get("token_stats", {}),
                        metadata=data.get("data", {}).get("metadata", {}),
                    )

                    checkpoints.append(checkpoint_info)

                except Exception as e:
                    logger.warning(f"读取checkpoint文件失败 {checkpoint_file}: {e}")
                    continue

            # 按时间戳排序
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)

            # 应用过滤条件
            if filters:
                checkpoints = self._filter_checkpoints(checkpoints, filters)

            return checkpoints

        except Exception as e:
            logger.error(f"列出checkpoints失败: {e}")
            return []

    def delete_checkpoint(self, name: str) -> bool:
        """
        删除checkpoint

        Args:
            name: checkpoint名称

        Returns:
            是否成功删除
        """
        try:
            checkpoint_file = self.checkpoint_dir / f"{name}.json"

            if checkpoint_file.exists():
                checkpoint_file.unlink()
                logger.info(f"🗑️ 已删除checkpoint: {name}")
                return True
            else:
                logger.warning(f"Checkpoint不存在: {name}")
                return False

        except Exception as e:
            logger.error(f"删除checkpoint失败: {e}")
            return False

    # ==================== 经验文件管理 ====================

    def save_experience(
        self, experience: Experience, metadata: Optional[Dict] = None
    ) -> str:
        """
        保存单个经验到文件

        Args:
            experience: 经验对象
            metadata: 额外的元数据

        Returns:
            保存的文件路径
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"individual_interview_experience_{experience.record_id}_{timestamp}.json"
            experience_file = self.experience_dir / filename

            experience_data = {
                "id": experience.id,
                "record_id": experience.record_id,
                "content": experience.content,
                "timestamp": experience.timestamp.isoformat(),
                "token_stats": experience.token_stats,
                "metadata": {**experience.metadata, **(metadata or {})},
            }

            with open(experience_file, "w", encoding="utf-8") as f:
                json.dump(experience_data, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 经验已保存: {experience_file.name}")
            return str(experience_file)

        except Exception as e:
            logger.error(f"保存经验失败: {e}")
            raise

    def load_experiences(self, filters: Optional[Dict] = None) -> List[Experience]:
        """
        加载经验文件

        Args:
            filters: 过滤条件（如日期范围、记录ID等）

        Returns:
            Experience对象列表
        """
        try:
            experiences = []
            pattern = "individual_interview_experience_*.json"

            for file_path in self.experience_dir.glob(pattern):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    experience = Experience(
                        id=data["id"],
                        record_id=data["record_id"],
                        content=data["content"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        token_stats=data.get("token_stats", {}),
                        metadata=data.get("metadata", {}),
                    )

                    experiences.append(experience)

                except Exception as e:
                    logger.warning(f"读取经验文件失败 {file_path}: {e}")
                    continue

            # 按时间戳排序
            experiences.sort(key=lambda x: x.timestamp)

            # 应用过滤条件
            if filters:
                experiences = self._filter_experiences(experiences, filters)

            logger.info(f"📁 已加载 {len(experiences)} 个经验文件")
            return experiences

        except Exception as e:
            logger.error(f"加载经验文件失败: {e}")
            return []

    def delete_experiences(self, experience_ids: List[str]) -> int:
        """
        删除指定的经验文件

        Args:
            experience_ids: 经验ID列表

        Returns:
            成功删除的文件数量
        """
        deleted_count = 0

        for experience_id in experience_ids:
            try:
                # 查找匹配的文件
                for file_path in self.experience_dir.glob(f"*{experience_id}*.json"):
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ 已删除经验文件: {file_path.name}")

            except Exception as e:
                logger.error(f"删除经验文件失败 {experience_id}: {e}")
                continue

        return deleted_count

    # ==================== 报告导出 ====================

    def export_report(
        self,
        data: Any,
        export_type: ExportType,
        export_format: ExportFormat,
        output_path: Optional[str] = None,
    ) -> str:
        """
        导出报告

        Args:
            data: 要导出的数据
            export_type: 导出类型
            export_format: 导出格式
            output_path: 输出路径，如果为None则自动生成

        Returns:
            导出文件路径
        """
        try:
            # 生成文件名
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{export_type.value}_{timestamp}.{export_format.value}"
                output_path = str(self.report_dir / filename)

            # 根据格式导出
            if export_format == ExportFormat.JSON:
                self._export_json(data, output_path)
            elif export_format == ExportFormat.CSV:
                self._export_csv(data, output_path)
            elif export_format == ExportFormat.EXCEL:
                self._export_excel(data, output_path)
            elif export_format == ExportFormat.MARKDOWN:
                self._export_markdown(data, output_path)
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")

            logger.info(f"📋 报告已导出: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"导出报告失败: {e}")
            raise

    def export_extraction_result(
        self,
        result: ExtractionResult,
        export_format: ExportFormat = ExportFormat.JSON,
        output_path: Optional[str] = None,
    ) -> str:
        """
        导出经验提取结果

        Args:
            result: 提取结果
            export_format: 导出格式
            output_path: 输出路径

        Returns:
            导出文件路径
        """
        # 转换为可序列化的字典
        data = result.to_dict()

        return self.export_report(
            data=data,
            export_type=ExportType.EXPERIENCES,
            export_format=export_format,
            output_path=output_path,
        )

    # ==================== 私有方法 ====================

    def _export_json(self, data: Any, output_path: str):
        """导出为JSON格式"""
        # 转换numpy类型
        data = self._convert_numpy(data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _export_csv(self, data: Any, output_path: str):
        """导出为CSV格式"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            raise ValueError("无法将数据转换为CSV格式")

    def _export_excel(self, data: Any, output_path: str):
        """导出为Excel格式"""
        if isinstance(data, pd.DataFrame):
            data.to_excel(output_path, index=False, engine="openpyxl")
        elif isinstance(data, list):
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False, engine="openpyxl")
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
            df.to_excel(output_path, index=False, engine="openpyxl")
        else:
            raise ValueError("无法将数据转换为Excel格式")

    def _export_markdown(self, data: Any, output_path: str):
        """导出为Markdown格式"""
        with open(output_path, "w", encoding="utf-8") as f:
            if isinstance(data, dict):
                f.write("# 报告\n\n")
                for key, value in data.items():
                    f.write(f"## {key}\n\n")
                    f.write(f"{value}\n\n")
            elif isinstance(data, str):
                f.write(data)
            else:
                f.write(str(data))

    def _convert_numpy(self, obj: Any) -> Any:
        """转换numpy类型为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (float, np.floating)):
            try:
                if pd.isna(obj) or np.isnan(obj):
                    return None
            except (ValueError, TypeError):
                pass
            return float(obj)
        elif obj is None:
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        else:
            try:
                if pd.isna(obj):
                    return None
            except (ValueError, TypeError):
                pass
        return obj

    def _filter_checkpoints(
        self, checkpoints: List[CheckpointInfo], filters: Dict
    ) -> List[CheckpointInfo]:
        """根据条件过滤checkpoints"""
        filtered = checkpoints

        if "created_after" in filters:
            after_date = datetime.fromisoformat(filters["created_after"])
            filtered = [cp for cp in filtered if cp.timestamp >= after_date]

        if "created_before" in filters:
            before_date = datetime.fromisoformat(filters["created_before"])
            filtered = [cp for cp in filtered if cp.timestamp <= before_date]

        return filtered

    def _filter_experiences(
        self, experiences: List[Experience], filters: Dict
    ) -> List[Experience]:
        """根据条件过滤经验"""
        filtered = experiences

        if "record_ids" in filters:
            record_ids = set(filters["record_ids"])
            filtered = [exp for exp in filtered if exp.record_id in record_ids]

        if "date_range" in filters:
            date_range = filters["date_range"]
            if ":" in date_range:
                start_date, end_date = date_range.split(":")
                start_date = datetime.fromisoformat(start_date)
                end_date = datetime.fromisoformat(end_date)
                filtered = [
                    exp for exp in filtered if start_date <= exp.timestamp <= end_date
                ]

        return filtered
