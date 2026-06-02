# InterviewDataManager - 面试数据管理器

## 📋 概述

`InterviewDataManager` 是对原有 `InterviewDataProcessor` 的重构升级版本，采用模块化设计，提供了更清晰的API接口和更强大的管理功能。

### 主要改进

✅ **模块化架构** - 清晰的职责分离，易于维护和扩展  
✅ **统一API接口** - 一致的调用方式，降低学习成本  
✅ **完善的数据模型** - 使用dataclass定义，类型安全  
✅ **增强的存储管理** - 独立的checkpoint和经验文件管理  
✅ **灵活的导出功能** - 支持多种格式（JSON/CSV/Excel/Markdown）  
✅ **保留原有逻辑** - 所有原有的处理功能都得到保留  

## 🏗️ 架构设计

```
InterviewDataManager (核心管理器)
├── DataLoader (数据加载层)
│   ├── CSV加载
│   ├── 编码处理
│   └── 数据验证
│
├── DataProcessor (数据处理层)
│   ├── 数据清洗
│   ├── 统计分析
│   └── 数据可视化
│
├── ExperienceExtractor (经验提取层)
│   ├── AI调用管理
│   ├── Token统计
│   └── 提示词构建
│
└── StorageManager (存储管理层)
    ├── Checkpoint管理
    ├── 经验文件管理
    └── 报告导出
```

## 📦 模块说明

### 1. models.py - 数据模型
定义所有数据类型：
- `Record` - 面试记录
- `Experience` - 提取的经验
- `ExtractionParams` - 提取参数
- `QueryResult` - 查询结果
- `StatisticsResult` - 统计结果
- `ManagerConfig` - 管理器配置

### 2. data_loader.py - 数据加载器
负责数据加载和验证：
- 多编码格式支持
- 自动数据清洗
- 数据完整性验证

### 3. data_processor.py - 数据处理器
保留原有的分析逻辑：
- 岗位分布分析
- 面试结果分析
- 候选人搜索
- 数据可视化

### 4. experience_extractor.py - 经验提取器
AI驱动的经验提取：
- 单个/批量提取
- 三种整合模式
- Token使用统计

### 5. storage_manager.py - 存储管理器
统一的持久化管理：
- Checkpoint CRUD
- 经验文件管理
- 多格式导出

### 6. interview_data_manager.py - 主管理器
统一的API入口

## 🚀 快速开始

### 基础使用

```python
from interview_data_manager import InterviewDataManager
from models import ManagerConfig

# 1. 创建管理器
config = ManagerConfig(data_source="interview_data.csv")
manager = InterviewDataManager(config)

# 2. 加载数据
result = manager.load_data()
print(f"加载了 {result.total_records} 条记录")

# 3. 获取概览
overview = manager.get_overview()
print(overview)
```

## 📖 API 文档

### 数据管理

#### load_data()
```python
result = manager.load_data(source="interview_data.csv")
# 返回 DataLoadResult
```

#### get_overview()
```python
overview = manager.get_overview()
# 返回包含数据概览的字典
```

#### get_summary()
```python
summary = manager.get_summary()
# 返回数据摘要
```

### 查询接口

#### query()
```python
result = manager.query(
    filters={"岗位名称": "产品经理"},
    limit=10,
    offset=0
)
# 返回 QueryResult
```

#### search()
```python
result = manager.search(
    keyword="市场",
    fields=["候选人脱敏简历", "岗位脱敏jd"],
    limit=10
)
# 返回 QueryResult
```

### 分析接口

#### get_statistics()
```python
# 岗位分布分析
stats = manager.get_statistics(analysis_type="position_distribution")

# 面试结果分析
stats = manager.get_statistics(analysis_type="interview_results")
# 返回 StatisticsResult
```

#### visualize()
```python
manager.visualize(
    chart_type="position_distribution",
    save_path="chart.png"
)
```

### 经验提取

#### extract_experiences()
```python
result = manager.extract_experiences(
    record_ids=[0, 1, 2, 3, 4],
    mode="incremental",  # incremental | review | full_refresh
    save_individual=True,
    auto_integrate=True
)
# 返回 ExtractionResult
```

**提取模式说明：**
- `incremental` - 增量更新（只整合新经验）
- `review` - 温故知新（新经验 + 部分历史经验）
- `full_refresh` - 全量重新总结（所有经验）

#### integrate_experiences()
```python
integrated = manager.integrate_experiences(
    experience_ids=["exp_001", "exp_002"],
    mode="review",
    output_format="markdown"
)
```

### Checkpoint管理

#### create_checkpoint()
```python
path = manager.create_checkpoint(
    name="my_checkpoint",
    include_state=True
)
```

#### restore_checkpoint()
```python
success = manager.restore_checkpoint(name="my_checkpoint")
```

#### list_checkpoints()
```python
checkpoints = manager.list_checkpoints(
    filters={"created_after": "2024-01-01"}
)
```

#### delete_checkpoint()
```python
success = manager.delete_checkpoint(name="old_checkpoint")
```

### 导出功能

#### export()
```python
# 导出报告为JSON
path = manager.export(
    export_type="report",
    export_format="json",
    output_path="report.json"
)

# 导出数据为CSV
path = manager.export(
    export_type="data",
    export_format="csv"
)

# 导出经验为Excel
path = manager.export(
    export_type="experiences",
    export_format="excel",
    filters={"record_ids": [1, 2, 3]}
)
```

**支持的格式：**
- `json` - JSON格式
- `csv` - CSV格式
- `excel` - Excel格式
- `markdown` - Markdown格式

## ⚙️ 配置选项

```python
config = ManagerConfig(
    # 数据源配置
    data_source="interview_data.csv",
    encoding="auto",  # auto | utf-8 | gbk | gb2312
    
    # AI配置
    ai_model="us.anthropic.claude-sonnet-4-20250514-v1:0",
    max_tokens=4096,
    temperature=0.7,
    
    # 存储配置
    checkpoint_dir="checkpoints",
    experience_dir="experiences",
    report_dir="reports",
    
    # 性能配置
    batch_size=10,
    max_workers=4,
    timeout=300,
    
    # 缓存配置
    enable_cache=True,
    cache_ttl=3600,
    
    # 日志配置
    log_level="INFO",
    log_file="manager.log"
)
```

## 💡 使用示例

### 示例1: 完整工作流

```python
from interview_data_manager import InterviewDataManager
from models import ManagerConfig

# 初始化
config = ManagerConfig(data_source="interview_data.csv")
manager = InterviewDataManager(config)

# 加载数据
manager.load_data()

# 查询分析
stats = manager.get_statistics(analysis_type="position_distribution")
print(stats.summary)

# 搜索候选人
result = manager.search(keyword="Python", limit=5)
print(f"找到 {result.total} 个匹配的候选人")

# 提取经验
extraction_result = manager.extract_experiences(
    record_ids=[0, 1, 2, 3, 4],
    mode="incremental"
)
print(f"成功提取 {extraction_result.successful_extractions} 条经验")

# 创建checkpoint
manager.create_checkpoint(name="after_extraction")

# 导出报告
manager.export(
    export_type="report",
    export_format="json",
    output_path="analysis_report.json"
)
```

### 示例2: 批量处理

```python
# 批量提取经验
record_ids = list(range(0, 20))  # 提取前20条记录

for i in range(0, len(record_ids), 5):
    batch = record_ids[i:i+5]
    
    result = manager.extract_experiences(
        record_ids=batch,
        mode="incremental",
        save_individual=True
    )
    
    # 每批次创建checkpoint
    manager.create_checkpoint(name=f"batch_{i//5}")
    
    print(f"批次 {i//5}: 成功 {result.successful_extractions}/{len(batch)}")
```

### 示例3: 高级筛选和分析

```python
# 查询特定岗位
pm_records = manager.query(
    filters={"岗位名称": "产品经理"},
    limit=100
)

# 分析该岗位的统计信息
# (这里可以对查询结果进行进一步处理)

# 导出筛选后的数据
manager.export(
    export_type="data",
    export_format="excel",
    filters={"岗位名称": "产品经理"},
    output_path="product_manager_data.xlsx"
)
```

## 🔄 与原版本的对比

| 功能 | InterviewDataProcessor | InterviewDataManager |
|------|----------------------|---------------------|
| 数据加载 | ✅ | ✅ (增强的验证) |
| 统计分析 | ✅ | ✅ (保留原逻辑) |
| 经验提取 | ✅ | ✅ (三种模式) |
| Checkpoint | ✅ 基础功能 | ✅ 完整的CRUD |
| 导出功能 | ⚠️ 仅JSON | ✅ 4种格式 |
| API设计 | ⚠️ 过程式 | ✅ 面向对象 |
| 模块化 | ❌ | ✅ |
| 类型安全 | ❌ | ✅ |
| 错误处理 | ⚠️ 基础 | ✅ 完善 |

## 📝 注意事项

1. **向后兼容**: 保留了原有的所有功能，可以平滑迁移
2. **数据格式**: 使用相同的CSV格式，无需修改数据文件
3. **AI配置**: 经验提取功能需要配置menglong AI模型
4. **日志**: 默认使用INFO级别，可通过配置调整

## 🐛 故障排除

### 问题1: 数据加载失败
```python
# 尝试指定编码
result = manager.load_data()
if not result.success:
    print(result.errors)
```

### 问题2: 经验提取超时
```python
# 减小批次大小
config = ManagerConfig(batch_size=3, timeout=600)
```

### 问题3: Token成本过高
```python
# 使用checkpoint断点续传
# 在提取过程中定期创建checkpoint
manager.create_checkpoint(name="progress_checkpoint")
```

## 🔗 相关文件

- `interview_processor.py` - 原版本（保留，不修改）
- `example_usage.py` - 完整的使用示例
- 各个模块文件（models.py, data_loader.py等）

## 📄 许可证

与原项目保持一致

## 👥 贡献

欢迎提交Issue和Pull Request！
