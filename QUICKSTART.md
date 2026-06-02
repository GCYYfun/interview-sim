# 🚀 快速开始指南

欢迎使用**面试模拟系统 v2.0**！

## 📋 前置要求

- Python >= 3.13
- 依赖包已安装（见下方）

## 🔧 安装

### 1. 克隆项目

```bash
git clone <repository-url>
cd interview_sim
```

### 2. 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

**核心依赖**:
- `rich` - 美观的控制台界面
- `menglong` - AI 模型框架
- `pandas` - 数据处理
- `matplotlib`, `seaborn` - 数据可视化

## ▶️ 运行应用

### 方式 1: 直接运行主程序

```bash
python main.py
```

### 方式 2: 模块方式运行

```bash
python -m cli.app
```

## 🎯 功能菜单

启动后，您将看到：

```
═══════════════════════════════════════════════════════
            面试模拟系统
        Interview Simulation System
═══════════════════════════════════════════════════════

选项    功能
────────────────────────────────────────────────────
1       🎭 面试模拟 - 运行完整面试流程
2       📊 数据分析 - 查看统计和分析数据
3       💡 经验提取 - 从面试中提取经验
4       🤝 面试助手 - 准备面试问题和分析
5       📄 报告查看 - 浏览和导出报告
0       ❌ 退出系统
```

## 📝 快速示例

### 示例 1: 运行自动面试

1. 选择 `1` - 面试模拟
2. 模式选择 `auto` (AI 自动对话)
3. 输入简历路径: `temp/demo_data.csv`
4. JD 路径: 按 Enter 跳过
5. 面试轮次: `5`

### 示例 2: 查看数据统计

1. 选择 `2` - 数据分析
2. 选择 `1` - 显示统计信息

### 示例 3: 生成面试问题

1. 选择 `4` - 面试助手
2. 选择 `1` - 生成面试问题
3. 输入简历和 JD 路径

## 📁 数据准备

### 简历文件 (CSV 格式)

确保 `interview_data.csv` 包含以下字段：

```csv
id,name,position,education,experience,skills,...
```

### JD 文件 (文本格式)

岗位描述文件，包含：
- 岗位名称
- 职责要求
- 技能要求
- 经验要求

## 🎨 Rich 界面特性

- ✅ 彩色输出（成功/错误/信息提示）
- ✅ 交互式菜单导航
- ✅ 表格数据展示
- ✅ 进度反馈
- ✅ 美观的面板布局

## 🐛 故障排查

### 问题 1: Rich 未安装

```bash
uv add rich
# 或
pip install rich
```

### 问题 2: 找不到模块

确保在项目根目录运行：

```bash
cd /path/to/interview_sim
python main.py
```

### 问题 3: CSV 文件错误

检查 `interview_data.csv` 是否存在且格式正确。

## 📚 更多文档

- [agents/README.md](agents/README.md) - Agent 系统文档
- [modules/README.md](modules/README.md) - 功能模块文档
- [cli/README.md](cli/README.md) - CLI 应用文档
- [manager/README.md](manager/README.md) - 数据管理文档

## 🎉 开始使用

```bash
python main.py
```

**祝您使用愉快！** 🚀
