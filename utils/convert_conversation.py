"""
Conversation JSON 转换脚本（支持 TXT 和 Markdown 格式）

用法:
    # 转换为 TXT 格式
    python convert_conversation.py conversation.json --format txt
    
    # 转换为 Markdown 格式
    python convert_conversation.py conversation.json --format md
    
    # 指定输出文件和名称
    python convert_conversation.py conversation.json output.md --format md --name 张三
"""

import argparse
import json
import os
import sys
from typing import List, Dict


def conversation_to_txt(
    conversation: List[Dict[str, str]], 
    candidate_name: str = "候选人",
    interviewer_name: str = "面试官"
) -> str:
    """
    将对话记录转换为 TXT 格式。
    
    格式: name(time):content（每人一行，无换行）
    """
    lines = []
    time_offset = 0
    
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # 格式化时间 HH:MM:SS
        hours = time_offset // 3600
        minutes = (time_offset % 3600) // 60
        seconds = time_offset % 60
        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # 确定显示名称
        if role == "interviewer":
            display_name = interviewer_name
        elif role == "candidate":
            display_name = candidate_name
        else:
            display_name = role
        
        # 移除换行，确保每人说话只占一行
        content_single_line = content.replace('\n', ' ').replace('\r', '')
        lines.append(f"{display_name}({time_str}): {content_single_line}")
        
        # 根据内容长度估算时间偏移
        time_offset += max(30, len(content) // 100 * 30 + 30)
    
    return "\n".join(lines)


def conversation_to_markdown(
    conversation: List[Dict[str, str]], 
    candidate_name: str = "候选人",
    interviewer_name: str = "面试官",
    title: str = "模拟面试记录"
) -> str:
    """
    将对话记录转换为 Markdown 格式（可视化友好）。
    
    使用引用块区分面试官和候选人，保留原始格式。
    """
    lines = []
    
    # 标题
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**面试官**: {interviewer_name}")
    lines.append(f"**候选人**: {candidate_name}")
    lines.append(f"**对话轮次**: {len(conversation)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    turn = 0
    for i, msg in enumerate(conversation):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # 每轮对话（面试官问题开始）标记轮次
        if role == "interviewer":
            turn += 1
            lines.append(f"## 第 {turn} 轮")
            lines.append("")
        
        # 确定显示名称和样式
        if role == "interviewer":
            display_name = f"🎤 **{interviewer_name}**"
            # 面试官使用普通格式
            lines.append(display_name)
            lines.append("")
            lines.append(content)
        elif role == "candidate":
            display_name = f"💬 **{candidate_name}**"
            # 候选人使用引用块
            lines.append(display_name)
            lines.append("")
            # 将内容转换为引用格式
            quoted_content = "\n".join([f"> {line}" for line in content.split("\n")])
            lines.append(quoted_content)
        else:
            lines.append(f"**{role}**: {content}")
        
        lines.append("")
    
    # 结尾
    lines.append("---")
    lines.append("")
    lines.append("*面试记录结束*")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Conversation JSON 转换脚本（支持 TXT 和 Markdown 格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("input", help="输入的 conversation.json 文件路径")
    parser.add_argument("output", nargs="?", help="输出文件路径（默认根据格式自动生成）")
    parser.add_argument("--format", "-f", choices=["txt", "md"], default="txt",
                        help="输出格式: txt 或 md（默认: txt）")
    parser.add_argument("--name", default="候选人", help="候选人名称（默认：候选人）")
    parser.add_argument("--interviewer", default="面试官", help="面试官名称（默认：面试官）")
    parser.add_argument("--title", default="模拟面试记录", help="Markdown 标题（默认：模拟面试记录）")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        sys.exit(1)
    
    # 确定输出路径
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input)[0]
        ext = ".md" if args.format == "md" else ".txt"
        output_path = f"{base_name}{ext}"
    
    # 读取 JSON
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            conversation = json.load(f)
    except json.JSONDecodeError as e:
        print(f"错误: JSON 解析失败: {e}")
        sys.exit(1)
    
    # 验证格式
    if not isinstance(conversation, list):
        print("错误: JSON 应该是一个数组")
        sys.exit(1)
    
    # 转换
    if args.format == "md":
        output_content = conversation_to_markdown(
            conversation, 
            candidate_name=args.name,
            interviewer_name=args.interviewer,
            title=args.title
        )
    else:
        output_content = conversation_to_txt(
            conversation, 
            candidate_name=args.name,
            interviewer_name=args.interviewer
        )
    
    # 写入输出
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_content)
    
    print(f"转换完成！")
    print(f"  格式: {args.format.upper()}")
    print(f"  输入: {args.input} ({len(conversation)} 条消息)")
    print(f"  输出: {output_path}")


if __name__ == "__main__":
    main()
