"""
面试模拟启动器

用法:
    # 无 transcript 模式（自主面试）
    python simulator.py --jd rm_jd --resume zhangsan_resume
    
    # 有 transcript 模式（参考面试记录）
    python simulator.py --jd rm_jd --resume zhangsan_resume --transcript zhangsan_rm_transcript_1
    
    # 指定最大轮数
    python simulator.py --jd rm_jd --resume zhangsan_resume --max-turns 10
"""

import argparse
import os
import sys

from simulation.interview_simulator import InterviewSimulator
from components.file_parser import FileParser


def load_jd(jd_name: str, resource_path: str = "data/resources/jd") -> str:
    """加载 JD 文件"""
    txt_file = os.path.join(resource_path, f"{jd_name}.txt")
    parser = FileParser()
    
    if os.path.exists(txt_file):
        print(f"读取 JD: {txt_file}")
        return parser.read_file(txt_file)
    
    raise FileNotFoundError(f"JD 未找到: {jd_name} (路径: {resource_path})")


def load_resume(resume_name: str, resource_path: str = "data/resources/candidate_resumes") -> str:
    """加载简历文件"""
    md_file = os.path.join(resource_path, f"{resume_name}.md")
    pdf_file = os.path.join(resource_path, f"{resume_name}.pdf")
    parser = FileParser()
    
    if os.path.exists(md_file):
        print(f"读取简历: {md_file}")
        return parser.read_file(md_file)
    elif os.path.exists(pdf_file):
        print(f"读取简历: {pdf_file}")
        return parser.read_file(pdf_file)
    
    raise FileNotFoundError(f"简历未找到: {resume_name} (路径: {resource_path})")


def load_transcript(transcript_name: str, resource_path: str = "data/resources/conversations") -> str:
    """加载 Transcript 文件"""
    pdf_file = os.path.join(resource_path, f"{transcript_name}.pdf")
    txt_file = os.path.join(resource_path, f"{transcript_name}.txt")
    parser = FileParser()
    
    if os.path.exists(pdf_file):
        print(f"读取 Transcript: {pdf_file}")
        return parser.read_file(pdf_file)
    elif os.path.exists(txt_file):
        print(f"读取 Transcript: {txt_file}")
        return parser.read_file(txt_file)
    
    raise FileNotFoundError(f"Transcript 未找到: {transcript_name} (路径: {resource_path})")


def parse_name_from_resume(resume_name: str) -> str:
    """从简历名称解析候选人名称"""
    # 格式: name_resume -> name
    parts = resume_name.split('_')
    if len(parts) >= 1:
        return parts[0]
    return resume_name


def parse_jd_short_name(jd_name: str) -> str:
    """从 JD 名称解析简称"""
    # 格式: desc_jd -> desc
    if jd_name.endswith('_jd'):
        return jd_name[:-3]
    return jd_name


def main():
    parser = argparse.ArgumentParser(
        description="面试模拟启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--jd", required=True, help="JD 名称（如 rm_jd）")
    parser.add_argument("--resume", required=True, help="简历名称（如 zhangsan_resume）")
    parser.add_argument("--transcript", help="可选的参考 Transcript 名称")
    parser.add_argument("--max-turns", type=int, default=20, help="最大面试轮数（默认 20）")
    parser.add_argument("--output-dir", default="data/generated/simulations", help="输出目录")
    parser.add_argument("--quiet", action="store_true", help="安静模式，减少输出")
    parser.add_argument("--temp", action="store_true", help="临时生成模式")
    parser.add_argument(
        "--interviewer-model", 
        default="anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="面试官使用的模型"
    )
    parser.add_argument(
        "--candidate-model",
        default="anthropic/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="候选人使用的模型"
    )
    
    args = parser.parse_args()
    
    # 加载资源
    try:
        jd_content = load_jd(args.jd)
        resume_content = load_resume(args.resume)
        
        transcript_content = None
        if args.transcript:
            transcript_content = load_transcript(args.transcript)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    
    # 解析名称用于保存
    candidate_name = parse_name_from_resume(args.resume)
    jd_short_name = parse_jd_short_name(args.jd)
    
    # 创建模拟器
    simulator = InterviewSimulator(
        jd=jd_content,
        resume=resume_content,
        transcript=transcript_content,
        interviewer_model=args.interviewer_model,
        candidate_model=args.candidate_model
    )
    
    # 运行模拟
    result = simulator.run(
        max_turns=args.max_turns,
        verbose=not args.quiet
    )
    
    # 保存结果
    output_dir = args.output_dir
    if args.temp:
        output_dir = "data/generated/temp"

    save_path = simulator.save(
        output_dir=output_dir,
        name=candidate_name,
        jd_name=jd_short_name
    )
    
    print(f"\n模拟完成！")
    print(f"  - 总轮数: {result['metadata']['total_turns']}")
    print(f"  - 面试官主动结束: {'是' if result['metadata']['ended_by_interviewer'] else '否'}")
    print(f"  - 结果保存: {save_path}")


if __name__ == "__main__":
    main()
