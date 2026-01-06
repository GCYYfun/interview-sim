
import traceback
import argparse
import traceback
import os
import json
import datetime
import sys
from pathlib import Path

from agents.eval_agent import EvalAgent
from components.file_parser import FileParser
from components.data_manager import DataManager
from components.selector import InterviewSelector

def load_transcript(transcript_name, resource_path=None):
    """Loads transcript from PDF or falls back to text."""

    if resource_path is None:
        resource_path = "data/resources/conversations"

    # Try both direct path (if selector gave full path) or constructed path
    candidates = [
        os.path.join(resource_path, f"{transcript_name}.pdf"),
        os.path.join(resource_path, f"{transcript_name}.txt"),
         # In case name implies full path
        f"{transcript_name}.pdf" if not transcript_name.endswith('.pdf') else transcript_name, 
        f"{transcript_name}.txt" if not transcript_name.endswith('.txt') else transcript_name
    ]
    
    parser = FileParser()
    
    for f in candidates:
        if os.path.exists(f):
            print(f"Reading {f}...")
            return parser.read_file(f)
        
    raise FileNotFoundError(f"Transcript not found for {transcript_name}")

def load_jd(jd_name, resource_path=None):

    if resource_path is None:
        resource_path = "data/resources/jd"

    md_file = os.path.join(resource_path, f"{jd_name}.txt")
    parser = FileParser()
    
    if os.path.exists(md_file):
        print(f"Reading {md_file}...")
        return parser.read_file(md_file)

    raise FileNotFoundError(f"JD not found for {jd_name} in {resource_path}")

def load_resume(resume_name, resource_path=None):

    if resource_path is None:
        resource_path = "data/resources/candidate_resumes"


    md_file = os.path.join(resource_path, f"{resume_name}.md")
    pdf_file = os.path.join(resource_path, f"{resume_name}.pdf")
    parser = FileParser()
    
    if os.path.exists(md_file):
        print(f"Reading {md_file}...")
        return parser.read_file(md_file)
    elif os.path.exists(pdf_file):
        print(f"Reading {pdf_file}...")
        return parser.read_file(pdf_file)
    else:
        raise FileNotFoundError(f"Resume not found for {resume_name} in {resource_path}")


def get_output_paths(transcript_name, temp=False):
    """Determines output paths for topic and report.
    
    Topic Analysis: Always uses standard cleaned/topic/ directory to allow reuse.
    Report: Uses standard reports/ directory by default, or temp/ if temp=True.
    """
    
    # Topic is stable, always use standard path
    topic_dir = "data/generated/cleaned/topic/"
    topic_filename = f"topic_{transcript_name}.json"
    topic_path = os.path.join(topic_dir, topic_filename)

    # Report depends on temp flag
    if temp:
        timestamp = f"_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir = "data/generated/temp/"
        report_filename = f"report_{transcript_name}{timestamp}.json"
    else:
        report_dir = "data/generated/reports/"
        report_filename = f"report_{transcript_name}.json"

    os.makedirs(topic_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    return topic_path, os.path.join(report_dir, report_filename)

def run_topic_analysis(agent, transcript_content, info, output_path, data_manager, force=False):
    """Runs or retrieves topic analysis."""
    
    if not force and os.path.exists(output_path):
        print(f"Topic analysis already exists at {output_path}")
        return data_manager.load_json(output_path)
    
    print("\nRunning Analyze Topics...")
    topic_analysis = agent.analyze_topics(transcript_content, info)
    data_manager.save_json(topic_analysis, output_path)
    print(f"Saved topic analysis to {output_path}")
    return topic_analysis

def run_evaluation(agent, topic_analysis, transcript_content, info, output_path, data_manager, force=False,stage="1"):
    """Runs or retrieves evaluation report."""
    
    if not force and os.path.exists(output_path):
        print(f"Evaluation report already exists at {output_path}")
        return
        
    print("\nRunning Evaluation...")
    
    # Prepare content from topics
    topics_content = ""
    if topic_analysis and "topics" in topic_analysis:
        for topic in topic_analysis["topics"]:
            topics_content += f"## 主题: {topic.get('topic_name', 'Unknown')}\n"
            for msg in topic.get("dialogue", []):
                sender = msg.get('name', msg.get('role', 'Unknown'))
                timestamp = f" ({msg.get('timestamp')})" if msg.get('timestamp') else ""
                content = msg.get('content', '')
                topics_content += f"{sender}{timestamp}: {content}\n"
            topics_content += "\n\n"
            
    if not topics_content:
        print("Warning: No topics found in analysis, falling back to raw transcript.")
        topics_content = transcript_content

    summary = None
    if stage != "1":
        prev_report_path = f"{output_path[:-6]}{int(stage)-1}.json"
        try:
            prev_report = data_manager.load_json(prev_report_path)
            prev_res = prev_report.get("response")
            summary = prev_res.get('summary', '没有获取到上一次评价总结，直接开始二面')
        except:
            traceback.print_exc()

    evaluation_report = agent.evaluate_interview(topics_content, info,True,stage,summary)
    data_manager.save_json(evaluation_report, output_path)
    print(f"Saved evaluation report to {output_path}")

def process_interview(transcript_name, args, data_manager, path_override=None):
    """Processes a single interview: loads data, runs topic analysis, runs evaluation."""
    print(f"\n{'='*50}")
    print(f"Processing: {transcript_name}")
    print(f"{'='*50}")

    try:
        # 2. Load Transcript
        # If path_override is provided (e.g. valid file path), use directory of that file
        resource_path = args.path
        
        # Determine actual file name for loading if full path is passed
        load_name = transcript_name
        
        transcript_content = load_transcript(load_name, resource_path)
        print(f"Transcript read, length: {len(transcript_content)} chars")

        # Parse transcript_name
        # Rule: name_desc_transcript_x
        # We need the basename without extension for split logic if transcript_name is full path
        basename = os.path.splitext(os.path.basename(transcript_name))[0]
        
        parts = basename.split('_')
        candidate_name = parts[0]
        job_desc = parts[1] if len(parts) > 1 else "unknown"
        # Avoid 'transcript' being the job description if format is name_transcript_x
        if job_desc == 'transcript': 
             job_desc = "unknown"
             
        stage = parts[3] if len(parts) > 3 else "unknown"

        # 2.1 Load JD
        jd_name = f"{job_desc}_jd"
        try:
            jd_content = load_jd(jd_name, "data/resources/jd")
            print(f"JD read ({jd_name}), length: {len(jd_content)}")
        except Exception:
            print(f"JD not found: {jd_name}")
            jd_content = None

        # 2.2 Load Resume
        resume_name = f"{candidate_name}_resume"
        try:
            resume_content = load_resume(resume_name, "data/resources/candidate_resumes")
            print(f"Resume read ({resume_name}), length: {len(resume_content)}")
        except Exception:
            print(f"Resume not found: {resume_name}")
            resume_content = None
        
        # Save raw text backup (always good to have)
        data_manager.save_txt(transcript_content, f"data/generated/cleaned/raw_text/raw_{basename}.txt")

        
        info = {
            "jd": jd_content if jd_content else "通用岗位面试（未提供详细JD）",
            "resume": resume_content if resume_content else "未提供简历"
        }
        
        topic_analysis = None
        agent = EvalAgent()
        
        # Determine paths using basename (clean id)
        topic_path, report_path = get_output_paths(basename, args.temp)
        
        force_topic = args.force or args.force_topic
        force_report = args.force or args.force_report

        # Execute Pipeline
        if args.step in ["all", "topic"]:
            topic_analysis = run_topic_analysis(
                agent, transcript_content, info, topic_path, data_manager, force=force_topic
            )
            
        if args.step in ["all", "report"]:
            if not topic_analysis:
                if os.path.exists(topic_path):
                     topic_analysis = data_manager.load_json(topic_path)
                else:
                    standard_topic_path = f"data/generated/cleaned/topic/topic_{basename}.json"
                    if os.path.exists(standard_topic_path):
                         print(f"Using topic analysis from standard location: {standard_topic_path}")
                         topic_analysis = data_manager.load_json(standard_topic_path)
                    else:
                        print("Topic analysis missing. Running report on raw transcript.")
                        topic_analysis = None

            run_evaluation(
                agent, topic_analysis, transcript_content, info, report_path, data_manager, force=force_report,stage=stage
            )
            
    except Exception as e:
        print(f"Error processing {transcript_name}: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Interview Evaluation Pipeline")
    parser.add_argument("--name", help="Base name of the transcript (without extension)")
    parser.add_argument("--path", default=f"data/resources/conversations/", help="Path to transcript directory")
    parser.add_argument("--step", choices=["all", "topic", "report"], default="all", help="Pipeline step to run")
    parser.add_argument("--force", action="store_true", help="Force overwrite all steps")
    parser.add_argument("--force-topic", action="store_true", help="Force overwrite topic analysis")
    parser.add_argument("--force-report", action="store_true", help="Force overwrite evaluation report")
    parser.add_argument("--temp", action="store_true", help="Save output to temp dir with timestamp")
    
    # Filter arguments
    parser.add_argument("--jd", help="Filter by JD (for batch selector)")
    parser.add_argument("--candidate", help="Filter by Candidate Name (for batch selector)")
    
    args = parser.parse_args()
    
    data_manager = DataManager()
    selector = InterviewSelector(args.path)

    try:
        if args.name:
            # Single file mode (legacy compatible)
            process_interview(args.name, args, data_manager)
        else:
            # Batch/Selector mode
            selector.scan()
            
            # Apply CLI args filters if present
            filters = {}
            if args.jd:
                filters['jd'] = args.jd
            if args.candidate:
                filters['candidate'] = args.candidate
            
            if filters:
                selected_transcripts = selector.filter(filters)
                if not selected_transcripts:
                     print(f"No transcripts matched filters: {filters}")
                     return
                print(f"Found {len(selected_transcripts)} matching transcripts.")
            else:
                # Interactive mode
                selected_transcripts = selector.interactive_select()
            
            if not selected_transcripts:
                print("No transcripts selected.")
                return

            print(f"\nStarting batch processing for {len(selected_transcripts)} interviews...")
            for item in selected_transcripts:
                process_interview(item['name'], args, data_manager)

    except Exception as e:
        print(f"Global Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()