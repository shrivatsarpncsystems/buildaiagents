import os
import sys
import subprocess
import re
from typing import Optional, Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Clean headless rendering; GUI viewer runs in isolated subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import analytics

# Set professional, modern styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.autolayout": True,
    "figure.titlesize": 14,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


def _finalize_plot(fig, filename: str, show: bool = True, title: str = "Academic Analytics Chart") -> str:
    """Save plot to CHARTS_DIR and optionally launch standalone non-interfering GUI viewer."""
    filepath = os.path.join(CHARTS_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Always close figure in main process to free memory and prevent GUI interference
    
    if show:
        try:
            viewer_script = os.path.join(os.path.dirname(__file__), "chart_viewer.py")
            py_exe = sys.executable
            # Launch isolated GUI viewer subprocess completely independent of the main terminal
            subprocess.Popen(
                [py_exe, viewer_script, filepath, title],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            try:
                os.startfile(filepath)
            except Exception as e2:
                print(f"[Note: GUI display window: {e2}]")
                
    return filepath


def plot_student_semester_marks(identifier: str, semester: int, show: bool = True) -> Optional[str]:
    """
    Bar graph showing individual subject marks out of 100 for a student in a semester.
    Includes pass line (40), distinction line (90), individual percentages, and aggregate total.
    """
    st = analytics.find_student(identifier)
    if not st or str(semester) not in st["semesters"]:
        return None
        
    sem_data = st["semesters"][str(semester)]
    subs = sem_data["subjects"]
    
    names = [s["name"] for s in subs]
    marks = [s["marks"] for s in subs]
    grades = [s["grade"] for s in subs]
    codes = [s["code"] for s in subs]
    
    total_obtained = sum(marks)
    total_max = len(subs) * 100
    sem_pct = round((total_obtained / total_max) * 100, 2)
    
    # Color bars based on grade
    colors = []
    for g in grades:
        if g in ["O", "A+"]:
            colors.append("#10B981")  # Emerald green
        elif g in ["A", "B+"]:
            colors.append("#2563EB")  # Royal blue
        elif g in ["B", "C"]:
            colors.append("#F59E0B")  # Amber
        else:
            colors.append("#EF4444")  # Rose red (Fail)
            
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x_positions = np.arange(len(subs))
    bars = ax.bar(x_positions, marks, color=colors, width=0.55, edgecolor="#1F2937", linewidth=1.2, zorder=3)
    
    # Reference threshold lines
    ax.axhline(40, color="#EF4444", linestyle="--", linewidth=1.2, label="Pass Mark (40%)", zorder=2)
    ax.axhline(75, color="#F59E0B", linestyle=":", linewidth=1.2, label="Distinction (75%)", zorder=2)
    ax.axhline(90, color="#10B981", linestyle="--", linewidth=1.2, label="Outstanding (90%)", zorder=2)
    
    # Value annotations on top of bars
    for bar, mark, grade in zip(bars, marks, grades):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2.0,
            yval + 1.5,
            f"{mark}/100\n({grade})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#111827"
        )
        
    ax.set_ylim(0, 115)
    ax.set_xticks(x_positions)
    # Shorten names for cleaner ticks
    tick_labels = [f"{c}\n" + (n[:20] + "..." if len(n) > 20 else n) for c, n in zip(codes, names)]
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")
    ax.set_ylabel("Marks Obtained (out of 100)", fontweight="bold")
    
    title = f"Semester {semester} Subject Marks Breakdown: {st['name']} ({st['usn']})"
    subtitle = f"Total Marks: {total_obtained} / {total_max}  |  Semester Percentage: {sem_pct:.2f}%  |  SGPA: {sem_data['sgpa']:.2f} / 10.0"
    ax.set_title(f"{title}\n{subtitle}", pad=14, fontweight="bold", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    
    filename = f"sem_{semester}_marks_{st['usn']}.png"
    return _finalize_plot(fig, filename, show=show)


def plot_student_comparison(id1: str, id2: str, show: bool = True) -> Optional[str]:
    """
    Side-by-side grouped bar chart comparing two students across all 8 semesters (SGPA progression).
    Highlights cumulative CGPAs and class ranks.
    """
    st1 = analytics.find_student(id1)
    st2 = analytics.find_student(id2)
    if not st1 or not st2:
        return None
        
    sems = [f"Sem {i}" for i in range(1, 9)]
    sgpa1 = [st1["semesters"][str(i)]["sgpa"] for i in range(1, 9)]
    sgpa2 = [st2["semesters"][str(i)]["sgpa"] for i in range(1, 9)]
    
    fig, ax = plt.subplots(figsize=(11, 5.8))
    width = 0.35
    x = np.arange(len(sems))
    
    color1 = "#2563EB"  # Blue
    color2 = "#7C3AED"  # Purple
    
    rects1 = ax.bar(x - width/2, sgpa1, width, label=f"{st1['name']} ({st1['usn']}) - CGPA: {st1['cgpa']:.2f}", color=color1, edgecolor="#1F2937", linewidth=1.1, zorder=3)
    rects2 = ax.bar(x + width/2, sgpa2, width, label=f"{st2['name']} ({st2['usn']}) - CGPA: {st2['cgpa']:.2f}", color=color2, edgecolor="#1F2937", linewidth=1.1, zorder=3)
    
    # Add CGPA horizontal lines
    ax.axhline(st1["cgpa"], color=color1, linestyle="--", alpha=0.7, linewidth=1.3, label=f"{st1['name']} Avg CGPA ({st1['cgpa']:.2f})")
    ax.axhline(st2["cgpa"], color=color2, linestyle=":", alpha=0.8, linewidth=1.5, label=f"{st2['name']} Avg CGPA ({st2['cgpa']:.2f})")
    
    # Value annotations on bars
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color1)
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.1, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color2)
        
    ax.set_ylim(0, 11.2)
    ax.set_ylabel("Semester SGPA (out of 10.0)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(sems, fontweight="bold")
    
    title = f"Academic Progression Comparison (8 Semesters)"
    subtitle = f"Candidate 1: {st1['name']} (Rank #{st1['rank']}, CGPA {st1['cgpa']:.2f})  vs  Candidate 2: {st2['name']} (Rank #{st2['rank']}, CGPA {st2['cgpa']:.2f})"
    ax.set_title(f"{title}\n{subtitle}", pad=14, fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    
    filename = f"comparison_{st1['usn']}_vs_{st2['usn']}.png"
    return _finalize_plot(fig, filename, show=show)


def plot_student_scorecard(identifier: str, show: bool = True) -> Optional[str]:
    """
    8-Semester SGPA progression chart with cumulative CGPA line and semester-by-semester bars.
    """
    st = analytics.find_student(identifier)
    if not st:
        return None
        
    sc = analytics.get_student_scorecard(identifier)
    sems = [f"Sem {i}" for i in range(1, 9)]
    sgpas = [st["semesters"][str(i)]["sgpa"] for i in range(1, 9)]
    
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bars = ax.bar(sems, sgpas, color="#0EA5E9", width=0.5, edgecolor="#0369A1", linewidth=1.2, zorder=3)
    ax.plot(sems, sgpas, color="#0369A1", marker="o", markersize=8, linewidth=2.5, zorder=4)
    
    # Cumulative CGPA reference line
    ax.axhline(st["cgpa"], color="#EF4444", linestyle="--", linewidth=1.5, label=f"Cumulative CGPA: {st['cgpa']:.2f} / 10.0", zorder=2)
    
    for bar, val in zip(bars, sgpas):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.15, f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        
    ax.set_ylim(0, 11.2)
    ax.set_ylabel("SGPA (Scale 0 - 10.0)", fontweight="bold")
    
    title = f"8-Semester Academic Scorecard: {st['name']} ({st['usn']})"
    subtitle = f"Class Rank: #{st['rank']} of 200  |  CGPA: {st['cgpa']:.2f}  |  8-Sem Average: {sc.get('average_marks', 'N/A')}/100  |  Backlogs: {st['backlogs']}"
    ax.set_title(f"{title}\n{subtitle}", pad=14, fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    
    filename = f"scorecard_{st['usn']}.png"
    return _finalize_plot(fig, filename, show=show)


def plot_top_scorers(n: int = 5, semester: Optional[int] = None, show: bool = True) -> Optional[str]:
    """
    Horizontal bar chart showing Top N students by SGPA (semester) or CGPA (overall).
    """
    data = analytics.get_top_scorers(n=n, semester=semester)
    students = data["top_students"]
    if not students:
        return None
        
    # Reverse so Rank 1 is at top
    students = list(reversed(students))
    
    labels = [f"#{s['rank']} {s['name']}\n({s['usn']})" for s in students]
    if semester:
        scores = [s["sgpa"] for s in students]
        metric_label = f"Semester {semester} SGPA (out of 10.0)"
        chart_title = f"Top {n} Scorers - Semester {semester} (Ranked by SGPA)"
        filename = f"top_{n}_sem_{semester}.png"
    else:
        scores = [s["cgpa"] for s in students]
        metric_label = "Cumulative CGPA (out of 10.0)"
        chart_title = f"Top {n} Batch Scorers (Autonomous Merit Gazette)"
        filename = f"top_{n}_overall_batch.png"
        
    fig, ax = plt.subplots(figsize=(10, max(4.5, n * 0.75)))
    colors = sns.color_palette("viridis", len(scores))
    bars = ax.barh(labels, scores, color=colors, height=0.55, edgecolor="#1F2937", linewidth=1.1, zorder=3)
    
    for bar, val in zip(bars, scores):
        ax.text(val + 0.08, bar.get_y() + bar.get_height()/2.0, f"{val:.2f}", ha="left", va="center", fontsize=9.5, fontweight="bold", color="#111827")
        
    ax.set_xlim(0, 10.6)
    ax.set_xlabel(metric_label, fontweight="bold")
    ax.set_title(chart_title, pad=14, fontweight="bold", fontsize=13)
    
    return _finalize_plot(fig, filename, show=show)


def plot_grade_distribution(subject_query: str, show: bool = True) -> Optional[str]:
    """
    Bar chart showing Grade Distribution (O, A+, A, B+, B, C, F) for a course or domain.
    """
    students = analytics._load_data()
    q = subject_query.lower().strip()
    
    target_name = None
    for k in sorted(analytics.DOMAIN_SUBJECT_MAPPINGS.keys(), key=len, reverse=True):
        pattern = rf"\b{re.escape(k)}\b" if len(k) <= 4 else rf"\b{re.escape(k)}"
        if re.search(pattern, q):
            target_name = analytics.DOMAIN_SUBJECT_MAPPINGS[k][0]
            break
            
    if not target_name:
        for sm in range(1, 9):
            for sub in students[0]["semesters"][str(sm)]["subjects"]:
                if sub["name"].lower() in q or sub["code"].lower() in q:
                    target_name = sub["name"]
                    break
            if target_name:
                break
                
    if not target_name:
        return None
        
    all_grades = []
    all_marks = []
    for s in students:
        for sm in s["semesters"].values():
            for sub in sm["subjects"]:
                if sub["name"] == target_name:
                    all_grades.append(sub["grade"])
                    all_marks.append(sub["marks"])
                    
    total = len(all_grades)
    grade_order = ["O", "A+", "A", "B+", "B", "C", "F"]
    counts = [all_grades.count(g) for g in grade_order]
    percentages = [(c / total) * 100 for c in counts]
    
    grade_colors = ["#10B981", "#34D399", "#2563EB", "#60A5FA", "#F59E0B", "#FBBF24", "#EF4444"]
    
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(grade_order, counts, color=grade_colors, width=0.55, edgecolor="#1F2937", linewidth=1.1, zorder=3)
    
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(
            bar.get_x() + bar.get_width()/2.0,
            count + 1.2,
            f"{count}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9.0,
            fontweight="bold"
        )
        
    ax.set_ylim(0, max(counts) * 1.25)
    ax.set_xlabel("Letter Grade Awarded", fontweight="bold")
    ax.set_ylabel("Student Count (Cohort = 200)", fontweight="bold")
    
    avg_score = sum(all_marks) / total
    pass_pct = ((total - all_grades.count("F")) / total) * 100
    title = f"Grade Distribution: {target_name.upper()}"
    subtitle = f"Course Average: {avg_score:.2f} / 100  |  Pass Rate: {pass_pct:.1f}%  |  Backlogs: {all_grades.count('F')} Students"
    ax.set_title(f"{title}\n{subtitle}", pad=14, fontweight="bold", fontsize=12)
    
    safe_name = re.sub(r"[^\w\-]", "_", target_name.lower())
    filename = f"grade_dist_{safe_name}.png"
    return _finalize_plot(fig, filename, show=show)


def plot_failed_students(semester: Optional[int] = None, show: bool = True) -> Optional[str]:
    """
    Bar chart showing failed students and active arrears per subject in a semester,
    or backlog distribution across all semesters.
    """
    students = analytics._load_data()
    
    if semester:
        fail_data = analytics.get_failed_students(semester=semester)
        failed_list = fail_data["students"]
        if not failed_list:
            return None
            
        names = [f"{s['name']}\n({s['usn']})" for s in failed_list]
        sgpas = [s["sgpa"] for s in failed_list]
        
        fig, ax = plt.subplots(figsize=(10, 5.2))
        bars = ax.bar(names, sgpas, color="#EF4444", width=0.5, edgecolor="#7F1D1D", linewidth=1.2, zorder=3)
        ax.axhline(4.0, color="#F59E0B", linestyle="--", linewidth=1.2, label="Pass Threshold SGPA (4.0)")
        
        for bar, s in zip(bars, failed_list):
            ax.text(
                bar.get_x() + bar.get_width()/2.0,
                bar.get_height() + 0.1,
                f"SGPA: {s['sgpa']:.2f}\nArrears: {len(s['failed_subjects'])}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold"
            )
            
        ax.set_ylim(0, 10.0)
        ax.set_ylabel(f"Semester {semester} SGPA", fontweight="bold")
        ax.set_title(f"Semester {semester} Backlog Candidates ({len(failed_list)} Students with Arrears)", pad=14, fontweight="bold", fontsize=12)
        ax.legend(loc="upper right", frameon=True)
        filename = f"failed_students_sem_{semester}.png"
        return _finalize_plot(fig, filename, show=show)
    else:
        # Distribution of backlogs across semesters 1 to 8
        sem_backlog_counts = {i: 0 for i in range(1, 9)}
        for s in students:
            for sm_idx in range(1, 9):
                sem_obj = s["semesters"][str(sm_idx)]
                backlog_in_sem = any(sub["grade"] == "F" for sub in sem_obj["subjects"])
                if backlog_in_sem:
                    sem_backlog_counts[sm_idx] += 1
                    
        sems = [f"Sem {i}" for i in range(1, 9)]
        counts = [sem_backlog_counts[i] for i in range(1, 9)]
        
        fig, ax = plt.subplots(figsize=(9.5, 5))
        bars = ax.bar(sems, counts, color="#F97316", width=0.55, edgecolor="#9A3412", linewidth=1.2, zorder=3)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2.0, c + 0.3, f"{c} students", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
            
        ax.set_ylabel("Number of Students with Active Arrears", fontweight="bold")
        ax.set_title(f"Cohort Arrears & Remediation Distribution Across 8 Semesters", pad=14, fontweight="bold", fontsize=12)
        filename = "overall_semester_arrears_distribution.png"
        return _finalize_plot(fig, filename, show=show)


def plot_batch_averages(show: bool = True) -> Optional[str]:
    """
    Bar chart showing batch average SGPA progression across Semesters 1 to 8.
    """
    students = analytics._load_data()
    sems = [f"Sem {i}" for i in range(1, 9)]
    avg_sgpas = []
    
    for i in range(1, 9):
        sgpas = [s["semesters"][str(i)]["sgpa"] for s in students]
        avg_sgpas.append(round(sum(sgpas) / len(sgpas), 2))
        
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bars = ax.bar(sems, avg_sgpas, color="#3B82F6", width=0.5, edgecolor="#1E40AF", linewidth=1.2, zorder=3)
    ax.plot(sems, avg_sgpas, color="#1E40AF", marker="s", markersize=8, linewidth=2.5, zorder=4)
    
    overall_cgpa_avg = round(sum(s["cgpa"] for s in students) / len(students), 2)
    ax.axhline(overall_cgpa_avg, color="#10B981", linestyle="--", linewidth=1.4, label=f"Cohort Cumulative CGPA Mean: {overall_cgpa_avg}")
    
    for bar, val in zip(bars, avg_sgpas):
        ax.text(bar.get_x() + bar.get_width()/2.0, val + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Mean SGPA (Scale 0 - 10.0)", fontweight="bold")
    ax.set_title(f"Cohort Academic Trend - Mean Semester SGPA Progression (200 Candidates)", pad=14, fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", frameon=True)
    
    filename = "batch_semester_averages.png"
    return _finalize_plot(fig, filename, show=show)


def plot_placement_eligibility(min_cgpa: float = 7.0, max_backlogs: int = 0, show: bool = True) -> Optional[str]:
    """
    Donut / bar chart displaying corporate campus placement drive screening.
    """
    students = analytics._load_data()
    eligible = sum(1 for s in students if s["cgpa"] >= min_cgpa and s["backlogs"] <= max_backlogs)
    ineligible = len(students) - eligible
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    
    # Donut Chart
    labels = [f"Eligible ({eligible})", f"Remediation Required ({ineligible})"]
    sizes = [eligible, ineligible]
    colors = ["#10B981", "#EF4444"]
    ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, wedgeprops=dict(width=0.4, edgecolor="#FFFFFF", linewidth=2))
    ax1.set_title(f"Drive Eligibility Breakdown (Cutoff: CGPA >= {min_cgpa:.1f})", fontweight="bold", fontsize=11)
    
    # CGPA Tier Breakdown
    tiers = ["Tier 1 (>=8.5)", "Tier 2 (7.5-8.49)", "Tier 3 (6.5-7.49)", "Below 6.5"]
    tier_counts = [
        sum(1 for s in students if s["cgpa"] >= 8.5),
        sum(1 for s in students if 7.5 <= s["cgpa"] < 8.5),
        sum(1 for s in students if 6.5 <= s["cgpa"] < 7.5),
        sum(1 for s in students if s["cgpa"] < 6.5)
    ]
    tier_colors = ["#3B82F6", "#60A5FA", "#93C5FD", "#F87171"]
    bars = ax2.bar(tiers, tier_counts, color=tier_colors, edgecolor="#1F2937", linewidth=1.1, width=0.55)
    for bar, c in zip(bars, tier_counts):
        ax2.text(bar.get_x() + bar.get_width()/2.0, c + 1.5, f"{c}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Student Count", fontweight="bold")
    ax2.set_title("Cohort CGPA Tier Distribution", fontweight="bold", fontsize=11)
    ax2.tick_params(axis="x", rotation=15)
    
    fig.suptitle(f"Corporate Placement & TPO Screening Analytics (Cohort = 200 Students)", fontsize=13, fontweight="bold")
    
    filename = f"placement_eligibility_cgpa_{min_cgpa}.png"
    return _finalize_plot(fig, filename, show=show)


GRAPH_KEYWORDS = [
    "graph", "plot", "chart", "visualize", "visualization", "diagram",
    "bar graph", "bar chart", "line chart", "histogram", "show graph",
    "give graph", "draw graph", "display graph", "make graph", "plot graph",
    "groph", "groff"
]


def is_graph_requested(text: str) -> bool:
    """Check if the user explicitly asked for a graph, plot, chart, or visualization."""
    if not text:
        return False
    t = text.lower()
    return any(re.search(rf"\b{re.escape(w)}\b", t) or w in t for w in GRAPH_KEYWORDS)


def dispatch_graph_for_query(user_query: str, history: Optional[List[Dict[str, Any]]] = None, show: bool = True) -> Optional[Dict[str, Any]]:
    """
    Intelligently analyzes user query and multi-turn conversation history to determine
    the appropriate academic visualization (bar graph, line progression, distribution).
    Triggers ONLY when the user asks for a graph or chart.
    """
    if not is_graph_requested(user_query):
        return None
        
    q_lower = user_query.lower().strip()
    if history is None:
        history = []
        
    # 1. Extract entities from current query
    direct_usns = [u.upper() for u in list(dict.fromkeys(re.findall(r"1rv21cs\d{3}", q_lower)))]
    
    direct_sem = analytics.extract_semester(q_lower)
    
    # 2. Extract entities from conversation history (k turns)
    recent_usns = []
    recent_sems = []
    last_turn_q = ""
    last_turn_a = ""
    if history:
        last_turn = history[-1]
        last_turn_q = last_turn.get("question", "").lower()
        last_turn_a = last_turn.get("answer", "").lower()
        for turn in reversed(history[-6:]):
            full_txt = f"{turn.get('question', '')} {turn.get('answer', '')}".lower()
            for u in re.findall(r"1rv21cs\d{3}", full_txt):
                u_upper = u.upper()
                if u_upper not in recent_usns:
                    recent_usns.append(u_upper)
            h_sem = analytics.extract_semester(turn.get("question", "")) or analytics.extract_semester(turn.get("answer", ""))
            if h_sem and h_sem not in recent_sems:
                recent_sems.append(h_sem)
            for sm in re.findall(r"(?:sem|semester)\s*([1-8])", full_txt):
                sm_int = int(sm)
                if sm_int not in recent_sems:
                    recent_sems.append(sm_int)

    # Check student by name if not in direct USN
    target_st_name = None
    if not direct_usns:
        clean = re.sub(r"\b(?:graph|plot|chart|visualize|bar|line|diagram|show|give|for|previous|conversation|in|of|sem|semester|5th|4th|3rd|2nd|1st|[1-8])\b", " ", q_lower).strip()
        if len(clean) >= 3:
            matched = analytics.find_students_by_name(clean)
            if matched:
                target_st_name = matched[0]["usn"]

    target_usn = direct_usns[0] if direct_usns else (target_st_name or (recent_usns[0] if recent_usns else None))
    target_sem = direct_sem or (recent_sems[0] if recent_sems else None)

    # -------------------------------------------------------------
    # CASE 1: Student Comparison (2 students)
    # -------------------------------------------------------------
    is_compare = any(w in q_lower for w in ["compare", "comparison", "difference", "versus", "vs", "between", "both"]) or (
        ("previous" in q_lower or "them" in q_lower or "their" in q_lower) and (len(recent_usns) >= 2 or (last_turn_q and any(w in last_turn_q for w in ["compare", "vs"])))
    )
    if is_compare or len(direct_usns) >= 2:
        u1, u2 = None, None
        if len(direct_usns) >= 2:
            u1, u2 = direct_usns[0], direct_usns[1]
        elif len(direct_usns) == 1 and len(recent_usns) >= 1:
            u1, u2 = recent_usns[0], direct_usns[0]
        elif len(recent_usns) >= 2:
            u1, u2 = recent_usns[0], recent_usns[1]
            
        if u1 and u2:
            p = plot_student_comparison(u1, u2, show=show)
            if p:
                return {
                    "chart_path": p,
                    "chart_type": "student_comparison",
                    "description": f"8-Semester Academic Performance Comparison: {u1} vs {u2}",
                    "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
                }

    # -------------------------------------------------------------
    # CASE 2: Single Student Semester Subject Marks Breakdown
    # -------------------------------------------------------------
    has_sem_intent = target_sem is not None and (
        direct_sem is not None or any(w in q_lower for w in ["subject", "subjects", "mark", "marks", "sem", "semester", "score", "percentage"])
        or (last_turn_q and any(w in last_turn_q for w in ["mark", "marks", "score", "sem", "percentage"]))
    )
    if target_usn and target_sem and has_sem_intent:
        p = plot_student_semester_marks(target_usn, target_sem, show=show)
        if p:
            return {
                "chart_path": p,
                "chart_type": "student_semester_marks",
                "description": f"Semester {target_sem} Subject Marks and Percentage Breakdown for {target_usn}",
                "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
            }

    # -------------------------------------------------------------
    # CASE 3: Top Scorers (Semester or Overall Batch)
    # -------------------------------------------------------------
    if any(w in q_lower for w in ["top", "topper", "toppers", "rank", "merit", "highest"]) or (last_turn_q and any(w in last_turn_q for w in ["top", "topper", "toppers", "rank"])):
        num_match = re.search(r"(?:top|first)\s*(\d+)", q_lower) or (re.search(r"(?:top|first)\s*(\d+)", last_turn_q) if last_turn_q else None)
        n_count = int(num_match.group(1)) if num_match else 5
        p = plot_top_scorers(n=n_count, semester=target_sem if (direct_sem or (last_turn_q and "sem" in last_turn_q)) else None, show=show)
        if p:
            return {
                "chart_path": p,
                "chart_type": "top_scorers",
                "description": f"Top {n_count} Scorers {'(Semester ' + str(target_sem) + ')' if target_sem else '(Overall Batch)'}",
                "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
            }

    # -------------------------------------------------------------
    # CASE 4: Failed Students / Backlog Arrears
    # -------------------------------------------------------------
    if any(w in q_lower for w in ["fail", "failed", "backlog", "backlogs", "arrear", "arrears"]) or (last_turn_q and any(w in last_turn_q for w in ["fail", "failed", "backlog", "arrear"])):
        p = plot_failed_students(semester=target_sem, show=show)
        if p:
            return {
                "chart_path": p,
                "chart_type": "failed_students",
                "description": f"Arrears and Remediation Distribution {'(Semester ' + str(target_sem) + ')' if target_sem else '(Overall)'}",
                "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
            }

    # -------------------------------------------------------------
    # CASE 5: Subject / Domain Grade Distribution
    # -------------------------------------------------------------
    sub_candidate = user_query if not any(w in q_lower for w in ["previous", "history"]) else last_turn_q
    if any(w in q_lower for w in ["grade", "distribution"]) or any(k in sub_candidate.lower() for k in analytics.DOMAIN_SUBJECT_MAPPINGS.keys()):
        p = plot_grade_distribution(sub_candidate, show=show)
        if p:
            return {
                "chart_path": p,
                "chart_type": "grade_distribution",
                "description": "Course Grade Distribution across 200 candidates",
                "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
            }

    # -------------------------------------------------------------
    # CASE 6: Placement Eligibility / TPO Screening
    # -------------------------------------------------------------
    if any(w in q_lower for w in ["placement", "campus drive", "tier 1", "tier-1", "cutoff", "tpo"]) or (last_turn_q and any(w in last_turn_q for w in ["placement", "drive", "cutoff"])):
        cutoff = 8.5 if ("tier 1" in q_lower or "tier-1" in q_lower or "tier 1" in last_turn_q) else 7.0
        p = plot_placement_eligibility(min_cgpa=cutoff, show=show)
        if p:
            return {
                "chart_path": p,
                "chart_type": "placement_eligibility",
                "description": f"Corporate Placement Screening Analytics (Cutoff CGPA >= {cutoff})",
                "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
            }

    # -------------------------------------------------------------
    # CASE 7: Single Student 8-Semester Progression Scorecard
    # -------------------------------------------------------------
    if target_usn:
        p = plot_student_scorecard(target_usn, show=show)
        if p:
            return {
                "chart_path": p,
                "chart_type": "student_scorecard",
                "description": f"8-Semester Academic Progression Scorecard for {target_usn}",
                "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
            }

    # -------------------------------------------------------------
    # CASE 8: Cohort / Batch Average Progression
    # -------------------------------------------------------------
    p = plot_batch_averages(show=show)
    if p:
        return {
            "chart_path": p,
            "chart_type": "batch_averages",
            "description": "Cohort Academic Trend - Mean Semester SGPA Progression",
            "markdown_notice": f"\n\n📊 **[Interactive Chart Displayed & Generated]**: Saved to `{os.path.basename(p)}`"
        }

    return None
