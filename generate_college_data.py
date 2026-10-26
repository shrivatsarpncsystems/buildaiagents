import os
import json
import random
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Seed for reproducible realistic data
random.seed(42)

CURRICULUM = {
    1: {
        "name": "Semester 1",
        "subjects": [
            {"code": "21MAT11", "name": "Engineering Mathematics I", "credits": 4},
            {"code": "21PHY12", "name": "Engineering Physics", "credits": 4},
            {"code": "21ELE13", "name": "Basic Electrical Engineering", "credits": 3},
            {"code": "21CIV14", "name": "Engineering Graphics", "credits": 3},
            {"code": "21PHYL15", "name": "Engineering Physics Lab", "credits": 1.5},
            {"code": "21ENG16", "name": "Technical English", "credits": 1.5},
        ]
    },
    2: {
        "name": "Semester 2",
        "subjects": [
            {"code": "21MAT21", "name": "Engineering Mathematics II", "credits": 4},
            {"code": "21CHE22", "name": "Engineering Chemistry", "credits": 4},
            {"code": "21PSP23", "name": "C Programming & Problem Solving", "credits": 3},
            {"code": "21ELN24", "name": "Basic Electronics Engineering", "credits": 3},
            {"code": "21CPL25", "name": "C Programming Lab", "credits": 1.5},
            {"code": "21CHEL26", "name": "Engineering Chemistry Lab", "credits": 1.5},
        ]
    },
    3: {
        "name": "Semester 3",
        "subjects": [
            {"code": "21CS31", "name": "Discrete Mathematical Structures", "credits": 4},
            {"code": "21CS32", "name": "Data Structures & Applications", "credits": 4},
            {"code": "21CS33", "name": "Computer Organization & Architecture", "credits": 3},
            {"code": "21CS34", "name": "Object Oriented Programming with Java", "credits": 3},
            {"code": "21CSL35", "name": "Data Structures Laboratory", "credits": 1.5},
            {"code": "21CSL36", "name": "Java Laboratory", "credits": 1.5},
        ]
    },
    4: {
        "name": "Semester 4",
        "subjects": [
            {"code": "21CS41", "name": "Design & Analysis of Algorithms", "credits": 4},
            {"code": "21CS42", "name": "Operating Systems", "credits": 4},
            {"code": "21CS43", "name": "Database Management Systems", "credits": 3},
            {"code": "21CS44", "name": "Theory of Computation & Automata", "credits": 3},
            {"code": "21CSL45", "name": "Algorithms Laboratory", "credits": 1.5},
            {"code": "21CSL46", "name": "DBMS Laboratory", "credits": 1.5},
        ]
    },
    5: {
        "name": "Semester 5",
        "subjects": [
            {"code": "21CS51", "name": "Computer Networks", "credits": 4},
            {"code": "21CS52", "name": "Software Engineering & Agile", "credits": 3},
            {"code": "21CS53", "name": "Web Technologies & Frameworks", "credits": 3},
            {"code": "21CS54", "name": "Microprocessors & Embedded Systems", "credits": 3},
            {"code": "21CSL55", "name": "Computer Networks Lab", "credits": 1.5},
            {"code": "21CSL56", "name": "Web Development Lab", "credits": 1.5},
        ]
    },
    6: {
        "name": "Semester 6",
        "subjects": [
            {"code": "21CS61", "name": "Compiler Design", "credits": 4},
            {"code": "21CS62", "name": "Artificial Intelligence & Expert Systems", "credits": 4},
            {"code": "21CS63", "name": "Cryptography & Network Security", "credits": 3},
            {"code": "21CS64", "name": "Cloud Computing & Virtualization", "credits": 3},
            {"code": "21CSL65", "name": "AI & Intelligence Lab", "credits": 1.5},
            {"code": "21CSL66", "name": "Security & Pentesting Lab", "credits": 1.5},
        ]
    },
    7: {
        "name": "Semester 7",
        "subjects": [
            {"code": "21CS71", "name": "Machine Learning & Pattern Recognition", "credits": 4},
            {"code": "21CS72", "name": "Big Data Analytics & Spark", "credits": 3},
            {"code": "21CS73", "name": "Internet of Things & Robotics", "credits": 3},
            {"code": "21CSE74", "name": "Elective I: Natural Language Processing", "credits": 3},
            {"code": "21CSL75", "name": "Machine Learning Laboratory", "credits": 1.5},
            {"code": "21CSP76", "name": "Major Project Phase I", "credits": 3},
        ]
    },
    8: {
        "name": "Semester 8",
        "subjects": [
            {"code": "21CS81", "name": "Deep Learning & Generative AI", "credits": 4},
            {"code": "21CS82", "name": "Distributed Systems & Blockchain", "credits": 3},
            {"code": "21CSE83", "name": "Elective II: DevOps & Microservices", "credits": 3},
            {"code": "21CSS84", "name": "Technical Seminar & Paper Publication", "credits": 2},
            {"code": "21CSP85", "name": "Major Project Phase II", "credits": 6},
        ]
    }
}

FIRST_NAMES = [
    "Aarav", "Aditi", "Aditya", "Akash", "Ananya", "Aniket", "Ankita", "Arjun", "Deepak", "Divya",
    "Gaurav", "Harsh", "Ishaan", "Kavya", "Manish", "Meera", "Mohit", "Neha", "Nikhil", "Nisha",
    "Pooja", "Pranav", "Prateek", "Priya", "Rahul", "Rakesh", "Riya", "Rohan", "Rohit", "Sahil",
    "Sameer", "Sanjay", "Sanjana", "Shreya", "Siddharth", "Sneha", "Tanvi", "Varun", "Vikram", "Yash"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Reddy", "Iyer", "Gupta", "Nair", "Singh", "Joshi", "Bose",
    "Kulkarni", "Deshmukh", "Choudhury", "Bhat", "Menon", "Rao", "Mishra", "Pandey", "Mehta", "Agarwal"
]

def calculate_grade(marks):
    if marks >= 90:
        return "O", 10
    elif marks >= 80:
        return "A+", 9
    elif marks >= 70:
        return "A", 8
    elif marks >= 60:
        return "B+", 7
    elif marks >= 50:
        return "B", 6
    elif marks >= 40:
        return "C", 5
    else:
        return "F", 0

def generate_dataset(num_students=200):
    students = []
    
    for i in range(1, num_students + 1):
        usn = f"1RV21CS{i:03d}"
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"
        
        # Student baseline performance profile (from high achiever to average to struggling)
        # 15% top achievers (85-98), 65% above average/good (60-84), 15% average (45-65), 5% at-risk
        profile_rand = random.random()
        if profile_rand < 0.15:
            base_mean = random.uniform(85, 94)
            spread = 5
        elif profile_rand < 0.80:
            base_mean = random.uniform(68, 83)
            spread = 7
        elif profile_rand < 0.95:
            base_mean = random.uniform(52, 67)
            spread = 8
        else:
            base_mean = random.uniform(38, 52)
            spread = 10
            
        semesters_data = {}
        total_credit_points = 0.0
        total_credits = 0.0
        backlogs_count = 0
        
        for sem_num in range(1, 9):
            sem_info = CURRICULUM[sem_num]
            subjects_record = []
            sem_credits = 0.0
            sem_points = 0.0
            
            for subj in sem_info["subjects"]:
                # Introduce slight variation per semester/subject
                marks = int(random.gauss(base_mean, spread))
                marks = max(25, min(99, marks))
                grade, gp = calculate_grade(marks)
                if grade == "F":
                    backlogs_count += 1
                
                credits = subj["credits"]
                sem_credits += credits
                sem_points += credits * gp
                
                subjects_record.append({
                    "code": subj["code"],
                    "name": subj["name"],
                    "credits": credits,
                    "marks": marks,
                    "grade": grade,
                    "grade_point": gp
                })
                
            sgpa = round(sem_points / sem_credits, 2)
            total_credits += sem_credits
            total_credit_points += sem_points
            
            semesters_data[str(sem_num)] = {
                "semester_name": sem_info["name"],
                "sgpa": sgpa,
                "credits": sem_credits,
                "subjects": subjects_record
            }
            
        cgpa = round(total_credit_points / total_credits, 2)
        
        if backlogs_count > 0:
            standing = "Active Backlog"
        elif cgpa >= 8.5:
            standing = "First Class with Distinction"
        elif cgpa >= 6.75:
            standing = "First Class"
        elif cgpa >= 5.0:
            standing = "Second Class"
        else:
            standing = "Pass"
            
        students.append({
            "usn": usn,
            "name": full_name,
            "branch": "Computer Science & Engineering",
            "batch": "2021-2025",
            "total_credits": total_credits,
            "cgpa": cgpa,
            "backlogs": backlogs_count,
            "standing": standing,
            "semesters": semesters_data
        })
        
    # Sort students by CGPA to compute ranks
    students.sort(key=lambda x: x["cgpa"], reverse=True)
    for rank, st in enumerate(students, 1):
        st["rank"] = rank
        
    return students

def build_pdf(students, pdf_filename="college_exam_results.pdf"):
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=1,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        alignment=1,
        textColor=colors.HexColor("#2B6CB0"),
        spaceAfter=10
    )
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        alignment=1,
        textColor=colors.HexColor("#4A5568"),
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2D3748"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2D3748")
    )
    body_bold = ParagraphStyle(
        'BodyCustomBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    elements = []
    
    # ------------------ COVER / PREAMBLE PAGE ------------------
    elements.append(Paragraph("APEX INSTITUTE OF TECHNOLOGY & ADVANCED STUDIES", title_style))
    elements.append(Paragraph("OFFICE OF THE CONTROLLER OF EXAMINATIONS | AUTONOMOUS ACADEMIC GAZETTE", subtitle_style))
    elements.append(Paragraph("Official Cumulative Transcript & Grade Record: B.Tech Computer Science & Engineering (Batch 2021-2025)", meta_style))
    elements.append(Spacer(1, 10))
    
    # Policy and Grading System
    elements.append(Paragraph("Academic Regulation & Grading Scheme Policy", h2_style))
    policy_text = (
        "This official academic gazette details the comprehensive 8-semester examination results for 200 undergraduate "
        "candidates registered under the Computer Science & Engineering program (2021-2025). "
        "The curriculum comprises 170.0 total cumulative credits spanning core theoretical foundations, engineering mathematics, "
        "advanced computational systems, laboratory practices, and capstone engineering projects."
    )
    elements.append(Paragraph(policy_text, body_style))
    elements.append(Spacer(1, 8))
    
    grading_data = [
        ["Grade", "Grade Point", "Marks Range (%)", "Description", "Award Criteria"],
        ["O", "10", "90 - 100", "Outstanding", "Distinction in subject mastery"],
        ["A+", "9", "80 - 89", "Excellent", "Superior academic performance"],
        ["A", "8", "70 - 79", "Very Good", "Above average competency"],
        ["B+", "7", "60 - 69", "Good", "Sound standard competency"],
        ["B", "6", "50 - 59", "Above Average", "Satisfactory completion"],
        ["C", "5", "40 - 49", "Average", "Pass threshold criteria"],
        ["F", "0", "< 40", "Fail", "Course Re-examination Required (Backlog)"],
    ]
    grade_table = Table(grading_data, colWidths=[50, 70, 90, 110, 220])
    grade_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(grade_table)
    elements.append(Spacer(1, 14))
    
    # Batch Statistics Overview
    avg_cgpa = round(sum(s["cgpa"] for s in students) / len(students), 2)
    top_cgpa = max(s["cgpa"] for s in students)
    distinction_cnt = sum(1 for s in students if s["cgpa"] >= 8.5)
    first_class_cnt = sum(1 for s in students if 6.75 <= s["cgpa"] < 8.5)
    backlog_cnt = sum(1 for s in students if s["backlogs"] > 0)
    
    elements.append(Paragraph("Batch Performance & Metric Highlights (200 Students)", h2_style))
    summary_data = [
        ["Total Enrolled Candidates", "200 Students"],
        ["Program & Specialization", "B.Tech Computer Science & Engineering (CSE)"],
        ["Batch Academic Cycle", "2021 - 2025 (Semesters 1 through 8)"],
        ["Overall Batch Average CGPA", f"{avg_cgpa:.2f} / 10.0"],
        ["Highest CGPA Achieved", f"{top_cgpa:.2f} (Rank 1: {students[0]['name']} - {students[0]['usn']})"],
        ["Distinction Achievers (CGPA >= 8.5)", f"{distinction_cnt} Candidates ({distinction_cnt/len(students)*100:.1f}%)"],
        ["First Class Achievers (6.75 <= CGPA < 8.5)", f"{first_class_cnt} Candidates ({first_class_cnt/len(students)*100:.1f}%)"],
        ["Candidates with Active Backlogs", f"{backlog_cnt} Candidates"],
    ]
    summary_table = Table(summary_data, colWidths=[240, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EDF2F7")),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 14))
    
    # Top 10 Merit List
    elements.append(Paragraph("Merit Honor Roll: Top 10 Graduating CSE Engineers", h2_style))
    merit_rows = [["Rank", "USN", "Candidate Name", "Sem 1-4 SGPA", "Sem 5-8 SGPA", "Cumulative CGPA", "Standing"]]
    for s in students[:10]:
        s1_4 = round(sum(s["semesters"][str(sm)]["sgpa"] for sm in range(1, 5)) / 4, 2)
        s5_8 = round(sum(s["semesters"][str(sm)]["sgpa"] for sm in range(5, 9)) / 4, 2)
        merit_rows.append([
            f"#{s['rank']}",
            s["usn"],
            s["name"],
            f"{s1_4:.2f}",
            f"{s5_8:.2f}",
            f"{s['cgpa']:.2f}",
            s["standing"]
        ])
    merit_table = Table(merit_rows, colWidths=[40, 80, 140, 75, 75, 70, 60])
    merit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A365D")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    elements.append(merit_table)
    elements.append(PageBreak())
    
    # ------------------ DETAILED STUDENT GRADE TRANSCRIPTS ------------------
    # For every student, create a structured student transcript entry.
    # Group students cleanly so page breaks are neat and documents retain clear citations.
    for idx, s in enumerate(students):
        student_elements = []
        
        # Student Header
        header_text = (
            f"<b>USN:</b> {s['usn']} | <b>Name:</b> {s['name']} | "
            f"<b>Program:</b> {s['branch']} | <b>Batch:</b> {s['batch']} | "
            f"<b>Rank:</b> #{s['rank']} | <b>CGPA:</b> {s['cgpa']:.2f} | <b>Status:</b> {s['standing']}"
        )
        student_elements.append(Paragraph(header_text, ParagraphStyle(
            'StHeader',
            parent=body_style,
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#1A365D"),
            backColor=colors.HexColor("#E2E8F0"),
            borderPadding=4,
            spaceAfter=6
        )))
        
        # Table of 8 Semesters SGPA & Subject Highlights
        sem_rows = [["Sem", "Semester Name", "SGPA", "Credits", "Key Subjects & Scores (Marks/100, Grade)"]]
        for sem_num in range(1, 9):
            s_data = s["semesters"][str(sem_num)]
            # Format subject list cleanly
            subj_strs = [f"{sub['code']} {sub['name']}: {sub['marks']}({sub['grade']})" for sub in s_data["subjects"]]
            subj_compact = ", ".join(subj_strs)
            sem_rows.append([
                f"S{sem_num}",
                s_data["semester_name"],
                f"{s_data['sgpa']:.2f}",
                f"{s_data['credits']:.1f}",
                Paragraph(subj_compact, ParagraphStyle('SubjStyle', parent=body_style, fontSize=7, leading=9))
            ])
            
        sem_table = Table(sem_rows, colWidths=[28, 85, 42, 45, 340])
        sem_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2D3748")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7.5),
            ('ALIGN', (0, 0), (3, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        student_elements.append(sem_table)
        student_elements.append(Spacer(1, 8))
        
        # Keep each student's transcript together so it doesn't break awkwardly across pages
        elements.append(KeepTogether(student_elements))
        
        # Add a page break every 2 students for clean layout
        if (idx + 1) % 2 == 0 and (idx + 1) < len(students):
            elements.append(PageBreak())
            
    doc.build(elements)
    print(f"PDF Successfully Generated: {pdf_filename} ({len(students)} students)")

if __name__ == "__main__":
    print("Generating comprehensive college examination records for 200 CSE students...")
    students = generate_dataset(200)
    
    # Save structured json for instantaneous exact calculations
    with open("students_results.json", "w", encoding="utf-8") as f:
        json.dump(students, f, indent=2)
    print("Saved students_results.json")
    
    # Build official PDF
    build_pdf(students, "college_exam_results.pdf")
    print("Dataset & PDF Generation Complete!")
