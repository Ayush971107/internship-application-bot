import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize the GPT-4 model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Create a prompt template for ATS keywords extraction
ats_prompt = ChatPromptTemplate.from_template(
    """You are an expert ATS (Applicant Tracking System) analyst. Given the following job description, identify and list the top 5 most important technical terms or skills that should be present in a resume for a good ATS match. Only provide the list of 5 terms, nothing else.

Job Description:
{job_description}

Top 5 Technical Terms:"""
)

# Create an LLMChain for ATS keywords extraction
ats_chain = LLMChain(llm=llm, prompt=ats_prompt)

# Function to process job description and get top 5 terms
def get_top_5_terms(job_description):
    result = ats_chain.invoke({"job_description": job_description})
    if isinstance(result, dict):
        output_text = result.get("text", "")
    else:
        output_text = result

    return output_text.strip().split("\n")

# Create a prompt template for cover letter modification
cover_letter_prompt = ChatPromptTemplate.from_template(
    """You are an expert in crafting professional cover letters. Given the job description and the existing cover letter below, modify the cover letter to align with the job description while maintaining the original flow and tone of the letter. 
    Only make changes that improve relevance to the new job role. 
    The first paragraph's first couple lines should be changed to show how my passion resonates with the company and what excites me, express it creatively and share genuine enthusiasm and passion, followed by my experience at Humana.
    The 2nd paragraph must remain unchanged always. The third paragraph should be completely changed. Start with stating why I would like to work for the company and what opportunity will I take/ would align with my career goal. The last line should be why I would be an asset to them.
    
Job Description:
{job_description}

Original Cover Letter:
{cover_letter}

Modified Cover Letter:"""
)

# Create an LLMChain for cover letter modification
cover_letter_chain = LLMChain(llm=llm, prompt=cover_letter_prompt)

# Function to modify the cover letter based on the job description
def modify_cover_letter(job_description, cover_letter):
    result = cover_letter_chain.invoke({
        "job_description": job_description,
        "cover_letter": cover_letter
    })
    # Assuming the result is directly the modified cover letter
    modified_cover_letter = result.get("text", "") if isinstance(result, dict) else result
    return modified_cover_letter.strip()

# Create a prompt template for visa sponsorship detection
visa_sponsorship_prompt = ChatPromptTemplate.from_template(
    """You are an expert in analyzing job descriptions. Given the job description below, determine if the company offers visa sponsorship for this position. 
    If the job description mentions that sponsorship is available, or that the company sponsors visas, state "This job offers visa sponsorship."
    If the job description mentions that only citizens, permanent residents, or those with security clearance can apply, or if there is any indication that sponsorship is not available, state "This job does not sponsor work visas."
    If the job description does not mention anything about visa sponsorship, state "No mention of sponsorship."

Job Description:
{job_description}

Visa Sponsorship Information:"""
)

# Create an LLMChain for visa sponsorship detection
visa_sponsorship_chain = LLMChain(llm=llm, prompt=visa_sponsorship_prompt)

# Function to check visa sponsorship availability using natural language understanding
def check_visa_sponsorship(job_description):
    result = visa_sponsorship_chain.invoke({"job_description": job_description})
    sponsorship_info = result.get("text", "") if isinstance(result, dict) else result
    return sponsorship_info.strip()

# Function to apply further feedback to the cover letter
def apply_feedback_to_cover_letter(modified_cover_letter, feedback):
    feedback_prompt_template = ChatPromptTemplate.from_template(
        """You are a skilled cover letter editor. Given the modified cover letter below, apply the following feedback to make further improvements while maintaining the overall flow and tone.

        Modified Cover Letter:
        {modified_cover_letter}

        Feedback:
        {feedback}

        Updated Cover Letter:"""
    )
    feedback_chain = LLMChain(llm=llm, prompt=feedback_prompt_template)
    result = feedback_chain.invoke({
        "modified_cover_letter": modified_cover_letter,
        "feedback": feedback
    })
    updated_cover_letter = result.get("text", "") if isinstance(result, dict) else result
    return updated_cover_letter.strip()

# Function to perform skill gap analysis
def perform_skill_gap_analysis(job_description, resume):
    skill_gap_prompt_template = ChatPromptTemplate.from_template(
        """You are an expert in skill gap analysis. Given the job description and the resume below, identify 3-5 potential skill gaps between the resume and the job description. 
        For each skill gap, provide a brief explanation and suggest ways to address the gap, such as acquiring new skills, gaining experience, or highlighting related skills that may not be explicitly mentioned in the resume.

        Job Description:
        {job_description}

        Resume:
        {resume}

        Skill Gap Analysis:"""
    )
    skill_gap_chain = LLMChain(llm=llm, prompt=skill_gap_prompt_template)
    result = skill_gap_chain.invoke({
        "job_description": job_description,
        "resume": resume
    })
    skill_gap_analysis = result.get("text", "") if isinstance(result, dict) else result
    return skill_gap_analysis.strip()

# Hardcoded resume in LaTeX
resume_latex = r"""
%-------------------------
% Resume in Latex
% Author : Ayush Roy
% License : MIT
%------------------------

\documentclass[letterpaper,11pt]{article}

\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
\input{glyphtounicode}


%----------FONT OPTIONS----------
% Uncomment one of the following lines for sans-serif fonts
%\usepackage[sfdefault]{FiraSans}
%\usepackage[sfdefault]{roboto}
%\usepackage[sfdefault]{noto-sans}
%\usepackage[default]{sourcesanspro}

% Uncomment for serif fonts
%\usepackage{CormorantGaramond}
%\usepackage{charter}

\pagestyle{fancy}
\fancyhf{} % clear all header and footer fields
\fancyfoot{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

% Adjust margins
\addtolength{\oddsidemargin}{-0.5in}
\addtolength{\evensidemargin}{-0.5in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.5in}
\addtolength{\textheight}{1.0in}

\urlstyle{same}

\raggedbottom
\raggedright
\setlength{\tabcolsep}{0in}

% Sections formatting
\titleformat{\section}{
  \vspace{-4pt}\scshape\raggedright\large
}{}{0em}{}[\color{black}\titlerule \vspace{-5pt}]

% Ensure that generated PDF is machine readable/ATS parsable
\pdfgentounicode=1

%-------------------------
% Custom commands
\newcommand{\resumeItem}[1]{
  \item\small{
    {#1 \vspace{-2pt}}
  }
}

\newcommand{\resumeSubheading}[4]{
  \vspace{-2pt}\item
    \begin{tabular*}{0.97\textwidth}[t]{l@{\extracolsep{\fill}}r}
      \textbf{#1} & #2 \\
      \textit{\small#3} & \textit{\small #4} \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubSubheading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \textit{\small#1} & \textit{\small #2} \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeProjectHeading}[2]{
    \item
    \begin{tabular*}{0.97\textwidth}{l@{\extracolsep{\fill}}r}
      \small#1 & #2 \\
    \end{tabular*}\vspace{-7pt}
}

\newcommand{\resumeSubItem}[1]{\resumeItem{#1}\vspace{-4pt}}

\renewcommand\labelitemii{$\vcenter{\hbox{\tiny$\bullet$}}$}

\newcommand{\resumeSubHeadingListStart}{\begin{itemize}[leftmargin=0.15in, label={}]}
\newcommand{\resumeSubHeadingListEnd}{\end{itemize}}
\newcommand{\resumeItemListStart}{\begin{itemize}}
\newcommand{\resumeItemListEnd}{\end{itemize}\vspace{-5pt}}

%-------------------------------------------
%%%%%%  RESUME STARTS HERE  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{document}
%----------HEADING----------
\begin{center}
    \textbf{\Huge \scshape Ayush Roy} \\ \vspace{1pt}
    \small (240) 491 7045 $|$ \href{mailto:aroy9711@umd.edu}{\underline{aroy9711@umd.edu}} $|$ 
    \href{https://linkedin.com/in/ayushroy9711}{\underline{linkedin.com/in/ayushroy9711}} $|$ \href{https://github.com/Ayush971107}{\underline{github.com/Ayush971107}}
\end{center}
%-----------EDUCATION-----------
\section{Education}
  \resumeSubHeadingListStart
    \resumeSubheading
      {University of Maryland }{College Park, MD}
      {Bachelor of Science in \textbf{Computer Science}, Minor in \textbf{Statistics}}{Expected Graduation: May 2026}
    \resumeItemListStart
        \resumeItem{Organizations: Google Developer Student Club, Big Th!nk AI}
        \resumeItem{Relevant Coursework: Object-Oriented Programming I and II, Applied Probability and Statistics, Linear Algebra}
    \resumeItemListEnd
  \resumeSubHeadingListEnd

%-----------EXPERIENCE-----------
\section{Experience}
  \resumeSubHeadingListStart

    \resumeSubheading
    {Humana }{May 2024 -- Aug 2024}
    {Software Engineering Intern}{Philadelphia, PA}

    \resumeItemListStart
        \resumeItem{
Developed and implemented internal server applications using \textbf{Angular} for frontend and \textbf{C\# in .NET framework} for backend.}
        \resumeItem{Enhanced code quality metrics by \textbf{20\%} through best practices and regular code audits.}
        \resumeItem{Leveraged \textbf{Qlik} for data visualization, designing interactive dashboards and analytics applications, and conducted \textbf{quality testing for 10+} applications, increasing data accuracy by \textbf{30\%}.}
        \resumeItem{Engineered and optimized \textbf{SQL database} solutions, reducing query time and achieving a \textbf{35\% increase in processing speed} for high-volume datasets.}
    \resumeItemListEnd

    \resumeSubheading
      {Sansar Tec }{Sep 2022 -- Oct 2022}
      {Software Development Intern}{Jersey City, NJ}

      \resumeItemListStart
        \resumeItem{Deployed an \textbf{automated system} using \textbf{Google Apps Script} for email attachment segregation, categorizing incoming emails and routing them to designated Google Drive folders, saving \textbf{15 hours} per week.}
        \resumeItem{Rigorously tested the  program using \textbf{unit tests} through multiple rounds of real-world attachment processing.}
      \resumeItemListEnd
%     \resumeSubheading
%       {Birlasoft }{Nov 2022 -- Dec 2022}
%       {Product Research Intern }{Raleigh, NC}

%       \resumeItemListStart
%         \resumeItem{Conducted comprehensive research to evaluate the technological feasibility of an In-Car payment system}
%         \resumeItem{Formulated a comprehensive set of business use cases and implementation strategies for the In-Car payment system, outlining practical scenarios and deployment roadmaps 
% }
%         \resumeItem{Devised advanced data security strategies, including tokenization and blockchain, bolstering encryption and authentication to protect sensitive user data proposing a 40\% reduction in security breaches}
%     \resumeItemListEnd

  \resumeSubHeadingListEnd

%-----------PROJECTS-----------
\section{Projects}
  \resumeSubHeadingListStart
      \resumeProjectHeading{\textbf{Face Detection and Age Classification} \emph{}}{Aug 2024}
      \resumeItemListStart
        \resumeItem{Developed an age prediction tool using \textbf{TensorFlow} to train a \textbf{ResNet-9 CNN model}, achieving \textbf{85\% accuracy}, processing over \textbf{100,000} photos.}
        \resumeItem{Implemented feature extraction, individual face detection in group photo and data augmentation increasing dataset size by \textbf{10X} with \textbf{CV2}.}
        \resumeItem{Built a user-friendly interface with \textbf{Streamlit} for seamless photo uploads and real-time age estimation.}
      \resumeItemListEnd

      \resumeProjectHeading{\textbf{FraudGuardPro - Fraudulent Insurance Claims Detection System (Hacklytics, Georgia Tech)} \emph{}}{Feb 2024}
      \resumeItemListStart
        \resumeItem{Led the development of a comprehensive fraudulent insurance claims detection system, implementing \textbf{decision trees} and \textbf{random forests} using \textbf{Scikit-learn} achieving a \textbf{95\% fraud detection rate} and reducing false positives by \textbf{20\%.}}
        \resumeItem{Employed \textbf{Pandas} and \textbf{NumPy} for data preprocessing, processing datasets with over \textbf{100,000 claims}, and \textbf{Pickle} for seamless model loading.}
        \resumeItem{Integrated \textbf{Langchain} for \textbf{SQL} query execution and implementation via natural language prompts.}
      \resumeItemListEnd

      \resumeProjectHeading{\textbf{Yousure? - AI-Powered Insurance Insights} \emph{}}{March 2024}
      \resumeItemListStart
        \resumeItem{Developed with\textbf{ React} for a user-friendly interface, enabling policy summarization, file uploads and policy comparison. Utilized \textbf{Flask} for API development and user authentication.}
        \resumeItem{Implemented an \textbf{XGBoost} model for user risk prediction, achieving \textbf{92\%} accuracy through automated hyperparameter tuning and ensemble techniques,  improving prediction reliability by \textbf{25\%}.}
        \resumeItem{Used \textbf{Pinecone} for efficient PDF data chunk storage, integrated \textbf{LangChain with RAG }and used\textbf{ GPT-3.5 Turbo} to deliver precise policy insights and qna abilities.}
      \resumeItemListEnd
  \resumeSubHeadingListEnd

%-----------TECHNICAL SKILLS-----------
\section{Technical Skills}
  \begin{itemize}[leftmargin=0.15in, label={}]
    \small{\item{
     \textbf{Languages}{: Python, Java, JavaScript, SQL, C\#, HTML/CSS} \\
     \textbf{Full Stack}{: Angular, React, Node.js, MongoDB, SSIS, .NET, Flask, FastAPI, RestAPIs} \\
     \textbf{ML Stack}{: PyTorch, TensorFlow, Scikit-learn, Langchain, Matplotlib, NumPy, Pandas, OpenCV} \\
     \textbf{Other Tools/Platforms}{: Linux, Docker, Sagemaker, Git, CI/CD, Azure, GCP, SAS} \\
     \textbf{Concepts}{: Object-Oriented Programming, Deep Learning, Machine Learning, Data Structures, Algorithms, Exploratory Data Analysis, Statistical Methods, Data Science, A/B Testing}
    }}
  \end{itemize}


    % %  \item \textbf{Availability}: 22nd May 2025 to 31st August 2025
    % % }}
  \end{itemize}

% %-----------AVAILABILITY-----------
% \section{Availability}
%   \begin{itemize}[leftmargin=0.15in, label={}]
%     \small{\item \textbf{Availability}: 22nd May 2025 to 31st August 2025}
  % \end{itemize}

\end{document}

"""

# Backend cover letter (hardcoded)
cover_letter = f"""Builders FirstSource's commitment to transforming the future of home building resonates deeply with my passion for using data to drive impactful, business-centric solutions. The opportunity to join your Data Management & Stewardship Summer Internship program excites me, as it offers the chance to contribute directly to critical transformation initiatives within a leading company. As a software engineer at Humana, I utilized Angular for the frontend and the .NET framework for the backend to improve code quality metrics by 20% and engineered optimized SQL database solutions that increased processing speeds by 35% for high-volume datasets.
Beyond software development, I have a strong interest in deep learning and have completed multiple projects in this field. For instance, I developed an age prediction tool using TensorFlow and a ResNet-9 CNN model, achieving 85% accuracy in estimating ages from group photos. I also led the creation of FraudGuardPro, a system for detecting fraudulent insurance claims using decision trees and random forests with Scikit-learn. Participating in various hackathons and collaborating with diverse teams has honed my ability to thrive in dynamic, fast-paced environments.
I am particularly excited about the opportunity to collaborate with seasoned professionals in a dynamic, growth-oriented environment. Builders FirstSource’s focus on continuous learning and innovation aligns perfectly with my career aspirations. I am confident that my strong analytical skills, coupled with my dedication to data-driven decision-making, will allow me to be an asset to your team.
"""

# Streamlit App
def main():
    st.title("Job Application Assistant")
    st.write("Paste the job description below and generate the top 5 ATS keywords and a modified cover letter.")

    # Input field for the job description
    job_description = st.text_area("Job Description", height=300)

    # Initialize session state for modified cover letter
    if "modified_cover_letter" not in st.session_state:
        st.session_state.modified_cover_letter = ""

    if st.button("Generate"):
        if job_description:
            # Generate top 5 ATS terms
            top_terms = get_top_5_terms(job_description)
            st.subheader("Top 5 Technical Terms for ATS Match:")
            for i, term in enumerate(top_terms, 1):
                st.write(f"{i}. {term}")

            # Modify the cover letter
            st.session_state.modified_cover_letter = modify_cover_letter(job_description, cover_letter)
            st.subheader("Modified Cover Letter:")
            st.write(st.session_state.modified_cover_letter)

            # Check for visa sponsorship availability using natural language prompt
            sponsorship_info = check_visa_sponsorship(job_description)
            st.subheader("Visa Sponsorship Information:")
            st.write(sponsorship_info)

            # Perform skill gap analysis
            skill_gap_analysis = perform_skill_gap_analysis(job_description, resume_latex)
            st.subheader("Skill Gap Analysis:")
            st.write(skill_gap_analysis)
        else:
            st.error("Please paste a job description.")

    # Check if a modified cover letter exists to display feedback options
    if st.session_state.modified_cover_letter:
        st.subheader("Provide Feedback for Further Modifications:")
        feedback = st.text_area("Feedback", height=100)

        if st.button("Apply Feedback"):
            if feedback:
                st.session_state.modified_cover_letter = apply_feedback_to_cover_letter(
                    st.session_state.modified_cover_letter, feedback
                )
                st.subheader("Updated Cover Letter:")
                st.write(st.session_state.modified_cover_letter)
            else:
                st.error("Please enter your feedback.")

if __name__ == "__main__":
    main()
