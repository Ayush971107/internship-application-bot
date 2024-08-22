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

# Backend cover letter (hardcoded)
cover_letter = f"""Builders FirstSource's commitment to transforming the future of home building resonates deeply with my passion for using data to drive impactful, business-centric solutions. The opportunity to join your Data Management & Stewardship Summer Internship program excites me, as it offers the chance to contribute directly to critical transformation initiatives within a leading company. As a full-stack and data visualization internship at Humana, I utilized Angular for the frontend and the .NET framework for the backend to improve code quality metrics by 20% and engineered optimized SQL database solutions that increased processing speeds by 35% for high-volume datasets.
Beyond software development, I have a strong interest in deep learning and have completed multiple projects in this field. For instance, I developed an age prediction tool using TensorFlow and a ResNet-9 CNN model, achieving 85% accuracy in estimating ages from group photos. I also led the creation of FraudGuardPro, a system for detecting fraudulent insurance claims using decision trees and random forests with Scikit-learn. Participating in various hackathons and collaborating with diverse teams has honed my ability to thrive in dynamic, fast-paced environments.
I am particularly excited about the opportunity to collaborate with seasoned professionals in a dynamic, growth-oriented environment. Builders FirstSource’s focus on continuous learning and innovation aligns perfectly with my career aspirations. I am confident that my strong analytical skills, coupled with my dedication to data-driven decision-making, will allow me to be an asset to your team.
"""
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

