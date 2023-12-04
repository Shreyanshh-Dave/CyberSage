import streamlit as st
import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load the sentence transformer model for semantic similarity calculation
model = SentenceTransformer('all-mpnet-base-v2')

# Load the dataset containing potential interview questions and their answers
df = pd.read_csv('interviewQnA.csv')

# Standardize the 'domain' and 'difficulty' fields for consistent processing
df['domain'] = df['domain'].str.lower().str.strip()
df['difficulty'] = df['difficulty'].str.lower().str.strip()

def evaluate_answer(user_answer, correct_answer):
    """
    Function to evaluate the user's answer by comparing semantic similarity with the correct answer.
    A similarity score of 0.45 or above is considered as a correct answer.
    """
    user_embedding = model.encode([user_answer])
    correct_embedding = model.encode([correct_answer])
    similarity = cosine_similarity(user_embedding, correct_embedding)
    return similarity >= 0.45

def fetch_questions(role):
    """
    Function to fetch a set of questions based on the selected role.
    The questions are selected from different domains and difficulties based on the role.
    """
    if role.lower() == 'blue team':
        domains = ['cyber security fundamentals', 'computer networks', 'soc analyst', 'logical aptitude']
    elif role.lower() == 'red team':
        domains = ['ethical questions', 'penetration testing', 'cyber security fundamentals', 'computer networks']
    elif role.lower() == 'digital forensics':
        domains = ['cyber security fundamentals', 'digital forensics', 'compliance basics', 'incident response']
    
    # Define the distribution of question difficulties
    difficulties = ['easy']*3 + ['intermediate']*8 + ['difficult']*4
    
    selected_questions = []
    
    for difficulty in difficulties:
        domain = random.choice(domains)
        available_questions = df[(df['domain'] == domain) & (df['difficulty'] == difficulty)]
        
        if not available_questions.empty:
            question = available_questions.sample(1)
            selected_questions.append(question)
        else:
            print(f"No questions available for domain {domain} and difficulty {difficulty}")
    
    return selected_questions

def app():
    """
    Main function to define the Streamlit app.
    The app provides an interactive platform for users to practice interview questions based on their selected role.
    """
    
    st.set_page_config(page_title='CyberSage Interviewer', page_icon=":briefcase:", layout='wide')
    
    st.title("CyberSage: Your AI Interviewer")
    
    st.markdown("""
                Welcome to CyberSage, your AI-driven interview preparation tool for cybersecurity and IT professionals.
                Select your desired role from the dropdown menu and start   your interview.
                """)
    
    role = st.sidebar.selectbox("Select a role", ["Blue Team", "Red Team", "Digital Forensics"])
    
    # Check if role already exists in session state
    if "role" not in st.session_state or st.session_state.role != role:
        # If role has changed, refresh the questions
        st.session_state.role = role
        st.session_state.questions_answers = fetch_questions(role)
    
    for i, qa in enumerate(st.session_state.questions_answers, start=1):
        question = qa['question'].values[0]
        correct_answer = qa['answer'].values[0]
        
        st.markdown(f"**Question {i}:** {question}")
        
        user_answer_key = f"user_answer_{i}"
        
        # Check if user_answer_key already exists in session state
        if user_answer_key not in st.session_state:
            st.session_state[user_answer_key] = ""
        
        user_answer = st.text_input(f"Your answer to Question {i}:", value=st.session_state[user_answer_key])
        
        # Update session state with user's answer
        st.session_state[user_answer_key] = user_answer
        
        if user_answer:
            is_correct = evaluate_answer(user_answer, correct_answer)
            
            if is_correct:
                st.success("Correct!")
            else:
                st.error("Incorrect.")

if __name__ == '__main__':
    app()
