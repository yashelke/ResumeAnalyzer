import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import spacy
from spacy.matcher import PhraseMatcher
from sentence_transformers import SentenceTransformer, util

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None

# Load BERT model for semantic similarity
@st.cache_resource
def load_bert_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Download necessary NLTK resources (only if not already present)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

# Load stopwords once globally
stop_words = set(stopwords.words("english"))

# Skills database for spaCy PhraseMatcher
skills_db = [
    "python", "java", "javascript", "typescript", "csharp", "c++", "c#", "rust",
    "go", "kotlin", "swift", "php", "ruby", "scala", "groovy", "sql", "r",
    "react", "angular", "vue", "vuejs", "django", "flask", "fastapi", "spring", "spring boot",
    ".net", "dotnet", "aws", "amazon web services", "azure", "gcp", "google cloud",
    "docker", "kubernetes", "git", "jenkins", "gitlab", "github", "terraform", "ansible",
    "vagrant", "linux", "windows", "macos", "ios", "android", "html", "html5", "css", "scss", "sass",
    "webpack", "npm", "yarn", "pip", "maven", "gradle", "mysql", "postgresql", "mongodb",
    "redis", "elasticsearch", "graphql", "rest api", "restful", "soap", "microservices", "api",
    "nodejs", "node.js", "express", "fastify", "nest", "nestjs", "rails", "ruby on rails", "laravel",
    "golang", "rust", "elixir", "haskell", "clojure", "perl", "bash", "shell scripting", "powershell",
    "svn", "perforce", "jira", "confluence", "slack", "tableau", "power bi", "looker",
    "splunk", "datadog", "prometheus", "grafana", "junit", "pytest", "mockito", "mocha", "jest",
    "cypress", "selenium", "postman", "insomnia", "vim", "vscode", "intellij", "sublime",
    "kafka", "rabbitmq", "activemq", "nats", "memcached", "solr", "lucene", "sphinx",
    "xml", "json", "yaml", "toml", "machine learning", "deep learning", "ml", "ai", "nlp",
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "pandas", "numpy", "scipy",
    "matplotlib", "seaborn", "plotly", "jupyter", "anaconda", "miniconda", "airflow", "dbt",
    "apache spark", "spark", "hadoop", "hive", "pig", "flume", "sqoop", "hbase",
    "cassandra", "dynamodb", "s3", "snowflake", "bigquery", "redshift",
    "data analysis", "data science", "data engineering", "etl", "data pipeline",
    "ci/cd", "devops", "cloud computing", "virtualization", "containerization",
    "api development", "web development", "mobile development", "frontend", "backend",
    "full stack", "database design", "system design", "software architecture"
]

# Initialize spaCy PhraseMatcher for skill extraction
if nlp:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills_db]
    matcher.add("SKILLS", patterns)
else:
    matcher = None




# Page setup

st.set_page_config(page_title="Resume Job Match Scorer", page_icon=":briefcase:", layout="wide")

# st.title("Hello")

st.markdown(
    """
    Upload your resume and paste a job description to see how well they match!\n
    Try it out and see how your resume stacks up against your **dream job!** 
    """
)

with st.sidebar:
    st.title("Resume Job Match Scorer") 
    
    st.header("About")
    st.info("""
            This tool helps you:-
            - Measures how your resume matches a job description.
            - Identify important job keywords.
            - Improve your resume based on missing terms.
            
            
            """
            )
    
    # st.markdown("<h2><i>How It Works</i></h2>",unsafe_allow_html=True)
    st.header("How It Works")
    st.write(""" 
            1. Upload your resume in PDF format.
            2. Paste the job description in the text box.
            3. Click **Analyze Match**.
            4. Review Score and Suggestions. 
             """)
    
    

# helper function


def extract_text_from_pdf(uploaded_file):
    
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    

def clean_text(text):
    
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_bert_similarity(resume_text, job_description):
    """Calculate semantic similarity using BERT"""
    model = load_bert_model()
    
    emb1 = model.encode(resume_text, convert_to_tensor=True)
    emb2 = model.encode(job_description, convert_to_tensor=True)
    
    score = util.cos_sim(emb1, emb2).item() * 100
    
    return round(score, 2)

def extract_sections(text):
    """Extract resume sections using regex patterns"""
    text_lower = text.lower()
    
    sections = {
        "skills": "",
        "experience": "",
        "projects": "",
        "education": ""
    }
    
    patterns = {
        "skills": r"(skills|technical skills|technical expertise)(.*?)(experience|projects|education|summary|$)",
        "experience": r"(experience|work experience|employment|professional experience)(.*?)(skills|projects|education|$)",
        "projects": r"(projects|personal projects|portfolio)(.*?)(experience|skills|education|$)",
        "education": r"(education|degree|certification)(.*?)(experience|skills|projects|$)"
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match and len(match.groups()) > 1:
            sections[key] = match.group(2).strip()
    
    return sections

def section_similarity(resume_sections, job_description):
    """Calculate similarity scores for each section"""
    scores = {}
    
    for section, content in resume_sections.items():
        if content.strip() and len(content) > 10:
            scores[section] = calculate_bert_similarity(content, job_description)
        else:
            scores[section] = 0
    
    return scores

def calculate_skill_score(resume_skills, job_skills):
    """Calculate skill match percentage"""
    if not job_skills:
        return 0
    
    matched = len(resume_skills & job_skills)
    total = len(job_skills)
    
    return (matched / total) * 100

def calculate_final_score(bert_score, skill_score, section_scores):
    """Calculate final ATS score combining all metrics"""
    
    final_score = (
        0.4 * bert_score +                              # semantic similarity
        0.3 * skill_score +                             # skill match
        0.2 * section_scores.get("experience", 0) + 
        0.1 * section_scores.get("projects", 0)
    )
    
    return round(final_score, 2)


# example

# tfidf_matix = [
#     [0.1, 0.2, 0.3], resume_processed (row 0)
#     [0.2, 0.1, 0.4]  job_processed (row 1)
# ]


def extract_keywords(text, num_keywords=10):
    words =word_tokenize(text)
        
    words = [w for w in words if len(w) > 2]
    tagged_words = pos_tag(words)
    nouns = [w for w, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
    
    word_freq = Counter(nouns)
    
    return word_freq.most_common(num_keywords)


def extract_skills_spacy(text):
    """Extract skills from text using spaCy PhraseMatcher"""
    if not nlp or not matcher:
        return set()
    
    doc = nlp(text.lower())
    matches = matcher(doc)
    
    found_skills = set()
    
    for match_id, start, end in matches:
        skill = doc[start:end].text.lower()
        found_skills.add(skill)
    
    return found_skills


# Main app


def main():
    st.header("Your Resume")
    uploaded_file = st.file_uploader("Upload your resume(.pdf) :- ",type=["pdf"])
    
    # job_description = st.text_area("Paste the job description here:- ", height=200)
    
    # the users can either upload the jd .pdf file or paste the jd in the text area
    
    st.subheader("Job Description")
    col1 , col2 = st.columns(2)
    
    with col1:
        job_description_file = st.file_uploader("Upload the job description(.pdf) :- ", type=["pdf"])
        
    with col2:
        job_description_text = st.text_area("Paste the job description here:- ", height=200)
    
    job_description = ""
    if job_description_file:
        job_description = extract_text_from_pdf(job_description_file)
        if job_description_text:
            st.info("ℹ️ Using job description from PDF file (text area input ignored).")
    else:
        job_description = job_description_text
    
    if st.button("Analyze Match", key="analyze_match"):
        # Validate inputs
        if not uploaded_file:
            st.warning("Please upload your resume in PDF format.")
        elif not job_description:
            st.warning("Please paste the job description in the text area or upload it as a PDF.")
        else:
            # Process the analysis
            with st.spinner("🔄 Analyzing your resume with AI..."):
                resume_text = extract_text_from_pdf(uploaded_file)
                if not resume_text:
                    st.error("Failed to extract text from the PDF. Please check the file and try again.")
                    return
                
                # 1. Calculate BERT semantic similarity
                bert_score = calculate_bert_similarity(resume_text, job_description)
                
                # 2. Extract resume sections
                resume_sections = extract_sections(resume_text)
                
                # 3. Calculate section-wise similarity
                section_scores = section_similarity(resume_sections, job_description)
                
                # 4. Extract skills
                resume_skills = extract_skills_spacy(resume_text)
                job_skills = extract_skills_spacy(job_description)
                
                # 5. Calculate skill match score
                skill_score = calculate_skill_score(resume_skills, job_skills)
                
                # 6. Calculate final ATS score
                final_score = calculate_final_score(bert_score, skill_score, section_scores)
                
                # Display results
                st.subheader("📊 ATS Analysis Results")
                
                # Main score display
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Final ATS Score", f"{final_score}%")
                with col2:
                    st.metric("🧠 Semantic Match", f"{bert_score}%")
                with col3:
                    st.metric("💼 Skill Match", f"{skill_score:.1f}%")
                with col4:
                    st.metric("📍 Experience Match", f"{section_scores.get('experience', 0):.1f}%")
                
                # Gauge chart for final score
                fig, ax = plt.subplots(figsize=(10, 2))
                
                colors=["#ff4b4b","#ffa726","#0f9d58"]
                
                color_index = min (int(final_score // 33), 2)
                
                ax.barh([0], [final_score], color=colors[color_index], height=0.5)
                
                ax.set_xlim(0, 100)
                
                ax.set_xlabel("Match Score (%)", fontsize=12)
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.set_yticks([])
                ax.set_title("Resume-Job Match", fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Score feedback
                if final_score < 40:
                    st.warning("Your resume has a low match score. Consider adding more relevant keywords and skills from the job description.", icon="⚠️")
                elif final_score < 70:
                    st.info("Your resume has a moderate match score. You can improve it by including more specific terms from the job description.", icon="ℹ️")
                else:
                    st.success("Great job! Your resume has a high match score. It closely aligns with the job description.", icon="✅")
                
                # Detailed Score Breakdown
                st.subheader("📈 Detailed Score Breakdown")
                
                breakdown_col1, breakdown_col2 = st.columns(2)
                
                with breakdown_col1:
                    st.write("**Score Components:**")
                    st.write(f"• BERT Semantic Similarity: **{bert_score}%**")
                    st.write(f"• Skill Match: **{skill_score:.1f}%**")
                    st.write(f"• Experience Match: **{section_scores.get('experience', 0):.1f}%**")
                    st.write(f"• Projects Match: **{section_scores.get('projects', 0):.1f}%**")
                
                with breakdown_col2:
                    st.write("**Weighting in ATS Score:**")
                    st.write("• Semantic Similarity: 40%")
                    st.write("• Skill Match: 30%")
                    st.write("• Experience: 20%")
                    st.write("• Projects: 10%")
                
                # Display matched skills
                st.subheader("✅ Matched Skills")
                matched_skills = resume_skills & job_skills
                if matched_skills:
                    matched_list = sorted(list(matched_skills))
                    st.success(f"**{len(matched_skills)} Skills Found** in your resume")
                    cols = st.columns(4)
                    for idx, skill in enumerate(matched_list):
                        with cols[idx % 4]:
                            st.write(f"✔️ {skill}")
                else:
                    st.warning("No matching skills found between resume and job description.")
                
                # Display missing skills
                st.subheader("💼 Missing Skills")
                st.write("These technical skills are in the job description but not found in your resume. Consider adding them to improve your match score:")
                
                missing_skills = job_skills - resume_skills
                
                if missing_skills:
                    missing_list = sorted(list(missing_skills))[:20]
                    st.error(f"**{len(missing_skills)} Technical Skills Missing**")
                    
                    # Display in a nice format with columns
                    cols = st.columns(4)
                    for idx, skill in enumerate(missing_list):
                        with cols[idx % 4]:
                            st.write(f"🔸 {skill}")
                else:
                    st.success("✅ Great! Your resume contains all the technical skills from the job description!")
            
     
if __name__ == "__main__":
    main()