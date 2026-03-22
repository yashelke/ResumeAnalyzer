# 🎯 Resume Job Match Scorer

An AI-powered resume analyzer that uses BERT semantic similarity and skill matching to provide an ATS-style resume analysis against job descriptions.

## Features

- 📊 **ATS-Style Scoring**: 4-component scoring system mimicking real ATS systems
- 🧠 **BERT Semantic Analysis**: Understands context and meaning beyond keywords
- 💼 **Skill Extraction**: Uses spaCy PhraseMatcher to identify 100+ technical skills
- ✅ **Matched Skills Display**: Shows which skills you have
- 🔸 **Missing Skills**: Identifies skills you need to add
- 📈 **Detailed Breakdown**: Section-wise analysis (experience, projects, education)
- 📄 **PDF Support**: Upload resume and job description as PDFs or paste text

## How It Works

### Scoring Components (40-30-20-10 weighting):
1. **BERT Semantic Similarity (40%)**: Uses transformer models to understand meaning
2. **Skill Match (30%)**: Matches technical skills between resume and job description
3. **Experience Match (20%)**: Analyzes work experience content relevance
4. **Projects Match (10%)**: Evaluates project descriptions

## Installation

### Local Setup
```bash
git clone https://github.com/[your-username]/ResumeAnalyzer.git
cd ResumeAnalyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
streamlit run app.py
```



## Project Structure

```
ResumeAnalyzer/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── .streamlit/
│   ├── config.toml       # Streamlit configuration
│   └── secrets.toml      # Sensitive data (not tracked)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Requirements

- Python 3.8+
- 4GB RAM minimum (for BERT model)
- 2GB disk space

## Performance Tips

- BERT model downloads on first run (~400MB)
- Subsequent analyses are faster due to caching
- Use the cloud deployment for 24/7 availability

## License

MIT License - feel free to use for personal or commercial projects.

## Contributing

Feel free to submit issues and enhancement requests!

---

**Created with ❤️ using Streamlit & AI**
