# VisionTrackAI

## Intelligent AI Candidate Ranking System

VisionTrackAI is a rule-based AI candidate ranking engine developed for the Redrob AI Hiring Challenge. It analyzes candidate profiles, evaluates them against the job description, detects suspicious profiles, generates explainable reasoning, and produces a ranked submission.csv.

---

## Team

Team Name: VisionTrackAI

Members:
- Supriya M
- Varshitha KS
- Dhanya KE
- Krishnendu GS

---

## Features

- Candidate data loading
- Structured profile scoring
- AI/ML skill matching
- Semantic similarity scoring
- Behavioral signal evaluation
- Job description matching
- Honeypot & keyword stuffing detection
- Candidate reasoning generation
- Final ranking
- Submission validation

---

## Project Structure

```
VisionTrackAI/
│
├── data/
│   ├── candidates.jsonl
│   ├── sample_candidates.json
│   ├── candidate_schema.json
│   └── job_description.docx
│
├── output/
│   └── submission.csv
│
├── src/
│   ├── behavioral_scoring.py
│   ├── data_loading.py
│   ├── disqualifiers.py
│   ├── honeypot.py
│   ├── jd_profile.py
│   ├── ranker.py
│   ├── reasoning.py
│   ├── semantic_similarity.py
│   ├── skill_matching.py
│   └── structured_scoring.py
│
├── rank.py
├── validate_submission.py
├── requirements.txt
├── submission_metadata.yaml
└── README.md
```

---

## Scoring Methodology

Each candidate is evaluated using multiple weighted signals:

- AI/ML skill match
- Semantic similarity with job description
- Years of experience
- Current designation
- Education relevance
- Behavioral signals
- Recruiter engagement
- Company quality
- Profile completeness
- Relocation preference
- Notice period

Penalty scores are applied for:

- Keyword stuffing
- Honeypot detection
- Suspicious profiles
- Missing information
- Low-quality candidate profiles

The final score is used to rank candidates.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

Generate ranked candidates:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./output/submission.csv
```

Validate the submission:

```bash
python validate_submission.py
```

---

## Output

The system generates:

- output/submission.csv
- Candidate ranking
- Candidate reasoning
- Validation-ready submission

---

## Technologies Used

- Python 3.10+
- Pandas
- NumPy
- scikit-learn
- python-docx

---

## Future Improvements

- LLM-based reasoning
- Resume embeddings
- Learning-to-rank model
- Interactive dashboard
- Better semantic matching

---

## License

Developed for the Redrob AI Hiring Challenge.