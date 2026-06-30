"""
Explicit encoding of the Redrob 'Senior AI Engineer - Founding Team' JD.

This module exists so the JD's qualitative reasoning ("we don't want
title-chasers", "framework enthusiasts are fine but not what we need") is
written down as concrete, auditable rules instead of being implicit in a
black-box model. Every constant here can be pointed back to a specific
sentence in job_description.md - that traceability is what we want to be
able to defend in the Stage 5 interview.

NOTE: this file encodes ONE job description. If Redrob swaps the JD for a
different role, only this file (plus maybe the consulting-firm list) should
need to change - the rest of the pipeline is JD-agnostic.
"""

from __future__ import annotations
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Core JD text - kept here so the embedding step has clean text to encode.
# Trimmed to the substantive requirements; drops the "vibe check" / culture
# prose, which doesn't have a profile-side signal to match against.
# ---------------------------------------------------------------------------
JD_TEXT = """
Senior AI Engineer, Founding Team, at Redrob AI, a Series A AI-native talent
intelligence platform. Location Pune or Noida, India, hybrid, with candidates
from Hyderabad, Mumbai, Delhi NCR also welcome. 5 to 9 years experience,
treated as a flexible band.

Core mandate: own the intelligence layer of the product - the ranking,
retrieval, and matching systems that decide what recruiters see when they
search for candidates. Will audit an existing BM25 plus rule-based system,
ship a v2 ranking system using embeddings, hybrid retrieval, and LLM-based
re-ranking, and build offline and online evaluation infrastructure.

Required: production experience with embeddings-based retrieval systems
such as sentence-transformers, OpenAI embeddings, BGE, or E5, deployed to
real users, including handling embedding drift, index refresh, and
retrieval-quality regression in production. Production experience with
vector databases or hybrid search infrastructure such as Pinecone, Weaviate,
Qdrant, Milvus, OpenSearch, Elasticsearch, or FAISS. Strong Python with
attention to code quality. Hands-on experience designing evaluation
frameworks for ranking systems: NDCG, MRR, MAP, offline to online
correlation, A/B test interpretation.

Nice to have: LLM fine-tuning experience such as LoRA, QLoRA, or PEFT.
Experience with learning-to-rank models, XGBoost-based or neural.
Prior exposure to HR-tech, recruiting tech, or marketplace products.
Background in distributed systems or large-scale inference optimization.
Open-source contributions in the AI and ML space.

Ideal candidate profile: 6 to 8 years total experience, of which 4 to 5 years
are in applied machine learning or AI roles at product companies, not pure
services companies. Has shipped at least one end-to-end ranking, search, or
recommendation system to real users at meaningful scale. Has strong, defensible
opinions about retrieval, evaluation, and LLM integration grounded in systems
they actually built. Located in or willing to relocate to Noida or Pune.
""".strip()


# ---------------------------------------------------------------------------
# Disqualifiers - JD section "Things we explicitly do NOT want" plus the
# experience-band disqualifiers. These map to HARD or SOFT penalties, not
# automatic removal - the JD's own framing is "will probably not move
# forward", which is a strong down-weight, not an instant zero. A candidate
# can still claw back rank with an exceptional profile elsewhere; this
# mirrors how a human recruiter actually reads a JD's "red flags" section.
# ---------------------------------------------------------------------------

# JD: "people who have only worked at consulting firms ... in their entire
# career. If you're currently at one of these companies but have prior
# product-company experience, that's fine." -> the penalty is about the
# *entire* career history being services-only, not current employer alone.
CONSULTING_FIRMS = {
    "tcs", "tata consultancy services", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "mindtree", "hcl", "hcltech", "tech mahindra",
    "ibm services", "genpact", "wns",
}

# JD: "people whose primary expertise is computer vision, speech, or
# robotics without significant NLP/IR exposure"
CV_SPEECH_ROBOTICS_TITLES = {
    "computer vision engineer", "robotics engineer", "speech recognition engineer",
    "autonomous systems engineer", "perception engineer",
}
NLP_IR_RESCUE_SKILLS = {
    "nlp", "natural language processing", "information retrieval", "embeddings",
    "search", "ranking", "retrieval", "transformers", "bert", "llm",
    "vector search", "semantic search", "recommendation systems",
}

# JD: required skills, used for the "must have at least some of these,
# grounded in career history, not just a skills-list" check.
REQUIRED_SKILL_FAMILIES = {
    "embeddings": {"embeddings", "sentence transformers", "bge", "e5", "openai embeddings"},
    "vector_db": {"pinecone", "weaviate", "qdrant", "milvus", "opensearch",
                  "elasticsearch", "faiss", "vector search", "vector database"},
    "python": {"python"},
    "eval": {"ndcg", "mrr", "map", "a/b testing", "evaluation framework",
              "offline evaluation", "learning to rank", "learning-to-rank"},
}

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning llms", "xgboost", "lightgbm",
    "distributed systems", "mlops", "open source",
}

TARGET_CITIES = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "gurgaon",
    "gurugram", "new delhi",
}

EXPERIENCE_SWEET_SPOT = (5.0, 9.0)   # JD's stated band
EXPERIENCE_IDEAL = (6.0, 8.0)        # JD's "ideal candidate" band

# JD: "if your 'AI experience' consists primarily of recent (under 12
# months) projects using LangChain to call OpenAI -- we will probably not
# move forward, unless you can demonstrate substantial pre-LLM-era ML
# production experience."
RECENT_LLM_WRAPPER_MONTHS = 12
LLM_WRAPPER_SKILLS = {"langchain", "openai api", "prompt engineering", "llamaindex"}
PRE_LLM_ML_SKILLS = {
    "machine learning", "scikit-learn", "xgboost", "lightgbm",
    "recommendation systems", "feature engineering", "information retrieval",
    "search", "ranking", "nlp",
}


@dataclass
class ScoringWeights:
    """All tunable weights in one place, so the methodology summary can just
    point here instead of scattering magic numbers through the codebase."""
    semantic_similarity: float = 0.30
    skill_match: float = 0.25
    experience_fit: float = 0.15
    career_trajectory: float = 0.10
    location: float = 0.05
    education: float = 0.05
    behavioral_multiplier_weight: float = 0.10  # applied multiplicatively, see ranker.py

    disqualifier_penalty: dict = field(default_factory=lambda: {
        "pure_research_only": 0.85,       # multiplicative penalty (0.85 = -15%)
        "recent_llm_wrapper_only": 0.80,
        "stale_senior_no_code": 0.85,
        "cv_speech_robotics_no_nlp": 0.75,
        "consulting_only_career": 0.80,
        "closed_source_no_validation": 0.90,
        "title_chaser": 0.85,
    })