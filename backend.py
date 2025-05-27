# backend/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Initialize pipelines
qa_pipeline = pipeline("question-answering")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Sample concepts dictionary
concepts = {
    "os": "An operating system (OS) manages computer hardware and software resources and provides common services for computer programs.",
    "data structure": "A data structure is a storage format that enables efficient access and modification of data.",
    "recursion": "Recursion is a method of solving a problem where the solution depends on solving smaller instances of the same problem."
}

sample_text = """
Python is an interpreted, high-level, general-purpose programming language. Its design philosophy emphasizes code readability.
Python supports multiple programming paradigms, including structured, object-oriented, and functional programming.
"""

class Question(BaseModel):
    question: str

class ConceptRequest(BaseModel):
    topic: str

@app.post("/explain")
def explain_concept(req: ConceptRequest):
    topic = req.topic.lower()
    explanation = concepts.get(topic, "Sorry, I don't have info on that.")
    return {"explanation": explanation}

@app.post("/summarize")
def summarize():
    summary = summarizer(sample_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
    return {"summary": summary}

@app.post("/ask")
def ask_question(req: Question):
    answer = qa_pipeline({'question': req.question, 'context': sample_text})['answer']
    return {"answer": answer}

@app.post("/mcq")
def generate_mcq():
    question = "What is Python primarily used for?"
    options = ['Low-level programming', 'Database management', 'Web and software development', 'Network configuration']
    correct = 2
    return {
        "question": question,
        "options": options,
        "answer": options[correct]
    }
