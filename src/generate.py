import os
from dotenv import load_dotenv
import groq

# Load .env explicitly
load_dotenv()

class Generator:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")
        
        self.client = groq.Groq(api_key=api_key)
        self.model_name = model_name
        
        self.system_prompt = (
            "You are a strict academic assistant. "
            "You must ONLY answer using the provided context. "
            "You must strictly refuse to answer if the information is not present in the context, "
            "outputting exactly: 'I cannot answer this based on the provided context.'"
        )

    def answer(self, question: str, retrieved_chunks: list) -> dict:
        """
        Receives a question and a list of top-k retrieved chunks.
        Formats the context and requests an answer from the Groq API.
        Returns a dict conforming to {answer, retrieved_chunks}.
        """
        context_str = "\n\n".join([f"Chunk {i+1}:\n{c['chunk']['text']}" for i, c in enumerate(retrieved_chunks)])
        
        user_message = f"Context:\n{context_str}\n\nQuestion: {question}"

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,  # CRITICAL: 0 for structured reproducible evaluation
            max_tokens=500
        )

        answer_text = response.choices[0].message.content.strip()

        return {
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks
        }
