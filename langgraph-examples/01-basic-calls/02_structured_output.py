"""02_structured_output.py -- with_structured_output() for typed Pydantic results.

Replaces the root project's manual JSON Schema flattening in 03-structured-output.py.
LangChain handles schema construction internally -- you just pass a Pydantic model.
"""

from pydantic import BaseModel, Field

from shared import get_openai_llm


class Reviewer(BaseModel):
    name: str
    publication: str


class MovieReview(BaseModel):
    title: str
    year: int
    reviewer: Reviewer
    rating: float = Field(ge=0.0, le=10.0)
    summary: str
    genre: list[str]
    recommended: bool


def main():
    llm = get_openai_llm()
    structured_llm = llm.with_structured_output(MovieReview)

    review_text = (
        "Avatar (2009) directed by James Cameron. "
        "Reviewed by Roger Ebert for the Chicago Sun-Times. "
        "Rating: 7.5/10. A groundbreaking sci-fi epic with stunning visuals. "
        "Genres: action, sci-fi, adventure. Highly recommended."
    )

    print("=== Structured Output ===")
    review = structured_llm.invoke(review_text)

    # Returns a typed Pydantic instance directly -- no manual parsing
    print(review.model_dump_json(indent=2))
    print(f"\nType: {type(review).__name__}")
    print(f"Rating: {review.rating}/10")
    print(f"Recommended: {review.recommended}")


if __name__ == "__main__":
    main()
