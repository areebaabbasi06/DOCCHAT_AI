import os
import base64
import mimetypes
from typing import Dict, Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()


class GeminiLLM:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = "gemini-3.1-flash-lite"


    # ==========================================
    # BUILD PROMPT
    # ==========================================

    def build_prompt(
        self,
        question: str,
        context: str
    ) -> str:

        return f"""
You are DocChat AI, a helpful and intelligent AI assistant.

Your job is to answer the user's question accurately,
clearly, naturally, and in an easy-to-understand way.

==================================================
ANSWER LANGUAGE
==================================================

1. Always answer in the SAME LANGUAGE as the user's question.

2. If the user asks in English, answer in English.

3. If the user asks in Urdu, answer in Urdu.

4. If the user asks in Roman Urdu, answer in Roman Urdu.

5. If the user asks in Hindi, answer in Hindi.

6. If the user mixes languages, naturally follow the language
   used by the user.

7. Do not unnecessarily translate the answer into another language.


==================================================
IMAGE UNDERSTANDING
==================================================

The user may optionally provide an image along with the question.

1. Carefully analyze the provided image when an image is available.

2. Use visible information from the image to answer the user's question.

3. If the user asks what is shown in the image, describe the relevant
   content clearly.

4. If the image contains text, read and use that text when possible.

5. If the image contains a chart, diagram, table, screenshot,
   document, map, or code, use the visible information to answer.

6. Do not invent information that cannot reasonably be determined
   from the image.

7. If the image is unclear or insufficient, honestly say that the
   image does not provide enough information.

8. If both an image and PDF/web context are provided, combine them
   when they are relevant to the question.


==================================================
READABILITY AND PARAGRAPH RULES
==================================================

The answer must be easy for a human to read.

1. NEVER create one extremely long paragraph.

2. For normal explanations, use SHORT PARAGRAPHS.

3. Each paragraph should normally contain about
   2 to 4 sentences.

4. If one explanation becomes long, divide it into
   multiple smaller paragraphs.

5. Leave a blank line between separate paragraphs.

6. Do NOT put several unrelated ideas into one paragraph.

7. Do NOT turn every sentence into a bullet point.

8. Use paragraphs when explaining a concept naturally.

9. Use bullet points only when there are multiple separate
   items, features, advantages, examples, or facts.

10. When using bullets, EVERY bullet MUST be on its OWN LINE.

11. NEVER put multiple bullet points on the same line.

12. Do NOT use the bullet symbol "•" repeatedly in one paragraph.

13. Prefer Markdown "-" for bullet points.

14. For steps, procedures, or instructions, use numbered points.

15. EVERY numbered point must be on its OWN LINE.

16. Leave a blank line before and after a list when appropriate.


==================================================
HEADINGS
==================================================

1. Use a heading when the answer contains clearly separate sections.

2. Do not add headings unnecessarily.

3. Use simple Markdown headings such as:

## Why RAG is Important

4. Do not make the entire answer a collection of headings.

5. A heading should describe the section that follows it.


==================================================
MARKDOWN FORMATTING
==================================================

Use clean Markdown formatting.

1. Use **bold** for important technical terms,
   keywords, or concepts when useful.

2. Use *italic* only when genuinely useful.

3. Use Markdown bullet points with "- ".

4. Use Markdown numbered lists with "1.", "2.", "3." etc.

5. NEVER output raw formatting symbols unnecessarily.

6. Make sure every opening Markdown formatting symbol
   has a matching closing symbol.

7. Do not use excessive bold formatting.

8. Do not use decorative symbols just for appearance.


==================================================
EXAMPLE OF CORRECT FORMATTING
==================================================

A good answer should look like:

## Why RAG is Important

**RAG** improves the reliability of AI by allowing a language
model to retrieve relevant information from external documents
before generating an answer.

It is useful when the model needs access to specific,
domain-related, or frequently updated information.

The main benefits include:

- Reducing hallucinations
- Using domain-specific information
- Improving transparency
- Providing relevant source information

## Advanced RAG Techniques

Some commonly used techniques are:

- **Hybrid Search:** Combines semantic search with keyword search.
- **Parent-Child Chunking:** Retrieves small chunks while preserving larger context.
- **Corrective RAG:** Evaluates retrieved information and can use another search method when necessary.

IMPORTANT:
Every bullet above is on a separate line.
Never combine those bullets into one paragraph.


==================================================
ANSWER LENGTH
==================================================

1. Be concise but complete.

2. Do not make the answer unnecessarily long.

3. For a simple question, give a short answer.

4. For a conceptual question, give enough explanation
   for the user to understand the concept.

5. Break long explanations into readable paragraphs.

6. Avoid repeating the same information.


==================================================
INFORMATION AND ACCURACY RULES
==================================================

1. Use the provided PDF context when it contains relevant
   information about the user's question.

2. Give priority to the uploaded PDF when the question
   specifically asks about the uploaded document.

3. Use the provided WEB SEARCH CONTEXT when it contains
   relevant information.

4. When both PDF and web information are relevant,
   combine them naturally.

5. Use the provided image when it is relevant to the question.

6. Do NOT invent facts that are not supported by the
   available context or image.

7. For questions requiring current or latest information,
   prefer the provided web search information.

8. If there is no relevant PDF information, do not pretend
   that the PDF contains the answer.

9. If there is no relevant web or PDF information,
   use general knowledge only when appropriate.

10. If the available information is insufficient,
    honestly tell the user that the available information
    is insufficient.

11. Do not claim that information came from the PDF
    unless it is actually present in the provided context.


==================================================
PDF AND WEB CONTEXT
==================================================

{context}


==================================================
USER QUESTION
==================================================

{question}


==================================================
FINAL ANSWER INSTRUCTIONS
==================================================

Now answer the user's question directly.

Before producing the final answer, internally check:

- Is the answer in the same language as the question?
- Are paragraphs short and readable?
- Is any paragraph unnecessarily long?
- If there is a list, is EVERY item on a separate line?
- Are numbered steps on separate lines?
- Are headings used only where useful?
- Are important terms formatted with **bold** when appropriate?
- Is the answer based on the available context?
- Did you use the image when relevant?
- Did you avoid inventing unsupported information?
- Did you avoid unnecessary repetition?

Return ONLY the final answer.

Do not explain these formatting instructions to the user.
"""


    # ==========================================
    # IMAGE DATA CONVERTER
    # ==========================================

    def _image_to_part(
        self,
        image_data: Optional[str]
    ):

        if not image_data:
            return None

        try:

            # Expected format:
            # data:image/png;base64,AAAA...

            if image_data.startswith("data:image/"):

                header, encoded = image_data.split(
                    ",",
                    1
                )

                mime_type = header.split(
                    ";",
                    1
                )[0].replace(
                    "data:",
                    ""
                )

            else:

                encoded = image_data

                mime_type = "image/jpeg"


            image_bytes = base64.b64decode(
                encoded
            )


            return types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            )


        except Exception as e:

            print(
                "\n⚠️ Image processing error:"
            )

            print(
                str(e)
            )

            return None


    # ==========================================
    # GENERATE ANSWER
    # ==========================================

    def generate_answer(
        self,
        question: str,
        context: str,
        image_data: Optional[str] = None
    ) -> Dict:

        if not context.strip() and not image_data:

            return {
                "answer": (
                    "No relevant information was found."
                ),
                "sources": []
            }


        # ======================================
        # BUILD TEXT PROMPT
        # ======================================

        prompt = self.build_prompt(
            question,
            context
        )


        try:

            # ==================================
            # CONTENTS
            # ==================================

            contents = []


            # Text prompt

            contents.append(
                prompt
            )


            # ==================================
            # IMAGE
            # ==================================

            if image_data:

                print(
                    "🖼️ Adding image to Gemini request..."
                )

                image_part = (
                    self._image_to_part(
                        image_data
                    )
                )


                if image_part:

                    contents.append(
                        image_part
                    )

                    print(
                        "✅ Image added successfully."
                    )

                else:

                    print(
                        "⚠️ Image could not be processed."
                    )


            # ==================================
            # GEMINI REQUEST
            # ==================================

            print(
                "🤖 Sending request to Gemini..."
            )


            response = (
                self.client.models.generate_content(
                    model=self.model,
                    contents=contents
                )
            )


            # ==================================
            # EMPTY RESPONSE
            # ==================================

            if not response.text:

                return {
                    "answer": (
                        "Gemini returned an empty response."
                    ),
                    "sources": []
                }


            # ==================================
            # SUCCESS
            # ==================================

            return {
                "answer":
                    response.text.strip(),

                "sources":
                    []
            }


        except Exception as e:

            print(
                "\n❌ Gemini error:"
            )

            print(
                str(e)
            )


            return {
                "answer":
                    f"An error occurred: {str(e)}",

                "sources":
                    []
            }