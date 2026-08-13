import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


class WebSearch:

    def __init__(self):

        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY not found in .env"
            )

        self.client = TavilyClient(
            api_key=api_key
        )

        print("✅ Tavily Web Search Initialized")


    def search(
        self,
        query,
        max_results=3
    ):

        print(
            f"\n🌐 Searching web for: {query}"
        )

        try:

            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results
            )

            results = []

            for item in response.get(
                "results",
                []
            ):

                results.append({

                    "title":
                        item.get(
                            "title",
                            ""
                        ),

                    "url":
                        item.get(
                            "url",
                            ""
                        ),

                    "content":
                        item.get(
                            "content",
                            ""
                        )
                })


            print(
                f"✅ Web search completed: "
                f"{len(results)} results"
            )

            return results


        except Exception as e:

            print(
                "\n⚠️ Web search failed:"
            )

            print(
                str(e)
            )

            # Important:
            # Web search failure should NOT
            # stop Gemini from answering.

            return []


# ==========================================
# TEST
# ==========================================

if __name__ == "__main__":

    web = WebSearch()

    results = web.search(
        "What is artificial intelligence?",
        max_results=3
    )

    for result in results:

        print(
            "\n----------------------"
        )

        print(
            "Title:",
            result["title"]
        )

        print(
            "URL:",
            result["url"]
        )

        print(
            "Content:"
        )

        print(
            result["content"][:500]
        )