import os
from dotenv import load_dotenv
from src.workflow import Workflow

load_dotenv()


def main():
    workflow = Workflow(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini")
    print("VPBANK ASSISTANT")

    while True:
        query = input("\nUser: ").strip()
        if query.lower() in {"quit", "exit"}:
            break

        if query:
            print(f"\nBot:")
            result = workflow.run(query)
            print(result.output)
            print("=" * 60)

            print(f"Search Result:", result.search_results)

            if result.analysis:
                print("Question Analysis: ")
                print("-" * 40)
                print(result.analysis)
            if result.need_human:
                print("Human Interfere............")


if __name__ == "__main__":
    main()