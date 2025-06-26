from dotenv import load_dotenv
from src.workflow import Workflow

load_dotenv()


def main():
    workflow = Workflow()
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


if __name__ == "__main__":
    main()