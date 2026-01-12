import json
from agent import plan_actions, generate_answer
from memory import save_message, get_relevant_history
from search import web_search

def run():
    print("🤖 Free MCP QA Bot (type 'exit' to quit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break

        # Greeting shortcut
        if question.lower() in ["hi", "hello", "hey"]:
            response = {
                "question": question,
                "action_taken": ["llm"],
                "answer": "Hello! How can I help you?",
                "confidence": 100
            }
            print("\nBot Response:")
            print(json.dumps(response, indent=2))
            print("-" * 60)
            save_message("assistant", json.dumps(response))
            continue

        save_message("user", question)

        memory_context = get_relevant_history(question)

        plan_result = plan_actions(question, memory_context)
        print(plan_result)
        plan = list(dict.fromkeys(plan_result["plan"]))  # dedupe

        combined_context = ""

        if "memory" in plan and memory_context:
            combined_context += "\n[MEMORY]\n" + memory_context

        if "web" in plan:
            combined_context += "\n[WEB]\n" + web_search(question)

        answer = generate_answer(question, combined_context)

        final_response = {
            "question": question,
            "action_taken": plan,
            "answer": answer["answer"],
            "confidence": answer["confidence"]
        }

        save_message("assistant", json.dumps(final_response))

        print("\nBot Response:")
        print(json.dumps(final_response, indent=2))
        print("-" * 60)

if __name__ == "__main__":
    run()
