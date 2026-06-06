from database import initialize_database

from memory import (
    save_message,
    get_recent_history,
    save_preference,
    save_summary,
    get_long_term_memory,
    build_context
)

from search_tool import web_search


def main():

    print("=" * 50)
    print("INITIALIZING DATABASE")
    print("=" * 50)

    initialize_database()

    user_id = "user123"
    session_id = "session001"

    print("\nSaving chat messages...")

    save_message(
        user_id,
        session_id,
        "user",
        "What is Artificial Intelligence?"
    )

    save_message(
        user_id,
        session_id,
        "assistant",
        "AI is the simulation of human intelligence."
    )

    print("Messages saved successfully.")

    print("\nSaving long-term memory...")

    save_preference(
        user_id,
        "Prefers concise answers"
    )

    save_summary(
        user_id,
        "Discussed AI fundamentals"
    )

    print("Memory saved successfully.")

    print("\n" + "=" * 50)
    print("RECENT HISTORY")
    print("=" * 50)

    history = get_recent_history(
        user_id,
        session_id
    )

    print(history)

    print("\n" + "=" * 50)
    print("LONG TERM MEMORY")
    print("=" * 50)

    memories = get_long_term_memory(
        user_id
    )

    print(memories)

    print("\n" + "=" * 50)
    print("CONTEXT BUILDER")
    print("=" * 50)

    context = build_context(
        user_id,
        session_id
    )

    print(context)

    print("\n" + "=" * 50)
    print("WEB SEARCH TEST")
    print("=" * 50)

    search_results = web_search(
        "Latest developments in Artificial Intelligence"
    )

    print(search_results)

    print("\n" + "=" * 50)
    print("ALL TESTS COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()