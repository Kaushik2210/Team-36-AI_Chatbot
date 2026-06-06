from database import get_connection


def save_message(
    user_id,
    session_id,
    role,
    message
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO chat_history(
            user_id,
            session_id,
            role,
            message
        )
        VALUES (?, ?, ?, ?)
    """,
    (
        user_id,
        session_id,
        role,
        message
    ))

    conn.commit()
    conn.close()


def get_recent_history(
    user_id,
    session_id,
    limit=10
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT role, message
        FROM chat_history
        WHERE user_id=? AND session_id=?
        ORDER BY id DESC
        LIMIT ?
    """,
    (
        user_id,
        session_id,
        limit
    ))

    rows = cursor.fetchall()

    conn.close()

    rows.reverse()

    history = ""

    for role, msg in rows:
        history += f"{role}: {msg}\n"

    return history


def save_preference(
    user_id,
    preference
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO long_term_memory(
            user_id,
            memory_type,
            memory
        )
        VALUES (?, ?, ?)
    """,
    (
        user_id,
        "preference",
        preference
    ))

    conn.commit()
    conn.close()


def save_summary(
    user_id,
    summary
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO long_term_memory(
            user_id,
            memory_type,
            memory
        )
        VALUES (?, ?, ?)
    """,
    (
        user_id,
        "summary",
        summary
    ))

    conn.commit()
    conn.close()


def get_long_term_memory(
    user_id,
    limit=10
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT memory_type,memory
        FROM long_term_memory
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?
    """,
    (
        user_id,
        limit
    ))

    rows = cursor.fetchall()

    conn.close()

    memory_text = ""

    for mem_type, mem in rows:
        memory_text += (
            f"{mem_type.upper()}: "
            f"{mem}\n"
        )

    return memory_text


def build_context(
    user_id,
    session_id
):

    history = get_recent_history(
        user_id,
        session_id
    )

    memories = get_long_term_memory(
        user_id
    )

    return f"""
USER MEMORY:
{memories}

RECENT CHAT:
{history}
"""