chat_history = []

def add_to_memory(question, answer):
    chat_history.append((question, answer))

def get_memory_text():
    history = ""
    for q, a in chat_history:
        history += f"User: {q}\nAI: {a}\n"
    return history
