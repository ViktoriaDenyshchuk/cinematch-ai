
import os
import json
from dotenv import load_dotenv
from groq import Groq

from movie_tools import search_movies, compare_movies

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 4

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_movies",
            "description": "Знайти фільми за жанром, мінімальним рейтингом і роком.",
            "parameters": {
                "type": "object",
                "properties": {
                    "genre": {
                        "type": "string",
                        "description": "Жанр англійською, наприклад Comedy, Drama, Sci-Fi."
                    },
                    "min_rating": {
                        "type": "number",
                        "description": "Мінімальний рейтинг від 0 до 5."
                    },
                    "year_from": {
                        "type": "integer",
                        "description": "Початковий рік випуску."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Кількість фільмів, від 1 до 20."
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_movies",
            "description": "Порівняти два фільми за середнім рейтингом та кількістю оцінок.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title1": {"type": "string"},
                    "title2": {"type": "string"}
                },
                "required": ["title1", "title2"]
            }
        }
    }
]


def run_agent(user_query: str):
    messages = [
        {
            "role": "system",
            "content": (
                "Ти CineMatch AI — помічник із вибору фільмів. "
                "Відповідай українською мовою. "
                "Для пошуку та порівняння фільмів обов'язково використовуй інструменти. "
                "Використовуй тільки факти, отримані з MovieLens. "
                "Не вигадуй рейтинги, фільми або посилання. "
                "Рейтинги MovieLens мають шкалу від 0.5 до 5. "
                "Якщо фільм відсутній, чесно повідом про це. "
                "Включай IMDb або TMDB посилання у відповідь."
            )
        },
        {"role": "user", "content": user_query}
    ]

    logs = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0
        )

        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return {
                "answer": message.content or "Не вдалося сформувати відповідь.",
                "logs": logs,
                "status": "success"
            }

        for call in message.tool_calls:
            name = call.function.name

            try:
                args = json.loads(call.function.arguments)

                if name == "search_movies":
                    result = search_movies(**args)
                elif name == "compare_movies":
                    result = compare_movies(**args)
                else:
                    result = {"error": f"Невідомий інструмент: {name}"}

                logs.append({
                    "step": step + 1,
                    "tool": name,
                    "arguments": args,
                    "result": result
                })

            except Exception as exc:
                result = {"error": str(exc)}
                logs.append({
                    "step": step + 1,
                    "tool": name,
                    "error": str(exc)
                })

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(result, ensure_ascii=False)
            })

    return {
        "answer": "Досягнуто максимальну кількість кроків агента.",
        "logs": logs,
        "status": "max_steps"
    }


if __name__ == "__main__":
    print("CineMatch AI запущено!")
    query = input("Введи запит: ")

    try:
        result = run_agent(query)
        print("\n=== ВІДПОВІДЬ ===")
        print(result["answer"])

        print("\n=== ЖУРНАЛ АГЕНТА ===")
        print(json.dumps(result["logs"], ensure_ascii=False, indent=2))

        print("\nSTATUS:", result["status"])

    except Exception as exc:
        print(f"Помилка API або агента: {exc}")
