import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY не найден!")

genai.configure(api_key=GEMINI_API_KEY)

# Обновляем модель на ту, что требует Google
model = genai.GenerativeModel('gemini-3.8-flash')

class UserText(BaseModel):
    text: str

SYSTEM_PROMPT = """
Ты — экспертный ИИ-ассистент приложения "Second Brain". Твоя задача — анализировать текст пользователя и структурировать его строго по правилам.

ПРАВИЛА ОПРЕДЕЛЕНИЯ ТИПА (type):
1. "task" — разовое действие с конкретной датой.
2. "habit" — регулярное действие.
3. "idea" — мысль, заметка или идея без привязки ко времени.

ПРАВИЛА ЗАПОЛНЕНИЯ ПОЛЕЙ:
- category: СТРОГО из списка ["Здоровье и Спорт", "Психология и Отношения", "Работа и Финансы", "Личное развитие", "Быт и Покупки", "Идеи и Инсайты"].
- due_date: Формат "YYYY-MM-DD". Если даты нет — null.
- due_time: Формат "HH:MM". Если время не указано — null.
- repeat: Для "habit" укажи периодичность ("daily", "weekly", "monthly"). Для остальных — "none".
- energy: "high", "medium", "low".
- duration: "5min", "30min", "1hour+".
- priority: "high", "medium", "low".
- status: Всегда "pending".
- missing_fields: Массив строк. Если для "task" нет даты — добавь "date". Если для "habit" нет repeat — добавь "repeat". Если это "idea", массив пустой [].

ФОРМАТ ОТВЕТА:
Верни СТРОГО валидный JSON без markdown-разметки и без лишних слов.

{
  "items": [
    {
      "type": "task" | "habit" | "idea",
      "title": "Краткое название (до 50 символов)",
      "description": "Полное описание или контекст",
      "category": "Одна из рубрик",
      "due_date": "YYYY-MM-DD" или null,
      "due_time": "HH:MM" или null,
      "repeat": "none" | "daily" | "weekly" | "monthly",
      "energy": "high" | "medium" | "low",
      "duration": "5min" | "30min" | "1hour+",
      "priority": "high" | "medium" | "low",
      "status": "pending",
      "missing_fields": []
    }
  ]
}
"""

@app.post("/process")
async def process_text(user_input: UserText):
    try:
        response = model.generate_content(
            f"{SYSTEM_PROMPT}\n\nТекст пользователя: {user_input.text}"
        )
        
        clean_response = response.text.strip()
        
        start_idx = clean_response.find('{')
        end_idx = clean_response.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_response = clean_response[start_idx:end_idx+1]
        
        parsed_data = json.loads(clean_response)
        return parsed_data
        
    except json.JSONDecodeError as e:
        return {"error": "ИИ вернул некорректный JSON", "raw_text": response.text if 'response' in locals() else "Нет ответа"}
    except Exception as e:
        return {"error": "Произошла ошибка", "details": str(e)}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Second Brain API is running"}
