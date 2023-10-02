1. Реєструємось на https://huggingface.co/ та отримуємо API.
2. Встановлюємо залежності з pyproject.toml.
3. Створюємо файл .env та закидуємо туди API

OPENAI_API_KEY=

HUGGINGFACEHUB_API_TOKEN=

4. Запускаємо код командою "streamlit run LLM_Chat_v2.py" в терміналі. В браузері повинна відкритись сторінка з чатом.
5. Спочатку завантажуємо PDF файл(и).
6. Натискаємо "Process" і чекаємо поки фийл обробиться.
7. Задаємо питання в чаті.


ОСОБЛИВОСТІ

1. Перший запуск буде довгим. Буде качатись модель (приблизно 5 Гб). Далі буде швидше.
2. Модель розуміє мови: English, German, French.
3. Якщо задавати питання без завантаженого файлу, буде помилка.
4. На моделі llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 1024}) довго обробляється файл (2 сторінки тексту приблизно 4 хв.).
5. В коді закоментовані рядки для перемикання моделі HuggingFaceHub та OpenAI.

