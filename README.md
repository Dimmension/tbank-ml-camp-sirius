# LLM-разметка данных

## Актуальность
Когда пользователь обращается в поддержку, нам необходимо определить его намерение, интент.
Один из способов решения – модель классификации.
При разметке данных разметчики часто допускают ошибки. Загрязнение датасета приводит к плохому качеству итоговой модели.

## Задача
Реализовать пайплайнна базе LLM для исправления ошибок разметки.
На исправленных данных нужно обучить модель классификатора и сравнить с бейзлайном обучении классификатора на исходном датасете с ошибками.

## Детали

### Датасет
* Исходный: пары «query-intent» (интенты могут быть ошибочными). Всего 150 классов интентов + 1 метка "oos" для текстов, не относящихся ни к одному из классов.
* Итоговый: мы сгенерировали с помощью Chat GPT 4o для каждого интента его описание. Для каждого интента нашли по 3 наиболее семантически похощих по описаниям лейбла

#### Замечание
Даже в валидационном датасете заметны не совсем логичные и очевидные ответы, например:
- **query**: "I repeat cancel", **true** label: "cancel", **predicted** by us: "repeat"
- **query**: "Is it possible to make a reservation at sushi king in Virginia beach?", **true** label: "accept_reservation", **predicted** by us: "restaraunt_reservation
- **query**: "You were made where?", **true** label: "where_are_you_from", **predicted** by us: "who_made_you"

### Варианты решений:
* Использовать модель классификации для разметки. Где она будет больше ошибаться, там скорее всего допущена ошибка. Очевидными ограничениями являются:
* * При появлении новых классов придется переобучаться, а тк они появляются в реальных кейсах постоянно, переобучатьяс также придется постоянно
  * Необходимо время на обучение
  * Модель классификации часто очень уверена. Даже на бейзлайне, она показывает высокие метрики, что делает данную идею почти невыполнимой
* Не использовать RAG, на вход LLM подавать все 150 лейблов с их описаниями с "просьбой" выбрать наиболее подходящую метку нашему запросу. Минусы:
* * Если меток слишком много, контекстное окно может выйти очень большим, что к тому же увеличит время и стоимость инференса
  * Модель будет чаще ошибаться, тк одна из главных проблем задачи заключается в том, что в данных очень много похожих интентов
* **Наше решение**: RAG система на базе LLM с многофакторной фильтрацией примеров out of domain ("oos").
* * С помощью retrieval достаем из всего пула интентов, только топ n наиболее релевантных
  * С помощью reranker ранжируем их, пропускаем дальше только топ k наиболее релевантных. Здесь же происходит первый этап отсеивания "oos"
  * Делаем промпт в LLM с запросом выбрать наиболее подходящую из представленных меток, иначе выбрать "oos"
 
#### Замечание
С просьбой LLM выбрать "oos" в случае, когда она думает, что ничего не подходит, нужно быть аккуратными, потому что ей свойственно часто сомневаться в сложных случаях и выбирать этот варинат

### Архитектура решения.
![image](https://github.com/user-attachments/assets/5bc6769f-8fcf-44ab-9c05-66fc7c5ebc77)
![image](https://github.com/user-attachments/assets/d397ff1d-41d1-49e6-8253-55e34970f066)

* **Embedder**: дообученный BAAI/bge-large-en-v1.1 на train датасете. Использовали Modified triplet loss:
* * MultipleNegative. Anchor: query; positive: описание; negatives: похожие описания (почему то показал себя хуже, поэтому не использовали)
  * Pairwise, similarity-based. Anchor: query; positive: описание.
* **Ensemble Retrieval**: Взвешенное комбинирование результатов BM25 и FAISS (веса соответственно 0.3 и 0.7). Также вместо FAISS пробовали другие векторные базы, например, Chroma
Метрики: ![image](https://github.com/user-attachments/assets/8b5ccbd7-d4f2-4431-a118-c97bba6b32ab)
* **Reranker**: BAAI/bge-reranker-v2-my
* Гиперпараметры: n = 20, k = 10, reranker filtering threshold = 1e-6. Подбирали

* LLm generation: ![image](https://github.com/user-attachments/assets/db93f8f0-1165-4bac-9c27-4c850d07d361)
* Использовали ансамбль LLM (все модели были квантизованные с использованием llama_cpp): 
* * Qwen-2.5 7B
  * LLama-3.1 8B
  * Gemma-2 9B
* Метрики: ![image](https://github.com/user-attachments/assets/f9bc9387-7456-4095-a790-d91814b0814e)
* Время инференса 1 примера для каждой модели ~ 1с

### Итог разметки
Разпознавание "oos" стало более стабильным. В некоторых примерах удалось генерировать даже более логичные интенты для валидационного датасета, из-за чего в конечном решении при прогоне по нему, метрики упали. Но при онлайн тестировании, наш тестировщик ввел новую метрику "по ощущениям" и заметил, что она высока.

На размеченных данных мы обучили новую модель классификации
![image](https://github.com/user-attachments/assets/909d197b-e1af-462a-81ab-67fa72bfcd3c)
![image](https://github.com/user-attachments/assets/12b4f319-dd06-41db-814f-edf2d921f3dc)

[Обученные модели](https://huggingface.co/chinchilla04)

### Интеграция
Мы интегрировали итоговую классификационную модель в телеграмм бота, завернутого в docker с использованием triton. В нем можно посмотреть разницу в работе итоговой модели и бейзлайна, обученного на "грязном" датасете

![image](https://github.com/user-attachments/assets/7215bac8-2dc0-4733-8cae-31866e4e128b)
![image](https://github.com/user-attachments/assets/2ef32f14-bcdf-4a9f-b0f5-43370aadf7a8)

## Структура проекта
```
tbank-ml-camp-sirius/
├── data/                   # Датасеты и дополнительные данные
├── preprocess_data/        # Пред- и постобработка данных
└── src/                    # Исходный код приложения
    ├── ensemble/           # Система разметки
    ├── finetune/           # Обучение классификаторов и дообучение эмбеддингов
    ├── tg_app/             # Телеграмм бот 
```
