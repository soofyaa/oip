import re
from collections import defaultdict
from pathlib import Path

import pymorphy3


# -----------------------------
# Пути
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
LEMMAS_DIR = BASE_DIR.parent / "task2" / "lemmas"
INDEX_FILE = BASE_DIR / "inverted_index.txt"

ENCODING = "utf-8"

# Приоритет операций
PRIORITY = {
    "NOT": 3,
    "AND": 2,
    "OR": 1,
}

OPERATORS = {"AND", "OR", "NOT"}
PARENTHESES = {"(", ")"}

QUERY_TOKEN_RE = re.compile(r"\(|\)|AND|OR|NOT|[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*", re.IGNORECASE)
RUSSIAN_WORD_RE = re.compile(r"^[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*$")


def extract_doc_id(file_path: Path) -> str:
    """
    Берём идентификатор документа из имени файла.
    Например:
    1.txt -> "1"
    page_12.txt -> "page_12"
    """
    return file_path.stem


def load_lemma_files(lemmas_dir: Path) -> dict[str, set[str]]:
    """
    Читает файлы из папки lemmas и строит индекс:
    lemma -> set(doc_ids)
    """
    if not lemmas_dir.exists():
        raise FileNotFoundError(f"Папка с леммами не найдена: {lemmas_dir}")

    lemma_to_docs = defaultdict(set)

    lemma_files = sorted(lemmas_dir.glob("*.txt"))
    if not lemma_files:
        raise FileNotFoundError(f"В папке нет txt-файлов: {lemmas_dir}")

    for file_path in lemma_files:
        doc_id = extract_doc_id(file_path)

        with file_path.open("r", encoding=ENCODING) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if not parts:
                    continue

                lemma = parts[0].lower()
                lemma_to_docs[lemma].add(doc_id)

    return lemma_to_docs


def save_inverted_index(index: dict[str, set[str]], output_file: Path) -> None:
    """
    Сохраняет индекс в файл.
    Формат:
    <лемма> <doc1> <doc2> ... <docN>
    """
    with output_file.open("w", encoding=ENCODING) as f:
        for lemma in sorted(index):
            docs = sorted(index[lemma], key=doc_sort_key)
            f.write(f"{lemma} {' '.join(docs)}\n")


def doc_sort_key(value: str):
    """
    Чтобы документы вида 1,2,10 сортировались как числа, если это возможно.
    Иначе сортируем как строки.
    """
    return (0, int(value)) if value.isdigit() else (1, value)


def normalize_query_token(token: str, morph: pymorphy3.MorphAnalyzer) -> str:
    """
    Если token — оператор/скобка, возвращаем как есть.
    Если слово — приводим к лемме.
    """
    upper = token.upper()

    if upper in OPERATORS:
        return upper

    if token in PARENTHESES:
        return token

    if RUSSIAN_WORD_RE.fullmatch(token):
        parsed = morph.parse(token.lower())
        if parsed:
            return parsed[0].normal_form

    return token.lower()


def tokenize_query(query: str) -> list[str]:
    """
    Разбивает запрос на токены:
    слова, AND, OR, NOT, скобки.
    """
    raw_tokens = QUERY_TOKEN_RE.findall(query)
    return [token.strip() for token in raw_tokens if token.strip()]


def insert_implicit_and(tokens: list[str]) -> list[str]:
    """
    При желании можно поддержать неявное AND:
    например, "цезарь помпей" -> "цезарь AND помпей"
    Но здесь включаем только для соседних слов/скобок,
    чтобы пользователь мог писать запросы удобнее.
    """
    if not tokens:
        return []

    result = [tokens[0]]

    for current in tokens[1:]:
        prev = result[-1]

        prev_is_term = prev not in OPERATORS and prev != "(" and prev != ")"
        prev_is_close = prev == ")"

        curr_is_term = current not in OPERATORS and current != "(" and current != ")"
        curr_is_open = current == "("
        curr_is_not = current == "NOT"

        if (prev_is_term or prev_is_close) and (curr_is_term or curr_is_open or curr_is_not):
            result.append("AND")

        result.append(current)

    return result


def validate_tokens(tokens: list[str]) -> None:
    """
    Базовая проверка корректности запроса.
    """
    balance = 0

    for token in tokens:
        if token == "(":
            balance += 1
        elif token == ")":
            balance -= 1
            if balance < 0:
                raise ValueError("Ошибка запроса: лишняя закрывающая скобка.")

    if balance != 0:
        raise ValueError("Ошибка запроса: несбалансированные скобки.")


def to_postfix(tokens: list[str]) -> list[str]:
    """
    Алгоритм сортировочной станции:
    перевод инфиксной записи в постфиксную.
    """
    output = []
    stack = []

    for token in tokens:
        if token == "(":
            stack.append(token)

        elif token == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())

            if not stack:
                raise ValueError("Ошибка запроса: не найдена открывающая скобка.")

            stack.pop()

        elif token in OPERATORS:
            while (
                stack
                and stack[-1] in OPERATORS
                and PRIORITY[stack[-1]] >= PRIORITY[token]
            ):
                output.append(stack.pop())

            stack.append(token)

        else:
            output.append(token)

    while stack:
        top = stack.pop()
        if top in PARENTHESES:
            raise ValueError("Ошибка запроса: несбалансированные скобки.")
        output.append(top)

    return output


def evaluate_postfix(postfix_tokens: list[str], index: dict[str, set[str]], all_docs: set[str]) -> set[str]:
    """
    Вычисляет булево выражение в постфиксной форме.
    """
    stack = []

    for token in postfix_tokens:
        if token == "NOT":
            if not stack:
                raise ValueError("Ошибка запроса: оператор NOT без операнда.")

            operand = stack.pop()
            stack.append(all_docs - operand)

        elif token in {"AND", "OR"}:
            if len(stack) < 2:
                raise ValueError(f"Ошибка запроса: оператор {token} без двух операндов.")

            right = stack.pop()
            left = stack.pop()

            if token == "AND":
                stack.append(left & right)
            else:
                stack.append(left | right)

        else:
            stack.append(set(index.get(token, set())))

    if len(stack) != 1:
        raise ValueError("Ошибка запроса: выражение не удалось корректно вычислить.")

    return stack[0]


def parse_and_search(query: str, index: dict[str, set[str]], all_docs: set[str], morph: pymorphy3.MorphAnalyzer) -> set[str]:
    raw_tokens = tokenize_query(query)

    if not raw_tokens:
        raise ValueError("Пустой запрос.")

    normalized_tokens = [normalize_query_token(token, morph) for token in raw_tokens]
    normalized_tokens = insert_implicit_and(normalized_tokens)
    validate_tokens(normalized_tokens)

    postfix = to_postfix(normalized_tokens)
    return evaluate_postfix(postfix, index, all_docs)


def print_index_stats(index: dict[str, set[str]], all_docs: set[str]) -> None:
    print(f"Документов: {len(all_docs)}")
    print(f"Лемм в индексе: {len(index)}")
    print(f"Файл индекса: {INDEX_FILE}")


def interactive_search(index: dict[str, set[str]], all_docs: set[str], morph: pymorphy3.MorphAnalyzer) -> None:
    print("\nБулев поиск по индексу")
    print("Поддерживаются операторы: AND, OR, NOT")
    print("Поддерживаются скобки.")
    print("Пример:")
    print("(Клеопатра AND Цезарь) OR (Антоний AND Цицерон) OR Помпей")
    print("Для выхода введите: exit\n")

    while True:
        query = input("Введите запрос: ").strip()

        if not query:
            print("Пустой запрос.\n")
            continue

        if query.lower() in {"exit", "quit", "выход"}:
            print("Завершение работы.")
            break

        try:
            result_docs = parse_and_search(query, index, all_docs, morph)
            sorted_docs = sorted(result_docs, key=doc_sort_key)

            print(f"Найдено документов: {len(sorted_docs)}")
            if sorted_docs:
                print("Документы:", " ".join(sorted_docs))
            else:
                print("Совпадений нет.")
            print()

        except ValueError as e:
            print(f"{e}\n")


def main():
    morph = pymorphy3.MorphAnalyzer()

    index = load_lemma_files(LEMMAS_DIR)
    all_docs = set()

    for docs in index.values():
        all_docs.update(docs)

    save_inverted_index(index, INDEX_FILE)

    print("Инвертированный индекс построен.")
    print_index_stats(index, all_docs)

    interactive_search(index, all_docs, morph)


if __name__ == "__main__":
    main()