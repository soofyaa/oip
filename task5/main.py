import math
import re
from collections import Counter
from pathlib import Path

import pymorphy3


# -----------------------------
# Пути
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
TASK4_DIR = BASE_DIR.parent / "task4"

# Основной вариант для поиска — по леммам
LEMMAS_TFIDF_DIR = TASK4_DIR / "lemmas_tfidf"

ENCODING = "utf-8"

QUERY_TOKEN_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
RUSSIAN_WORD_RE = re.compile(r"^[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*$")


def doc_sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def load_document_vectors(tfidf_dir: Path) -> dict[str, dict[str, float]]:
    """
    Загружает векторы документов из файлов task4/lemmas_tfidf/*.txt

    Формат строки:
    <лемма> <idf> <tf-idf>
    """
    if not tfidf_dir.exists():
        raise FileNotFoundError(f"Папка TF-IDF не найдена: {tfidf_dir}")

    document_vectors = {}

    tfidf_files = sorted(tfidf_dir.glob("*.txt"), key=lambda p: doc_sort_key(p.stem))
    if not tfidf_files:
        raise FileNotFoundError(f"В папке нет txt-файлов: {tfidf_dir}")

    for file_path in tfidf_files:
        doc_id = file_path.stem
        vector = {}

        with file_path.open("r", encoding=ENCODING) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 3:
                    continue

                term, idf_value, tfidf_value = parts

                try:
                    vector[term] = float(tfidf_value)
                except ValueError:
                    continue

        document_vectors[doc_id] = vector

    return document_vectors


def compute_idf_from_document_vectors(document_vectors: dict[str, dict[str, float]]) -> dict[str, float]:
    """
    Восстанавливаем idf для терминов из множества документов:
    df(term) = число документов, где термин присутствует в tf-idf файле
    idf(term) = ln(N / df)
    """
    total_docs = len(document_vectors)
    df_counter = Counter()

    for vector in document_vectors.values():
        for term in vector.keys():
            df_counter[term] += 1

    idf_map = {}
    for term, df in df_counter.items():
        idf_map[term] = math.log(total_docs / df) if df > 0 else 0.0

    return idf_map


def preprocess_query(query: str, morph: pymorphy3.MorphAnalyzer) -> list[str]:
    """
    Токенизация и лемматизация запроса.
    """
    raw_tokens = QUERY_TOKEN_RE.findall(query)
    result = []

    for token in raw_tokens:
        token = token.lower()

        if not RUSSIAN_WORD_RE.fullmatch(token):
            continue

        parsed = morph.parse(token)
        if not parsed:
            continue

        lemma = parsed[0].normal_form
        if lemma:
            result.append(lemma)

    return result


def create_query_vector(query_terms: list[str], idf_map: dict[str, float]) -> dict[str, float]:
    """
    Строим вектор запроса.
    Используем:
    tf(term, query) = count(term) / total_terms_in_query
    tf-idf(term, query) = tf * idf
    """
    if not query_terms:
        return {}

    query_counter = Counter(query_terms)
    total_terms = sum(query_counter.values())

    query_vector = {}
    for term, count in query_counter.items():
        tf = count / total_terms
        idf = idf_map.get(term, 0.0)
        query_vector[term] = tf * idf

    return query_vector


def cosine_similarity(vector_a: dict[str, float], vector_b: dict[str, float]) -> float:
    """
    Косинусное сходство двух разреженных векторов.
    """
    if not vector_a or not vector_b:
        return 0.0

    dot_product = 0.0
    for term, value in vector_a.items():
        dot_product += value * vector_b.get(term, 0.0)

    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def find_relevant_documents(
    query: str,
    document_vectors: dict[str, dict[str, float]],
    morph: pymorphy3.MorphAnalyzer,
    top_k: int | None = None,
) -> list[tuple[str, float]]:
    """
    Возвращает список документов, отсортированный по убыванию релевантности.
    """
    query_terms = preprocess_query(query, morph)
    if not query_terms:
        return []

    idf_map = compute_idf_from_document_vectors(document_vectors)
    query_vector = create_query_vector(query_terms, idf_map)

    scores = []
    for doc_id, doc_vector in document_vectors.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:
            scores.append((doc_id, score))

    scores.sort(key=lambda item: (-item[1], doc_sort_key(item[0])))

    if top_k is not None:
        return scores[:top_k]

    return scores


def print_results(results: list[tuple[str, float]]):
    if not results:
        print("Ничего не найдено.")
        return

    print("\nРезультаты поиска:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        print(f"{rank}. Документ {doc_id} — релевантность: {score:.6f}")


def main():
    document_vectors = load_document_vectors(LEMMAS_TFIDF_DIR)
    morph = pymorphy3.MorphAnalyzer()

    print("Векторный поиск по документам")
    print(f"Источник TF-IDF: {LEMMAS_TFIDF_DIR}")
    print(f"Документов загружено: {len(document_vectors)}")
    print("Введите запрос. Для выхода: q")
    print("-" * 50)

    while True:
        query = input("\nВаш запрос: ").strip()

        if query.lower() == "q":
            print("Выход.")
            break

        if not query:
            print("Пустой запрос.")
            continue

        results = find_relevant_documents(query, document_vectors, morph, top_k=10)
        print_results(results)


if __name__ == "__main__":
    main()