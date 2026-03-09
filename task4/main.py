import math
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
TASK2_DIR = BASE_DIR.parent / "task2"

TOKENS_DIR = TASK2_DIR / "tokens"
LEMMAS_DIR = TASK2_DIR / "lemmas"

TERMS_OUTPUT_DIR = BASE_DIR / "terms_tfidf"
LEMMAS_OUTPUT_DIR = BASE_DIR / "lemmas_tfidf"

ENCODING = "utf-8"


def ensure_dirs():
    TERMS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LEMMAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clear_outputs():
    ensure_dirs()
    for f in TERMS_OUTPUT_DIR.glob("*.txt"):
        f.unlink()
    for f in LEMMAS_OUTPUT_DIR.glob("*.txt"):
        f.unlink()


def read_tokens_file(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding=ENCODING).splitlines()
        if line.strip()
    ]


def read_lemmas_file(path: Path) -> Counter:
    """
    Формат строки в task2/lemmas/<doc>.txt:
    <лемма> <токен1> <токен2> ... <токенN>

    Для tf леммы по заданию нужно суммировать количество терминов,
    относящихся к этой лемме.
    Значит частота леммы в документе = количество токенов в строке после леммы.
    """
    lemma_counter = Counter()

    lines = path.read_text(encoding=ENCODING).splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if not parts:
            continue

        lemma = parts[0]
        forms = parts[1:]

        lemma_counter[lemma] += len(forms)

    return lemma_counter


def compute_idf(doc_counters: dict[str, Counter], total_docs: int) -> dict[str, float]:
    df_counter = Counter()

    for counter in doc_counters.values():
        for term in counter.keys():
            if counter[term] > 0:
                df_counter[term] += 1

    idf = {}
    for term, df in df_counter.items():
        idf[term] = math.log(total_docs / df) if df > 0 else 0.0

    return idf


def save_tfidf_file(path: Path, counter: Counter, idf_map: dict[str, float]):
    total_terms = sum(counter.values())
    lines = []

    for item in sorted(counter.keys()):
        tf = counter[item] / total_terms if total_terms > 0 else 0.0
        idf = idf_map.get(item, 0.0)
        tf_idf = tf * idf
        lines.append(f"{item} {idf:.6f} {tf_idf:.6f}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding=ENCODING)


def main():
    if not TOKENS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка токенов: {TOKENS_DIR}")
    if not LEMMAS_DIR.exists():
        raise FileNotFoundError(f"Не найдена папка лемм: {LEMMAS_DIR}")

    clear_outputs()

    token_files = sorted(TOKENS_DIR.glob("*.txt"))
    lemma_files = sorted(LEMMAS_DIR.glob("*.txt"))

    if not token_files:
        print("Нет файлов токенов")
        return
    if not lemma_files:
        print("Нет файлов лемм")
        return

    # -----------------------------
    # Термины
    # -----------------------------
    term_doc_counters: dict[str, Counter] = {}

    for file_path in token_files:
        doc_id = file_path.stem
        tokens = read_tokens_file(file_path)
        term_doc_counters[doc_id] = Counter(tokens)

    total_docs_terms = len(term_doc_counters)
    term_idf = compute_idf(term_doc_counters, total_docs_terms)

    for doc_id, counter in term_doc_counters.items():
        save_tfidf_file(TERMS_OUTPUT_DIR / f"{doc_id}.txt", counter, term_idf)

    # -----------------------------
    # Леммы
    # -----------------------------
    lemma_doc_counters: dict[str, Counter] = {}

    for file_path in lemma_files:
        doc_id = file_path.stem
        lemma_doc_counters[doc_id] = read_lemmas_file(file_path)

    total_docs_lemmas = len(lemma_doc_counters)
    lemma_idf = compute_idf(lemma_doc_counters, total_docs_lemmas)

    for doc_id, counter in lemma_doc_counters.items():
        save_tfidf_file(LEMMAS_OUTPUT_DIR / f"{doc_id}.txt", counter, lemma_idf)

    print("Готово.")
    print(f"Файлы терминов: {TERMS_OUTPUT_DIR}")
    print(f"Файлы лемм: {LEMMAS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()