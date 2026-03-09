import math
import re
from collections import Counter
from pathlib import Path

import pymorphy3
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TASK4_DIR = PROJECT_ROOT / "task4"
TASK1_DIR = PROJECT_ROOT / "task1"

LEMMAS_TFIDF_DIR = TASK4_DIR / "lemmas_tfidf"
INDEX_FILE = TASK1_DIR / "index.txt"

ENCODING = "utf-8"
QUERY_TOKEN_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
RUSSIAN_WORD_RE = re.compile(r"^[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*$")
TOP_K = 10

app = FastAPI(title="Vector Search Demo")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
morph = pymorphy3.MorphAnalyzer()


def doc_sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def load_document_vectors(tfidf_dir: Path) -> dict[str, dict[str, float]]:
    if not tfidf_dir.exists():
        raise FileNotFoundError(
            f"Папка TF-IDF не найдена: {tfidf_dir}. Сначала запусти task4/main.py"
        )

    document_vectors: dict[str, dict[str, float]] = {}
    tfidf_files = sorted(tfidf_dir.glob("*.txt"), key=lambda p: doc_sort_key(p.stem))

    if not tfidf_files:
        raise FileNotFoundError(
            f"В папке {tfidf_dir} нет .txt файлов. Сначала сформируй результаты task4."
        )

    for file_path in tfidf_files:
        doc_id = file_path.stem
        vector: dict[str, float] = {}

        with file_path.open("r", encoding=ENCODING) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue

                term, _idf_value, tfidf_value = parts
                try:
                    vector[term] = float(tfidf_value)
                except ValueError:
                    continue

        document_vectors[doc_id] = vector

    return document_vectors


def load_document_links(index_file: Path) -> dict[str, str]:
    links: dict[str, str] = {}

    if not index_file.exists():
        return links

    with index_file.open("r", encoding=ENCODING) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue

            filename, url = parts[0].strip(), parts[1].strip()
            doc_id = Path(filename).stem
            links[doc_id] = url

    return links


def compute_idf_from_document_vectors(document_vectors: dict[str, dict[str, float]]) -> dict[str, float]:
    total_docs = len(document_vectors)
    df_counter = Counter()

    for vector in document_vectors.values():
        for term in vector.keys():
            df_counter[term] += 1

    idf_map = {}
    for term, df in df_counter.items():
        idf_map[term] = math.log(total_docs / df) if df > 0 else 0.0

    return idf_map


def preprocess_query(query: str) -> list[str]:
    lemmas = []
    for token in QUERY_TOKEN_RE.findall(query):
        token = token.lower()
        if not RUSSIAN_WORD_RE.fullmatch(token):
            continue
        parsed = morph.parse(token)
        if parsed:
            lemmas.append(parsed[0].normal_form)
    return lemmas


def create_query_vector(query_terms: list[str], idf_map: dict[str, float]) -> dict[str, float]:
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
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(value * vector_b.get(term, 0.0) for term, value in vector_a.items())
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def search_documents(query: str, top_k: int = TOP_K) -> tuple[list[str], list[dict]]:
    query_terms = preprocess_query(query)
    if not query_terms:
        return [], []

    query_vector = create_query_vector(query_terms, IDF_MAP)
    scored = []

    for doc_id, doc_vector in DOCUMENT_VECTORS.items():
        score = cosine_similarity(query_vector, doc_vector)
        if score > 0:
            scored.append({
                "doc_id": doc_id,
                "score": score,
                "url": DOCUMENT_LINKS.get(doc_id, ""),
            })

    scored.sort(key=lambda item: (-item["score"], doc_sort_key(item["doc_id"])))
    return query_terms, scored[:top_k]


DOCUMENT_VECTORS = load_document_vectors(LEMMAS_TFIDF_DIR)
DOCUMENT_LINKS = load_document_links(INDEX_FILE)
IDF_MAP = compute_idf_from_document_vectors(DOCUMENT_VECTORS)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "query": "",
            "normalized_query": [],
            "results": [],
            "results_count": 0,
            "documents_count": len(DOCUMENT_VECTORS),
        },
    )


@app.post("/search", response_class=HTMLResponse)
def search(request: Request, query: str = Form(...)):
    query = query.strip()
    normalized_query, results = search_documents(query, top_k=TOP_K)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "query": query,
            "normalized_query": normalized_query,
            "results": results,
            "results_count": len(results),
            "documents_count": len(DOCUMENT_VECTORS),
        },
    )
