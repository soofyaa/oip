import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
import pymorphy3


# -----------------------------
# Пути
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
SOURCE_DIR = BASE_DIR.parent / "task1" / "vykachka"

TOKENS_DIR = BASE_DIR / "tokens"
LEMMAS_DIR = BASE_DIR / "lemmas"

ALL_TOKENS_FILE = BASE_DIR / "all_tokens.txt"
ALL_LEMMAS_FILE = BASE_DIR / "all_lemmas.txt"

ENCODING = "utf-8"


# -----------------------------
# Регулярки
# -----------------------------
TOKEN_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
RUSSIAN_WORD_RE = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$", re.IGNORECASE)


# -----------------------------
# Фильтры
# -----------------------------
EXCLUDED_POS = {
    "PREP",   # предлог
    "CONJ",   # союз
    "PRCL",   # частица
    "INTJ",   # междометие
    "NPRO",   # местоимение
    "NUMR",   # числительное
    "PRED",   # предикатив
}

STOP_WORDS = {
    "и", "а", "но", "или", "да", "же", "ли", "бы", "чтобы", "как", "что", "это",
    "тот", "та", "те", "то", "этот", "эта", "эти", "такой", "такая", "такие",
    "кто", "где", "куда", "откуда", "почему", "зачем", "когда",
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "меня", "тебя", "нас", "вас", "них",
    "мой", "моя", "моё", "мои", "твой", "твоя", "его", "ее", "её", "их", "наш", "ваш",
    "себя", "свой", "своя", "свои",
    "в", "во", "на", "под", "над", "у", "к", "ко", "с", "со", "от", "до", "из", "по",
    "за", "для", "о", "об", "обо", "про", "при", "через", "без", "между",
    "не", "ни", "нет", "есть",
    "был", "была", "было", "были", "будет", "будут",
    "очень", "просто", "тоже", "ещё", "еще", "уже", "лишь", "только",
    "там", "тут", "здесь", "сюда", "отсюда",
    "который", "которая", "которое", "которые",
}

# Служебные токены сайта
SITE_GARBAGE_TOKENS = {
    "автор", "авторы", "архив", "вход", "выход", "главная", "главный",
    "комментарий", "комментарии", "контакт", "контакты",
    "конфиденциальность", "политика", "сайт", "форум", "поиск",
    "меню", "пожаловаться", "цитировать", "ответить", "оценка",
    "рецепт", "рецепты", "кулинария", "академия", "категория", "категории",
    "поделиться", "лайк", "новость", "новости", "вакансия", "вакансии",
    "канал", "каналы", "полезное", "инструмент", "инструменты",
    "компания", "правило", "правила", "условие", "условия",
    "реклама", "соцсеть", "соцсети", "сервис", "сервисы",
    "страница", "раздел", "разделы", "фотоотчет", "фотоотчеты",
    "совет", "советы", "конкурс", "конкурсы", "калькулятор", "калькуляторы",
    "таблица", "таблицы", "поисковый", "поиска", "кнопка",
}

# Единицы измерения / краткие мусорные слова
JUNK_TOKENS = {
    "г", "гр", "кг", "мг", "л", "мл", "см", "мм", "шт",
    "ккал", "мин", "сек", "ч", "час",
    "др", "пр", "ул", "рис", "табл",
}

# Целые строки, которые почти точно являются мусором
BOILERPLATE_PHRASES = {
    "главная",
    "поиск",
    "действия",
    "войти",
    "меню",
    "пожаловаться",
    "комментарии",
    "похожие рецепты",
    "советы к рецепту",
    "показать комментарии",
    "ссылка на комментарий",
    "цитировать",
    "ответить",
    "автор",
    "популярно сейчас",
    "расширенный поиск",
    "фотоотчеты и комментарии",
    "кулинарная академия",
    "категории рецептов",
    "калькуляторы",
    "таблица калорийности продуктов",
    "таблица содержания белков в продуктах",
    "таблица содержания жиров в продуктах",
    "таблица содержания углеводов в продуктах",
    "моя оценка рецепта",
    "наши вакансии",
    "наши соцсети",
    "полезное",
    "конкурсы",
    "инструменты",
    "каналы",
    "работа",
    "форум",
    "архив",
    "контакты",
    "политика конфиденциальности",
    "условия использования",
    "вернуться наверх",
}

# Строки, содержащие такие фрагменты, чаще всего не нужны
BAD_SUBSTRINGS = {
    "комментар",
    "форум",
    "конкурс",
    "инструмент",
    "категори",
    "калькулятор",
    "соцсет",
    "ваканси",
    "архив",
    "политик",
    "конфиденциаль",
    "поиск",
    "оценка рецепта",
    "показать комментарии",
    "ссылка на комментарий",
    "наши соцсети",
    "наши вакансии",
}

# Чтобы не тащить имена пользователей и короткий шум
DISALLOWED_NAME_LIKE = {
    "анна", "мария", "елена", "ольга", "ирина", "наталья", "светлана",
    "алексей", "андрей", "дмитрий", "сергей", "иван", "татьяна",
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def ensure_dirs():
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    LEMMAS_DIR.mkdir(parents=True, exist_ok=True)


def clear_outputs():
    ensure_dirs()

    for path in TOKENS_DIR.glob("*.txt"):
        path.unlink()

    for path in LEMMAS_DIR.glob("*.txt"):
        path.unlink()

    for path in [ALL_TOKENS_FILE, ALL_LEMMAS_FILE]:
        if path.exists():
            path.unlink()


def extract_lines_from_html(file_path: Path) -> list[str]:
    html = file_path.read_text(encoding=ENCODING)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = []
    for text in soup.stripped_strings:
        cleaned = " ".join(text.split())
        if cleaned:
            lines.append(cleaned)

    return lines


def is_noise_line(line: str) -> bool:
    normalized = normalize_text(line)

    if not normalized:
        return True

    if normalized in BOILERPLATE_PHRASES:
        return True

    if any(part in normalized for part in BAD_SUBSTRINGS):
        return True

    if re.fullmatch(r"[\d\s:./\\,#\-–—]+", normalized):
        return True

    if re.fullmatch(r"[^\wа-яё]+", normalized, flags=re.IGNORECASE):
        return True

    words = TOKEN_RE.findall(line)

    # Совсем короткие и пустые строки
    if len(words) == 0:
        return True

    # Одно короткое слово
    if len(words) == 1 and len(words[0]) <= 2:
        return True

    # Строка из коротких служебных слов
    if words and all(len(w) <= 3 for w in words) and len(words) <= 3:
        return True

    return False


def select_main_text(lines: list[str]) -> str:
    """
    Стараемся оставить содержательные строки.
    """
    filtered = []
    line_counts = Counter(normalize_text(line) for line in lines if line.strip())

    for line in lines:
        if is_noise_line(line):
            continue

        normalized = normalize_text(line)

        # Повторяющиеся элементы интерфейса обычно встречаются много раз
        if line_counts[normalized] > 2 and len(normalized.split()) <= 6:
            continue

        words = TOKEN_RE.findall(line)
        long_words = [w for w in words if len(w) >= 4]

        # Очень короткие строки без смысла отбрасываем
        if len(words) < 2 and len(long_words) == 0:
            continue

        filtered.append(line)

    if filtered:
        return "\n".join(filtered)

    return "\n".join(lines)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def is_valid_token(token: str, morph: pymorphy3.MorphAnalyzer) -> bool:
    token = token.lower()

    if not token or len(token) < 3:
        return False

    if token in STOP_WORDS:
        return False

    if token in JUNK_TOKENS:
        return False

    if token in SITE_GARBAGE_TOKENS:
        return False

    if token in DISALLOWED_NAME_LIKE:
        return False

    if not RUSSIAN_WORD_RE.fullmatch(token):
        return False

    if re.search(r"\d", token):
        return False

    parses = morph.parse(token)
    if not parses:
        return False

    best = parses[0]
    lemma = best.normal_form

    if not lemma:
        return False

    if best.tag.POS in EXCLUDED_POS:
        return False

    if "LATN" in best.tag or "UNKN" in best.tag:
        return False

    if lemma in STOP_WORDS:
        return False

    if lemma in JUNK_TOKENS:
        return False

    if lemma in SITE_GARBAGE_TOKENS:
        return False

    if lemma in DISALLOWED_NAME_LIKE:
        return False

    if not RUSSIAN_WORD_RE.fullmatch(lemma):
        return False

    return True


def build_tokens_and_lemmas(text: str, morph: pymorphy3.MorphAnalyzer):
    page_tokens = set()
    page_lemmas = defaultdict(set)

    for token in tokenize(text):
        if not is_valid_token(token, morph):
            continue

        best = morph.parse(token)[0]
        lemma = best.normal_form

        page_tokens.add(token)
        page_lemmas[lemma].add(token)

    return page_tokens, page_lemmas


def save_tokens(tokens: set[str], output_path: Path):
    data = sorted(tokens)
    output_path.write_text(
        "\n".join(data) + ("\n" if data else ""),
        encoding=ENCODING
    )


def save_lemmas(lemma_to_tokens: dict[str, set[str]], output_path: Path):
    lines = []
    for lemma in sorted(lemma_to_tokens):
        tokens = sorted(lemma_to_tokens[lemma])
        lines.append(f"{lemma} {' '.join(tokens)}")

    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding=ENCODING
    )


def merge_lemma_maps(target: dict[str, set[str]], source: dict[str, set[str]]):
    for lemma, tokens in source.items():
        target[lemma].update(tokens)


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Папка с HTML не найдена: {SOURCE_DIR}")

    html_files = sorted(SOURCE_DIR.glob("*.html"))
    if not html_files:
        print(f"В папке нет html-файлов: {SOURCE_DIR}")
        return

    clear_outputs()
    morph = pymorphy3.MorphAnalyzer()

    all_tokens = set()
    all_lemmas = defaultdict(set)

    print(f"Источник: {SOURCE_DIR}")
    print(f"Найдено HTML-файлов: {len(html_files)}")
    print("-" * 50)

    for html_file in html_files:
        raw_lines = extract_lines_from_html(html_file)
        page_text = select_main_text(raw_lines)

        page_tokens, page_lemmas = build_tokens_and_lemmas(page_text, morph)

        base_name = html_file.stem
        save_tokens(page_tokens, TOKENS_DIR / f"{base_name}.txt")
        save_lemmas(page_lemmas, LEMMAS_DIR / f"{base_name}.txt")

        all_tokens.update(page_tokens)
        merge_lemma_maps(all_lemmas, page_lemmas)

        print(
            f"[OK] {html_file.name}: "
            f"токенов={len(page_tokens)}, лемм={len(page_lemmas)}"
        )

    save_tokens(all_tokens, ALL_TOKENS_FILE)
    save_lemmas(all_lemmas, ALL_LEMMAS_FILE)

    print("-" * 50)
    print("Готово.")
    print(f"Токены: {TOKENS_DIR}")
    print(f"Леммы: {LEMMAS_DIR}")
    print(f"Общий список токенов: {ALL_TOKENS_FILE}")
    print(f"Общий список лемм: {ALL_LEMMAS_FILE}")


if __name__ == "__main__":
    main()