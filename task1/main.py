import time
import zipfile
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# -----------------------------
# Настройки
# -----------------------------
URLS_FILE = "urls.txt"              # файл со списком ссылок
OUTPUT_DIR = "vykachka"             # папка для скачанных страниц
INDEX_FILE = "index.txt"            # файл индексирования
ZIP_FILE = "vykachka.zip"           # итоговый zip-архив
MIN_PAGES = 100                     # минимум страниц по заданию
REQUEST_TIMEOUT = 15                # таймаут запроса в секундах
DELAY_BETWEEN_REQUESTS = 0.5        # задержка между запросами
MIN_TEXT_LENGTH = 300               # минимум символов текста, чтобы страница считалась нормальной

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def read_urls(file_path: str) -> list[str]:
    """
    Читает ссылки из файла.
    Поддерживает:
    - одна ссылка в строке
    - несколько ссылок через запятую
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл со ссылками не найден: {file_path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    raw_parts = []
    for line in content.splitlines():
        raw_parts.extend(line.split(","))

    urls = []
    seen = set()

    for part in raw_parts:
        url = part.strip()
        if not url:
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def is_text_page(response: requests.Response) -> bool:
    """
    Проверяем, что это HTML-страница.
    """
    content_type = response.headers.get("Content-Type", "").lower()
    return "text/html" in content_type or "application/xhtml+xml" in content_type


def download_page(url: str) -> str | None:
    """
    Скачивает страницу и возвращает HTML.
    """
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True
        )
        response.raise_for_status()

        if not is_text_page(response):
            print(f"[SKIP] Не HTML-страница: {url}")
            return None

        if not response.encoding:
            response.encoding = response.apparent_encoding or "utf-8"

        return response.text

    except requests.RequestException as e:
        print(f"[ERROR] {url} -> {e}")
        return None


def extract_clean_text_html(raw_html: str, url: str) -> str | None:
    """
    Извлекает только текст страницы и формирует новый HTML,
    где в body находится только текст.
    """
    soup = BeautifulSoup(raw_html, "html.parser")

    # Удаляем заведомо недопустимые и служебные элементы
    for tag in soup([
        "script", "style", "noscript", "iframe", "svg", "canvas",
        "img", "picture", "source", "video", "audio",
        "link", "meta", "object", "embed", "form", "button"
    ]):
        tag.decompose()

    # Удаляем комментарии
    for text_node in soup.find_all(string=lambda t: t and t.__class__.__name__ == "Comment"):
        text_node.extract()

    # Берём title, если есть
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        title = url

    # Извлекаем текст
    text = soup.get_text(separator="\n", strip=True)

    # Чистим пустые/мусорные строки
    lines = []
    prev = None
    for line in text.splitlines():
        cleaned = " ".join(line.split())
        if not cleaned:
            continue
        # убираем подряд идущие дубли
        if cleaned == prev:
            continue
        lines.append(cleaned)
        prev = cleaned

    final_text = "\n".join(lines).strip()

    # Если текста слишком мало, такая страница не подходит
    if len(final_text) < MIN_TEXT_LENGTH:
        return None

    # Формируем "чистый" html только с текстом
    paragraphs = []
    for line in lines:
        safe_line = (
            line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        paragraphs.append(f"<p>{safe_line}</p>")

    clean_html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
</head>
<body>
    <h1>{title}</h1>
    {'\n    '.join(paragraphs)}
</body>
</html>
"""
    return clean_html


def save_page(output_dir: Path, file_number: int, html_text: str) -> Path:
    """
    Сохраняет очищенную страницу в html-файл.
    """
    file_path = output_dir / f"{file_number}.html"
    file_path.write_text(html_text, encoding="utf-8")
    return file_path


def create_zip_archive(output_dir: Path, index_file: Path, zip_file: Path) -> None:
    """
    Создаёт zip-архив:
    - папка с html-файлами
    - index.txt
    """
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(output_dir.parent)
                zf.write(file_path, arcname)

        if index_file.exists():
            zf.write(index_file, index_file.name)


def main():
    urls = read_urls(URLS_FILE)

    if not urls:
        print("Список ссылок пуст.")
        return

    output_dir = Path(OUTPUT_DIR)

    # очищаем старую папку, чтобы не было старых файлов
    if output_dir.exists():
        for file_path in output_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    index_lines = []

    print(f"Найдено ссылок: {len(urls)}")
    print(f"Нужно скачать минимум: {MIN_PAGES}")
    print("-" * 50)

    for url in urls:
        if saved_count >= MIN_PAGES:
            break

        print(f"[LOAD] {url}")
        raw_html = download_page(url)

        if raw_html is None:
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        clean_html = extract_clean_text_html(raw_html, url)

        if clean_html is None:
            print(f"[SKIP] Недостаточно текстового содержимого: {url}")
            time.sleep(DELAY_BETWEEN_REQUESTS)
            continue

        saved_count += 1
        save_page(output_dir, saved_count, clean_html)
        index_lines.append(f"{saved_count}.html\t{url}")

        print(f"[OK] Сохранено: {saved_count}.html")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    # сохраняем индекс
    index_path = Path(INDEX_FILE)
    index_path.write_text("\n".join(index_lines), encoding="utf-8")

    # создаём zip
    zip_path = Path(ZIP_FILE)
    create_zip_archive(output_dir, index_path, zip_path)

    print("-" * 50)
    print(f"Итог: скачано страниц = {saved_count}")
    print(f"Папка с файлами: {output_dir.resolve()}")
    print(f"Индекс-файл: {index_path.resolve()}")
    print(f"ZIP-архив: {zip_path.resolve()}")

    if saved_count < MIN_PAGES:
        print(
            f"ВНИМАНИЕ: удалось скачать только {saved_count} страниц "
            f"из требуемых {MIN_PAGES}."
        )


if __name__ == "__main__":
    main()