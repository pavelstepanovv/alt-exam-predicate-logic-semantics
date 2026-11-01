"""Локальный интерфейс перевода в логику предикатов."""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import webbrowser

from .lexicon import load_lexicon
from .logic import format_formula
from .output import parse_response
from .serialization import DEFAULT_LIMITS, tokenize_input
from .text import normalize_sentence

ROOT = Path(__file__).resolve().parents[1]


EXAMPLES = [
    ("Два участника", "Мария уважает Анну."),
    ("Другой порядок слов", "Анну уважает Мария."),
    ("Отрицание", "Ни один учитель не является аккуратным."),
    ("Два действия", "Кира одновременно танцует и читает."),
    ("Два квантора", "Каждый врач знает хотя бы одного студента."),
    ("Общий объект", "Каждый преподаватель рецензирует одну и ту же статью."),
    ("Новые имена", "Олега консультирует Алиса."),
    ("Новое название", "Каждый лабубу любит Алису."),
]

RELATIONS = {"root": "Главное слово", "nsubj": "Кто выполняет действие", "obj": "На кого направлено действие",
             "acl:relcl": "Уточнение", "det": "Определение", "nummod": "Количество", "advmod": "Уточняет",
             "fixed": "Часть выражения", "conj": "Однородная часть", "cc": "Союз", "cop": "Связка", "xcomp": "Свойство"}


def display_analysis(record):
    if record["status"] != "ok":
        return []
    by_id = {t["id"]: t["text"] for t in record["tokens"]}
    return [{"word": t["text"], "lemma": t["lemma"], "head": by_id.get(t["head_id"], "—"),
             "relation": RELATIONS.get(t["rel"], "Связь со словом")}
            for t in record["tokens"] if t["pos"] != "PUNCT"]


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Translator:
    def __init__(self):
        self.config = json.loads((ROOT / "models/model.json").read_text(encoding="utf-8"))
        if self.config["representation"] != "abstract-roles-v2":
            raise ValueError("Неподдерживаемый формат модели.")
        self.lexicon = load_lexicon()
        self.model = None

    def load(self):
        if self.model is not None:
            return
        base_path = ROOT / self.config["base_directory"]
        for name, expected in self.config["base_sha256"].items():
            path = base_path / name
            if not path.is_file():
                raise ValueError("Сначала скачайте базовую модель: python scripts/download_model.py")
            if file_digest(path) != expected:
                raise ValueError("Файл базовой модели повреждён: " + name)
        adapter_path = ROOT / self.config["adapter_directory"]
        for name, expected in self.config["adapter_sha256"].items():
            if file_digest(adapter_path / name) != expected:
                raise ValueError("Файл обученной модели повреждён: " + name)
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer, MT5ForConditionalGeneration
        from .nlp import CachedParser, NatashaAnalyzer
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False, legacy=True, local_files_only=True)
        tokenizer.model_max_length = DEFAULT_LIMITS["max_input_tokens"]
        base = MT5ForConditionalGeneration.from_pretrained(base_path, dtype=torch.float32, local_files_only=True)
        model = PeftModel.from_pretrained(base, adapter_path, adapter_name="syntax", is_trainable=False)
        model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        model.requires_grad_(False)
        parser = CachedParser(NatashaAnalyzer(), ROOT / ".cache/parses")
        saved = json.loads((ROOT / "data/parses.json").read_text(encoding="utf-8"))
        if parser.analyzer.identity != saved["identity"]:
            raise ValueError("Версии компонентов разбора отличаются от сохранённых. Установите зависимости из requirements.txt.")
        self.reviewed = {key: {**record, "identity": saved["identity"], "parameters": saved["parameters"]}
                         for key, record in saved["records"].items()}
        self.parser, self.tokenizer, self.model = parser, tokenizer, model

    def analysis(self, text):
        from .nlp import cache_id, validate_parse
        identifier = cache_id(text, self.parser.analyzer.identity)
        record = self.reviewed.get(identifier)
        if record is not None:
            validate_parse(record)
            if record["text"] != text:
                raise ValueError("Разбор относится к другому предложению.")
            return record, "Сохранённый разбор с исправлениями"
        return self.parser.parse_text(text), "Автоматический разбор"

    def translate(self, text):
        from .abstraction import rename_formula
        from .abstract_roles import abstract_input
        from .constrained import constraints_from_entities
        from .prediction import generate_responses
        text = normalize_sentence(text)
        self.load()
        analysis, source = self.analysis(text)
        config = self.config
        # То же закреплённое распределение номеров, что в primary итогового теста.
        context = abstract_input(text, analysis, self.lexicon, seed=config["seed"])
        constraints = None
        if config.get("constrained_decoding", False):
            constraints = constraints_from_entities(self.tokenizer, context["lexicon"], {"demo": context["entities"]})
        ids = tokenize_input(context["input"], self.tokenizer, DEFAULT_LIMITS["max_input_tokens"])
        self.model.set_adapter("syntax")
        self.model.requires_grad_(False)
        response = generate_responses(self.model, self.tokenizer, [{"id": "demo", "input_ids": ids}],
                                      batch_size=1, constraints=constraints)[0]
        parsed = parse_response(response["raw_response"], context["lexicon"], context["entities"])
        valid = parsed["validation_status"] == "valid"
        formula = rename_formula(parsed["parsed_ast"], context["inverse"]) if valid else None
        result = {"formula": format_formula(formula, context["original_entities"]) if valid else None,
                  "error": None if valid else "Модель не сформировала полноценную формулу. Попробуйте другую формулировку."}
        symbols = "; ".join(symbol + " — " + context["original_entities"].get(original, original)
                            for symbol, original in sorted(context["inverse"].items()))
        return {"sentence": text, "results": {"translation": result}, "analysis": display_analysis(analysis),
                "preparation": "Обобщённая запись: " + context["normalized_text"], "symbols": symbols,
                "analysis_source": source,
                "analysis_message": "" if analysis["status"] == "ok" else "Связи недоступны; использованы слова и грамматические признаки."}


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, address, translator=None):
        super().__init__(address, Handler)
        self.translator = translator
        self.translation_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, status, body, content_type="application/json; charset=utf-8"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            return self.respond(200, Path(__file__).with_suffix(".html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/examples":
            return self.respond(200, EXAMPLES)
        if self.path == "/health":
            return self.respond(200, {"app": "predicate-translator", "representation": "abstract-roles-v2"})
        self.respond(404, {"error": "Страница не найдена."})

    def do_POST(self):
        if self.path != "/translate":
            return self.respond(404, {"error": "Страница не найдена."})
        if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"):
            return self.respond(403, {"error": "Откройте локальную страницу переводчика."})
        if not self.server.translation_lock.acquire(blocking=False):
            return self.respond(409, {"error": "Предыдущий перевод ещё выполняется."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Некорректный размер запроса.")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict) or set(body) != {"text"}:
                raise ValueError("Нужно передать одно предложение.")
            normalize_sentence(body["text"])
            if self.server.translator is None:
                self.server.translator = Translator()
            result = self.server.translator.translate(body["text"])
        except (ValueError, UnicodeError, RecursionError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            print(f"Ошибка перевода: {type(error).__name__}: {error}", flush=True)
            self.respond(500, {"error": "Не удалось выполнить перевод. Подробности указаны в окне запуска программы."})
        else:
            self.respond(200, result)
        finally:
            self.server.translation_lock.release()


def main():
    parser = argparse.ArgumentParser(description="Перевод предложения в логику предикатов")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    address = f"http://127.0.0.1:{args.port}"
    try:
        server = DemoServer(("127.0.0.1", args.port))
    except OSError:
        from urllib.request import urlopen
        with urlopen(address + "/health", timeout=2) as response:
            health = json.load(response)
            if health.get("app") != "predicate-translator":
                raise RuntimeError("Порт занят другой программой. Используйте --port с другим номером.")
            if health.get("representation") != "abstract-roles-v2":
                raise RuntimeError("Открыта старая версия переводчика. Закройте её окно запуска и откройте переводчик заново.")
        if not args.no_browser:
            webbrowser.open(address)
        print("Переводчик уже открыт:", address)
        return
    print("Переводчик:", address, "\nДля завершения нажмите Ctrl+C.", flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
