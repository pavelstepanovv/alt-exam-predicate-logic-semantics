import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.abstraction import rename_formula
from src.abstract_roles import abstract_input
from src.constrained import FormulaGrammar
from src.gui import Handler, Translator, file_digest
from src.lexicon import load_lexicon
from src.logic import format_formula, validate_formula
from src.nlp import CachedParser
from src.output import parse_response
from src.text import normalize_sentence

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {r["text"]: r for filename in ("natasha_parses.json", "open_vocabulary_parses.json", "abstract_roles_parses.json")
            for r in json.loads((ROOT / "tests/fixtures" / filename).read_text(encoding="utf-8"))}


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = load_lexicon()

    def test_sentence_validation(self):
        self.assertEqual(normalize_sentence("  Мария   уважает Анну "), "Мария уважает Анну.")
        for text in ("", "123", "а" * 401, "Мария читает. Анна пишет."):
            with self.subTest(text=text[:30]), self.assertRaises(ValueError):
                normalize_sentence(text)

    def test_only_the_published_model_is_selected_and_weights_match(self):
        translator = Translator()
        self.assertEqual(translator.config["representation"], "abstract-roles-v2")
        self.assertEqual(translator.config["selected_step"], 3500)
        self.assertIsNone(translator.model)
        for name, digest in translator.config["adapter_sha256"].items():
            self.assertEqual(file_digest(ROOT / translator.config["adapter_directory"] / name), digest)

    def test_word_order_preserves_names_and_explicit_roles(self):
        first = abstract_input("Мария уважает Анну.", FIXTURES["Мария уважает Анну."], self.lexicon)
        second = abstract_input("Анну уважает Мария.", FIXTURES["Анну уважает Мария."], self.lexicon)
        self.assertEqual(first["mapping"], second["mapping"])
        self.assertIn("дополнение", second["inline_annotations"]["1_1"])
        self.assertIn("подлежащее", second["inline_annotations"]["1_3"])
        self.assertNotIn("Мария", second["input"])

    def test_unknown_names_and_relations_use_symbols_without_word_substitution(self):
        text = "Анна любит Лабубу."
        context = abstract_input(text, FIXTURES[text], self.lexicon)
        self.assertTrue(context["mapping"]["Любить"].startswith("R"))
        self.assertEqual(len(context["entities"]), 2)
        for word in ("Анна", "Лабубу", "Консультирует"):
            self.assertNotIn(word, context["input"])

    def test_unknown_quantified_name_is_a_class(self):
        text = "Все Лабубу бегают."
        context = abstract_input(text, FIXTURES[text], self.lexicon)
        self.assertTrue(context["mapping"]["Лабубу"].startswith("C"))
        self.assertEqual(context["entities"], {})

    def test_missing_parse_does_not_invent_links(self):
        text = "Хотя бы один лабубу любит хотя бы одного студента."
        context = abstract_input(text, FIXTURES[text], self.lexicon)
        self.assertIn("SYNTAX_UNAVAILABLE", context["input"])
        self.assertTrue(all("связь=недоступна" in annotation for annotation in context["inline_annotations"].values()))

    def test_formula_validation_and_bracket_repair(self):
        formula = ["pred", "Уважает", ["const", "maria"], ["const", "anna"]]
        names = {"maria": "Мария", "anna": "Анна"}
        validate_formula(formula, self.lexicon, names)
        parsed = parse_response(json.dumps(formula)[:-1], self.lexicon, names)
        self.assertEqual(parsed["parsed_ast"], formula)
        self.assertTrue(parsed["bracket_repair"]["applied"])
        self.assertEqual(format_formula(formula, names), "Уважает(Мария, Анна)")
        with self.assertRaises(ValueError):
            validate_formula(["pred", "Уважает", ["var", "x"], ["const", "anna"]], self.lexicon, names)

    def test_generation_grammar_binds_variables_and_checks_arity(self):
        grammar = FormulaGrammar({"predicates": {"R1": {"arity": 2}}}, {"a1": "a1"})
        self.assertTrue(grammar.accepts('["forall","x",["pred","R1",["var","x"],["const","a1"]]]'))
        self.assertFalse(grammar.accepts('["pred","R1",["var","x"],["const","a1"]]'))
        self.assertFalse(grammar.accepts('["pred","R1",["const","a1"]]'))

    def test_gui_restores_names_without_correcting_wrong_roles(self):
        fixture = FIXTURES["Анну уважает Мария."]
        translator = Translator()
        translator.load, translator.model, translator.tokenizer = Mock(), Mock(), object()
        translator.analysis = Mock(return_value=(fixture, "Автоматический разбор"))
        context = abstract_input(fixture["text"], fixture, self.lexicon, seed=translator.config["seed"])
        wrong = ["pred", "Уважает", ["const", "anna"], ["const", "maria"]]
        response = json.dumps(rename_formula(wrong, context["mapping"]))
        with patch("src.gui.tokenize_input", return_value=[3, 1]) as tokenize, \
             patch("src.constrained.constraints_from_entities", return_value={"demo": object()}), \
             patch("src.prediction.generate_responses", return_value=[{"raw_response": response}]) as generate:
            result = translator.translate(fixture["text"])
        self.assertEqual(result["results"]["translation"]["formula"], "Уважает(Анна, Мария)")
        self.assertEqual(tokenize.call_args.args[0], context["input"])
        self.assertEqual(generate.call_args.args[2], [{"id": "demo", "input_ids": [3, 1]}])
        translator.model.requires_grad_.assert_called_once_with(False)

    def test_parser_cache_reuses_the_same_record(self):
        fixture = FIXTURES["Мария уважает Анну."]
        analyzer = SimpleNamespace(identity={"implementation": "test"}, analyze=Mock(return_value={k: fixture[k] for k in ("tokens", "sentences")}))
        (ROOT / ".cache").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            parser = CachedParser(analyzer, directory)
            first = parser.parse_text(fixture["text"])
            second = parser.parse_text(fixture["text"])
            self.assertEqual(first, second)
            analyzer.analyze.assert_called_once()

    def test_bad_http_requests_fail_before_model_loading(self):
        for body in ('{', '{"text":""}', '{"text":"Анна читает.","gold":[]}'):
            handler = object.__new__(Handler)
            encoded = body.encode("utf-8")
            handler.path, handler.headers = "/translate", {"Content-Length": str(len(encoded))}
            handler.rfile, handler.respond = io.BytesIO(encoded), Mock()
            handler.server = SimpleNamespace(translation_lock=threading.Lock(), translator=None, server_port=8765)
            with patch("src.gui.Translator") as translator:
                handler.do_POST()
                translator.assert_not_called()
            self.assertEqual(handler.respond.call_args.args[0], 400)
            self.assertFalse(handler.server.translation_lock.locked())


if __name__ == "__main__":
    unittest.main()
