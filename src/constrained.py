"""Ограничения записи JSON-формулы без эталона и выбора правильного смысла."""

from functools import lru_cache
import json


def literal(text):
    return ("L", text)


class FormulaGrammar:
    def __init__(self, lexicon, entities, max_depth=8):
        self.predicates = tuple(sorted((name, entry["arity"]) for name, entry in lexicon["predicates"].items()))
        self.entities = tuple(sorted(entities))
        self.max_depth = max_depth
        self.initial = frozenset({(("F", 0, ()),)})

    def expand(self, item):
        if item[0] == "T":
            env = item[1]
            return [(literal(json.dumps(["var", name], separators=(",", ":"))),) for name in env] + [
                (literal(json.dumps(["const", name], ensure_ascii=False, separators=(",", ":"))),) for name in self.entities]
        _, depth, env = item
        alternatives = []
        for name, arity in self.predicates:
            if arity and not (env or self.entities):
                continue
            parts = [literal('["pred",' + json.dumps(name, ensure_ascii=False))]
            for _ in range(arity):
                parts.extend((literal(","), ("T", env)))
            alternatives.append(tuple(parts + [literal("]")]))
        if depth >= self.max_depth:
            return alternatives
        # Не создаём ветку, у которой на пределе глубины нет допустимых термов.
        can_nest = depth + 1 < self.max_depth or bool(env or self.entities) or any(arity == 0 for _, arity in self.predicates)
        if can_nest:
            child = ("F", depth + 1, env)
            alternatives.append((literal('["not",'), child, literal("]")))
            for op in ("and", "or", "implies"):
                alternatives.append((literal('["' + op + '",'), child, literal(","), child, literal("]")))
        for variable in ("x", "y"):
            if variable not in env:
                for op in ("forall", "exists"):
                    alternatives.append((literal('["' + op + '","' + variable + '",'),
                                         ("F", depth + 1, env + (variable,)), literal("]")))
        return alternatives

    @lru_cache(maxsize=8192)
    def frontier(self, states):
        pending, seen, output = list(states), set(), set()
        while pending:
            stack = pending.pop()
            if stack in seen:
                continue
            seen.add(stack)
            if not stack or stack[0][0] == "L":
                output.add(stack)
            else:
                pending.extend(expansion + stack[1:] for expansion in self.expand(stack[0]))
        return frozenset(output)

    @lru_cache(maxsize=16384)
    def advance(self, states, char):
        output = set()
        for stack in self.frontier(states):
            if stack and stack[0][1].startswith(char):
                remaining = stack[0][1][1:]
                output.add(((literal(remaining),) if remaining else ()) + stack[1:])
        return frozenset(output)

    @lru_cache(maxsize=4096)
    def consume(self, text):
        states, quoted, escaped = self.initial, False, False
        for char in text:
            if not quoted and char.isspace():
                continue
            states = self.advance(states, char)
            if not states:
                break
            if escaped:
                escaped = False
            elif quoted and char == "\\":
                escaped = True
            elif char == '"':
                quoted = not quoted
        return states, quoted, escaped

    def accepts(self, text):
        states, quoted, _ = self.consume(text)
        return not quoted and () in self.frontier(states)


def grammar_characters(lexicon, entities):
    return set('[]",: \\ntxy predvarconstnotandorimpliesforallexists' + "".join(lexicon["predicates"]) + "".join(entities))


class TokenTrie:
    def __init__(self, tokenizer, allowed_characters=None):
        self.root = {}
        special = set(tokenizer.all_special_ids)
        for identifier in sorted(set(tokenizer.get_vocab().values())):
            if identifier in special:
                continue
            piece = tokenizer.convert_ids_to_tokens(identifier).replace("▁", " ")
            if not piece.strip() or piece.startswith("<"):
                continue
            if allowed_characters is not None and not set(piece) <= allowed_characters:
                continue
            node = self.root
            for char in piece:
                node = node.setdefault(char, {})
            node.setdefault(None, []).append(identifier)


class ConstrainedDecoder:
    def __init__(self, tokenizer, lexicon, entities, *, trie=None):
        self.tokenizer = tokenizer
        self.grammar = FormulaGrammar(lexicon, entities)
        self.trie = TokenTrie(tokenizer, grammar_characters(lexicon, entities)) if trie is None else trie

    def allowed(self, identifiers):
        return self._allowed(tuple(identifiers))

    @lru_cache(maxsize=4096)
    def _allowed(self, identifiers):
        ids = list(identifiers)
        if ids and ids[0] == self.tokenizer.pad_token_id:
            ids = ids[1:]
        if self.tokenizer.eos_token_id in ids:
            return [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]
        text = self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        states, quoted, escaped = self.grammar.consume(text)
        allowed = []
        if self.grammar.accepts(text):
            allowed.append(self.tokenizer.eos_token_id)
        pending = [(self.trie.root, states, quoted, escaped, False)]
        while pending:
            node, current, inside, escape, progress = pending.pop()
            if progress:
                for identifier in node.get(None, []):
                    candidate = self.tokenizer.decode(ids + [identifier], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    if self.grammar.consume(candidate)[0] and candidate != text:
                        allowed.append(identifier)
            for char, child in node.items():
                if char is None:
                    continue
                if not inside and char.isspace():
                    pending.append((child, current, inside, escape, progress))
                    continue
                after = self.grammar.advance(current, char)
                if not after:
                    continue
                next_inside, next_escape = inside, False
                if escape:
                    pass
                elif inside and char == "\\":
                    next_escape = True
                elif char == '"':
                    next_inside = not inside
                pending.append((child, after, next_inside, next_escape, True))
        if not allowed:
            raise ValueError("Нет допустимого продолжения формулы; произвольный ответ не подставляется")
        return sorted(set(allowed))


def constraints_from_entities(tokenizer, lexicon, entities_by_id):
    """Только публичный словарь и таблицы имён; правильные ответы не принимаются."""
    all_entities = {key: value for entities in entities_by_id.values() for key, value in entities.items()}
    trie = TokenTrie(tokenizer, grammar_characters(lexicon, all_entities))
    decoders, result = {}, {}
    for identifier, entities in entities_by_id.items():
        key = tuple(sorted(entities))
        if key not in decoders:
            decoders[key] = ConstrainedDecoder(tokenizer, lexicon, entities, trie=trie)
        result[identifier] = decoders[key]
    return result
