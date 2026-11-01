"""Генерация только из разрешённых входных токенов. Эталоны не читаются."""


def generation_call(model, batch, max_new_tokens, *, prefix_allowed_tokens_fn=None):
    if set(batch) != {"input_ids", "attention_mask"}:
        raise ValueError("При генерации разрешены только входные токены и маска, без labels или префикса ответа")
    options = {} if prefix_allowed_tokens_fn is None else {"prefix_allowed_tokens_fn": prefix_allowed_tokens_fn}
    return model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True, **options)


def decode_generated(ids, tokenizer, decoder_start_id):
    ids = list(ids)
    if ids and ids[0] == decoder_start_id:
        ids = ids[1:]
    while ids and ids[-1] == tokenizer.pad_token_id:
        ids.pop()
    if ids and ids[-1] == tokenizer.eos_token_id:
        ids.pop()
    return tokenizer.decode(ids, skip_special_tokens=False)


def generate_responses(model, tokenizer, inputs, batch_size=4, *, constraints=None):
    import torch
    if any(set(row) != {"id", "input_ids"} for row in inputs):
        raise ValueError("Генератор не принимает цели, метки или дерево происхождения")
    model.eval()
    results = []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            subset = inputs[start:start + batch_size]
            width = (max(len(row["input_ids"]) for row in subset) + 7) // 8 * 8
            batch = {
                "input_ids": [r["input_ids"] + [tokenizer.pad_token_id] * (width - len(r["input_ids"])) for r in subset],
                "attention_mask": [[1] * len(r["input_ids"]) + [0] * (width - len(r["input_ids"])) for r in subset],
            }
            batch = {key: torch.tensor(value, device=device) for key, value in batch.items()}
            prefix_fn = None
            if constraints is not None:
                prefix_fn = lambda batch_id, ids: constraints[subset[batch_id]["id"]].allowed(ids.tolist())
            outputs = generation_call(model, batch, 128, prefix_allowed_tokens_fn=prefix_fn)
            for row, output in zip(subset, outputs):
                ids = output.cpu().tolist()
                results.append({"id": row["id"], "token_ids": ids,
                                "raw_response": decode_generated(ids, tokenizer, model.config.decoder_start_token_id)})
    return results
