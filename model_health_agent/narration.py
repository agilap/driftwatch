"""Optional provider-neutral narration of an existing report."""

import json
import logging
import os
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import httpx

logger = logging.getLogger(__name__)
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)")


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield Decimal(str(value))
    elif isinstance(value, str):
        for match in NUMBER.findall(value):
            yield Decimal(match)


def _supported(narrative, report):
    evidence = list(_numbers(report))
    for text in NUMBER.findall(narrative):
        number = Decimal(text)
        if number == number.to_integral_value() and 1 <= number <= 5:
            continue
        precision = Decimal(1).scaleb(number.as_tuple().exponent)
        try:
            if not any(
                item.quantize(precision, rounding=ROUND_HALF_UP) == number
                for item in evidence
            ):
                return False
        except InvalidOperation:
            return False
    return True


async def narrate_report(report, client=None):
    base = os.getenv("DRIFTWATCH_LLM_BASE_URL")
    model = os.getenv("DRIFTWATCH_LLM_MODEL")
    key = os.getenv("DRIFTWATCH_LLM_API_KEY")
    if not all((base, model, key)):
        return report
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Explain the structured report. Do not add evidence or claims not present in it.",
            },
            {"role": "user", "content": json.dumps(report, sort_keys=True)},
        ],
    }
    try:

        async def request(http):
            response = await http.post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Narrative is not text")
            return content

        if client is None:
            async with httpx.AsyncClient(timeout=30) as http:
                narrative = await request(http)
        else:
            narrative = await request(client)
        # This guard checks numbers, not meaning.
        if not _supported(narrative, report):
            return {
                **report,
                "narrative_rejected": "Narrative contains a number unsupported by the report.",
            }
        return {**report, "mode": "llm", "narrative": narrative}
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        logger.warning("Narration failed: %s", exc)
        return report
