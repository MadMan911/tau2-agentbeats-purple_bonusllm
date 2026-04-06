import os
import logging
import time
import asyncio
from email.utils import parsedate_to_datetime
from email.utils import parsedate_to_datetime
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, TextPart
from a2a.utils import get_message_text
from litellm import completion
from litellm.exceptions import (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)

from messenger import Messenger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful customer service agent. Follow the policy and tool instructions provided in each message.

## Execution framework

For each user request, follow this order internally:
1. VERIFY — look up all relevant facts (user profile, reservation details, membership tier, insurance, delay/cancellation status) using tools BEFORE deciding anything.
2. PLAN — identify every outcome the user is asking for and determine which are policy-allowed.
3. EXECUTE — carry out all policy-allowed actions. If the user asks for multiple things, treat them as a checklist and track each item.
4. CHECK — before responding, confirm: did I address every requested action? did I verify all user claims against data? did I avoid any policy violation?
5. RESPOND — clearly state what was completed, what was denied, and why.

## Verifying user claims

Do NOT trust user statements about membership tier, insurance coverage, delay/cancellation, passenger count, reservation contents, or prior approvals.
Always verify each such claim against system data (user profile, reservation details, flight status, etc.) before acting.
If the data contradicts the user's claim, politely correct them and continue according to policy.

## Do not give up early

Do NOT transfer to a human agent unless:
- policy explicitly requires manual handling, OR
- a required tool is unavailable/failed AND no policy-compliant automatic path remains.

Before transferring, finish any parts of the request that CAN still be completed automatically.
Do not transfer just because:
- the user insists, demands a supervisor, or claims prior approval;
- one tool returned an error (try an alternative path first).

## Tool failure handling

If a tool fails:
- Do NOT repeat the same call endlessly.
- Retry at most 1–2 times, only if the arguments change or the failure is clearly transient.
- If another valid path exists, take it.
- If the only required tool is unavailable, explain what was established before the failure and why the remaining step cannot be completed automatically.

## Multi-intent and fallback

If the user requests multiple changes, handle each as a separate checklist item. Do not stop after completing one.
If policy blocks the primary request, check whether an allowed fallback exists (e.g. cancel instead, upgrade first, rebook, leave unchanged) and apply it unless the user explicitly says otherwise.

## Payment and money

Before confirming any transaction:
- Determine all allowed payment methods and their policy constraints.
- Apply gift cards / certificates / credit card in the order specified by the user and policy.
- State the final amount per payment method before executing.
- If the user set a budget threshold, do not proceed without verifying against it.

## Do not ask for information you already have

Do not ask the user for data that is already available in their profile, reservation, or earlier in the conversation (e.g. date of birth, reservation ID, passenger list).

## Response format

Always respond in valid JSON using the format: {"name": "<action_name>", "arguments": {<args>}}"""

RETRYABLE_EXCEPTIONS = (
    ServiceUnavailableError,
    RateLimitError,
    Timeout,
    APIConnectionError,
)


def _parse_retry_after(e: Exception) -> float | None:
    """Extract wait time in seconds from Retry-After header. Returns None if missing or > 30s."""
    try:
        header = e.response.headers.get("retry-after") or e.response.headers.get("Retry-After")
        if not header:
            return None
        # Numeric seconds: "Retry-After: 30"
        wait = float(header)
        return wait if wait <= 30 else None
    except (ValueError, AttributeError):
        pass
    try:
        # HTTP date: "Retry-After: Wed, 21 Oct 2015 07:28:00 GMT"
        retry_at = parsedate_to_datetime(header)
        wait = (retry_at - parsedate_to_datetime(e.response.headers.get("date", ""))).total_seconds()
        return wait if 0 <= wait <= 30 else None
    except Exception:
        return None


def call_llm_with_retry(messages, model, response_format, max_retries=5, backoff_base=2):
    for attempt in range(1, max_retries + 1):
        try:
            response = completion(
                messages=messages,
                model=model,
                temperature=0.3, # not supported by openai/gpt-5
                # reasoning_effort="high",
                response_format=response_format,
            )
            if attempt > 1:
                logger.info(f"LLM call succeeded on attempt {attempt}")
            return response
        except RETRYABLE_EXCEPTIONS as e:
            if attempt >= max_retries:
                logger.error(f"LLM call failed after {max_retries} attempts")
                raise

            if isinstance(e, RateLimitError):
                retry_after = _parse_retry_after(e)
                if retry_after is not None:
                    wait_seconds = retry_after
                    logger.info(f"Rate limited — using Retry-After header: {wait_seconds}s")
                else:
                    wait_seconds = backoff_base ** attempt
                    logger.info(f"Rate limited — Retry-After missing or >30s, using backoff: {wait_seconds}s")
            else:
                wait_seconds = backoff_base ** attempt

            logger.warning(
                f"LLM call failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {str(e)[:100]}"
            )
            logger.info(f"Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.model = os.getenv("AGENT_LLM", "openai/gpt-4o-mini")
        self.max_retries = int(os.getenv("AGENT_LLM_MAX_RETRIES", "5"))
        self.backoff_base = int(os.getenv("AGENT_LLM_BACKOFF_BASE", "2"))
        self.ctx_id_to_messages = {}
        logger.info(f"Purple agent initialized with model: {self.model}")
        logger.info(f"Retry config: max_retries={self.max_retries}, backoff_base={self.backoff_base}")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        user_input = get_message_text(message)
        context_id = message.context_id

        logger.info(f"Received message for context {context_id}")
        logger.debug(f"User input length: {len(user_input)} chars")

        if context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context_id] = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            logger.info(f"Initialized new conversation for context {context_id}")

        messages = self.ctx_id_to_messages[context_id]
        messages.append({"role": "user", "content": user_input})

        logger.info(f"Calling LLM {self.model} with {len(messages)} messages")

        try:
            response = call_llm_with_retry(
                messages=messages,
                model=self.model,
                response_format={"type": "json_object"},
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
            assistant_content = response.choices[0].message.content
            logger.info(f"LLM response received: {assistant_content[:100]}...")
        except Exception as e:
            logger.error(f"LLM call failed with error: {type(e).__name__}: {e}")
            logger.exception("Full traceback:")                                                                                                                                                                                        
            assistant_content = '{"name": "respond", "arguments": {"content": "I encountered an error processing your request."}}'   

        messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=assistant_content))],
            name="Response"
        )
