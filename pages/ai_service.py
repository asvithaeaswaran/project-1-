import os
import re
import json
import math
from django.conf import settings


GROQ_MODELS = [
    'openai/gpt-oss-120b',
    'qwen/qwen3.6-27b',
    'openai/gpt-oss-20b',
    'groq/compound',
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'mixtral-8x7b-32768',
    'gemma2-9b-it'
]

GEMINI_MODELS = [
    'gemini-2.0-flash',
    'gemini-1.5-flash'
]

OPENAI_MODELS = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-3.5-turbo',
    'gpt-4-turbo'
]


def get_configured_api_key(request=None):
    """
    Look for API key in:
    1. User's active session
    2. Environment variables
    3. Django settings
    """

    if request and request.session.get('openai_api_key'):
        return request.session.get('openai_api_key').strip()

    for var_name in [
        'OPENAI_API_KEY',
        'GROQ_API_KEY',
        'GEMINI_API_KEY',
        'GOOGLE_API_KEY',
        'DEEPSEEK_API_KEY',
        'OPENROUTER_API_KEY'
    ]:
        val = os.environ.get(var_name)

        if val and val.strip():
            return val.strip()

    return getattr(settings, 'OPENAI_API_KEY', None)


def detect_provider_and_base_url(
    api_key,
    custom_base_url=None,
    custom_model=None
):
    """
    Auto-detect provider, base URL, and compatible model.
    """

    if not api_key:
        return None, None, None

    key = api_key.strip()

    # Groq
    if key.startswith('gsk_'):
        base_url = (
            custom_base_url
            or 'https://api.groq.com/openai/v1'
        )

        model = (
            custom_model
            if custom_model
            and (
                'gpt-oss' in custom_model
                or 'qwen' in custom_model
                or 'llama' in custom_model
                or 'groq' in custom_model
            )
            else 'openai/gpt-oss-120b'
        )

        return base_url, model, 'Groq (Free Live AI)'

    # Google Gemini
    if key.startswith('AIza'):
        base_url = (
            custom_base_url
            or 'https://generativelanguage.googleapis.com/v1beta/openai/'
        )

        model = (
            custom_model
            if custom_model in GEMINI_MODELS
            else 'gemini-2.0-flash'
        )

        return base_url, model, 'Google Gemini'

    # OpenRouter
    if key.startswith('sk-or-'):
        base_url = (
            custom_base_url
            or 'https://openrouter.ai/api/v1'
        )

        model = (
            custom_model
            if custom_model and '/' in custom_model
            else 'meta-llama/llama-3.3-70b-instruct:free'
        )

        return base_url, model, 'OpenRouter'

    # Custom provider
    if custom_base_url and custom_base_url.strip():
        return (
            custom_base_url.strip(),
            custom_model or 'gpt-4o-mini',
            'Custom Provider'
        )

    # OpenAI
    model = (
        custom_model
        if custom_model in OPENAI_MODELS
        else 'gpt-4o-mini'
    )

    return (
        'https://api.openai.com/v1',
        model,
        'OpenAI'
    )


def get_configured_model(request=None):
    if request and request.session.get('openai_model'):
        return request.session.get('openai_model')

    return os.environ.get('OPENAI_MODEL', None)


def get_configured_base_url(request=None):
    if request and request.session.get('openai_base_url'):
        return request.session.get('openai_base_url')

    return os.environ.get('OPENAI_BASE_URL', None)


def generate_ai_response(
    conversation,
    new_user_message_content,
    request=None,
    document_context=None
):
    """
    Generate AI response using live LLM.

    Optimized for lower token usage:
    - Last 6 conversation messages only
    - Documents limited to 20,000 characters
    - Short system prompt
    - Maximum 800 output tokens
    """

    api_key = get_configured_api_key(request)

    custom_model = get_configured_model(request)

    custom_base_url = get_configured_base_url(request)

    # --------------------------------------------------
    # DOCUMENT CONTEXT
    # --------------------------------------------------

    if document_context and document_context.strip():

        doc_text = document_context.strip()

        # Reduced from 60,000 to 20,000 characters
        if len(doc_text) > 20000:
            doc_text = (
                doc_text[:20000]
                + "\n\n...[Document content truncated]..."
            )

        full_user_prompt = (
            f"Document:\n"
            f"```\n{doc_text}\n```\n\n"
            f"Question:\n"
            f"{new_user_message_content}"
        )

    else:
        full_user_prompt = new_user_message_content

    # --------------------------------------------------
    # LIVE API
    # --------------------------------------------------

    if api_key and api_key.strip():

        base_url, model, provider_name = (
            detect_provider_and_base_url(
                api_key,
                custom_base_url,
                custom_model
            )
        )

        # --------------------------------------------------
        # SHORT SYSTEM PROMPT
        # --------------------------------------------------

        messages_payload = [
            {
                "role": "system",
                "content": (
                    "You are Gelato, a helpful AI assistant. "
                    "Answer accurately and clearly. "
                    "Use provided document content when relevant. "
                    "Use concise Markdown and code blocks when needed."
                )
            }
        ]

        # --------------------------------------------------
        # LAST 6 MESSAGES ONLY
        # --------------------------------------------------

        history_messages = (
            conversation.messages
            .all()
            .order_by('-created_at')[:6]
        )

        # Restore chronological order
        history_messages = reversed(list(history_messages))

        for msg in history_messages:

            if msg.role in ['user', 'assistant', 'system']:

                messages_payload.append(
                    {
                        "role": msg.role,
                        "content": msg.content
                    }
                )

        # --------------------------------------------------
        # ADD CURRENT USER MESSAGE
        # --------------------------------------------------

        if not conversation.messages.filter(
            role='user',
            content=full_user_prompt
        ).exists():

            messages_payload.append(
                {
                    "role": "user",
                    "content": full_user_prompt
                }
            )

        # --------------------------------------------------
        # MODEL FALLBACK LIST
        # --------------------------------------------------

        candidate_models = [model]

        if 'groq' in (base_url or '').lower():

            candidate_models += [
                m for m in GROQ_MODELS
                if m != model
            ]

        elif 'openai.com' in (base_url or '').lower():

            candidate_models += [
                m for m in [
                    'gpt-4o-mini',
                    'gpt-3.5-turbo',
                    'gpt-4o',
                    'gpt-4'
                ]
                if m != model
            ]

        elif 'googleapis' in (base_url or '').lower():

            candidate_models += [
                m for m in GEMINI_MODELS
                if m != model
            ]

        # --------------------------------------------------
        # CALL API
        # --------------------------------------------------

        last_error = None

        for candidate_model in candidate_models:

            try:

                from openai import OpenAI

                client_kwargs = {
                    'api_key': api_key.strip()
                }

                if base_url:
                    client_kwargs['base_url'] = base_url

                client = OpenAI(**client_kwargs)

                response = client.chat.completions.create(
                    model=candidate_model,
                    messages=messages_payload,

                    # Lower output randomness
                    temperature=0.5,

                    # Reduced from 2048
                    max_tokens=800,
                )

                assistant_reply = (
                    response
                    .choices[0]
                    .message
                    .content
                )

                return {
                    "content": assistant_reply,
                    "provider": provider_name,
                    "model": candidate_model,
                    "status": "success"
                }

            except Exception as e:

                last_error = str(e)

                continue

        # --------------------------------------------------
        # BUILT-IN FALLBACK
        # --------------------------------------------------

        local_reply = generate_builtin_response(
            new_user_message_content,
            conversation,
            document_context
        )

        disclaimer = (
            f"\n\n> ⚠️ *Live API Notice:* "
            f"`{last_error}`. "
            f"*Answered using Gelato built-in engine.*"
        )

        return {
            "content": local_reply + disclaimer,
            "provider": f"{provider_name} (Fallback)",
            "model": model,
            "status": "fallback_error",
            "error": last_error
        }

    # --------------------------------------------------
    # NO API KEY
    # --------------------------------------------------

    local_reply = generate_builtin_response(
        new_user_message_content,
        conversation,
        document_context
    )

    return {
        "content": local_reply,
        "provider": "Gelato Built-in Engine",
        "model": "built-in-ai",
        "status": "success"
    }


# ======================================================
# BUILT-IN KNOWLEDGE BASE & DOCUMENT READER
# ======================================================

CAPITALS_DATA = {

    # Indian States & UTs

    "tamil nadu": ("Chennai", "State of India"),
    "tamilnadu": ("Chennai", "State of India"),
    "andhra pradesh": ("Amaravati", "State of India"),
    "arunachal pradesh": ("Itanagar", "State of India"),
    "assam": ("Dispur", "State of India"),
    "bihar": ("Patna", "State of India"),
    "chhattisgarh": ("Raipur", "State of India"),
    "goa": ("Panaji", "State of India"),
    "gujarat": ("Gandhinagar", "State of India"),
    "haryana": ("Chandigarh", "State of India"),
    "himachal pradesh": ("Shimla", "State of India"),
    "jharkhand": ("Ranchi", "State of India"),
    "karnataka": ("Bengaluru (Bangalore)", "State of India"),
    "kerala": ("Thiruvananthapuram (Trivandrum)", "State of India"),
    "madhya pradesh": ("Bhopal", "State of India"),
    "maharashtra": ("Mumbai", "State of India"),
    "manipur": ("Imphal", "State of India"),
    "meghalaya": ("Shillong", "State of India"),
    "mizoram": ("Aizawl", "State of India"),
    "nagaland": ("Kohima", "State of India"),
    "odisha": ("Bhubaneswar", "State of India"),
    "punjab": ("Chandigarh", "State of India"),
    "rajasthan": ("Jaipur", "State of India"),
    "sikkim": ("Gangtok", "State of India"),
    "telangana": ("Hyderabad", "State of India"),
    "tripura": ("Agartala", "State of India"),
    "uttar pradesh": ("Lucknow", "State of India"),
    "uttarakhand": ("Dehradun", "State of India"),
    "west bengal": ("Kolkata", "State of India"),
    "delhi": ("New Delhi", "National Capital Territory of India"),

    # World Countries

    "india": ("New Delhi", "Country in South Asia"),
    "france": ("Paris", "Country in Western Europe"),
    "united states": ("Washington, D.C.", "Country in North America"),
    "usa": ("Washington, D.C.", "Country in North America"),
    "united kingdom": ("London", "Country in Northwestern Europe"),
    "uk": ("London", "Country in Northwestern Europe"),
    "england": ("London", "Country in the United Kingdom"),
    "germany": ("Berlin", "Country in Central Europe"),
    "italy": ("Rome", "Country in Southern Europe"),
    "spain": ("Madrid", "Country in Southwestern Europe"),
    "japan": ("Tokyo", "Country in East Asia"),
    "china": ("Beijing", "Country in East Asia"),
    "russia": ("Moscow", "Transcontinental Country"),
    "canada": ("Ottawa", "Country in North America"),
    "australia": ("Canberra", "Country in Oceania"),
    "brazil": ("Brasília", "Country in South America"),
    "south africa": (
        "Pretoria (Admin), Cape Town (Legis), Bloemfontein (Judicial)",
        "Country in Africa"
    ),
    "egypt": ("Cairo", "Country in North Africa"),
    "saudi arabia": ("Riyadh", "Country in Middle East"),
    "uae": ("Abu Dhabi", "Country in Middle East"),
    "united arab emirates": ("Abu Dhabi", "Country in Middle East"),
    "singapore": (
        "Singapore",
        "Island city-state in Southeast Asia"
    ),
    "malaysia": ("Kuala Lumpur", "Country in Southeast Asia"),
    "indonesia": ("Jakarta", "Country in Southeast Asia"),
    "thailand": ("Bangkok", "Country in Southeast Asia"),
    "vietnam": ("Hanoi", "Country in Southeast Asia"),
    "south korea": ("Seoul", "Country in East Asia"),
    "north korea": ("Pyongyang", "Country in East Asia"),
    "switzerland": ("Bern", "Country in Europe"),
    "netherlands": ("Amsterdam", "Country in Europe"),
    "sweden": ("Stockholm", "Country in Northern Europe"),
    "norway": ("Oslo", "Country in Northern Europe"),
    "denmark": ("Copenhagen", "Country in Northern Europe"),
    "finland": ("Helsinki", "Country in Northern Europe"),
    "greece": ("Athens", "Country in Southeastern Europe"),
    "turkey": ("Ankara", "Country between Europe and Asia"),
    "mexico": ("Mexico City", "Country in North America"),
    "argentina": ("Buenos Aires", "Country in South America"),
    "new zealand": ("Wellington", "Country in Oceania"),
    "sri lanka": (
        "Sri Jayawardenepura Kotte / Colombo",
        "Country in South Asia"
    ),
    "pakistan": ("Islamabad", "Country in South Asia"),
    "bangladesh": ("Dhaka", "Country in South Asia"),
    "nepal": ("Kathmandu", "Country in South Asia"),
    "bhutan": ("Thimphu", "Country in South Asia"),
}


FACTS_DATA = {

    "largest planet":
        "The largest planet in our solar system is **Jupiter**.",

    "smallest planet":
        "The smallest planet in our solar system is **Mercury**.",

    "speed of light":
        "The speed of light in a vacuum is **299,792,458 meters per second** (~3 × 10⁸ m/s).",

    "speed of sound":
        "The speed of sound in air is approximately **343 meters per second** (1,235 km/h).",

    "tallest mountain":
        "The tallest mountain on Earth is **Mount Everest** (8,848.86 meters / 29,031.7 ft).",

    "longest river":
        "The longest river in the world is the **Nile River** (~6,650 km / 4,132 miles).",

    "deepest ocean":
        "The deepest ocean is the **Pacific Ocean** (Mariana Trench at ~11,034 meters deep).",

    "boiling point of water":
        "The boiling point of water is **100°C (212°F)** at standard atmospheric pressure.",

    "freezing point of water":
        "The freezing point of water is **0°C (32°F)**.",

    "father of computer":
        "**Charles Babbage** is known as the *Father of the Computer*.",

    "who invented python":
        "Python was created by **Guido van Rossum** in **1991**.",

    "who invented c":
        "The C programming language was created by **Dennis Ritchie** at Bell Labs in **1972**.",

    "who invented world wide web":
        "The World Wide Web was invented by **Sir Tim Berners-Lee** in **1989** at CERN.",

    "author of romeo and juliet":
        "**William Shakespeare** wrote *Romeo and Juliet*.",

    "who wrote romeo and juliet":
        "**William Shakespeare** wrote *Romeo and Juliet*.",

    "who is prime minister of india":
        "The Prime Minister of India is **Narendra Modi**.",

    "who is president of usa":
        "The President of the United States is the head of state and head of government of the USA.",

    "how many states in india":
        "India has **28 states** and **8 Union Territories**.",
}


def try_solve_math(prompt):

    clean = prompt.lower()

    clean = re.sub(
        r'^(what is|calculate|evaluate|solve|compute|\s)+',
        '',
        clean
    )

    clean = clean.strip().rstrip('?').strip()

    if re.match(
        r'^[\d\s\+\-\*\/\^\%\(\)\.\,]+$',
        clean
    ):

        try:

            expr = (
                clean
                .replace('^', '**')
                .replace(',', '')
            )

            allowed_names = {
                "sqrt": math.sqrt,
                "pi": math.pi,
                "sin": math.sin,
                "cos": math.cos,
                "pow": pow
            }

            code = compile(
                expr,
                "<string>",
                "eval"
            )

            for name in code.co_names:

                if name not in allowed_names:
                    return None

            result = eval(
                code,
                {"__builtins__": {}},
                allowed_names
            )

            return (
                f"**Result:**\n\n"
                f"$$\n"
                f"{clean} = {result}\n"
                f"$$"
            )

        except Exception:
            return None

    return None


def generate_builtin_response(
    prompt,
    conversation=None,
    document_context=None
):

    p_clean = prompt.strip()

    p_lower = p_clean.lower()

    # --------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------

    if document_context and document_context.strip():

        lines = [
            line.strip()
            for line in document_context.splitlines()
            if line.strip()
        ]

        preview_text = "\n".join(lines[:15])

        return (
            f"### Document Analysis\n\n"
            f"I reviewed the attached document "
            f"({len(document_context)} characters, "
            f"~{len(lines)} lines).\n\n"
            f"**Key Content Preview:**\n"
            f"```\n"
            f"{preview_text}\n"
            f"...\n"
            f"```\n\n"
            f"**Response regarding '{p_clean}':**\n"
            f"Based on the document content, "
            f"the information above is relevant to your request.\n\n"
            f"*(Live AI models can provide deeper document analysis.)*"
        )

    # --------------------------------------------------
    # MATH
    # --------------------------------------------------

    math_res = try_solve_math(p_clean)

    if math_res:
        return math_res

    # --------------------------------------------------
    # CAPITALS
    # --------------------------------------------------

    if "capital" in p_lower:

        for place, (capital, desc) in CAPITALS_DATA.items():

            if place in p_lower:

                return (
                    f"The capital of **{place.title()}** "
                    f"is **{capital}**.\n\n"
                    f"- **Region:** {desc}\n"
                    f"- **Capital City:** {capital}"
                )

    # --------------------------------------------------
    # FACTS
    # --------------------------------------------------

    for key, answer in FACTS_DATA.items():

        if key in p_lower:
            return answer

    # --------------------------------------------------
    # GREETINGS
    # --------------------------------------------------

    if p_lower in [
        "hello",
        "hi",
        "hey",
        "greetings",
        "good morning",
        "good evening",
        "how are you",
        "who are you"
    ]:

        return (
            "Hello! 👋 I am **Gelato**, your AI assistant.\n\n"
            "How can I help you today?"
        )

    # --------------------------------------------------
    # CODING
    # --------------------------------------------------

    if any(
        k in p_lower
        for k in [
            "code",
            "python",
            "javascript",
            "script",
            "function",
            "def ",
            "class ",
            "sql",
            "html",
            "css",
            "django",
            "react",
            "algorithm"
        ]
    ):

        if "fibonacci" in p_lower:

            return (
                "Here is an efficient **Fibonacci** implementation "
                "in Python:\n\n"
                "```python\n"
                "def fibonacci(n):\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    if n == 1:\n"
                "        return [0]\n\n"
                "    sequence = [0, 1]\n"
                "    while len(sequence) < n:\n"
                "        sequence.append("
                "sequence[-1] + sequence[-2])\n"
                "    return sequence\n"
                "```\n\n"
                "Time: **O(n)**"
            )

        elif "palindrome" in p_lower:

            return (
                "Here is a Python function to check a "
                "**palindrome**:\n\n"
                "```python\n"
                "def is_palindrome(s):\n"
                "    clean = ''.join("
                "c.lower() for c in str(s) "
                "if c.isalnum())\n"
                "    return clean == clean[::-1]\n"
                "```"
            )

        elif "sql" in p_lower:

            return (
                "```sql\n"
                "SELECT users.id, users.username, "
                "COUNT(orders.id) AS total_orders\n"
                "FROM users\n"
                "LEFT JOIN orders "
                "ON users.id = orders.user_id\n"
                "GROUP BY users.id, users.username\n"
                "ORDER BY total_orders DESC;\n"
                "```"
            )

        else:

            return (
                f"Here is a Python implementation for "
                f"**{p_clean}**:\n\n"
                "```python\n"
                "def solve():\n"
                f"    # Implementation for: {p_clean}\n"
                "    data = [1, 2, 3, 4, 5]\n"
                "    return [x * 2 for x in data]\n\n"
                "if __name__ == '__main__':\n"
                "    print(solve())\n"
                "```"
            )

    # --------------------------------------------------
    # DEFAULT
    # --------------------------------------------------

    return (
        f"**Answer:** *{p_clean}*\n\n"
        "I don't have enough built-in information to answer "
        "this fully. Please provide more details or use a "
        "configured AI API."
    )
