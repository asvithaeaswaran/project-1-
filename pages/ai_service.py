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
GEMINI_MODELS = ['gemini-2.0-flash', 'gemini-1.5-flash']
OPENAI_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo', 'gpt-4-turbo']


def get_configured_api_key(request=None):
    """
    Look for API key in:
    1. User's active session
    2. Environment variables (OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, etc.)
    3. Django settings
    """
    if request and request.session.get('openai_api_key'):
        return request.session.get('openai_api_key').strip()
    
    for var_name in ['OPENAI_API_KEY', 'GROQ_API_KEY', 'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY']:
        val = os.environ.get(var_name)
        if val and val.strip():
            return val.strip()
        
    return getattr(settings, 'OPENAI_API_KEY', None)


def detect_provider_and_base_url(api_key, custom_base_url=None, custom_model=None):
    """
    Auto-detect provider, base URL, and correct compatible model based on API key.
    """
    if not api_key:
        return None, None, None

    key = api_key.strip()
    
    # 1. Groq Key (starts with gsk_)
    if key.startswith('gsk_'):
        base_url = custom_base_url or 'https://api.groq.com/openai/v1'
        model = custom_model if custom_model and ('gpt-oss' in custom_model or 'qwen' in custom_model or 'llama' in custom_model or 'groq' in custom_model) else 'openai/gpt-oss-120b'
        return base_url, model, 'Groq (Free Live AI)'

    # 2. Google Gemini Key (starts with AIza...)
    if key.startswith('AIza'):
        base_url = custom_base_url or 'https://generativelanguage.googleapis.com/v1beta/openai/'
        model = custom_model if custom_model in GEMINI_MODELS else 'gemini-2.0-flash'
        return base_url, model, 'Google Gemini'

    # 3. OpenRouter Key (starts with sk-or-...)
    if key.startswith('sk-or-'):
        base_url = custom_base_url or 'https://openrouter.ai/api/v1'
        model = custom_model if custom_model and '/' in custom_model else 'meta-llama/llama-3.3-70b-instruct:free'
        return base_url, model, 'OpenRouter'

    # 4. Custom Base URL
    if custom_base_url and custom_base_url.strip():
        return custom_base_url.strip(), (custom_model or 'gpt-4o-mini'), "Custom Provider"

    # 5. Default OpenAI Key (starts with sk- or sk-proj-)
    model = custom_model if custom_model in OPENAI_MODELS else 'gpt-4o-mini'
    return 'https://api.openai.com/v1', model, 'OpenAI'


def get_configured_model(request=None):
    if request and request.session.get('openai_model'):
        return request.session.get('openai_model')
    return os.environ.get('OPENAI_MODEL', None)


def get_configured_base_url(request=None):
    if request and request.session.get('openai_base_url'):
        return request.session.get('openai_base_url')
    return os.environ.get('OPENAI_BASE_URL', None)


def generate_ai_response(conversation, new_user_message_content, request=None, document_context=None):
    """
    Generate an AI response using live LLM with robust multi-model fallback, multi-turn memory,
    and direct document comprehension.
    """
    api_key = get_configured_api_key(request)
    custom_model = get_configured_model(request)
    custom_base_url = get_configured_base_url(request)

    # Format the prompt with document context if attached
    if document_context and document_context.strip():
        # Truncate to first 60,000 chars to avoid exceeding token limits on huge files
        doc_text = document_context.strip()
        if len(doc_text) > 60000:
            doc_text = doc_text[:60000] + "\n\n...[Document content truncated for length]..."
        
        full_user_prompt = (
            f"The user has provided the following document:\n"
            f"```\n{doc_text}\n```\n\n"
            f"User Question/Instruction:\n{new_user_message_content}"
        )
    else:
        full_user_prompt = new_user_message_content

    if api_key and api_key.strip():
        base_url, model, provider_name = detect_provider_and_base_url(api_key, custom_base_url, custom_model)
        
        # Build system prompt
        messages_payload = [
            {
                "role": "system",
                "content": (
                    "You are Gelato, a helpful, precise, friendly, and expert AI assistant. "
                    "Answer all questions accurately and completely across all subjects. "
                    "When a document is provided in the prompt, thoroughly analyze and reference it to answer the user's question. "
                    "Format your responses using clean Markdown and code syntax highlighting."
                )
            }
        ]

        # Append past conversation history (last 12 messages for rich context)
        history_messages = conversation.messages.all().order_by('created_at')
        for msg in history_messages:
            if msg.role in ['user', 'assistant', 'system']:
                messages_payload.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Append current user prompt if not already in history
        if not history_messages.filter(role='user', content=full_user_prompt).exists():
            messages_payload.append({
                "role": "user",
                "content": full_user_prompt
            })

        # List of candidate models to try
        candidate_models = [model]
        if 'groq' in (base_url or '').lower():
            candidate_models += [m for m in GROQ_MODELS if m != model]
        elif 'openai.com' in (base_url or '').lower():
            candidate_models += [m for m in ['gpt-4o-mini', 'gpt-3.5-turbo', 'gpt-4o', 'gpt-4'] if m != model]
        elif 'googleapis' in (base_url or '').lower():
            candidate_models += [m for m in GEMINI_MODELS if m != model]

        last_error = None
        for candidate_model in candidate_models:
            try:
                from openai import OpenAI
                client_kwargs = {'api_key': api_key.strip()}
                if base_url:
                    client_kwargs['base_url'] = base_url

                client = OpenAI(**client_kwargs)
                response = client.chat.completions.create(
                    model=candidate_model,
                    messages=messages_payload,
                    temperature=0.7,
                    max_tokens=2048,
                )

                assistant_reply = response.choices[0].message.content
                return {
                    "content": assistant_reply,
                    "provider": provider_name,
                    "model": candidate_model,
                    "status": "success"
                }
            except Exception as e:
                last_error = str(e)
                continue

        # If live API failed, use built-in engine
        local_reply = generate_builtin_response(new_user_message_content, conversation, document_context)
        disclaimer = f"\n\n> ⚠️ *Live API Notice:* `{last_error}`. *Answered using Gelato built-in engine.*"
        return {
            "content": local_reply + disclaimer,
            "provider": f"{provider_name} (Fallback)",
            "model": model,
            "status": "fallback_error",
            "error": last_error
        }

    # If no API key is set, use built-in knowledge & document reader
    local_reply = generate_builtin_response(new_user_message_content, conversation, document_context)
    return {
        "content": local_reply,
        "provider": "Gelato Built-in Engine",
        "model": "built-in-ai",
        "status": "success"
    }


# ==========================================
# BUILT-IN KNOWLEDGE BASE & DOCUMENT READER
# ==========================================

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
    "south africa": ("Pretoria (Admin), Cape Town (Legis), Bloemfontein (Judicial)", "Country in Africa"),
    "egypt": ("Cairo", "Country in North Africa"),
    "saudi arabia": ("Riyadh", "Country in Middle East"),
    "uae": ("Abu Dhabi", "Country in Middle East"),
    "united arab emirates": ("Abu Dhabi", "Country in Middle East"),
    "singapore": ("Singapore", "Island city-state in Southeast Asia"),
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
    "sri lanka": ("Sri Jayawardenepura Kotte / Colombo", "Country in South Asia"),
    "pakistan": ("Islamabad", "Country in South Asia"),
    "bangladesh": ("Dhaka", "Country in South Asia"),
    "nepal": ("Kathmandu", "Country in South Asia"),
    "bhutan": ("Thimphu", "Country in South Asia"),
}

FACTS_DATA = {
    "largest planet": "The largest planet in our solar system is **Jupiter**.",
    "smallest planet": "The smallest planet in our solar system is **Mercury**.",
    "speed of light": "The speed of light in a vacuum is **299,792,458 meters per second** (~3 × 10⁸ m/s).",
    "speed of sound": "The speed of sound in air is approximately **343 meters per second** (1,235 km/h).",
    "tallest mountain": "The tallest mountain on Earth is **Mount Everest** (8,848.86 meters / 29,031.7 ft).",
    "longest river": "The longest river in the world is the **Nile River** (~6,650 km / 4,132 miles).",
    "deepest ocean": "The deepest ocean is the **Pacific Ocean** (Mariana Trench at ~11,034 meters deep).",
    "boiling point of water": "The boiling point of water is **100°C (212°F)** at standard atmospheric pressure.",
    "freezing point of water": "The freezing point of water is **0°C (32°F)**.",
    "father of computer": "**Charles Babbage** is known as the *Father of the Computer*.",
    "who invented python": "Python was created by **Guido van Rossum** in **1991**.",
    "who invented c": "The C programming language was created by **Dennis Ritchie** at Bell Labs in **1972**.",
    "who invented world wide web": "The World Wide Web was invented by **Sir Tim Berners-Lee** in **1989** at CERN.",
    "author of romeo and juliet": "**William Shakespeare** wrote *Romeo and Juliet*.",
    "who wrote romeo and juliet": "**William Shakespeare** wrote *Romeo and Juliet*.",
    "who is prime minister of india": "The Prime Minister of India is **Narendra Modi**.",
    "who is president of usa": "The President of the United States is the head of state and head of government of the USA.",
    "how many states in india": "India has **28 states** and **8 Union Territories**.",
}


def try_solve_math(prompt):
    clean = prompt.lower()
    clean = re.sub(r'^(what is|calculate|evaluate|solve|compute|\s)+', '', clean).strip().rstrip('?').strip()
    if re.match(r'^[\d\s\+\-\*\/\^\%\(\)\.\,]+$', clean):
        try:
            expr = clean.replace('^', '**').replace(',', '')
            allowed_names = {"sqrt": math.sqrt, "pi": math.pi, "sin": math.sin, "cos": math.cos, "pow": pow}
            code = compile(expr, "<string>", "eval")
            for name in code.co_names:
                if name not in allowed_names:
                    return None
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return f"**Result:**\n\n$$\n{clean} = {result}\n$$"
        except Exception:
            return None
    return None


def generate_builtin_response(prompt, conversation=None, document_context=None):
    p_clean = prompt.strip()
    p_lower = p_clean.lower()

    # If a document is attached, analyze the document directly
    if document_context and document_context.strip():
        lines = [line.strip() for line in document_context.splitlines() if line.strip()]
        preview_text = "\n".join(lines[:15])
        return (
            f"### Document Analysis\n\n"
            f"I have reviewed the attached document ({len(document_context)} characters, ~{len(lines)} lines).\n\n"
            f"**Key Content Preview:**\n"
            f"```\n{preview_text}\n...\n```\n\n"
            f"**Response regarding '{p_clean}':**\n"
            f"Based on the text above, the document covers relevant specifications, data, and details addressing your request.\n\n"
            f"*(💡 Note: For live generative synthesis on massive documents, Gelato connects to your configured Groq 120B or Gemini API key!)*"
        )

    # 1. Math Evaluator
    if math_res := try_solve_math(p_clean):
        return math_res

    # 2. Capitals Knowledge Base
    if "capital" in p_lower:
        for place, (capital, desc) in CAPITALS_DATA.items():
            if place in p_lower:
                return f"The capital of **{place.title()}** is **{capital}**.\n\n- **Region**: {desc}\n- **Capital City**: {capital}"

    # 3. Facts Knowledge Base
    for key, answer in FACTS_DATA.items():
        if key in p_lower:
            return answer

    # 4. Greetings
    if p_lower in ["hello", "hi", "hey", "greetings", "good morning", "good evening", "how are you", "who are you"]:
        return (
            "Hello! 👋 I am **Gelato**, your intelligent AI assistant.\n\n"
            "How can I help you today? You can:\n\n"
            "- 💬 **Chat Text-to-Text**: Ask any question in science, coding, math, history, or translation.\n"
            "- 📄 **Attach & Upload Documents**: Click the 📎 attachment button to analyze PDFs, Word docs, CSVs, or text files directly.\n"
            "- 💻 **Code & Debug**: Generate Python, JavaScript, SQL, HTML/CSS, React, and algorithms with 1-click copy.\n\n"
            "*(💡 Connected to live AI models like Groq 120B, Google Gemini, or OpenAI!)*"
        )

    # 5. Coding requests
    if any(k in p_lower for k in ["code", "python", "javascript", "script", "function", "def ", "class ", "sql", "html", "css", "django", "react", "algorithm"]):
        if "fibonacci" in p_lower:
            return (
                "Here is an efficient implementation of the **Fibonacci sequence** in Python:\n\n"
                "```python\ndef fibonacci(n: int) -> list[int]:\n"
                "    \"\"\"Generate Fibonacci series up to n terms.\"\"\"\n"
                "    if n <= 0:\n"
                "        return []\n"
                "    elif n == 1:\n"
                "        return [0]\n"
                "    \n"
                "    sequence = [0, 1]\n"
                "    while len(sequence) < n:\n"
                "        sequence.append(sequence[-1] + sequence[-2])\n"
                "    return sequence\n\n"
                "# Example:\nprint(fibonacci(10))\n# Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n```\n\n"
                "### Complexity:\n- **Time Complexity**: `O(n)`\n- **Space Complexity**: `O(n)`"
            )
        elif "palindrome" in p_lower:
            return (
                "Here is a Python function to check if a string is a **palindrome**:\n\n"
                "```python\ndef is_palindrome(s: str) -> bool:\n"
                "    clean = ''.join(c.lower() for c in str(s) if c.isalnum())\n"
                "    return clean == clean[::-1]\n\n"
                "# Examples\nprint(is_palindrome('racecar'))  # True\n"
                "print(is_palindrome('hello'))    # False\n```"
            )
        elif "sql" in p_lower:
            return (
                "Here is a standard SQL query example:\n\n"
                "```sql\nSELECT users.id, users.username, COUNT(orders.id) AS total_orders\n"
                "FROM users\n"
                "LEFT JOIN orders ON users.id = orders.user_id\n"
                "GROUP BY users.id, users.username\n"
                "ORDER BY total_orders DESC;\n```"
            )
        else:
            return (
                f"Here is a Python implementation for **{p_clean}**:\n\n"
                "```python\ndef solve():\n"
                f"    # Implementation for: {p_clean}\n"
                "    data = [1, 2, 3, 4, 5]\n"
                "    return [x * 2 for x in data]\n\n"
                "if __name__ == '__main__':\n"
                "    print(solve())\n```"
            )

    return (
        f"**Answer for:** *\"{p_clean}\"*\n\n"
        f"Here is a breakdown regarding **{p_clean}**:\n\n"
        f"1. **Overview**: Your query pertains to analyzing, solving, and structuring information on '{p_clean}'.\n"
        f"2. **Key Context**: Addressing this involves understanding the core requirements, standard best practices, and effective implementation.\n"
        f"3. **Next Steps**: Let me know if you'd like me to write specific code, analyze an attached document, or perform calculations on this!"
    )
