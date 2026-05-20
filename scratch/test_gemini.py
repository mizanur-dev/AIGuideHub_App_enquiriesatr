import os, django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'aiguidehub.settings')
django.setup()
from ai_chatbot.rag.pdf_structure_extractor import extract_pdf_structure
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json

s = extract_pdf_structure('media/enquiriesatr.pdf')
m = s[1]
txt = '\n'.join([sub['content'] for sub in m['subsections']])

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
prompt=f'''Return a JSON object ONLY with two fields: category and description.
category must be one of: FOUNDATION, TACTICAL, OPERATIONS, LEGAL.
description must be a short (one-sentence) summary of the module (max 200 characters).

IMPORTANT: You must return valid JSON. Do not use unescaped double quotes inside the description.

Module name: {m["module"]}

Module content (truncated):
{txt[:4000]}

Respond with strict JSON only, for example:
{{"category": "LEGAL", "description": "Short module summary"}}
'''

print("Invoking LLM...")
response = llm.invoke([HumanMessage(content=prompt)])
print("Response:", repr(response.content))
