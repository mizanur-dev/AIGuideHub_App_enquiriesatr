from rest_framework import status
from rest_framework.generics import CreateAPIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError, NotFound
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from .serializers import ChatRequestSerializer, ChatResponseSerializer, EmailSerializer
from django.conf import settings
from django.contrib.sessions.models import Session
import uuid
import json
from rest_framework.views import APIView
from .serializers import DocumentUploadSerializer
from .rag.ingestion import process_pdf, process_document, process_document_async
from .rag.embedding import embed_texts
from .rag.vector_store import retrieve_context, delete_document_vectors
from django.core.files.storage import FileSystemStorage
import tempfile
import logging

logger = logging.getLogger(__name__)

# Compact, production-oriented system prompt (single source)
SYSTEM_PROMPT = (
"You are a strict, document-based AI assistant. Your sole purpose is to answer user queries using EXACTLY AND ONLY the provided Context below.\n"
"If the Context provided does not contain the answer, or if the question is unrelated to the Context, "
"you MUST reply ONLY with the following exact sentence: 'I am unable to provide responses outside of the uploaded documents.'\n"
"Do NOT use your general knowledge. Do NOT hallucinate. Do NOT guess. Do NOT provide outside advice.\n"
"Respond in plain text only — no Markdown, no bullets, no numbered lists, no emojis, no code blocks, and no decorative symbols. Use simple sentences and paragraphs."
)

# History settings
HISTORY_MAX_TURNS = 20  # keep last 20 user+assistant exchanges

# Module-level LLM and chain for reuse
_LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=settings.GEMINI_API_KEY,
    temperature=0.5,
    max_output_tokens=2048,
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

_CHAIN = _PROMPT | _LLM

# Ensure responses are plain text without Markdown or decorative symbols
def _normalize_response(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        # Remove common markdown bullets and dots
        for prefix in ("- ", "* ", "• "):
            if l.startswith(prefix):
                l = l[len(prefix):].strip()
                break
        # Remove markdown headers
        while l.startswith('#'):
            l = l.lstrip('#').strip()
        cleaned_lines.append(l)
    cleaned = ' '.join(cleaned_lines)
    # Strip code fences/backticks and emphasis asterisks/underscores
    cleaned = cleaned.replace('```', '').replace('`', '').replace('*', '')
    cleaned = cleaned.replace('_', '')
    # Collapse excessive spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

class EmailView(CreateAPIView):
    serializer_class = EmailSerializer
    authentication_classes = []
    permission_classes = []

    def create(self, request, *args, **kwargs):
        role = self.kwargs.get('role', 'user')  # default to user
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        email = serializer.validated_data['email']
        
        # Create a new session
        request.session.create()
        
        if role == 'admin':
            session_id = f"admin_{email}_{uuid.uuid4().hex[:8]}"
            request.session['admin_email'] = email
            request.session['admin_session_id'] = session_id
            return Response({
                "message": f"Admin Email set successfully: {email}. You can now upload documents.",
                "admin_session_id": session_id
            }, status=status.HTTP_200_OK)
        else:
            # user or legacy
            session_id = f"user_{email}_{uuid.uuid4().hex[:8]}"
            request.session['user_email'] = email
            request.session['user_session_id'] = session_id
            
            if role == 'legacy':
                request.session['custom_session_id'] = session_id
                return Response({
                    "message": f"Email set successfully: {email}. You can now use the chatbot.",
                    "session_id": session_id
                }, status=status.HTTP_200_OK)
                
            return Response({
                "message": f"User Email set successfully: {email}. You can now use the chatbot.",
                "user_session_id": session_id
            }, status=status.HTTP_200_OK)

class GenerateAssessmentView(APIView):
    authentication_classes = []
    permission_classes = []

    def get_session_by_admin_id(self, session_id):
        from django.contrib.sessions.models import Session
        sessions = Session.objects.all()
        for session in sessions:
            session_data = session.get_decoded()
            if session_data.get('admin_session_id') == session_id or session_data.get('custom_session_id') == session_id:
                return session, session_data
        return None, None

    def post(self, request):
        from .serializers import AssessmentRequestSerializer
        from .models import Module
        import re

        serializer = AssessmentRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        session_id = serializer.validated_data['session_id']
        module_id = serializer.validated_data['module_id']

        # Admin Session authentication check
        session_obj, session_data = self.get_session_by_admin_id(session_id)
        if not session_obj:
            logger.warning(f"Unauthenticated attempt to generate_assessment with session: {session_id}")
            return Response({"error": "Unauthorized. Only admins can generate assessments. Please provide a valid admin session_id."}, status=status.HTTP_403_FORBIDDEN)

        try:
            module = Module.objects.get(module_id=module_id)
        except Module.DoesNotExist:
            return Response({"error": f"Module with id {module_id} not found."}, status=status.HTTP_404_NOT_FOUND)

        admin_email = session_data.get("admin_email", "Unknown Admin")
        logger.info(f"Admin {admin_email} generating assessment for module {module_id}")

        # 1. Gather text from the module to form a query, or directly form context
        subsections = module.subsections.all()
        module_text = f"Module: {module.name}\n" + "\n".join(
            [f"Subsection: {sub.name}\n{sub.content}" for sub in subsections]
        )

        # 2. Use module-specific retrieval first, then fallback to full module text
        query_text = module_text[:3000]
        retrieved_context = retrieve_context(query_text, namespace="global_knowledge", top_k=10, module_id=module_id)
        if retrieved_context.strip():
            retrieved_context = module_text + "\n\n---\n\n" + retrieved_context
        else:
            retrieved_context = module_text

        def _truncate_text(text, max_chars=6000):
            return text if len(text) <= max_chars else text[:max_chars]

        def _extract_json_from_text(text):
            start = text.find("{")
            if start == -1:
                raise ValueError("No JSON object found in model output.")

            depth = 0
            end = None
            for idx in range(start, len(text)):
                if text[idx] == "{":
                    depth += 1
                elif text[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx
                        break

            if end is None:
                raise ValueError("No balanced JSON object found in model output.")

            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                cleaned = re.sub(r",\s*}\s*$", "}", candidate)
                cleaned = re.sub(r",\s*\]", "]", cleaned)
                return json.loads(cleaned)

        def _normalize_parsed_questions(candidate):
            if not isinstance(candidate, dict):
                return None
            questions = candidate.get("questions")
            if not isinstance(questions, list):
                return None

            normalized = []
            for item in questions:
                if not isinstance(item, dict):
                    continue
                if not all(k in item for k in ("question", "options", "correct_option", "justification")):
                    continue
                if not isinstance(item["options"], list) or len(item["options"]) != 4:
                    continue
                normalized.append({
                    "question": str(item["question"]).strip(),
                    "options": [str(opt).strip() for opt in item["options"]],
                    "correct_option": str(item["correct_option"]).strip(),
                    "justification": str(item["justification"]).strip(),
                })

            return normalized if len(normalized) == 5 else None

        def _build_prompt(context, error_message=None):
            instructions = (
                "Return exactly one valid JSON object and nothing else. "
                "The top-level object must include a field named questions. "
                "Each item in questions must be an object with question, options, correct_option, and justification. "
                "Options must be exactly 4 items labeled A), B), C), and D). "
                "The correct_option must be a single letter: A, B, C, or D. "
                "Do not include any markdown, bullets, or additional explanation."
            )
            if error_message:
                instructions += f" Previous output failed parsing due to: {error_message}. Return valid JSON only."
            return f"""Generate 5 multiple-choice questions based strictly on the provided content.

{instructions}

Content:
{context}
"""

        from pydantic import BaseModel, Field
        from typing import List

        class MCQ(BaseModel):
            question: str
            options: List[str] = Field(description="Exactly 4 options.", min_length=4, max_length=4)
            correct_option: str = Field(description="The correct option letter or option text.")
            justification: str

        class AssessmentOutput(BaseModel):
            questions: List[MCQ] = Field(description="Exactly 5 multiple-choice questions.", min_length=5, max_length=5)

        def normalize_mcq(q):
            normalized_options = []
            for idx, opt in enumerate(q["options"]):
                text = opt.strip()
                label = chr(ord("A") + idx)
                if re.match(r"^[A-D]\)", text, re.I):
                    normalized_options.append(text)
                else:
                    stripped = re.sub(r'^[A-D]\)\s*', '', text, flags=re.I).strip()
                    normalized_options.append(f"{label}) {stripped}")

            raw_answer = q["correct_option"].strip()
            if re.match(r"^[A-D]$", raw_answer, re.I):
                correct_letter = raw_answer.upper()
            elif re.match(r"^[A-D]\)$", raw_answer, re.I):
                correct_letter = raw_answer[0].upper()
            else:
                correct_letter = None
                for idx, opt in enumerate(normalized_options):
                    option_text = opt[3:].strip()
                    if raw_answer.lower() == option_text.lower() or raw_answer.lower() == opt.lower():
                        correct_letter = chr(ord("A") + idx)
                        break
                if correct_letter is None:
                    raise ValueError("Correct option must be A, B, C, or D, or exactly match one listed option.")

            return {
                "question": q["question"].strip(),
                "options": normalized_options,
                "correct_option": correct_letter,
                "justification": q["justification"].strip(),
            }

        prompt_text = _build_prompt(_truncate_text(retrieved_context, 6000))
        max_retries = 3
        assessment_data = None
        raw_text = None
        last_error = None

        for attempt in range(max_retries):
            try:
                structured_llm = _LLM.with_structured_output(AssessmentOutput)
                response = structured_llm.invoke([HumanMessage(content=prompt_text)])

                if not response or not response.questions:
                    raise ValueError("LLM returned empty structured output.")

                assessment_data = [q.model_dump() for q in response.questions]
                break

            except Exception as e:
                last_error = str(e)
                logger.debug("Structured output parse attempt %s failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    prompt_text = _build_prompt(_truncate_text(retrieved_context, 6000), error_message=last_error)
                    continue

                try:
                    raw_response = _LLM.invoke([HumanMessage(content=prompt_text)])
                    raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
                    parsed = _extract_json_from_text(raw_text)
                    assessment_data = _normalize_parsed_questions(parsed)
                except Exception as fallback_error:
                    logger.warning("Structured output fallback parse attempt failed: %s", fallback_error)
                    last_error = str(fallback_error)

        if not assessment_data:
            logger.error("Failed to generate assessment after %s attempts. Last error: %s", max_retries, last_error)
            return Response({"error": "Failed to generate valid assessment format from LLM after multiple attempts."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        if len(assessment_data) != 5:
            logger.warning("LLM returned wrong number of valid questions: %s", len(assessment_data))
            return Response({"error": "Failed to generate exactly 5 valid questions."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        normalized_questions = []
        for q in assessment_data:
            if len(q["options"]) != 4:
                return Response({"error": "Failed to generate exactly 4 options per question."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            normalized_q = normalize_mcq(q)
            normalized_questions.append(normalized_q)

        assessment_data = normalized_questions

        return Response({
            "success": True,
            "module_id": module_id,
            "assessment": assessment_data
        }, status=status.HTTP_200_OK)

class ChatView(CreateAPIView):
    serializer_class = ChatRequestSerializer
    authentication_classes = []
    permission_classes = []

    def get_session_by_custom_id(self, session_id):
        """Find session by our custom session ID"""
        sessions = Session.objects.all()
        for session in sessions:
            session_data = session.get_decoded()
            if session_data.get('user_session_id') == session_id or session_data.get('custom_session_id') == session_id:
                return session, session_data
        return None, None

    def create(self, request, *args, **kwargs):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user_message = serializer.validated_data['message']
        session_id = serializer.validated_data['session_id']
        
        if not session_id:
            raise ValidationError("Session ID is required")
        
        # Find session by user session ID
        session_obj, session_data = self.get_session_by_custom_id(session_id)
        
        if not session_obj or not session_data:
            raise NotFound("Invalid session ID. Please set your user email first via /api/set_user_email/")
        
        if 'user_email' not in session_data:
            raise NotFound("Invalid session. Please set your user email first.")
        
        # Chat processing with trimmed history
        chat_history_key = f'chat_history_{session_id}'
        history_data = session_data.get(chat_history_key, [])

        # Trim to last N turns (each turn = human+ai → 2 messages)
        max_messages = HISTORY_MAX_TURNS * 2
        trimmed_history = history_data[-max_messages:]

        # Convert to LangChain messages
        history_msgs = []
        for item in trimmed_history:
            if item.get('type') == 'human':
                history_msgs.append(HumanMessage(content=item.get('content', '')))
            elif item.get('type') == 'ai':
                history_msgs.append(AIMessage(content=item.get('content', '')))

        # Fetch RAG context from the globally available knowledge base
        context = retrieve_context(user_message, "global_knowledge", top_k=12, score_threshold=0.25)

        # Fallback safeguard: If no context met the threshold, bypass LLM completely.
        if not context.strip():
            ai_text = "I am unable to provide the answer because it is outside of our knowledge of the app."
        else:
            # Prepare final prompt to include context
            final_prompt = f"""
Context:
{context}

User:
{user_message}
"""

            # Invoke chain with module-level prompt/llm and context combined in the input
            response = _CHAIN.invoke({
                "input": final_prompt,
                "history": history_msgs,
            })

            ai_text = response.content if hasattr(response, "content") else str(response)
            ai_text = _normalize_response(ai_text)

        # Append new turn and re-trim
        updated_history = trimmed_history + [
            {"type": "human", "content": user_message},
            {"type": "ai", "content": ai_text},
        ]
        updated_history = updated_history[-max_messages:]

        # Persist back to session
        session_data[chat_history_key] = updated_history
        session_obj.session_data = Session.objects.encode(session_data)
        session_obj.save()

        response_serializer = ChatResponseSerializer({'response': ai_text})
        return Response(response_serializer.data, status=status.HTTP_200_OK)



class DocumentUploadView(APIView):
    def get_session_by_admin_id(self, session_id):
        sessions = Session.objects.all()
        for session in sessions:
            session_data = session.get_decoded()
            if session_data.get('admin_session_id') == session_id or session_data.get('custom_session_id') == session_id:
                return session, session_data
        return None, None

    def post(self, request):
        from ai_chatbot.rag.pdf_structure_extractor import extract_pdf_structure
        from ai_chatbot.models import Document, Module, Subsection
        serializer = DocumentUploadSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        file = serializer.validated_data["file"]
        session_id = serializer.validated_data["session_id"]
        file_type = serializer.validated_data["file_type"]

        # Validate admin session
        session_obj, session_data = self.get_session_by_admin_id(session_id)
        if not session_obj:
            return Response({"error": "Unauthorized. Only admins can upload documents. Please set an admin email."}, status=403)

        # Use global namespace for indexing so all users can access it
        indexing_namespace = "global_knowledge"

        logger.info(f"Processing admin upload for session: {session_id}, type: {file_type}")

        try:
            from ai_chatbot.serializers import DocumentStructureSerializer
            # Save the uploaded file to the media directory
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            original_filename = file.name

            # Determine actual file type by extension as fallback
            is_slide_deck = file_type == "slide_deck" or file.name.lower().endswith(('.pptx', '.ppt'))

            if is_slide_deck:
                # Synchronous ingestion for slide deck content so PPTX text gets indexed immediately
                num_chunks = process_document(file_path, "slide_deck", indexing_namespace, original_filename)
                return Response(
                    {
                        "message": f"Slide deck uploaded successfully and indexed {num_chunks} chunks.",
                        "num_chunks": num_chunks
                    },
                    status=200
                )
            else:
                # --- PDF Structure Extraction and Storage ---
                structure = extract_pdf_structure(file_path)
                if structure:
                    doc_obj = Document.objects.create(file=filename)
                    module_order = 0
                    indexed_structure = []

                    for mod in structure:
                        module_obj = Module.objects.create(document=doc_obj, name=mod['module'], order=module_order)
                        module_order += 1
                        subsection_order = 0
                        module_entry = {
                            'module_id': module_obj.module_id,
                            'module_name': module_obj.name,
                            'module_order': module_obj.order,
                            'subsections': []
                        }

                        for sub in mod['subsections']:
                            subsection_obj = Subsection.objects.create(
                                module=module_obj,
                                name=sub['name'],
                                content=sub['content'],
                                order=subsection_order
                            )
                            module_entry['subsections'].append({
                                'subsection_id': subsection_obj.id,
                                'name': subsection_obj.name,
                                'content': subsection_obj.content,
                                'order': subsection_obj.order,
                            })
                            subsection_order += 1

                        indexed_structure.append(module_entry)

                    num_chunks = process_pdf(file_path, indexing_namespace, original_filename, structure=indexed_structure)

                    # Serialize the document structure for response
                    doc_obj.refresh_from_db()
                    serializer = DocumentStructureSerializer(doc_obj)
                    fs.delete(filename)
                    return Response(
                        {
                            "message": f"PDF processed, {num_chunks} chunks indexed, and structure extracted successfully.",
                            "structure": serializer.data
                        },
                        status=200
                    )
                else:
                    # Fall back to legacy full-document indexing when structure extraction fails
                    num_chunks = process_pdf(file_path, indexing_namespace, original_filename)
                    fs.delete(filename)
                    return Response(
                        {
                            "message": f"PDF processed with fallback indexing, {num_chunks} chunks indexed. No module structure could be extracted.",
                            "structure": None
                        },
                        status=200
                    )
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return Response({"error": "Failed to process document."}, status=500)