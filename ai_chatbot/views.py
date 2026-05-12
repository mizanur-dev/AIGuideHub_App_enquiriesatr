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
from rest_framework.views import APIView
from .serializers import DocumentUploadSerializer
from .rag.ingestion import process_pdf, process_document_async
from .rag.embedding import embed_texts
from .rag.vector_store import retrieve_context
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
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        email = serializer.validated_data['email']
        
        # Generate unique session ID based on email
        session_id = f"{email}_{uuid.uuid4().hex[:8]}"
        
        # Create a new session
        request.session.create()
        
        # Store email and our custom session ID
        request.session['user_email'] = email
        request.session['custom_session_id'] = session_id
        
        return Response({
            "message": f"Email set successfully: {email}. You can now use the chatbot.",
            "session_id": session_id
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
            if session_data.get('custom_session_id') == session_id:
                return session, session_data
        return None, None

    def create(self, request, *args, **kwargs):
        serializer = ChatRequestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user_message = serializer.validated_data['message']
        session_id = serializer.validated_data['session_id']
        
        if not session_id:
            raise ValidationError("Session ID is required")
        
        # Find session by custom session ID
        session_obj, session_data = self.get_session_by_custom_id(session_id)
        
        if not session_obj or not session_data:
            raise NotFound("Invalid session ID. Please set your email first via /api/set_email/")
        
        if 'user_email' not in session_data:
            raise NotFound("Invalid session. Please set your email first via /api/set_email/")
        
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

        # Fetch RAG context with threshold
        context = retrieve_context(user_message, session_id, top_k=12, score_threshold=0.40)

        # Fallback safeguard: If no context met the threshold, bypass LLM completely.
        if not context.strip():
            ai_text = "I am unable to provide the answer because it is out knowledge of the app."
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
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        file = serializer.validated_data["file"]
        session_id = serializer.validated_data["session_id"]
        file_type = serializer.validated_data["file_type"]

        logger.info(f"Processing upload for session: {session_id}, type: {file_type}")

        try:
            # Save the uploaded file to the media directory
            fs = FileSystemStorage()
            filename = fs.save(file.name, file)
            file_path = fs.path(filename)
            original_filename = file.name

            # Determine actual file type by extension as fallback
            is_slide_deck = file_type == "slide_deck" or file.name.lower().endswith(('.pptx', '.ppt'))
            
            if is_slide_deck:
                # Async ingestion
                process_document_async(file_path, "slide_deck", namespace=session_id, original_filename=original_filename)
                return Response(
                    {"message": "Document uploaded. Ingestion running in background."}, status=202
                )
            else:
                # Process the PDF synchronously
                num_chunks = process_pdf(file_path, session_id, original_filename)
                fs.delete(filename)
                return Response(
                    {"message": f"PDF processed and {num_chunks} chunks indexed successfully."}, status=200
                )
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return Response({"error": "Failed to process document."}, status=500)