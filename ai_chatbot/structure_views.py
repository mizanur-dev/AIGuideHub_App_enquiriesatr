from rest_framework.generics import RetrieveAPIView
from ai_chatbot.models import Document
from ai_chatbot.serializers import DocumentStructureSerializer

class DocumentStructureView(RetrieveAPIView):
    queryset = Document.objects.all()
    serializer_class = DocumentStructureSerializer
    lookup_field = 'id'
