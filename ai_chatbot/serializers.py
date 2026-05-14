from rest_framework import serializers
from ai_chatbot.models import Document, Module, Subsection

class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=1000)
    session_id = serializers.CharField(required=True)

class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()

class EmailSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)

class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    session_id = serializers.CharField()
    file_type = serializers.ChoiceField(
        choices=["pdf", "slide_deck"],
        default="pdf",
        required=False
    )

class SubsectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Subsection
        fields = ['id', 'name', 'order', 'content']

class ModuleSerializer(serializers.ModelSerializer):
    subsections = SubsectionSerializer(many=True, read_only=True)
    class Meta:
        model = Module
        fields = ['module_id', 'name', 'order', 'subsections']

class DocumentStructureSerializer(serializers.ModelSerializer):
    modules = ModuleSerializer(many=True, read_only=True)
    class Meta:
        model = Document
        fields = ['id', 'file', 'uploaded_at', 'modules']