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
    lesson = serializers.SerializerMethodField()
    topic = serializers.SerializerMethodField()
    use = serializers.SerializerMethodField()
    subsections = serializers.SerializerMethodField()

    class Meta:
        model = Module
        fields = ['module_id', 'name', 'order', 'category', 'description', 'lesson', 'topic', 'use', 'subsections']

    def _get_table_data(self, obj):
        if not hasattr(obj, '_table_data'):
            import json
            obj._table_data = {}
            for sub in obj.subsections.all():
                if sub.name == '_module_metadata_table_':
                    try:
                        obj._table_data = json.loads(sub.content)
                    except:
                        pass
                    break
        return obj._table_data

    def get_lesson(self, obj):
        return self._get_table_data(obj).get('lesson')

    def get_topic(self, obj):
        return self._get_table_data(obj).get('topic')

    def get_use(self, obj):
        return self._get_table_data(obj).get('use')

    def get_subsections(self, obj):
        subs = [sub for sub in obj.subsections.all() if sub.name != '_module_metadata_table_']
        return SubsectionSerializer(subs, many=True).data

class DocumentStructureSerializer(serializers.ModelSerializer):
    modules = ModuleSerializer(many=True, read_only=True)
    class Meta:
        model = Document
        fields = ['id', 'file', 'uploaded_at', 'modules']

class AssessmentRequestSerializer(serializers.Serializer):
    module_id = serializers.IntegerField(required=True, help_text="ID of the module to generate assessment for.")
    session_id = serializers.CharField(required=True, help_text="Admin session ID.")