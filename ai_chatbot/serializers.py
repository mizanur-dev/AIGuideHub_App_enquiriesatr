from rest_framework import serializers

class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)

class ChatResponseSerializer(serializers.Serializer):
    response = serializers.CharField()

class EmailSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)

class ChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(max_length=1000)
    session_id = serializers.CharField(required=True)

class PDFUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    session_id = serializers.CharField()