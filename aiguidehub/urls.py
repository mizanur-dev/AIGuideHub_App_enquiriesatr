from django.urls import path
from .views import ChatView, EmailView, PDFUploadAPIView

urlpatterns = [
    path('set_email/', EmailView.as_view(), name='set_email'),
    path('chat/', ChatView.as_view(), name='chat'),
    path("upload_pdf/", PDFUploadAPIView.as_view()),
]
