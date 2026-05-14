from django.urls import path
from .views import ChatView, EmailView, DocumentUploadView
from .structure_views import DocumentStructureView

urlpatterns = [
    path('set_email/', EmailView.as_view(), name='set_email'),
    path('chat/', ChatView.as_view(), name='chat'),
    path("upload_pdf/", DocumentUploadView.as_view()),
    path("document/<int:id>/structure/", DocumentStructureView.as_view(), name="document-structure"),
]
