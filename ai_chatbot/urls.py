from django.urls import path
from .views import ChatView, EmailView, DocumentUploadView, GenerateAssessmentView
from .structure_views import DocumentStructureView

urlpatterns = [
    path('set_admin_email/', EmailView.as_view(), {'role': 'admin'}, name='set_admin_email'),
    path('set_user_email/', EmailView.as_view(), {'role': 'user'}, name='set_user_email'),
    # Kept for backward compatibility
    path('set_email/', EmailView.as_view(), {'role': 'legacy'}, name='set_email'),
    path('chat/', ChatView.as_view(), name='chat'),
    path("upload_pdf/", DocumentUploadView.as_view()),
    path("document/<int:id>/structure/", DocumentStructureView.as_view(), name="document-structure"),
    path('generate_assessment/', GenerateAssessmentView.as_view(), name='generate_assessment'),
]
