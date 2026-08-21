from django.urls import path
from . import views

urlpatterns = [
    # UI Pages
    path('', views.index, name='index'),
    path('login/', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('demo-login/', views.demo_login_view, name='demo_login'),
    path('logout/', views.logout_view, name='logout'),

    # REST APIs for Gelato Chat & Conversations
    path('api/conversations/', views.api_get_conversations, name='api_get_conversations'),
    path('api/conversations/create/', views.api_create_conversation, name='api_create_conversation'),
    path('api/conversations/clear/', views.api_clear_all_conversations, name='api_clear_all_conversations'),
    path('api/conversations/<uuid:conversation_id>/', views.api_get_conversation_detail, name='api_get_conversation_detail'),
    path('api/conversations/<uuid:conversation_id>/rename/', views.api_rename_conversation, name='api_rename_conversation'),
    path('api/conversations/<uuid:conversation_id>/delete/', views.api_delete_conversation, name='api_delete_conversation'),
    path('api/conversations/<uuid:conversation_id>/messages/', views.api_send_message, name='api_send_message'),
    path('api/conversations/<uuid:conversation_id>/regenerate/', views.api_regenerate_message, name='api_regenerate_message'),

    # REST APIs for Document Attachments & Uploads
    path('api/documents/', views.api_get_documents, name='api_get_documents'),
    path('api/documents/upload/', views.api_upload_documents, name='api_upload_documents'),
    path('api/documents/<uuid:doc_id>/delete/', views.api_delete_document, name='api_delete_document'),

    # Settings API
    path('api/settings/', views.api_settings, name='api_settings'),
]