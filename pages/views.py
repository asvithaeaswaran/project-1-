import json
import uuid
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie

from .models import Conversation, Message, UploadedDocument
from .ai_service import generate_ai_response, get_configured_api_key, get_configured_model, get_configured_base_url
from .doc_service import extract_text_from_file


def ensure_user(request):
    """
    Helper to get the current authenticated user or return None.
    """
    if request.user.is_authenticated:
        return request.user
    return None


@ensure_csrf_cookie
def index(request):
    """
    Render main Gelato Chat interface.
    """
    if not request.user.is_authenticated:
        return redirect('login')
    
    docs_count = UploadedDocument.objects.filter(user=request.user).count()
    
    return render(request, 'pages/index.html', {
        'username': request.user.username,
        'has_api_key': bool(get_configured_api_key(request)),
        'docs_count': docs_count,
    })


def signup_view(request):
    """
    User registration view.
    """
    if request.user.is_authenticated:
        return redirect('index')

    error_message = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()

        if not username or not password:
            error_message = 'Please provide both username and password.'
        elif len(password) < 4:
            error_message = 'Password must be at least 4 characters long.'
        elif password != confirm_password:
            error_message = 'Passwords do not match.'
        elif User.objects.filter(username=username).exists():
            error_message = 'Username already taken. Please choose another one.'
        else:
            user = User.objects.create_user(username=username, password=password)
            login(request, user)
            return redirect('index')

    return render(request, 'pages/signup.html', {'error': error_message})


def login_view(request):
    """
    User login view.
    """
    if request.user.is_authenticated:
        return redirect('index')

    error_message = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            error_message = 'Invalid username or password.'

    return render(request, 'pages/login.html', {'error': error_message})


def demo_login_view(request):
    """
    One-click instant login for seamless testing without registering.
    """
    demo_username = 'demouser'
    user, created = User.objects.get_or_create(username=demo_username)
    if created:
        user.set_password('demo1234')
        user.save()
    login(request, user)
    return redirect('index')


def logout_view(request):
    """
    Log out user.
    """
    logout(request)
    return redirect('login')


# ==========================================
# CONVERSATION REST API ENDPOINTS
# ==========================================

@login_required
@require_http_methods(["GET"])
def api_get_conversations(request):
    """
    Fetch all conversations for the current user.
    """
    conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
    data = []
    for conv in conversations:
        data.append({
            'id': str(conv.id),
            'title': conv.title,
            'created_at': conv.created_at.isoformat(),
            'updated_at': conv.updated_at.isoformat(),
            'message_count': conv.messages.count(),
        })
    return JsonResponse({'conversations': data})


@login_required
@require_http_methods(["POST"])
def api_create_conversation(request):
    """
    Create a new empty conversation.
    """
    try:
        data = json.loads(request.body.decode('utf-8')) if request.body else {}
    except Exception:
        data = {}

    title = data.get('title', 'New Chat').strip() or 'New Chat'
    conv = Conversation.objects.create(user=request.user, title=title)
    return JsonResponse({
        'id': str(conv.id),
        'title': conv.title,
        'created_at': conv.created_at.isoformat(),
        'updated_at': conv.updated_at.isoformat(),
        'messages': []
    }, status=201)


@login_required
@require_http_methods(["GET"])
def api_get_conversation_detail(request, conversation_id):
    """
    Get all messages for a given conversation.
    """
    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid conversation ID'}, status=400)

    conv = get_object_or_404(Conversation, id=conv_uuid, user=request.user)
    return JsonResponse({
        'id': str(conv.id),
        'title': conv.title,
        'created_at': conv.created_at.isoformat(),
        'updated_at': conv.updated_at.isoformat(),
        'messages': conv.get_messages_data(),
    })


@login_required
@require_http_methods(["PATCH", "POST"])
def api_rename_conversation(request, conversation_id):
    """
    Rename a conversation.
    """
    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid conversation ID'}, status=400)

    conv = get_object_or_404(Conversation, id=conv_uuid, user=request.user)
    try:
        data = json.loads(request.body.decode('utf-8'))
        new_title = data.get('title', '').strip()
        if not new_title:
            return JsonResponse({'error': 'Title cannot be empty'}, status=400)
        conv.title = new_title[:250]
        conv.save()
        return JsonResponse({'id': str(conv.id), 'title': conv.title})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@login_required
@require_http_methods(["DELETE"])
def api_delete_conversation(request, conversation_id):
    """
    Delete a conversation and its messages.
    """
    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid conversation ID'}, status=400)

    conv = get_object_or_404(Conversation, id=conv_uuid, user=request.user)
    conv.delete()
    return JsonResponse({'success': True, 'id': str(conversation_id)})


@login_required
@require_http_methods(["DELETE"])
def api_clear_all_conversations(request):
    """
    Delete all conversations for the user.
    """
    Conversation.objects.filter(user=request.user).delete()
    return JsonResponse({'success': True, 'message': 'All conversations cleared.'})


@login_required
@require_http_methods(["POST"])
def api_send_message(request, conversation_id):
    """
    Post a user message to a conversation and get a Gelato AI response.
    Supports pure Text-to-Text messaging AND direct Document Attachments (PDF, DOCX, TXT, CSV).
    """
    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid conversation ID'}, status=400)

    conv = get_object_or_404(Conversation, id=conv_uuid, user=request.user)
    
    user_content = ""
    document_context = ""
    attachments_info = []

    # Check if request is multipart/form-data (with file upload) or JSON
    if request.content_type and 'multipart/form-data' in request.content_type:
        user_content = request.POST.get('content', '').strip()
        
        # Handle attached file(s)
        uploaded_files = request.FILES.getlist('files') or ([request.FILES['file']] if 'file' in request.FILES else [])
        for file_obj in uploaded_files:
            extracted = extract_text_from_file(file_obj, file_obj.name)
            doc_text = extracted.get('text', '')
            if doc_text:
                document_context += f"\n\n--- Document: {file_obj.name} ---\n{doc_text}\n"
            
            # Save document record in SQLite
            try:
                clean_title = file_obj.name.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
                doc_record = UploadedDocument.objects.create(
                    user=request.user,
                    title=clean_title,
                    original_filename=file_obj.name,
                    file=file_obj,
                    file_type=extracted.get('file_type', 'txt'),
                    file_size=file_obj.size,
                    char_count=len(doc_text),
                    status='ready'
                )
            except Exception:
                pass

            attachments_info.append({
                'name': file_obj.name,
                'type': extracted.get('file_type', 'txt').upper(),
                'size': f"{file_obj.size / 1024:.1f} KB" if file_obj.size >= 1024 else f"{file_obj.size} B"
            })
            
    else:
        try:
            body = json.loads(request.body.decode('utf-8'))
            user_content = body.get('content', '').strip()
            # If user selected an existing uploaded document ID
            doc_id = body.get('document_id', None)
            if doc_id:
                try:
                    doc = UploadedDocument.objects.get(id=doc_id, user=request.user)
                    extracted = extract_text_from_file(doc.file.path, doc.original_filename)
                    document_context = f"\n\n--- Document: {doc.original_filename} ---\n{extracted.get('text', '')}\n"
                    attachments_info.append({
                        'name': doc.original_filename,
                        'type': doc.file_type.upper(),
                        'size': doc.formatted_file_size
                    })
                except Exception:
                    pass
        except Exception:
            return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    if not user_content and not document_context:
        return JsonResponse({'error': 'Message content cannot be empty'}, status=400)

    if not user_content and document_context:
        user_content = "Please analyze and summarize the attached document."

    # Auto-title conversation if currently 'New Chat'
    if conv.title == 'New Chat':
        first_line = user_content.split('\n')[0].strip()
        conv.title = first_line[:40] if len(first_line) <= 40 else first_line[:37] + '...'
        conv.save()

    # 1. Save user message to database
    user_msg = Message.objects.create(
        conversation=conv,
        role='user',
        content=user_content,
        sources=attachments_info
    )

    # 2. Generate Gelato AI response (with conversation history & document context)
    ai_result = generate_ai_response(
        conversation=conv,
        new_user_message_content=user_content,
        request=request,
        document_context=document_context
    )

    assistant_content = ai_result.get('content', '')

    # 3. Save assistant message to database
    assistant_msg = Message.objects.create(
        conversation=conv,
        role='assistant',
        content=assistant_content
    )

    conv.save()

    return JsonResponse({
        'status': 'success',
        'conversation_id': str(conv.id),
        'conversation_title': conv.title,
        'user_message': {
            'id': user_msg.id,
            'role': user_msg.role,
            'content': user_msg.content,
            'attachments': attachments_info,
            'created_at': user_msg.created_at.isoformat(),
        },
        'assistant_message': {
            'id': assistant_msg.id,
            'role': assistant_msg.role,
            'content': assistant_msg.content,
            'created_at': assistant_msg.created_at.isoformat(),
        },
        'meta': {
            'provider': ai_result.get('provider'),
            'model': ai_result.get('model'),
            'status': ai_result.get('status'),
            'error': ai_result.get('error', None),
        }
    })


@login_required
@require_http_methods(["POST"])
def api_regenerate_message(request, conversation_id):
    """
    Regenerate the latest assistant response in the conversation.
    """
    try:
        conv_uuid = uuid.UUID(str(conversation_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid conversation ID'}, status=400)

    conv = get_object_or_404(Conversation, id=conv_uuid, user=request.user)

    last_user_msg = conv.messages.filter(role='user').last()
    if not last_user_msg:
        return JsonResponse({'error': 'No user message to regenerate response for'}, status=400)

    last_assistant_msg = conv.messages.filter(role='assistant').last()
    if last_assistant_msg and last_assistant_msg.id > last_user_msg.id:
        last_assistant_msg.delete()

    ai_result = generate_ai_response(conv, last_user_msg.content, request)
    assistant_content = ai_result.get('content', '')

    new_assistant_msg = Message.objects.create(
        conversation=conv,
        role='assistant',
        content=assistant_content
    )
    conv.save()

    return JsonResponse({
        'status': 'success',
        'assistant_message': {
            'id': new_assistant_msg.id,
            'role': new_assistant_msg.role,
            'content': new_assistant_msg.content,
            'created_at': new_assistant_msg.created_at.isoformat(),
        },
        'meta': {
            'provider': ai_result.get('provider'),
            'model': ai_result.get('model'),
        }
    })


# ==========================================
# DIRECT DOCUMENT UPLOAD & LIST ENDPOINTS
# ==========================================

@login_required
@require_http_methods(["GET"])
def api_get_documents(request):
    """
    List all uploaded documents for current user.
    """
    docs = UploadedDocument.objects.filter(user=request.user).order_by('-created_at')
    data = []
    for doc in docs:
        data.append({
            'id': str(doc.id),
            'title': doc.title,
            'original_filename': doc.original_filename,
            'file_type': doc.file_type,
            'file_size': doc.file_size,
            'formatted_size': doc.formatted_file_size,
            'char_count': doc.char_count,
            'created_at': doc.created_at.isoformat(),
        })
    return JsonResponse({'documents': data, 'total': len(data)})


@login_required
@require_http_methods(["POST"])
def api_upload_documents(request):
    """
    Upload and parse documents directly.
    """
    files = request.FILES.getlist('files') or ([request.FILES['file']] if 'file' in request.FILES else [])
    if not files:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    uploaded_results = []
    allowed_extensions = {'pdf', 'docx', 'doc', 'txt', 'csv', 'md', 'json', 'py', 'js', 'html', 'css', 'sql'}

    for file_obj in files:
        filename = file_obj.name
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'txt'

        if ext not in allowed_extensions:
            uploaded_results.append({
                'filename': filename,
                'status': 'error',
                'error': f"Unsupported file format '.{ext}'. Supported: PDF, DOCX, TXT, CSV, MD, JSON, Code"
            })
            continue

        clean_title = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ').title()
        extracted = extract_text_from_file(file_obj, filename)

        doc = UploadedDocument.objects.create(
            user=request.user,
            title=clean_title,
            original_filename=filename,
            file=file_obj,
            file_type=ext,
            file_size=file_obj.size,
            char_count=extracted.get('char_count', 0),
            status='ready'
        )

        uploaded_results.append({
            'id': str(doc.id),
            'title': doc.title,
            'filename': filename,
            'file_type': ext,
            'file_size': doc.file_size,
            'formatted_size': doc.formatted_file_size,
            'char_count': doc.char_count,
            'preview': extracted.get('preview', ''),
            'status': 'ready'
        })

    return JsonResponse({
        'status': 'success',
        'results': uploaded_results,
        'total_uploaded': len(uploaded_results)
    })


@login_required
@require_http_methods(["DELETE"])
def api_delete_document(request, doc_id):
    """
    Delete an uploaded document record and file from disk.
    """
    try:
        doc_uuid = uuid.UUID(str(doc_id))
    except ValueError:
        return JsonResponse({'error': 'Invalid document ID'}, status=400)

    doc = get_object_or_404(UploadedDocument, id=doc_uuid, user=request.user)

    try:
        if doc.file and os.path.exists(doc.file.path):
            os.remove(doc.file.path)
    except Exception:
        pass

    doc.delete()
    return JsonResponse({'success': True, 'id': str(doc_id), 'message': 'Document deleted.'})


@login_required
def api_settings(request):
    """
    Get or update user AI settings (Groq, OpenAI, Gemini).
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            api_key = data.get('api_key', '').strip()
            model = data.get('model', '').strip()
            base_url = data.get('base_url', '').strip()

            if api_key:
                request.session['openai_api_key'] = api_key
            elif 'api_key' in data and data['api_key'] == '':
                request.session.pop('openai_api_key', None)

            if model:
                request.session['openai_model'] = model
            if base_url:
                request.session['openai_base_url'] = base_url
            elif 'base_url' in data and data['base_url'] == '':
                request.session.pop('openai_base_url', None)

            return JsonResponse({'status': 'success', 'message': 'Settings updated successfully.'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    # GET request
    current_key = get_configured_api_key(request)
    masked_key = ''
    if current_key:
        if len(current_key) > 8:
            masked_key = current_key[:4] + '...' + current_key[-4:]
        else:
            masked_key = '••••••••'

    return JsonResponse({
        'has_api_key': bool(current_key),
        'masked_api_key': masked_key,
        'model': get_configured_model(request) or 'openai/gpt-oss-120b',
        'base_url': get_configured_base_url(request) or '',
    })
