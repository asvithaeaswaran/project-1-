from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from .models import Conversation, Message, UploadedDocument
import json


class GelatoApplicationTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='password123')
        self.client.login(username='testuser', password='password123')

    def test_index_authenticated(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'pages/index.html')

    def test_create_and_fetch_conversations(self):
        response = self.client.post(
            reverse('api_create_conversation'),
            data=json.dumps({'title': 'Test Chat'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 201)
        conv_id = response.json()['id']

        list_response = self.client.get(reverse('api_get_conversations'))
        self.assertEqual(list_response.status_code, 200)
        convs = list_response.json()['conversations']
        self.assertEqual(len(convs), 1)
        self.assertEqual(convs[0]['title'], 'Test Chat')

        detail_response = self.client.get(reverse('api_get_conversation_detail', kwargs={'conversation_id': conv_id}))
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()['title'], 'Test Chat')

    def test_send_message_text_to_text(self):
        conv = Conversation.objects.create(user=self.user, title='New Chat')
        
        response = self.client.post(
            reverse('api_send_message', kwargs={'conversation_id': conv.id}),
            data=json.dumps({'content': 'What is the capital of France?'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('user_message', data)
        self.assertIn('assistant_message', data)
        
        self.assertEqual(conv.messages.count(), 2)
        user_msg = conv.messages.filter(role='user').first()
        assistant_msg = conv.messages.filter(role='assistant').first()
        self.assertEqual(user_msg.content, 'What is the capital of France?')
        self.assertTrue(len(assistant_msg.content) > 0)

    def test_send_message_with_attached_document(self):
        conv = Conversation.objects.create(user=self.user, title='Doc Chat')
        sample_file = SimpleUploadedFile(
            'notes.txt',
            b'Project Apollo: Launch target is October 2026. Budget is $5 Million.',
            content_type='text/plain'
        )

        response = self.client.post(
            reverse('api_send_message', kwargs={'conversation_id': conv.id}),
            {
                'content': 'What is the budget and launch target?',
                'file': sample_file
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertTrue(len(data['user_message']['attachments']) >= 1)
        self.assertEqual(data['user_message']['attachments'][0]['name'], 'notes.txt')
        self.assertTrue(len(data['assistant_message']['content']) > 0)

    def test_document_upload_and_delete(self):
        test_file = SimpleUploadedFile('spec.txt', b'Specs: Version 2.0', content_type='text/plain')

        upload_resp = self.client.post(
            reverse('api_upload_documents'),
            {'files': [test_file]}
        )
        self.assertEqual(upload_resp.status_code, 200)
        data = upload_resp.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(len(data['results']), 1)
        doc_id = data['results'][0]['id']

        docs_resp = self.client.get(reverse('api_get_documents'))
        self.assertEqual(docs_resp.status_code, 200)
        self.assertEqual(docs_resp.json()['total'], 1)

        del_resp = self.client.delete(reverse('api_delete_document', kwargs={'doc_id': doc_id}))
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(UploadedDocument.objects.filter(id=doc_id).count(), 0)
