from django.http import JsonResponse
from django.shortcuts import render
from .logic import database_exists, setup_backend, get_similar_sentences


def index(request):
    if request.method == 'POST':
        query = request.POST.get('query')
        k = int(request.POST.get('k'))
        result = get_similar_sentences(query, k)
        return JsonResponse({'sentences': result})
    return render(request, 'similarity/index.html')


def db_status(request):
    ready = database_exists()
    status = {
        'exists': ready,
        'message': 'Database exists' if ready else 'Database is being built'
    }
    return JsonResponse(status)


def build_db(request):
    setup_backend()
    return JsonResponse({'status': 'Building database...'})
