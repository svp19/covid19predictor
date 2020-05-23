from django.shortcuts import render
from .forms import PredictForm


# Create your views here.
def home(request):
    context = dict()
    context['form'] = PredictForm()
    if request.method == 'POST':
        form = PredictForm(request.POST)
        if form.is_valid():
            context['predict'] = "Yes"
            context['form'] = form
            return render(request, 'app/home.html', context)
    return render(request, 'app/home.html', context)
