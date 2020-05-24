from django.shortcuts import render
from .forms import PredictForm
from .regions import DISTRICT_CHOICES, STATES


# Create your views here.
def home(request):
    context = dict()
    context['states'] = STATES
    if request.method == 'POST':
        context["predict"] = True
        context["state"] = request.POST.get('state-input')
        return render(request, 'app/home.html', context)
    return render(request, 'app/home.html', context)
