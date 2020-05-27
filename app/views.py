from django.shortcuts import render
from .read_predictions import read_predictions
from .districts import DISTRICTS

# Create your views here.
def home(request):
    context = dict()
    context['states'] = sorted(DISTRICTS)
    if request.method == 'POST':
        context["predict"] = True
        context["state"] = request.POST.get('state-input')
        context["predictions"] = read_predictions(context["state"]).values.tolist()
        # print(context["predictions"])
        return render(request, 'app/home.html', context)
    return render(request, 'app/home.html', context)
