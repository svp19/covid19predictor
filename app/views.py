from django.shortcuts import render
from .forms import PredictForm
from .regions import STATES
from .districts import DISTRICTS
from .seq2seq.predict import get_predictions

# Create your views here.
def home(request):
    context = dict()
    # context['states'] = STATES
    context['states'] = DISTRICTS
    if request.method == 'POST':
        context["predict"] = True
        context["state"] = request.POST.get('state-input')
        context["predictions"] = get_predictions(context["state"])

        return render(request, 'app/home.html', context)
    return render(request, 'app/home.html', context)
