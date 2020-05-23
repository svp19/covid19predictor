from django import forms
from .districts import DISTRICT_CHOICES


class PredictForm(forms.Form):
    district = forms.ChoiceField(choices=DISTRICT_CHOICES)
    lockdown = forms.BooleanField(required=False, initial=True)

