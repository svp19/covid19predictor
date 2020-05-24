from django import forms



class PredictForm(forms.Form):
    state = forms.HiddenInput()
    lockdown = forms.BooleanField(required=False, initial=True)

