from django import forms
from apps.etl.models import Paciente


def _classify_imc(imc: float) -> str:
    if imc < 18.5: return "bajo_peso"
    if imc < 25:   return "normal"
    if imc < 30:   return "sobrepeso"
    return "obesidad"


class PacienteForm(forms.ModelForm):
    class Meta:
        model = Paciente
        fields = [
            'id_paciente', 'nombres', 'apellidos', 'edad', 'sexo',
            'peso', 'altura',
            'presion_sistolica', 'presion_diastolica',
            'glucosa', 'colesterol', 'frecuencia_cardiaca',
            'saturacion_oxigeno', 'temperatura',
            'antecedentes_familiares', 'fumador', 'consumo_alcohol',
            'actividad_fisica',
            'diagnostico_preliminar', 'riesgo_enfermedad',
        ]
        widgets = {
            'id_paciente':            forms.NumberInput(attrs={'class': 'ha-input', 'min': 1}),
            'nombres':                forms.TextInput(attrs={'class': 'ha-input', 'placeholder': 'Ej. María Camila'}),
            'apellidos':              forms.TextInput(attrs={'class': 'ha-input', 'placeholder': 'Ej. Torres Ruiz'}),
            'edad':                   forms.NumberInput(attrs={'class': 'ha-input', 'min': 0, 'max': 120}),
            'sexo':                   forms.Select(attrs={'class': 'ha-input'}),
            'peso':                   forms.NumberInput(attrs={'class': 'ha-input', 'placeholder': 'kg', 'step': '0.1'}),
            'altura':                 forms.NumberInput(attrs={'class': 'ha-input', 'placeholder': 'm (ej. 1.75)', 'step': '0.01'}),
            'presion_sistolica':      forms.NumberInput(attrs={'class': 'ha-input'}),
            'presion_diastolica':     forms.NumberInput(attrs={'class': 'ha-input'}),
            'glucosa':                forms.NumberInput(attrs={'class': 'ha-input'}),
            'colesterol':             forms.NumberInput(attrs={'class': 'ha-input'}),
            'frecuencia_cardiaca':    forms.NumberInput(attrs={'class': 'ha-input'}),
            'saturacion_oxigeno':     forms.NumberInput(attrs={'class': 'ha-input', 'placeholder': '%', 'step': '0.1'}),
            'temperatura':            forms.NumberInput(attrs={'class': 'ha-input', 'placeholder': '°C', 'step': '0.1'}),
            'antecedentes_familiares': forms.CheckboxInput(attrs={'class': 'ha-checkbox'}),
            'fumador':                forms.CheckboxInput(attrs={'class': 'ha-checkbox'}),
            'consumo_alcohol':        forms.CheckboxInput(attrs={'class': 'ha-checkbox'}),
            'actividad_fisica':       forms.TextInput(attrs={'class': 'ha-input', 'placeholder': 'Ej. sedentario, moderada, activa'}),
            'diagnostico_preliminar': forms.TextInput(attrs={'class': 'ha-input'}),
            'riesgo_enfermedad':      forms.Select(attrs={'class': 'ha-input'}),
        }

    def clean(self):
        cleaned = super().clean()
        peso = cleaned.get("peso")
        altura = cleaned.get("altura")
        if peso and altura and altura > 0:
            imc = round(peso / (altura ** 2), 2)
            cleaned["imc"] = imc
            cleaned["imc_clasificacion"] = _classify_imc(imc)
        return cleaned

    def save(self, commit=True):
        instance = super().save(commit=False)
        peso = self.cleaned_data.get("peso")
        altura = self.cleaned_data.get("altura")
        if peso and altura and altura > 0:
            instance.imc = round(peso / (altura ** 2), 2)
            instance.imc_clasificacion = _classify_imc(instance.imc)
        if commit:
            instance.save()
        return instance
