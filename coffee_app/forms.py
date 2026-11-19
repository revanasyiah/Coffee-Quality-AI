"""Form prediksi. Pilihan dropdown dan batas angka diambil dari metadata model."""
from django import forms

from .services import get_metadata

CTRL = {"class": "form-control"}
SEL = {"class": "form-select"}


def _choices(values):
    return [(v, v) for v in values]


class PredictionForm(forms.Form):
    altitude = forms.FloatField(label="Ketinggian kebun (meter di atas permukaan laut)",
                                min_value=100, max_value=3000,
                                widget=forms.NumberInput(attrs={**CTRL, "step": "1", "placeholder": "mis. 1300"}))
    moisture = forms.FloatField(label="Kadar air biji (%)", min_value=5, max_value=20,
                                widget=forms.NumberInput(attrs={**CTRL, "step": "0.1", "placeholder": "mis. 10.8"}))
    category_one_defects = forms.IntegerField(label="Cacat Kategori 1 (jumlah biji)", min_value=0, max_value=100,
                                              initial=0, widget=forms.NumberInput(attrs=CTRL))
    category_two_defects = forms.IntegerField(label="Cacat Kategori 2 (jumlah biji)", min_value=0, max_value=100,
                                              initial=0, widget=forms.NumberInput(attrs=CTRL))
    quakers = forms.IntegerField(label="Quakers (jumlah biji belum matang)", min_value=0, max_value=100,
                                 initial=0, widget=forms.NumberInput(attrs=CTRL))
    country = forms.ChoiceField(label="Negara asal", widget=forms.Select(attrs=SEL))
    variety = forms.ChoiceField(label="Varietas kopi", widget=forms.Select(attrs=SEL))
    processing_method = forms.ChoiceField(label="Metode pengolahan", widget=forms.Select(attrs=SEL))
    color = forms.ChoiceField(label="Warna biji hijau (green bean)", widget=forms.Select(attrs=SEL))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ch = get_metadata()["choices"]
        self.fields["country"].choices = _choices(ch["Country of Origin"])
        self.fields["variety"].choices = _choices(ch["Variety"])
        self.fields["processing_method"].choices = _choices(ch["Processing Method"])
        self.fields["color"].choices = _choices(ch["Color"])

    def to_model_inputs(self):
        """Ubah cleaned_data menjadi dict berkunci nama kolom training."""
        d = self.cleaned_data
        return {
            "Altitude": d["altitude"], "Moisture Percentage": d["moisture"],
            "Category One Defects": d["category_one_defects"],
            "Category Two Defects": d["category_two_defects"], "Quakers": d["quakers"],
            "Country of Origin": d["country"], "Variety": d["variety"],
            "Processing Method": d["processing_method"], "Color": d["color"],
        }

    def out_of_range_warnings(self):
        """Peringatan (bukan error) bila input di luar rentang data latih."""
        meta = get_metadata()["numeric_ranges"]
        d = self.to_model_inputs()
        warn = []
        for col, label in [("Altitude", "Ketinggian"), ("Moisture Percentage", "Kadar air"),
                           ("Category Two Defects", "Cacat kategori 2"), ("Quakers", "Quakers"),
                           ("Category One Defects", "Cacat kategori 1")]:
            lo, hi = meta[col]
            if not lo <= d[col] <= hi:
                warn.append(f"{label} ({d[col]:g}) di luar rentang data latih ({lo:g} - {hi:g}); estimasi kurang andal.")
        return warn
