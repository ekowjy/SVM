from flask import Flask, render_template, request

app = Flask(__name__)

# Dataset sampel untuk visualisasi di dashboard
dataset = [
    {"id": 1, "total": 1702743, "items": 20, "age": 64, "hour": 10, "weekend": 1, "label": 1},
    {"id": 2, "total": 4314572, "items": 25, "age": 31, "hour": 8, "weekend": 1, "label": -1},
    {"id": 3, "total": 4936484, "items": 26, "age": 56, "hour": 6, "weekend": 0, "label": -1},
    {"id": 4, "total": 1200000, "items": 15, "age": 45, "hour": 12, "weekend": 0, "label": 1},
    {"id": 5, "total": 6000000, "items": 40, "age": 28, "hour": 19, "weekend": 1, "label": -1},
    {"id": 6, "total": 800000, "items": 10, "age": 70, "hour": 9, "weekend": 0, "label": 1},
    {"id": 7, "total": 3500000, "items": 22, "age": 33, "hour": 15, "weekend": 1, "label": -1},
    {"id": 8, "total": 1500000, "items": 18, "age": 52, "hour": 11, "weekend": 0, "label": 1},
    {"id": 9, "total": 2800000, "items": 30, "age": 40, "hour": 16, "weekend": 0, "label": -1},
    {"id": 10, "total": 950000, "items": 12, "age": 58, "hour": 10, "weekend": 1, "label": 1},
]

# Konstanta SVM berdasarkan perhitungan manual
W = [-2.6119, -5.0, 33.0, 2.0, 0.0]
B = -1465.14

def classify(features):
    """
    Menghitung f(x) = w . x + b
    features: [total, items, age, hour, weekend]
    """
    # Tahap 1: Normalisasi Fitur 'Total' (Dibagi 10^6)
    total_norm = features[0] / 1e6
    
    # Tahap 2: Dot Product (Perkalian Vektor w dan x)
    # Kita menggunakan indeks manual agar mahasiswa paham urutannya
    dot_product = (W[0] * total_norm) + \
                  (W[1] * features[1]) + \
                  (W[2] * features[2]) + \
                  (W[3] * features[3]) + \
                  (W[4] * features[4])
    
    # Tahap 3: Fungsi Keputusan
    score = dot_product + B
    label = "Loyal (+1)" if score >= 0 else "Reguler (-1)"
    
    return label, round(score, 4)

@app.route('/')
def dashboard():
    return render_template('dashboard.html', dataset=dataset)

@app.route('/langkah')
@app.route('/langkah/<int:sample_id>')
def langkah(sample_id=1):
    # Cari data berdasarkan ID, default ke ID 1 jika tidak ditemukan
    selected_sample = next((d for d in dataset if d['id'] == sample_id), dataset[0])
    return render_template('steps.html', w=W, b=B, sample=selected_sample, dataset=dataset)

@app.route('/pipeline')
def pipeline():
    return render_template('pipeline.html')

@app.route('/evaluasi')
def evaluasi():
    results = []
    correct = 0
    for d in dataset:
        # Normalisasi Total
        x_total = d['total'] / 1000000
        
        # Hitung Komponen Parsial
        p1 = W[0] * x_total
        p2 = W[1] * d['items']
        p3 = W[2] * d['age']
        p4 = W[3] * d['hour']
        p5 = W[4] * d['weekend']
        
        dot_product = p1 + p2 + p3 + p4 + p5
        final_score = dot_product + B
        
        pred_label = 1 if final_score >= 0 else -1
        is_correct = (pred_label == d['label'])
        if is_correct:
            correct += 1
            
        results.append({
            "id": d['id'],
            "actual": d['label'],
            "predicted": pred_label,
            "score": final_score,
            "is_correct": is_correct,
            "components": {
                "total": p1,
                "items": p2,
                "age": p3,
                "hour": p4,
                "weekend": p5
            },
            "features": d
        })
    
    accuracy = (correct / len(dataset)) * 100
    return render_template('evaluation.html', results=results, accuracy=accuracy, W=W, B=B)

@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    result = None
    input_data = None
    if request.method == 'POST':
        input_data = [
            float(request.form['total']),
            float(request.form['items']),
            float(request.form['age']),
            float(request.form['hour']),
            float(request.form['weekend'])
        ]
        result = classify(input_data)
    return render_template('predict.html', result=result, input_data=input_data)

if __name__ == '__main__':
    app.run(debug=True)