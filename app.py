from flask import Flask, request, jsonify
import joblib  


app = Flask(__name__)

model1 = joblib.load('trained_model_symptom.pkl')  
vectorizer = joblib.load('vectorizer_symptom.pkl')  
model2 = joblib.load('disease_prediction_model.pkl')
encoder=joblib.load('label_encoder.pkl')

@app.route('/predict_diseasem', methods=['POST'])
def predict_medical_term():
    try:
        input_synonyms = request.get_json(force=True)  
        
        if not isinstance(input_synonyms["symptoms"], list) or not input_synonyms["symptoms"]:
            return jsonify({"error": "Invalid input. Provide a list of synonyms."}), 400
        
        prediction = []
        
        for item in input_synonyms["symptoms"]:
            vectorized = vectorizer.transform([item])
            predict = model1.predict(vectorized)
            prediction.append(predict)
        #input_vectorized = vectorizer.transform(input_synonyms["symptoms"])
        #prediction = model1.predict(input_vectorized)
        #predicted_medical_term = prediction.tolist()
        
        #predicted_medical_term = prediction.tolist()
        
        try:
            encoded_terms = encoder.transform(prediction)  
        except Exception as e:
            return jsonify({"error": f"Encoding failed: {str(e)}"}), 500
        
        diseases_prediction = model2.predict(encoded_terms)

        return jsonify({
            "predicted_diseases": diseases_prediction.tolist()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
