from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/fire-alert', methods=['POST'])
def receive_fire_alert():
    data = request.json
    print("Alerta de incêndio recebido:")
    print(data)
    return jsonify({
        "alert": data
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
