from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

if __name__ == '__main__':
    print(f"Starting web server!")
    app.run(debug=False, host='0.0.0.0', port=5000)
