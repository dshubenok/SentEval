from flask import Flask, request, jsonify
import process_restraune_request as proc

app = Flask(__name__)


@app.route('/process')
def hello_world():
    return jsonify(proc.main(request.args.get('name', '')))

if __name__ == '__main__':
    app.run()

