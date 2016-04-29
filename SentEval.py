# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify
from flask_wtf import Form
from wtforms import StringField

from process_restaurant_request import main
import process_restaurant_request as proc

# BEGIN: UGLY MONKEYPATCH
import pkgutil
orig_get_loader = pkgutil.get_loader
def get_loader(name):
    try:
        return orig_get_loader(name)
    except AttributeError:
        pass
pkgutil.get_loader = get_loader
# END: UGLY MONKEYPATCH


class RestaurantForm(Form):
    restaurant = StringField('restaurant')


app = Flask(__name__)
app.secret_key = 'so secret'


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    form = RestaurantForm()

    if form.validate_on_submit():
        result = main(form.restaurant.data)
        return render_template('index.html', result=result)

    return render_template('index.html', form=form)

@app.route('/process')
def api():
    return jsonify(proc.main(request.args.get('name', '')))


if __name__ == '__main__':
    app.run()
