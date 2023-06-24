from flask import Flask, render_template, request
from event_predict import localhost
import pandas as pd

full_data = pd.read_csv('data/ga_sessions.csv')
example_data = pd.read_csv('data/ga_sessions.csv').iloc[:10].to_html()
app = Flask(__name__, template_folder='templates')



@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        number = request.form['nm']
        value = int(str((localhost(int(number))))[1])
        row = full_data.iloc[value]
        return render_template('action.html', value=value,row=row)
    else:
        return render_template('form.html',data=example_data)
if __name__ == '__main__ ':
    app.run()