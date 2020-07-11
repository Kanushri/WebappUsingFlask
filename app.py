import flask
import pickle
import pandas as pd

with open(f'model/bike_model_xgboost.pkl', 'rb') as f:model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
#@app.route('/')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
    	return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        Principal = flask.request.form['Principal']
        terms = flask.request.form['terms']
        age = flask.request.form['age']
        Gender = flask.request.form['Gender']
        weekend= flask.request.form['weekend']
        Bachelor= flask.request.form['Bachelor']
        HighSchoolorBelow= flask.request.form['HighSchoolorBelow']
        college=flask.request.form['college']
        
        input_variables = pd.DataFrame([[Principal, terms, age,Gender,weekend,Bachelor,HighSchoolorBelow,college]],
        columns=['Principal', 'terms', 'age','Gender','weekend','Bachelor','HighSchoolorBelow','college'],
                                       dtype=int)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Principal':Principal,
                                                     'terms':terms,
                                                            'age':age,
                                                     'Gender':Gender, 
                                                     'weekend':weekend,
                                                     'Bachelor':Bachelor,
                                                     'HighSchoolorBelow':HighSchoolorBelow,
                                                     'college':college},                                     
                                                     result=prediction,)



#def main():
    #return(flask.render_template('main.#html'))
if __name__ == '__main__':
    app.run()
    
    
    