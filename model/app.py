import flask
import pickle

with open(f"bike_model_xgboost.pkl", 'rb') as f:model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
#@app.route('/')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
    	return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        temperature = flask.request.form['temperature']
        humidity = flask.request.form['humidity']
        windspeed = flask.request.form['windspeed']
        '''Gender = flask.request.form['Gender']
        weekend= flask.request.form['weekend']
        bachelor= flask.request.form['bachelor']
        HighSchoolorBelow=form['HighSchoolorBelow']
        college= flask.request.form['college']'''
        
        #input_variables = pd.DataFrame([[temperature, humidity, windspeed,Gender,weekend,Bachelor,High School \or Below,college]],
        input_variables = pd.DataFrame([[temperature, humidity, windspeed]],
                                       #columns=['temperature', 'humidity', 'windspeed','Gender','weekend','Bachelor','High School or Below','college'],
                                       columns=['temperature', 'humidity', 'windspeed'],
                                       
                                       dtype=int64)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('main.html',
                                     original_input={'Temperature':temperature,
                                                     'Humidity':humidity,
                                                     'Windspeed':windspeed,
                                                     #'GENDER':gender, 
                                                     #'WEEKEND':weekend,
                                                     #'Bachelor':bachelor,
                                                   #'HighSchool'=HighSchoolorBelow,
                                                   #'College'=college
                                                   },                                     
                                                     result=prediction,)



#def main():
    #return(flask.render_template('main.#html'))
if __name__ == '__main__':
    app.run()
    
    
    