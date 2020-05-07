#create flask application

#flask imports
from flask import Flask,session, url_for, redirect, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

#model_import
from deployment import return_prediction


app= Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class FlowerForm(FlaskForm):

    sep_len = StringField("Sepal Length")
    sep_wid = StringField("Sepal Width")
    pet_len = StringField("Petal Length")
    pet_wid = StringField("Petal Width")
    submit = SubmitField("Analyze")



@app.route('/',methods=['GET', 'POST']) #run on home page
def index():
    form = FlowerForm()
    if form.validate_on_submit():
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data

        return redirect(url_for("prediction"))
    return render_template('home.html', form=form)




@app.route('/prediction')
def prediction():
    content = {} #create empty content dictionary
    content['sepal_length'] = float(session['sep_len'])
    content['sepal_width'] = float(session['sep_wid'])
    content['petal_length'] = float(session['pet_len'])
    content['petal_width'] = float(session['pet_wid'])

    results = return_prediction(json=content) #predict the flower

    return render_template('prediction.html', results=results)


if __name__=='__main__':
    app.run(debug=True)




