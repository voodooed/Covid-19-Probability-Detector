from flask import Flask, render_template,request
app = Flask(__name__)
import pickle

  # open a file, where you ant to store the data
file = open(('model.pkl', 'rb'))
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET","POST"])
def hello_world():
  if request.method == "POST":
    myDict = request.form
    fever = int(myDict['fever'])
    age = int(myDict['age'])
    pain = int(myDict['pain'])
    runnyNose = int(myDict['runnyNose'])
    diffBreath = int(myDict['diffBreath'])


    # Code for Inference
    inputfeatures = [[fever,pain,age,runnyNose,diffBreath]]
    infectProb = clf.predict_proba(inputfeatures)[0,1]
    print(infectProb)
    return render_template('show.html',inf=round(infectProb*100))
    
      
  return render_template('index.html')
    
  #return 'Hello, World!' + str(infectProb)

if __name__ == "__main__":
    app.run(debug=True)