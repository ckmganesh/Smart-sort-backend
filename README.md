# Smart-sort-backend

This is the backend for the SmartSort App built using python and deployed in heroku.

The model is trained using the Kaggle dataset whose link is as follows

```
https://www.kaggle.com/techsash/waste-classification-data
```
The code for the neural network model is not been shared but you can access the h5 file to deploy it locally.

The backend is live in the following link

```
https://smart-sort.herokuapp.com/
```
As of now the link is only for the API service but html and css would be added shortly.

If you want to test the functionality without deploying it locally change the PATH file in test.py to your image path and run the following command

```
python test.py
```
If you want to deploy it locally follow the following steps

1.Clone/Download this repository
2.Run the following commands
```
pip install -r requirements.txt
```
```
python app.py
```
3.Change the url in test.py to http://localhost:5000/process and change image path accordingly,then run the following command

```
python test.py
```
If you face any problem feel free to open an issue.


## Contributors

<a href="https://github.com/ckmganesh">Ganesh C K M</a><br>
<a href="https://github.com/siddharth1010">Siddharth M Nair</a><br>
<a href="https://github.com/ramcalm">Ram Gunasekaran A</a><br>
<a href="https://github.com/ssabarish2000">Sabarish S</a><br>
