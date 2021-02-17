import requests

'''
your localhost url. If running on port 5000
'''
url = "http://localhost:5000/process"
filess = {"img": open("D:/DATASET/TEST/R/R_10008.jpg", "rb")}
results = requests.post(url, files=filess)
print(results.text)