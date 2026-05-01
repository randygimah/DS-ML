import requests
result = requests.get(r"http://127.0.0.1:8000/predict/5?ds=BARCELONA&vc=Turista&fr=Promo&dur=160")
eval(result.content.decode())['model_out']
